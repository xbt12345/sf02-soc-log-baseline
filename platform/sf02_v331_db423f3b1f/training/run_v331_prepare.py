"""Build v3.3.1 from bound official bytes and v3.2 cache, retaining all rows."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import soc_v3_prepare as base
import v331_prepare as parser
from audit_v32_prepare import asa_observed_facts

PARENT_SHA = "95c79304fae8e60af997821b2ed279635e091ae74ffac3e694da29d3747d6c10"
DEPENDENCIES = ["soc_v3_prepare.py", "soc_v32_prepare.py", "v32_finalize.py",
                "v331_prepare.py", "run_v331_prepare.py", "audit_v32_prepare.py"]


def run(args):
    out = Path(args.output_dir)
    parent = Path(args.parent_run)
    started = time.perf_counter()
    previous = json.loads((parent/"result.json").read_text(encoding="utf-8"))
    if base.file_hash(args.train) != base.EXPECTED_SHA:
        raise ValueError("Official data identity mismatch")
    if base.file_hash(parent/"prepared_corpus.parquet") != PARENT_SHA:
        raise ValueError("Frozen parent corpus mismatch")
    for name in DEPENDENCIES[:3]:
        if base.file_hash(Path(__file__).with_name(name)) != previous["source_sha256"][name]:
            raise ValueError("Frozen parser dependency changed: " + name)
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in DEPENDENCIES:
        shutil.copyfile(Path(__file__).with_name(name), out/name)
        sources[name] = base.file_hash(out/name)
    base.save(out/"configuration.json", {
        "version": parser.VERSION, "input_sha256": base.EXPECTED_SHA, "parent_corpus_sha256": PARENT_SHA,
        "source_sha256": sources, "all_labels_and_row_weights_preserved": True,
        "route_uses_message_only": True, "official_only": True, "model_trained": False,
        "scope": "Incremental scoped input repair and complete regrouping. Not label-truth validation.",
    })
    pfile = pq.ParquetFile(parent/"prepared_corpus.parquet")
    rawfile = pq.ParquetFile(args.train)
    n = rawfile.metadata.num_rows
    assert pfile.metadata.num_rows == n == 2056871
    extra = [pa.field("repair_route", pa.string()), pa.field("asa_template", pa.string()),
             pa.field("boundary_verified", pa.bool_()), pa.field("authentication_result_unknown", pa.bool_()),
             pa.field("no_observable_auth_facts", pa.bool_()), pa.field("repair_spans", pa.string()),
             pa.field("auth_facts", pa.string()), pa.field("retained_body_sha256", pa.string())]
    schema = pa.schema(list(pfile.schema_arrow) + extra)
    y = np.empty(n, np.uint8); groups = np.empty(n, np.int32); info = np.empty(n, bool)
    oldgroups = np.empty(n, np.int32)
    next_group = 0; groupmap = {}; changed_cache = {}; nochange = set()
    offset = 0; counts = collections.Counter(); routes = collections.Counter()
    field_counts = collections.Counter(); changes = []; auth_reviews = []
    examples = []; coverage = collections.Counter(); facts_errors = []
    raw_iter = rawfile.iter_batches(batch_size=4096, columns=["event_id", "label_binary", "message_sanitized"], use_threads=False)
    parent_iter = pfile.iter_batches(batch_size=4096, use_threads=False)
    with pq.ParquetWriter(out/"prepared_corpus.parquet", schema, compression="zstd") as writer:
        for rb, pb in zip(raw_iter, parent_iter):
            rawrows = rb.to_pylist(); rows = pb.to_pylist()
            assert len(rawrows) == len(rows)
            for rawrow, row in zip(rawrows, rows):
                raw = base.string(rawrow["message_sanitized"])
                h = hashlib.sha256(raw.encode("utf-8")).digest()
                assert row["row_position"] == offset and row["raw_hash"] == h
                assert row["event_id"] == str(rawrow["event_id"]) and row["label"] == rawrow["label_binary"]
                oldtext = row["text"]; oldgroups[offset] = row["group_id"]
                p = changed_cache.get(h)
                if p is None and h not in nochange:
                    which = parser.route(raw)
                    if which == "unchanged":
                        nochange.add(h)
                    else:
                        p = parser.prepare_record({"message_sanitized": raw})
                        assert p["route"] == which
                        changed_cache[h] = p
                which = p["route"] if p else "unchanged"
                routes[(row["product"], row["format"], which)] += 1
                if row["format"] == "asa_like" and which != "asa":
                    raise AssertionError("Known ASA boundary not handled at " + str(offset))
                if row["product"] == "Duo" and which != "authentication":
                    raise AssertionError("Known Duo object not handled at " + str(offset))
                row.update(repair_route=which, asa_template=None, boundary_verified=False,
                           authentication_result_unknown=False, no_observable_auth_facts=False,
                           repair_spans="[]", auth_facts="{}", retained_body_sha256=None)
                if p:
                    for name in ["text", "original_empty", "filtered_empty", "unknown_format"]:
                        row[name] = p[name]
                    row.update(removed_field_count=len(p["removed_spans"]),
                               upstream_field_count=p["upstream_fields_removed"],
                               unverified_field_count=p["unverified_fields_isolated"],
                               invalid_string_values=p["invalid_string_values_preserved"],
                               final_quarantined_fragments=len(p["final_quarantined_fragments"]),
                               final_quarantine_spans=json.dumps(p["final_quarantined_fragments"], ensure_ascii=False),
                               asa_template=p["asa_template"], boundary_verified=p["boundary_verified"],
                               authentication_result_unknown=p["authentication_result_unknown"],
                               no_observable_auth_facts=p["no_observable_auth_facts"],
                               repair_spans=json.dumps(p["repair_spans"], ensure_ascii=False),
                               auth_facts=json.dumps(p["auth_facts"], ensure_ascii=False),
                               retained_body_sha256=p["retained_body_sha256"])
                    field_counts.update(s["reason"]+":"+s.get("path", "prefix") for s in p["repair_spans"])
                    if which == "asa":
                        if asa_observed_facts(raw) != asa_observed_facts(row["text"]):
                            facts_errors.append(offset)
                    if which == "authentication":
                        auth_reviews.append({"row_position":offset, "raw_hash":h.hex(),
                            "facts":p["auth_facts"], "fields":p["repair_spans"],
                            "result_unknown":p["authentication_result_unknown"],
                            "status":"observable_auth_facts_only_not_attack_label_confirmation"})
                    if len(examples) < 12 and which == "asa":
                        examples.append({"name":"asa_real_"+str(offset), "kind":"invariance",
                            "raw_a":raw, "raw_b":"<164>Sep 12 2026 12:00:00: USER-9999 "+parser.asa_parts(raw)["body"]})
                text = row["text"]
                if text != oldtext:
                    counts["changed_rows"] += 1
                    changes.append((offset, which, hashlib.sha256(oldtext.encode()).hexdigest(),
                                    hashlib.sha256(text.encode()).hexdigest()))
                if not p:
                    assert text == oldtext
                    counts["copied_unchanged_rows"] += 1
                if row["original_empty"]:
                    gid = next_group; next_group += 1
                else:
                    key = b"text:" + hashlib.sha256(text.encode()).digest() if text else b"raw:" + h
                    gid = groupmap.get(key)
                    if gid is None:
                        gid = next_group; next_group += 1; groupmap[key] = gid
                row["group_id"] = gid
                y[offset] = base.LABELS.index(row["label"]); groups[offset] = gid; info[offset] = bool(text)
                coverage[(row["product"], row["format"], row["label"], which)] += 1
                offset += 1
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            if offset % (4096*32) == 0:
                print(json.dumps({"stage":"prepare","rows":offset,"total":n,
                    "changed":counts["changed_rows"],"seconds":round(time.perf_counter()-started,1)}), flush=True)
    assert offset == n
    if facts_errors:
        raise AssertionError("Visible ASA facts changed: " + str(facts_errors[:20]))
    pq.write_table(pa.table({"row_position":np.arange(n,dtype=np.int32),"group_id":groups,
        "label_index":y,"informative":info}), out/"group_manifest.parquet", compression="zstd")
    pq.write_table(pa.table({k:[c[i] for c in changes] for i,k in enumerate(
        ["row_position","route","old_text_sha256","new_text_sha256"])}),out/"changed_rows.parquet",compression="zstd")
    c = np.bincount(groups[info].astype(np.int64)*3+y[info], minlength=next_group*3).reshape(-1,3)
    mixed = (c>0).sum(axis=1)>1
    ids = np.flatnonzero(mixed)
    base.save(out/"text_collisions.json", {
        "mixed_label_text_groups":int(mixed.sum()),"rows_in_mixed_label_text_groups":int(c[mixed].sum()),
        "minimum_observed_mistakes_deterministic_text_only":int((c.sum(axis=1)-c.max(axis=1)).sum()),
        "all_conflict_groups":[{"group_id":int(g),"counts":dict(zip(base.LABELS,map(int,c[g])))} for g in ids],
        "scope":"Observable representation conflict, not confirmed label noise; no labels changed.",
    })
    base.save(out/"repair_coverage.json", {"routes":[{"product":p,"format":f,"route":r,"rows":n}
        for (p,f,r),n in sorted(routes.items())], "field_counts":dict(field_counts),
        "visible_ASA_fact_mismatches":facts_errors, "counts":dict(counts)})
    base.save(out/"auth_field_review.json",auth_reviews)
    prior_pairs = json.loads((parent/"model_stress_fixtures.json").read_text(encoding="utf-8"))
    base.save(out/"model_stress_fixtures.json",prior_pairs+examples)
    base.save(out/"coverage_by_source_format.json",[dict(product=p,format=f,label=l,route=r,rows=n)
        for (p,f,l,r),n in sorted(coverage.items())])
    result = {"status":"prepared_pending_full_audit","version":parser.VERSION,"rows":n,
        "class_counts":dict(zip(base.LABELS,map(int,np.bincount(y,minlength=3)))),
        "groups":next_group,"informative_rows":int(info.sum()),"counts":dict(counts),
        "input_sha256":base.EXPECTED_SHA,"parent_corpus_sha256":PARENT_SHA,"source_sha256":sources,
        "corpus_sha256":base.file_hash(out/"prepared_corpus.parquet"),
        "group_manifest_sha256":base.file_hash(out/"group_manifest.parquet"),
        "model_stress_fixtures_sha256":base.file_hash(out/"model_stress_fixtures.json"),
        "model_trained":False,"transfer_validated":False,"elapsed_seconds":time.perf_counter()-started}
    base.save(out/"result.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",required=True);p.add_argument("--parent-run",required=True);p.add_argument("--output-dir",required=True)
    run(p.parse_args())

