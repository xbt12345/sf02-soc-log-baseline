"""Official-only v3.2 preparation with disk-backed unique-input cache. No fit/split."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback
import zlib

import soc_v3_prepare as old
import soc_v32_prepare as prep


def cached_item(item):
    h, raw = item
    result = prep.prepare_record({"message_sanitized": raw})
    data = zlib.compress(json.dumps({"raw": raw, "prepared": result}, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 1)
    return h, data


def run(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    train = Path(args.train)
    config = {"version": prep.VERSION, "stage": "L1 full preparation; split and fit require later checks",
              "train": str(train.resolve()), "input_sha256": old.file_hash(train),
              "workers": args.workers, "batch_size": 512, "seed": 20260912,
              "cache": "SQLite zlib unique raw strings, complete input key includes bound source/config hashes; no label-conditioned parsing",
              "row_weight": 1, "external_data": False, "python": sys.version}
    if config["input_sha256"] != old.EXPECTED_SHA:
        raise ValueError("Official training file identity mismatch")
    sources = {}
    for filename in ["soc_v3_prepare.py", "soc_v32_prepare.py", "run_v32_prepare.py"]:
        shutil.copyfile(Path(__file__).with_name(filename), out / filename)
        sources[filename] = old.file_hash(out / filename)
    config["source_sha256"] = sources
    old.save(out / "configuration.json", config)
    n = pq.ParquetFile(train).metadata.num_rows
    labels = np.empty(n, dtype=np.uint8)
    groups = np.empty(n, dtype=np.int32)
    informative = np.empty(n, dtype=bool)
    group_map, next_group = {}, 0
    db = sqlite3.connect(str(out / "input_cache.sqlite"))
    db.execute("PRAGMA cache_size=-16384")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("CREATE TABLE inputs (h BLOB PRIMARY KEY, data BLOB NOT NULL) WITHOUT ROWID")
    coverage, residuals, fields, counters = [collections.Counter() for _ in range(4)]
    examples = []
    residual_examples = []
    example_keys = set()
    pool = None
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=args.workers)
    schema = pa.schema([(k,t) for k,t in [
        ("row_position",pa.int32()),("event_id",pa.string()),("label",pa.string()),
        ("product",pa.string()),("format",pa.string()),("group_id",pa.int32()),
        ("original_empty",pa.bool_()),("filtered_empty",pa.bool_()),("unknown_format",pa.bool_()),
        ("text",pa.string()),("diagnostic_text",pa.string()),("raw_hash",pa.binary(32)),
        ("removed_field_count",pa.int32()),("upstream_field_count",pa.int32()),
        ("unverified_field_count",pa.int32()),("invalid_string_values",pa.int32())]])
    writer = pq.ParquetWriter(out / "prepared_corpus.parquet", schema, compression="zstd")
    offset = 0
    last_progress = started
    try:
        for batch in pq.ParquetFile(train).iter_batches(batch_size=512, columns=["event_id","message_sanitized","product_name","label_binary"], use_threads=False):
            rows = batch.to_pylist()
            raw_by_hash = {hashlib.sha256(old.string(r["message_sanitized"]).encode()).digest():old.string(r["message_sanitized"]) for r in rows}
            keys = list(raw_by_hash)
            found = dict(db.execute("SELECT h,data FROM inputs WHERE h IN ("+",".join("?" for _ in keys)+")", keys))
            missing = [(h,raw_by_hash[h]) for h in keys if h not in found]
            new_items = list(pool.map(cached_item, missing, chunksize=16)) if pool else list(map(cached_item, missing))
            db.executemany("INSERT INTO inputs VALUES (?,?)", new_items)
            found.update(new_items)
            counters["unique_parses"] += len(missing)
            parsed = {}
            for h, data in found.items():
                item = json.loads(zlib.decompress(data))
                if item["raw"] != raw_by_hash[h]:
                    raise AssertionError("Cache raw identity mismatch")
                parsed[h] = item["prepared"]
            output = []
            for row in rows:
                raw = old.string(row["message_sanitized"])
                h = hashlib.sha256(raw.encode()).digest()
                p = parsed[h]
                text = p["text"]
                key = prep.group_key(raw,p,offset)
                if p["original_empty"]:
                    g = next_group
                    next_group += 1
                else:
                    g = group_map.get(key)
                    if g is None:
                        g = next_group
                        next_group += 1
                        group_map[key] = g
                label = row["label_binary"]
                labels[offset], groups[offset], informative[offset] = old.LABELS.index(label),g,bool(text)
                source = old.string(row["product_name"]) or "<missing>"
                coverage[(source,p["format"],label,p["original_empty"],p["filtered_empty"],p["unknown_format"])] += 1
                counters["original_empty"] += p["original_empty"]
                counters["filtered_empty"] += p["filtered_empty"]
                counters["unknown_format"] += p["unknown_format"]
                counters["invalid_string_rows"] += p["invalid_string_values_preserved"] > 0
                counters["unverified_field_rows"] += p["unverified_fields_isolated"] > 0
                for s in p["removed_spans"]:
                    fields[(s["reason"],s["path"])] += 1
                for name, pattern in [("redaction_marker",old.MARKER),("ipv4_pattern",prep.IP_CANDIDATE),("iso_date",old.ISO)]:
                    hits = list(pattern.finditer(text))
                    residuals[name] += bool(hits)
                    if hits and len(residual_examples) < 60:
                        residual_examples.append({"row_position":offset,"pattern":name,"format":p["format"],"contexts":[text[max(0,m.start()-70):m.end()+70] for m in hits[:3]]})
                example_key = (source,p["format"],label)
                if example_key not in example_keys:
                    example_keys.add(example_key)
                    examples.append({"row_position":offset,"raw_hash":h.hex(),"raw_message":raw,**p})
                output.append({"row_position":offset,"event_id":old.string(row["event_id"]),"label":label,
                    "product":source,"format":p["format"],"group_id":g,"original_empty":p["original_empty"],
                    "filtered_empty":p["filtered_empty"],"unknown_format":p["unknown_format"],
                    "text":text,"diagnostic_text":p["diagnostic_text"],"raw_hash":h,
                    "removed_field_count":len(p["removed_spans"]),"upstream_field_count":p["upstream_fields_removed"],
                    "unverified_field_count":p["unverified_fields_isolated"],"invalid_string_values":p["invalid_string_values_preserved"]})
                offset += 1
            writer.write_table(pa.Table.from_pylist(output,schema=schema))
            db.commit()
            if time.perf_counter()-last_progress >= 20:
                last_progress = time.perf_counter()
                status = {"rows":offset,"total":n,"unique_parses":counters["unique_parses"],"elapsed_seconds":last_progress-started,"parent_peak_bytes":old.peak_memory_bytes()}
                old.save(out / "progress.json",status)
                print(json.dumps(status),flush=True)
    except Exception as e:
        old.save(out / "failure.json", {"rows_completed":offset,"error":str(e),"traceback":traceback.format_exc()})
        raise
    finally:
        writer.close()
        db.close()
        if pool:
            pool.shutdown(wait=True)
    if offset != n:
        raise AssertionError("Row coverage incomplete")
    del group_map
    counts = np.bincount(groups[informative]*3+labels[informative],minlength=next_group*3).reshape(-1,3)
    mixed = (counts>0).sum(axis=1)>1
    ids = np.flatnonzero(mixed)
    collisions = {"mixed_label_text_groups":len(ids),"rows_in_mixed_label_text_groups":int(counts[mixed].sum()),
      "minimum_observed_mistakes_deterministic_text_only":int((counts.sum(axis=1)-counts.max(axis=1)).sum()),
      "all_conflict_groups":[{"group_id":int(g),"counts":dict(zip(old.LABELS,map(int,counts[g])))} for g in ids],
      "scope":"Fixed prepared-text observation conflicts only; no relabeling, no universal error bound; empty inputs excluded"}
    pq.write_table(pa.table({"row_position":np.arange(n,dtype=np.int32),"group_id":groups,"label_index":labels,"informative":informative}),out / "group_manifest.parquet",compression="zstd")
    old.save(out / "text_collisions.json",collisions)
    old.save(out / "normalization_examples.json",examples)
    old.save(out / "residual_pattern_checks.json",{"counts":dict(residuals),"examples":residual_examples,"scope":"Pattern candidates only; IPv4-looking versions require context review"})
    old.save(out / "field_policy_coverage.json",[{"reason":k[0],"path":k[1],"rows":v} for k,v in sorted(fields.items())])
    old.save(out / "coverage_by_source_format.json",[{"product":k[0],"format":k[1],"label":k[2],"original_empty":k[3],"filtered_empty":k[4],"unknown_format":k[5],"rows":v} for k,v in sorted(coverage.items())])
    result={"status":"prepared_pending_independent_semantic_and_collision_review","version":prep.VERSION,"rows":n,
      "counters":dict(counters),"informative_rows":int(informative.sum()),"informative_groups":int((counts.sum(axis=1)>0).sum()),
      "mixed_label_text_groups":len(ids),"model_trained":False,"semantic_normalization_accepted":False,"transfer_validated":False,
      "elapsed_seconds":time.perf_counter()-started,"parent_peak_bytes":old.peak_memory_bytes(),
      "peak_scope":"Parent process high-water working set, excludes parser workers and other machine processes",
      "input_sha256":config["input_sha256"],"source_sha256":sources,
      "corpus_sha256":old.file_hash(out / "prepared_corpus.parquet"),"group_manifest_sha256":old.file_hash(out / "group_manifest.parquet")}
    old.save(out / "result.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--workers",type=int,default=2,choices=[1,2,3,4])
    run(p.parse_args())
