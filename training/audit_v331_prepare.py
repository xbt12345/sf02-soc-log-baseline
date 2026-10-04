"""Full independent content/role checks before releasing v3.3.1 prepared inputs."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pyarrow.parquet as pq
import soc_v3_prepare as base
import v331_prepare as parser
from audit_v32_prepare import asa_observed_facts


def audit(args):
    out=Path(args.run_dir); parent=Path(args.parent_run)
    result=json.loads((out/"result.json").read_text(encoding="utf-8"))
    started=time.perf_counter()
    checks={"official_identity":base.file_hash(args.train)==base.EXPECTED_SHA,
      "corpus_identity":base.file_hash(out/"prepared_corpus.parquet")==result["corpus_sha256"],
      "groups_identity":base.file_hash(out/"group_manifest.parquet")==result["group_manifest_sha256"]}
    for name,value in result["source_sha256"].items():
        checks["source_"+name]=base.file_hash(Path(__file__).with_name(name))==value==base.file_hash(out/name)
    if not all(checks.values()):raise AssertionError(checks)
    gm=pq.read_table(out/"group_manifest.parquet")
    y=gm["label_index"].to_numpy();g=gm["group_id"].to_numpy();info=gm["informative"].to_numpy()
    n=len(y);offset=0;seen_text={};seen_raw={};cache={};counts=collections.Counter()
    conflicts=collections.defaultdict(set);errors=[];unknown_auth=[];template_labels=collections.defaultdict(collections.Counter)
    iters=[pq.ParquetFile(args.train).iter_batches(batch_size=4096,columns=["event_id","message_sanitized","label_binary"],use_threads=False),
           pq.ParquetFile(parent/"prepared_corpus.parquet").iter_batches(batch_size=4096,columns=["text"],use_threads=False),
           pq.ParquetFile(out/"prepared_corpus.parquet").iter_batches(batch_size=4096,use_threads=False)]
    for rawb,oldb,newb in zip(*iters):
        raws=rawb.to_pylist();olds=oldb.column(0).to_pylist();rows=newb.to_pylist()
        assert len(raws)==len(olds)==len(rows)
        for rawrow,oldtext,row in zip(raws,olds,rows):
            raw=base.string(rawrow["message_sanitized"]);h=hashlib.sha256(raw.encode()).digest()
            assert row["row_position"]==offset and row["event_id"]==str(rawrow["event_id"])
            assert row["raw_hash"]==h and row["label"]==rawrow["label_binary"]==base.LABELS[int(y[offset])]
            assert row["group_id"]==g[offset] and bool(row["text"])==bool(info[offset])
            route=row["repair_route"]
            expected="asa" if row["format"]=="asa_like" else "authentication" if row["product"]=="Duo" else "unchanged"
            assert route==expected,(offset,route,expected)
            counts[route]+=1
            if route=="unchanged":
                assert row["text"]==oldtext
            else:
                if h not in cache:
                    p=parser.prepare_record({"message_sanitized":raw})
                    assert p["route"]==route
                    if route=="asa":
                        parts=parser.asa_parts(raw)
                        native=parser.ASA_CODE.search(raw[:parts["body_start"]])
                        native=native.group() if native else ""
                        for prefix in ["<164>Sep 12 2026 12:00:00: USER-9999 ",
                                       "<180>2026-09-12T12:00:00Z collector.example ",
                                       "<180>Sep 12 USER-7777 12:00:00: USER-9999 "]:
                            variant=prefix+native+parts["body"]
                            assert parser.prepare_record({"message_sanitized":variant})["text"]==p["text"]
                        assert asa_observed_facts(raw)==asa_observed_facts(p["text"])
                        counts["unique_asa_header_invariance_checked"]+=1
                    else:
                        found=parser.auth_object(raw)
                        roots=found[3]
                        facts={}
                        for a,b,path,begin in roots:
                            if path[0] in parser.AUTH_FACTS:
                                try:value=json.loads(found[2][begin:b])
                                except ValueError:value=None
                                facts[path[0]]="unknown_value" if not isinstance(value,str) or base.MARKER.search(value) else value
                        assert facts==p["auth_facts"]
                        if found[4] is not None:counts["partial_identity_tail_unique"]+=1
                        if re.search(r"adaptive_trust|trust_level|model_version|txid|[A-Z0-9]{20}",p["text"],re.I):
                            raise AssertionError("Auth upstream/identity candidate in retained text")
                    cache[h]=p["text"]
                assert row["text"]==cache[h]
            if row["authentication_result_unknown"]:unknown_auth.append(offset)
            if not row["original_empty"]:
                key=hashlib.sha256(row["text"].encode()).digest() if row["text"] else b"raw"+h
                if key in seen_text:assert seen_text[key]==g[offset]
                else:seen_text[key]=int(g[offset])
                if h in seen_raw:assert seen_raw[h]==g[offset]
                else:seen_raw[h]=int(g[offset])
            if row["asa_template"]:
                template_labels[row["asa_template"]][row["label"]]+=1
            if row["repair_route"]=="asa":
                conflicts[int(g[offset])].add(asa_observed_facts(row["text"]))
            offset+=1
    assert offset==n==2056871
    assert all(len(s)==1 for s in conflicts.values())
    checks.update(all_rows_labels_preserved=True,unchanged_routes_equal_frozen=True,
        repaired_rows_replayed=True,known_ASA_and_Duo_full_coverage=True,
        same_nonempty_text_and_raw_message_same_group=True,
        checked_ASA_facts_preserved=True,all_ASA_header_variants_equal=True,
        auth_complete_scalar_facts_verified=True)
    c=np.bincount(g[info].astype(np.int64)*3+y[info],minlength=(int(g.max())+1)*3).reshape(-1,3)
    mixed=(c>0).sum(axis=1)>1
    collisions=json.loads((out/"text_collisions.json").read_text(encoding="utf-8"))
    assert int(mixed.sum())==collisions["mixed_label_text_groups"]
    assert int(c[mixed].sum())==collisions["rows_in_mixed_label_text_groups"]
    checks["conflicts_recomputed"]=True
    receipt={"scope":"Full input/content replay and narrow facts; not label truth or transfer acceptance.",
        "all_checks_passed":all(checks.values()),"checks":checks,"counts":dict(counts),
        "rows":n,"corpus_sha256":result["corpus_sha256"],"group_manifest_sha256":result["group_manifest_sha256"],
        "script_sha256":base.file_hash(__file__),"elapsed_seconds":time.perf_counter()-started,
        "unknown_auth_result_rows":len(unknown_auth),"asa_templates":dict(template_labels),
        "limitations":["Official class definitions give no per-event independent truth.",
          "Scalar extraction does not repair the 3 malformed identity tails.",
          "Text groups and row counts do not establish incident independence.",
          "Unknown formats outside ASA/authentication retain previous limitations."]}
    base.save(out/"preparation_audit.json",receipt)
    base.save(out/"semantic_review.json",{
        "eligible_for_domain_development":True,"corpus_sha256":result["corpus_sha256"],
        "preparation_audit_sha256":base.file_hash(out/"preparation_audit.json"),
        "semantic_normalization_accepted_as_pure_facts":False,
        "scope":"Allow registered official-data development with known input limits; no label corrections.",
        "model_trained":False,"transfer_validated":False})
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",required=True);p.add_argument("--parent-run",required=True);p.add_argument("--run-dir",required=True)
    audit(p.parse_args())
