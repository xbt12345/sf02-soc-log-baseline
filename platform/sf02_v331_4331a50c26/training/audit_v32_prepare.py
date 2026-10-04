"""Independent row/evidence checks for v3.2 preparation. Never trains a model."""
import argparse
import collections
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import sqlite3
import time
import zlib

import numpy as np
import pyarrow.parquet as pq


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8388608),b""):
            h.update(b)
    return h.hexdigest()


def save_new(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def asa_observed_facts(text):
    """Separate narrow parser: only directly visible action/roles/ports/codes."""
    m=re.search(r'\b(deny|permit)\s+(tcp|udp|icmp)\s+src\s+([^: ]+):(\S+)\s+dst\s+([^: ]+):(\S+)',text,re.I)
    if not m:return None
    def port(endpoint):
        if '/' not in endpoint:return None
        value=endpoint.rsplit('/',1)[1]
        return int(value) if value.isdigit() and 0<=int(value)<=65535 else 'unknown'
    icmp=re.search(r'\(type\s+(\d+),\s*code\s+(\d+)\)',text,re.I)
    return (m[1].lower(),m[2].lower(),m[3].lower(),port(m[4]),m[5].lower(),port(m[6]),
      tuple(map(int,icmp.groups())) if icmp else None,tuple(re.findall(r'0x[0-9a-f]+',text,re.I)))


def audit(args):
    started=time.perf_counter()
    run=Path(args.run_dir)
    result=json.loads((run/"result.json").read_text(encoding="utf-8"))
    checks={"official_train_identity":sha(args.train)==result["input_sha256"],
            "prepared_identity":sha(run/"prepared_corpus.parquet")==result["corpus_sha256"],
            "group_manifest_identity":sha(run/"group_manifest.parquet")==result["group_manifest_sha256"]}
    gm=pq.read_table(run/"group_manifest.parquet")
    labels=gm["label_index"].to_numpy(); groups=gm["group_id"].to_numpy(); informative=gm["informative"].to_numpy()
    checks["group_manifest_row_order"]=np.array_equal(gm["row_position"].to_numpy(),np.arange(len(labels)))
    collisions=json.loads((run/"text_collisions.json").read_text(encoding="utf-8"))
    conflict_ids={v["group_id"] for v in collisions["all_conflict_groups"]}
    raw_groups={}; prepared_groups={}; class_counts=collections.Counter(); errors=[]
    ip_examples=[]; ip_rows=0; valid_ip_rows=0; upstream_examples=[]; upstream_rows=0
    conflict_rows=collections.defaultdict(list); sample_counts=[]; source_support=collections.Counter()
    conflict_fact_errors=[]; conflict_fact_rows=0; conflict_signatures=collections.defaultdict(set)
    raw_iter=pq.ParquetFile(args.train).iter_batches(batch_size=512,columns=["event_id","label_binary","message_sanitized"],use_threads=False)
    out_iter=pq.ParquetFile(run/"prepared_corpus.parquet").iter_batches(batch_size=512,use_threads=False)
    labels_order=["benign","malicious","suspicious"]
    offset=0; word=re.compile(r"(?u)\b\w+\b")
    # Different candidate detector from normalization: numeric runs, then actual IPv4 parsing.
    ip_scan=re.compile(r"(?<![0-9.])[0-9]+(?:\.[0-9]+){3}(?![0-9.])")
    upstream_scan=re.compile(r'"(?:tactic_id|technique_id|behavior_id|display_name|detection_id|parent_md5|parent_sha256|behaviors_processed)"\s*:',re.I)
    for rb,ob in zip(raw_iter,out_iter):
        raw_rows=rb.to_pylist(); out_rows=ob.to_pylist()
        if len(raw_rows)!=len(out_rows):
            raise AssertionError("Batch length mismatch")
        for raw,row in zip(raw_rows,out_rows):
            p=offset; g=int(groups[p]); text=row["text"]
            source_text="" if raw["message_sanitized"] is None else str(raw["message_sanitized"])
            raw_hash=hashlib.sha256(source_text.encode()).digest()
            if (row["row_position"]!=p or row["event_id"]!=str(raw["event_id"]) or row["label"]!=raw["label_binary"]
                or labels_order[int(labels[p])]!=row["label"] or row["group_id"]!=g or bool(text)!=bool(informative[p])
                or row["raw_hash"]!=raw_hash or row["original_empty"]!=(not source_text.strip())
                or row["filtered_empty"]!=(bool(source_text.strip()) and not bool(text))):
                errors.append(p)
            class_counts[row["label"]]+=1
            if source_text.strip():
                old=raw_groups.setdefault(raw_hash,g)
                if old!=g:
                    errors.append(p)
            if text:
                th=hashlib.sha256(text.encode()).digest()
                old=prepared_groups.setdefault(th,g)
                if old!=g:
                    errors.append(p)
            matches=list(ip_scan.finditer(text)); valid=[]
            for m in matches:
                try:
                    ipaddress.IPv4Address(m.group())
                    valid.append(m)
                except ipaddress.AddressValueError:
                    pass
            ip_rows+=bool(matches); valid_ip_rows+=bool(valid)
            if valid and len(ip_examples)<80:
                ip_examples.append({"row_position":p,"format":row["format"],"contexts":[text[max(0,m.start()-70):m.end()+70] for m in valid[:4]]})
            if upstream_scan.search(text):
                upstream_rows+=1
                if len(upstream_examples)<40:
                    upstream_examples.append({"row_position":p,"format":row["format"],"text":text})
            if g in conflict_ids:
                conflict_rows[g].append({"row_position":p,"label":row["label"],"raw_hash":raw_hash.hex(),"format":row["format"],"final_text":text})
                facts=asa_observed_facts(source_text);prepared_facts=asa_observed_facts(text)
                conflict_fact_rows+=1
                if facts is None or facts!=prepared_facts:conflict_fact_errors.append(p)
                conflict_signatures[g].add(facts)
            source_support[(row["product"],row["format"],row["label"])]+=1
            if int.from_bytes(hashlib.blake2b(str(p).encode(),digest_size=4).digest(),"little")%1024==0:
                tokens=word.findall(text)
                sample_counts.append(len(set(tokens))+len(set(zip(tokens,tokens[1:]))))
            offset+=1
    checks["complete_rows_and_original_content_alignment"]=not errors and offset==len(labels)==2056871
    checks["expected_class_counts"]=class_counts=={"benign":1899723,"malicious":111728,"suspicious":45420}
    checks["exact_raw_and_prepared_duplicates_stay_grouped"]=not errors
    counts=np.bincount(groups[informative]*3+labels[informative],minlength=(int(groups.max())+1)*3).reshape(-1,3)
    mixed=(counts>0).sum(axis=1)>1
    recomputed={"groups":int(mixed.sum()),"rows":int(counts[mixed].sum()),"fixed_text_min_errors":int((counts.sum(axis=1)-counts.max(axis=1)).sum())}
    checks["collision_counts_match"]=recomputed["groups"]==collisions["mixed_label_text_groups"] and recomputed["rows"]==collisions["rows_in_mixed_label_text_groups"] and recomputed["fixed_text_min_errors"]==collisions["minimum_observed_mistakes_deterministic_text_only"]
    cache_root=Path(result["parent_run"]) if "parent_run" in result else run
    repaired={}
    if (run/"root_repair_examples.json").exists():
        repaired={r["row_position"]:r["prepared"] for r in json.loads((run/"root_repair_examples.json").read_text(encoding="utf-8"))}
    db=sqlite3.connect("file:"+(cache_root/"input_cache.sqlite").resolve().as_posix()+"?mode=ro",uri=True)
    conflict_examples=[]
    for g,records in conflict_rows.items():
        representatives=[]; seen=set()
        for r in records:
            if r["label"] in seen:
                continue
            seen.add(r["label"])
            data=db.execute("SELECT data FROM inputs WHERE h=?",(bytes.fromhex(r["raw_hash"]),)).fetchone()
            item=json.loads(zlib.decompress(data[0]))
            prepared=repaired.get(r["row_position"],item["prepared"])
            representatives.append({**r,"raw_message":item["raw"],"text":r["final_text"],"removed_spans":prepared["removed_spans"]})
        conflict_examples.append({"group_id":g,"rows":len(records),"class_counts":dict(collections.Counter(r["label"] for r in records)),"representatives":representatives})
    db.close()
    save_new(run/"collision_review_examples.json",conflict_examples)
    save_new(run/"collision_fact_check.json",{"rows":conflict_fact_rows,"groups":len(conflict_signatures),
      "raw_to_prepared_fact_mismatches":conflict_fact_errors,
      "groups_with_multiple_checked_fact_signatures":[g for g,s in conflict_signatures.items() if len(s)!=1],
      "all_checked_facts_preserved":not conflict_fact_errors and all(len(s)==1 for s in conflict_signatures.values()),
      "checked_fields":"Action, protocol, src/dst interface direction, complete valid ports, ICMP type/code, visible hex codes",
      "limitations":"Partial redacted ports remain unknown. Narrow fact comparison is not full semantic equivalence, label correctness or missing-context proof."})
    report={"scope":"Independent original-row/hash/duplicate/collision verification and residual candidate scan, not semantic acceptance or model quality",
       "checks":checks,"all_integrity_checks_passed":all(checks.values()),"row_errors_first20":errors[:20],
       "rows":offset,"class_counts":dict(class_counts),"collision_counts":recomputed,
       "ip_scan":{"numeric_ipv4_like_rows":ip_rows,"valid_ipv4_candidate_rows":valid_ip_rows,"examples":ip_examples,"note":"Even valid IPv4 syntax can be a version; context requires review"},
       "upstream_named_property_candidates":{"rows":upstream_rows,"examples":upstream_examples,"note":"Quoted command/request fields require context review; these are not automatic truth"},
       "resource_sample":{"rows":len(sample_counts),"mean_word_1_2_nnz":float(np.mean(sample_counts)),
          "full_nnz_estimate_uncapped":float(np.mean(sample_counts)*offset),"note":"Resource sample only, no vocabulary fit or model; excludes typed punctuation tokens and solver peaks"},
       "elapsed_seconds":time.perf_counter()-started,"code_sha256":sha(__file__),"input_sha256":result["input_sha256"],"corpus_sha256":result["corpus_sha256"]}
    save_new(run/"independent_audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ["ip_scan","upstream_named_property_candidates"]},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",required=True)
    p.add_argument("--run-dir",required=True)
    audit(p.parse_args())
