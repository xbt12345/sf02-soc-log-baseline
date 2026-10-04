"""Final ambiguity quarantine on prepared text; preserves diagnostic evidence.

Composition, not a reparse: raw -> soc_v32_prepare -> this module. All rows are
processed, all normalized groups rebuilt. Offsets here refer to pre-final text.
"""
import argparse
import collections
import functools
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soc_v3_prepare as base
import soc_v32_prepare as prep

VERSION="v32-final-text-1.1"
QUAD=re.compile(r"(?<![0-9.])[0-9]+(?:\.[0-9]+){3}(?![0-9.])")
EXPLICIT_VERSION=re.compile(r"\b(?:version|file_version|product_version|build)\s*[:= ]\s*(?:\d+\.){2,}\d+",re.I)
COMPOUND_TAIL=re.compile(r"[0-9a-f]*:[0-9a-f:.]+",re.I)


@functools.lru_cache(maxsize=512)
def finalize_text(text):
    protected=[m.span() for m in EXPLICIT_VERSION.finditer(text)]
    changes=[]
    for m in QUAD.finditer(text):
        if any(a<=m.start()<b for a,b in protected):continue
        try:ipaddress.IPv4Address(m.group())
        except ValueError:continue
        end=m.end()
        tail=COMPOUND_TAIL.match(text,end)
        if tail:end=tail.end()
        changes.append({"start":m.start(),"end":end,"old_value":text[m.start():end],
          "reason":"unresolved_address_or_version_fragment","coordinate_space":"v32_stage1_text"})
    if not changes:return text,()
    chunks=[];pos=0
    for change in changes:
        if change["start"]<pos:continue
        chunks.extend((text[pos:change["start"]],"unknown_dotted_value"))
        pos=change["end"]
    chunks.append(text[pos:])
    return "".join(chunks),tuple(changes)


def prepare_record(row):
    p=prep.prepare_record(row)
    p["text"],spans=finalize_text(p["text"])
    p["final_quarantined_fragments"]=[dict(s) for s in spans]
    return p


def run(args):
    parent=Path(args.parent_run);out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    source=json.loads((parent/"result.json").read_text(encoding="utf-8"))
    if base.file_hash(parent/"prepared_corpus.parquet")!=source["corpus_sha256"]:
        raise ValueError("Stage 1 identity changed")
    started=time.perf_counter()
    sources={}
    for name in ["soc_v3_prepare.py","soc_v32_prepare.py"]:
        shutil.copyfile(parent/name,out/name);sources[name]=base.file_hash(out/name)
        if sources[name]!=source["source_sha256"][name]:raise AssertionError("Frozen stage 1 dependency changed")
    shutil.copyfile(__file__,out/"v32_finalize.py");sources["v32_finalize.py"]=base.file_hash(out/"v32_finalize.py")
    config={"version":VERSION,"parent_run":str(parent.resolve()),"parent_corpus_sha256":source["corpus_sha256"],
      "input_sha256":source["input_sha256"],"source_sha256":sources,
      "change":"Apply final ambiguity quarantine to EVERY prepared row and rebuild all groups. Stage 1 parser cache remains a dependency, not final text.",
      "diagnostic_text":"Retains pre-final ambiguity and upstream evidence; not eligible as primary model input",
      "labels_changed":False,"row_weight":1}
    base.save(out/"configuration.json",config)
    n=source["rows"];y=np.empty(n,dtype=np.uint8);groups=np.empty(n,dtype=np.int32);info=np.empty(n,dtype=bool)
    group_map={};next_group=0;offset=0;changed_rows=0;counts=collections.Counter();examples=[]
    writer=None
    for batch in pq.ParquetFile(parent/"prepared_corpus.parquet").iter_batches(batch_size=512,use_threads=False):
        rows=batch.to_pylist()
        for row in rows:
            oldtext=row["text"];row["text"],spans=finalize_text(oldtext)
            if spans:
                changed_rows+=1
                if len(examples)<100:
                    examples.append({"row_position":offset,"raw_hash":row["raw_hash"].hex(),"text_before":oldtext,"text_after":row["text"],"fragments":list(spans)})
                for span in spans:counts[span["reason"]]+=1
            if row["text"]:
                key=b"text:"+hashlib.sha256(row["text"].encode()).digest()
            else:key=b"raw:"+row["raw_hash"]
            if row["original_empty"]:
                gid=next_group;next_group+=1
            else:
                gid=group_map.get(key)
                if gid is None:
                    gid=next_group;next_group+=1;group_map[key]=gid
            row["group_id"]=gid
            row["final_quarantined_fragments"]=len(spans)
            row["final_quarantine_spans"]=json.dumps(list(spans),ensure_ascii=False,separators=(",",":"))
            y[offset]=base.LABELS.index(row["label"]);groups[offset]=gid;info[offset]=bool(row["text"])
            offset+=1
        table=pa.Table.from_pylist(rows)
        if writer is None:writer=pq.ParquetWriter(out/"prepared_corpus.parquet",table.schema,compression="zstd")
        writer.write_table(table)
    writer.close()
    if offset!=n:raise AssertionError("All rows must be retained")
    del group_map
    c=np.bincount(groups[info]*3+y[info],minlength=next_group*3).reshape(-1,3)
    mixed=(c>0).sum(axis=1)>1;ids=np.flatnonzero(mixed)
    collision={"mixed_label_text_groups":len(ids),"rows_in_mixed_label_text_groups":int(c[mixed].sum()),
      "minimum_observed_mistakes_deterministic_text_only":int((c.sum(axis=1)-c.max(axis=1)).sum()),
      "all_conflict_groups":[{"group_id":int(g),"counts":dict(zip(base.LABELS,map(int,c[g])))} for g in ids],
      "scope":"Final fixed text only; no relabeling; not a universal model bound"}
    pq.write_table(pa.table({"row_position":np.arange(n,dtype=np.int32),"group_id":groups,"label_index":y,"informative":info}),out/"group_manifest.parquet",compression="zstd")
    base.save(out/"text_collisions.json",collision)
    base.save(out/"final_quarantine_examples.json",examples)
    for name in ["coverage_by_source_format.json","field_policy_coverage.json"]:
        shutil.copyfile(parent/name,out/name)
    result={"status":"final_text_pending_independent_review","version":VERSION,"rows":n,
      "counters":{**source["counters"],"final_quarantine_rows":changed_rows,"final_quarantine_fragments":dict(counts)},
      "informative_rows":int(info.sum()),"informative_groups":int((c.sum(axis=1)>0).sum()),
      "mixed_label_text_groups":len(ids),"model_trained":False,"semantic_normalization_accepted":False,"transfer_validated":False,
      "parent_run":str(parent.resolve()),"parent_corpus_sha256":source["corpus_sha256"],"input_sha256":source["input_sha256"],
      "source_sha256":sources,"corpus_sha256":base.file_hash(out/"prepared_corpus.parquet"),
      "group_manifest_sha256":base.file_hash(out/"group_manifest.parquet"),"elapsed_seconds":time.perf_counter()-started}
    base.save(out/"result.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run",required=True)
    p.add_argument("--output-dir",required=True)
    run(p.parse_args())
