"""Invalidate cached parses when the reviewed root-routing dependency changes.

Every official row is checked, including rows outside the discovered examples.
Only the root router changed since stage 1; other parsing functions are bound.
"""
import argparse
import ast
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
import soc_v32_prepare as prep
import v32_finalize as final


def assert_only_router_change(old_path,new_path):
    def signature(path):
        tree=ast.parse(Path(path).read_text(encoding="utf-8"))
        tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=="object_root")
          and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="VERSION" for t in n.targets))]
        class Router(ast.NodeTransformer):
            def visit_Attribute(self,node):
                if isinstance(node.value,ast.Name) and node.value.id=="base" and node.attr=="object_root":
                    return ast.Name(id="object_root",ctx=node.ctx)
                return self.generic_visit(node)
        return ast.dump(Router().visit(tree),include_attributes=False)
    if signature(old_path)!=signature(new_path):raise AssertionError("Cache dependency changed beyond root routing")


def run(args):
    parent=Path(args.parent_run);out=Path(args.output_dir);started=time.perf_counter()
    source=json.loads((parent/"result.json").read_text(encoding="utf-8"))
    if base.file_hash(args.train)!=source["input_sha256"] or source["input_sha256"]!=base.EXPECTED_SHA:raise ValueError("Official input changed")
    if base.file_hash(parent/"prepared_corpus.parquet")!=source["corpus_sha256"]:raise ValueError("Parent corpus changed")
    assert_only_router_change(parent/"soc_v32_prepare.py",Path(prep.__file__))
    for name in ["soc_v3_prepare.py","v32_finalize.py"]:
        if base.file_hash(Path(__file__).with_name(name))!=source["source_sha256"][name]:raise AssertionError("Unreviewed dependency change")
    out.mkdir(parents=True,exist_ok=False)
    sources={}
    for name in ["soc_v3_prepare.py","soc_v32_prepare.py","v32_finalize.py","v32_root_repair.py"]:
        shutil.copyfile(Path(__file__).with_name(name),out/name);sources[name]=base.file_hash(out/name)
    base.save(out/"configuration.json",{"version":"v32-root-repair-1.0","parent_corpus_sha256":source["corpus_sha256"],
      "source_sha256":sources,"all_rows_router_comparison":True,"non_router_ast_equivalence_checked":True,
      "cache_invalidation":"Reparse every changed root; otherwise preserve validated cached parse. Rebuild all groups.",
      "labels_changed":False,"row_weight":1})
    n=source["rows"];y=np.empty(n,dtype=np.uint8);groups=np.empty(n,dtype=np.int32);info=np.empty(n,dtype=bool)
    group_map={};next_group=0;offset=0;changed=[];counters=collections.Counter();coverage=collections.Counter()
    writer=None;last_progress=started
    raw_iter=pq.ParquetFile(args.train).iter_batches(batch_size=512,columns=["message_sanitized"],use_threads=False)
    prev_iter=pq.ParquetFile(parent/"prepared_corpus.parquet").iter_batches(batch_size=512,use_threads=False)
    for rb,pb in zip(raw_iter,prev_iter):
        raw_rows=rb.to_pylist();rows=pb.to_pylist()
        if len(raw_rows)!=len(rows):raise AssertionError("Batch alignment")
        for raw_row,row in zip(raw_rows,rows):
            raw=base.string(raw_row["message_sanitized"])
            if hashlib.sha256(raw.encode()).digest()!=row["raw_hash"] or row["row_position"]!=offset:raise AssertionError("Original row mismatch")
            before=base.object_root(raw);after=prep.object_root(raw)
            if before!=after:
                p=final.prepare_record({"message_sanitized":raw});spans=p["final_quarantined_fragments"]
                changed.append({"row_position":offset,"raw_hash":row["raw_hash"].hex(),"root_before":before,"root_after":after,
                  "raw_message":raw,"text_before":row["text"],"prepared":p})
                for key in ["format","text","diagnostic_text","original_empty","filtered_empty","unknown_format"]:row[key]=p[key]
                row.update(removed_field_count=len(p["removed_spans"]),upstream_field_count=p["upstream_fields_removed"],
                  unverified_field_count=p["unverified_fields_isolated"],invalid_string_values=p["invalid_string_values_preserved"],
                  final_quarantined_fragments=len(spans),final_quarantine_spans=json.dumps(spans,ensure_ascii=False,separators=(",",":")))
            key=b"text:"+hashlib.sha256(row["text"].encode()).digest() if row["text"] else b"raw:"+row["raw_hash"]
            if row["original_empty"]:gid=next_group;next_group+=1
            else:
                gid=group_map.get(key)
                if gid is None:gid=next_group;next_group+=1;group_map[key]=gid
            row["group_id"]=gid;y[offset]=base.LABELS.index(row["label"]);groups[offset]=gid;info[offset]=bool(row["text"])
            for key in ["original_empty","filtered_empty","unknown_format"]:counters[key]+=row[key]
            counters["invalid_string_rows"]+=row["invalid_string_values"]>0
            counters["unverified_field_rows"]+=row["unverified_field_count"]>0
            counters["final_quarantine_rows"]+=row["final_quarantined_fragments"]>0
            counters["final_quarantine_fragments"]+=row["final_quarantined_fragments"]
            coverage[(row["product"],row["format"],row["label"],row["original_empty"],row["filtered_empty"],row["unknown_format"])]+=1
            offset+=1
        table=pa.Table.from_pylist(rows)
        if writer is None:writer=pq.ParquetWriter(out/"prepared_corpus.parquet",table.schema,compression="zstd")
        writer.write_table(table)
        if time.perf_counter()-last_progress>20:
            last_progress=time.perf_counter();print(json.dumps({"rows":offset,"new_routes":len(changed),"seconds":last_progress-started}),flush=True)
    writer.close()
    if offset!=n:raise AssertionError("Full row coverage")
    del group_map
    c=np.bincount(groups[info]*3+y[info],minlength=next_group*3).reshape(-1,3);mixed=(c>0).sum(axis=1)>1;ids=np.flatnonzero(mixed)
    pq.write_table(pa.table({"row_position":np.arange(n,dtype=np.int32),"group_id":groups,"label_index":y,"informative":info}),out/"group_manifest.parquet",compression="zstd")
    base.save(out/"text_collisions.json",{"mixed_label_text_groups":len(ids),"rows_in_mixed_label_text_groups":int(c[mixed].sum()),
      "minimum_observed_mistakes_deterministic_text_only":int((c.sum(axis=1)-c.max(axis=1)).sum()),
      "all_conflict_groups":[{"group_id":int(g),"counts":dict(zip(base.LABELS,map(int,c[g])))} for g in ids],"scope":"Fixed text only; all original rows and labels retained"})
    base.save(out/"root_repair_examples.json",changed)
    base.save(out/"coverage_by_source_format.json",[{"product":k[0],"format":k[1],"label":k[2],"original_empty":k[3],"filtered_empty":k[4],"unknown_format":k[5],"rows":v} for k,v in sorted(coverage.items())])
    result={"status":"repaired_text_pending_independent_review","version":"v32-root-repair-1.0","rows":n,
      "counters":dict(counters),"roots_compared":offset,"rows_reparsed":len(changed),"non_router_ast_equivalence_checked":True,
      "informative_rows":int(info.sum()),"informative_groups":int((c.sum(axis=1)>0).sum()),"mixed_label_text_groups":len(ids),
      "model_trained":False,"semantic_normalization_accepted":False,"transfer_validated":False,
      "parent_run":source["parent_run"],"previous_stage_run":str(parent.resolve()),"parent_corpus_sha256":source["corpus_sha256"],
      "input_sha256":source["input_sha256"],"source_sha256":sources,"corpus_sha256":base.file_hash(out/"prepared_corpus.parquet"),
      "group_manifest_sha256":base.file_hash(out/"group_manifest.parquet"),"elapsed_seconds":time.perf_counter()-started}
    base.save(out/"result.json",result);print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run",required=True);parser.add_argument("--train",required=True);parser.add_argument("--output-dir",required=True)
    run(parser.parse_args())
