"""Row-level independent split support and no-crossing verification."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soc_v3_prepare as base


def run(args):
    root=Path(args.run_dir)
    preparation=json.loads((root/"result.json").read_text(encoding="utf-8"))
    result=json.loads((root/"split_result.json").read_text(encoding="utf-8"))
    prior=json.loads((root/"independent_audit.json").read_text(encoding="utf-8"))
    checks={"prior_integrity_passed_for_this_corpus":prior["all_integrity_checks_passed"] and prior["corpus_sha256"]==preparation["corpus_sha256"],
      "corpus_identity":base.file_hash(root/"prepared_corpus.parquet")==preparation["corpus_sha256"],
      "group_identity":base.file_hash(root/"group_manifest.parquet")==preparation["group_manifest_sha256"],
      "split_identity":base.file_hash(root/"split_manifest.parquet")==result["split_sha256"]}
    gm=pq.read_table(root/"group_manifest.parquet");sm=pq.read_table(root/"split_manifest.parquet")
    g=gm["group_id"].to_numpy();y=gm["label_index"].to_numpy();info=gm["informative"].to_numpy();n=len(g)
    outer=sm["outer_fold"].to_numpy();ng=int(g.max())+1
    checks["original_row_order_and_full_coverage"]=bool(n==2056871 and np.array_equal(gm["row_position"].to_numpy(),np.arange(n)) and np.array_equal(sm["row_position"].to_numpy(),np.arange(n)))
    checks["three_outer_folds"]=bool(np.isin(outer,[0,1,2]).all())
    supports=[]
    metadata=pq.read_table(root/"prepared_corpus.parquet",columns=["product","format"],read_dictionary=["product","format"])
    partitions={}
    for name in ["product","format"]:
        col=metadata[name].combine_chunks().dictionary_encode()
        partitions[name]=(col.indices.to_numpy(),col.dictionary.to_pylist())
    source_support=[]
    for f in range(3):
        inner=sm["inner_for_outer_"+str(f)].to_numpy()
        checks["fold_{}_valid_inner_roles".format(f)]=bool(np.all(inner[outer==f]==-1) and np.isin(inner[outer!=f],[0,3,4]).all())
        role=np.where(outer==f,3,np.where(inner==0,0,np.where(inner==3,1,2)))
        lo=np.full(ng,9,dtype=np.int8);hi=np.full(ng,-1,dtype=np.int8)
        np.minimum.at(lo,g,role);np.maximum.at(hi,g,role)
        checks["fold_{}_whole_groups_only".format(f)]=bool(np.array_equal(lo,hi))
        for r,name in enumerate(["fit","selection","calibration","outer"]):
            for c,label in enumerate(base.LABELS):
                mask=(role==r)&(y==c)
                _,sizes=np.unique(g[mask&info],return_counts=True)
                fraction=float(sizes.max()/sizes.sum()) if len(sizes) else None
                checks["fold_{}_{}_{}_support".format(f,name,label)]=len(sizes)>=30
                if name in ["selection","calibration"]:
                    checks["fold_{}_{}_{}_no_majority_group".format(f,name,label)]=fraction is not None and fraction<=0.5
                supports.append({"fold":f,"role":name,"class":label,"rows":int(mask.sum()),"nonempty_groups":len(sizes),"largest_group_fraction":fraction})
            for category,(codes,names) in partitions.items():
                for code,value in enumerate(names):
                    subset=(role==r)&(codes==code)
                    for c,label in enumerate(base.LABELS):
                        mask=subset&(y==c)
                        _,sizes=np.unique(g[mask&info],return_counts=True)
                        source_support.append({"fold":f,"role":name,"category":category,"value":value,"class":label,
                          "rows":int(mask.sum()),"nonempty_groups":len(sizes),"largest_group_fraction":float(sizes.max()/sizes.sum()) if len(sizes) else None})
    with (root/"split_source_support.json").open("x",encoding="utf-8") as f:
        json.dump({"scope":"Support diagnostic only. Product and format are not independent organizations; absent classes remain zero support.","support":source_support},f,ensure_ascii=False,indent=2)
    report={"scope":"Full row coverage, whole-group roles and engineering support checks; no claim of true incident independence, classifier quality or blind testing",
      "checks":checks,"all_checks_passed":all(checks.values()),"supports":supports,
      "corpus_sha256":preparation["corpus_sha256"],"split_sha256":result["split_sha256"],"code_sha256":base.file_hash(__file__)}
    with (root/"split_verification.json").open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({"all_checks_passed":report["all_checks_passed"],"failed":[k for k,v in checks.items() if not v],"checks":len(checks)}),flush=True)
    if not report["all_checks_passed"]:raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-dir",required=True)
    run(p.parse_args())
