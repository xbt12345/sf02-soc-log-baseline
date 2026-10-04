"""Deterministic whole-group allocation; labels for support, never model scores."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soc_v3_prepare as base

SEED=20260912


def allocate(counts, icounts, available, ratios):
    """Largest groups first; minimize change in class Pearson target deviation."""
    ratios=np.asarray(ratios,dtype=float)
    total=counts[available].sum(axis=0)
    target=np.maximum(ratios[:,None]*total[None,:],1.0)
    roles=np.full(len(counts),-1,dtype=np.int8)
    current=np.zeros_like(target)
    sizes=counts[available].sum(axis=1)
    hashes=np.array([int.from_bytes(hashlib.blake2b((str(SEED)+":"+str(int(g))).encode(),digest_size=8).digest(),"little") for g in available],dtype=np.uint64)
    order=available[np.lexsort((hashes,-sizes))]
    for g in order:
        v=counts[g]
        delta=((2*(current-target)*v+v*v)/target).sum(axis=1)
        role=int(np.argmin(delta))
        roles[g]=role
        current[role]+=v
    return roles


def role_summary(roles,icounts,role):
    chosen=icounts[roles==role]
    return (chosen>0).sum(axis=0),chosen.sum(axis=0),chosen.max(axis=0) if len(chosen) else np.zeros(3,dtype=int)


def repair(roles,counts,icounts,limit=10000):
    """Support/dominance repair by whole-group moves; all choices score-free."""
    moves=[]
    checked=0
    for iteration in range(100):
        stats=[role_summary(roles,icounts,r) for r in range(3)]
        violation=None
        # Move a dominating selection/calibration group into fit before refilling.
        for r in (1,2):
            for c in range(3):
                if stats[r][1][c] and stats[r][2][c]>0.5*stats[r][1][c]:
                    candidates=np.flatnonzero((roles==r)&(icounts[:,c]==stats[r][2][c]))
                    g=int(candidates[0])
                    roles[g]=0
                    moves.append({"group_id":g,"from":r,"to":0,"reason":"single_group_majority"})
                    violation="dominance"
                    break
            if violation:
                break
        if violation:
            continue
        for r in range(3):
            for c in range(3):
                if stats[r][0][c]<30:
                    violation=(r,c)
                    break
            if violation:
                break
        if not violation:
            return roles,moves,checked,True
        r,c=violation
        candidates=np.flatnonzero((roles>=0)&(roles!=r)&(icounts[:,c]>0))
        # Prefer real-sized support, with candidate id as fixed tie-break.
        candidates=candidates[np.lexsort((candidates,-icounts[candidates,c]))]
        moved=False
        for g in candidates:
            checked+=1
            if checked>limit:
                return roles,moves,checked,False
            donor=int(roles[g]); flags=icounts[g]>0
            if ((stats[donor][0]-flags)<30).any():
                continue
            # Avoid moving an already dominant block back into an evaluation role.
            if r in (1,2) and (icounts[g] > 0.5*(stats[r][1]+icounts[g])).any():
                continue
            roles[g]=r
            moves.append({"group_id":int(g),"from":donor,"to":r,"reason":"class_group_support"})
            moved=True
            break
        if not moved:
            return roles,moves,checked,False
    return roles,moves,checked,False


def run(args):
    run=Path(args.run_dir)
    gate=json.loads((run / "semantic_review.json").read_text(encoding="utf-8"))
    if not gate.get("eligible_for_domain_development"):
        raise ValueError("Representation/collision review has not passed for domain development")
    result=json.loads((run / "result.json").read_text(encoding="utf-8"))
    if gate.get("corpus_sha256")!=result["corpus_sha256"]:
        raise ValueError("Semantic review does not belong to this corpus")
    if base.file_hash(run / "prepared_corpus.parquet")!=result["corpus_sha256"] or base.file_hash(run / "group_manifest.parquet")!=result["group_manifest_sha256"]:
        raise ValueError("Prepared input identity changed")
    if (run / "split_manifest.parquet").exists():
        raise FileExistsError("Never overwrite a split")
    started=time.perf_counter()
    shutil.copyfile(__file__,run / "v32_split.py")
    config={"seed":SEED,"outer_folds":3,"inner_targets":[0.6,0.2,0.2],"minimum_nonempty_groups":30,
      "selection_calibration_max_group_fraction":0.5,"repair_candidate_limit":10000,
      "order":"descending group row count, then BLAKE2b(seed:group_id)",
      "objective":"sum of squared class-row target deviations divided by target, incremental minimization; ratios soft",
      "source_balance":"source/format support reported, not optimized as labels or predictor features",
      "new_development_split_not_blind_test":True,"code_sha256":base.file_hash(Path(__file__))}
    base.save(run / "split_configuration.json",config)
    t=pq.read_table(run / "group_manifest.parquet")
    g=t["group_id"].to_numpy(); y=t["label_index"].to_numpy(); info=t["informative"].to_numpy()
    n=len(g); ng=int(g.max())+1
    counts=np.bincount(g*3+y,minlength=ng*3).reshape(-1,3)
    icounts=np.bincount(g[info]*3+y[info],minlength=ng*3).reshape(-1,3)
    outer_g=allocate(counts,icounts,np.arange(ng),[1/3]*3)
    outer=outer_g[g]
    table={"row_position":np.arange(n,dtype=np.int32),"outer_fold":outer}
    supports=[]; move_log=[]
    for fold in range(3):
        available=np.flatnonzero(outer_g!=fold)
        inner_g=allocate(counts,icounts,available,[0.6,0.2,0.2])
        inner_g,moves,checked,ok=repair(inner_g,counts,icounts)
        roles=inner_g[g]
        inner=np.where(roles==0,0,np.where(roles==1,3,np.where(roles==2,4,-1))).astype(np.int8)
        table["inner_for_outer_"+str(fold)]=inner
        masks={"fit":inner==0,"selection":inner==3,"calibration":inner==4,"outer_development":outer==fold}
        if not (sum(v.astype(np.uint8) for v in masks.values())==1).all():
            raise AssertionError("Every row needs exactly one role")
        support={name:base.group_support(y,g,mask,info) for name,mask in masks.items()}
        passed=all(v["engineering_support_30_groups"] for role in support.values() for v in role.values())
        passed &= all(v["largest_group_fraction_in_informative_rows"] is not None and v["largest_group_fraction_in_informative_rows"]<=0.5 for name in ["selection","calibration"] for v in support[name].values())
        support.update(fold=fold,support_gate_passed=bool(passed and ok))
        supports.append(support)
        move_log.append({"fold":fold,"candidate_checks":checked,"moves":moves})
        print(json.dumps({"fold":fold,"passed":bool(passed and ok),"repair_moves":len(moves)}),flush=True)
    pq.write_table(pa.table(table),run / "split_manifest.parquet",compression="zstd")
    base.save(run / "split_support.json",supports)
    base.save(run / "split_repairs.json",move_log)
    status={"status":"split_ready_for_independent_check" if all(s["support_gate_passed"] for s in supports) else "split_support_failed",
      "all_support_gates_passed":all(s["support_gate_passed"] for s in supports),"elapsed_seconds":time.perf_counter()-started,
      "split_sha256":base.file_hash(run / "split_manifest.parquet"),"model_trained":False}
    base.save(run / "split_result.json",status)
    print(json.dumps(status),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True)
    run(p.parse_args())
