"""Freeze all v3.3.1 development/holdout responsibilities before model results."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import soc_v3_prepare as base
from v32_split import allocate, repair

TASK_NAMES=["domain_0","domain_1","domain_2","asa_template_0","asa_template_1","asa_template_2",
            "source_ad","source_duo","source_waf"]
ROLE_NAMES={-1:"excluded",0:"fit",1:"selection",2:"calibration",3:"evaluation"}


def summary(y,g,info,roles):
    answer={}
    for role,name in ROLE_NAMES.items():
        mask=roles==role
        classes=base.group_support(y,g,mask,info)
        answer[name]={"rows":int(mask.sum()),"classes":classes}
    return answer


def assign_inner(counts,icounts,available):
    roles=allocate(counts,icounts,available,[0.6,0.2,0.2])
    roles,moves,checked,ok=repair(roles,counts,icounts)
    return roles,moves,checked,ok


def build(args):
    root=Path(args.run_dir)
    if (root/"protocol.json").exists():raise FileExistsError("Preserve frozen protocol")
    started=time.perf_counter()
    result=json.loads((root/"result.json").read_text(encoding="utf-8"))
    audit=json.loads((root/"preparation_audit.json").read_text(encoding="utf-8"))
    if not audit["all_checks_passed"] or audit["corpus_sha256"]!=result["corpus_sha256"]:
        raise ValueError("Full preparation audit required")
    assert base.file_hash(root/"prepared_corpus.parquet")==result["corpus_sha256"]
    assert base.file_hash(root/"group_manifest.parquet")==result["group_manifest_sha256"]
    t=pq.read_table(root/"group_manifest.parquet")
    y=t["label_index"].to_numpy();g=t["group_id"].to_numpy();info=t["informative"].to_numpy()
    meta=pq.read_table(root/"prepared_corpus.parquet",columns=["product","repair_route","asa_template"])
    products=np.array(meta["product"].to_pylist(),dtype=object)
    routes=np.array(meta["repair_route"].to_pylist(),dtype=object)
    templates=meta["asa_template"].to_pylist()
    n=len(y);ng=int(g.max())+1
    counts=np.bincount(g.astype(np.int64)*3+y,minlength=ng*3).reshape(-1,3)
    icounts=np.bincount(g[info].astype(np.int64)*3+y[info],minlength=ng*3).reshape(-1,3)
    outerg=allocate(counts,icounts,np.arange(ng),[1/3]*3)
    columns={"row_position":np.arange(n,dtype=np.int32)}
    tasks=[];supports={};repairs={}
    # Families are computed by field grammar before this module reads labels.
    family_names=sorted({k for k in templates if k is not None})
    mapping={k:i for i,k in enumerate(family_names)}
    family=np.array([mapping.get(k,-1) for k in templates],dtype=np.int32)
    asa=routes=="asa"
    assert np.array_equal(asa,family>=0)
    fcounts=np.bincount(family[asa].astype(np.int64)*3+y[asa],minlength=len(mapping)*3).reshape(-1,3)
    fouter=allocate(fcounts,fcounts,np.arange(len(mapping)),[1/3]*3)
    source_map={"source_ad":"Windows Active Directory","source_duo":"Duo","source_waf":"Barracuda WAF"}
    for name in TASK_NAMES:
        if name.startswith("domain"):
            fold=int(name[-1]);test=outerg[g]==fold
        elif name.startswith("asa_template"):
            fold=int(name[-1]);test=np.zeros(n,bool);test[asa]=fouter[family[asa]]==fold
        else:
            test=products==source_map[name]
        held_groups=np.unique(g[test])
        available=np.flatnonzero(~np.isin(np.arange(ng),held_groups))
        inner,moves,checked,ok=assign_inner(counts,icounts,available)
        roles=inner[g];roles[test]=3
        linked_excluded=np.isin(g,held_groups)&~test
        roles[linked_excluded]=-1
        assert np.all(roles[test]==3)
        assert not np.any(np.isin(g,held_groups)&(roles>=0)&(roles<3))
        # Generic vectorized group-role check, allowing only excluded/test coexistence.
        minrole=np.full(ng,127,np.int8);maxrole=np.full(ng,-127,np.int8)
        active=roles>=0
        np.minimum.at(minrole,g[active],roles[active]);np.maximum.at(maxrole,g[active],roles[active])
        assert np.all(minrole[maxrole>=0]==maxrole[maxrole>=0])
        support=summary(y,g,info,roles)
        reasons=[]
        if not ok:reasons.append("inner_support_repair_failed")
        for r in ["fit","selection","calibration"]:
            for label,v in support[r]["classes"].items():
                if not v["engineering_support_30_groups"]:reasons.append(r+":"+label+":fewer_than_30_nonempty_groups")
                frac=v["largest_group_fraction_in_informative_rows"]
                if r in ["selection","calibration"] and (frac is None or frac>0.5):
                    reasons.append(r+":"+label+":single_group_dominance")
        expected=[0,1,2] if name.startswith("domain") else [1,2] if name.startswith("asa") else [0,2]
        for c in expected:
            if not np.any(test&(y==c)):reasons.append("evaluation_missing_"+base.LABELS[c])
        if name.startswith("asa"):
            seen=set(family[test])
            assert not seen.intersection(family[(roles>=0)&(roles<3)])
        if name.startswith("source"):
            assert not np.any((products==source_map[name])&(roles>=0)&(roles<3))
        columns[name]=roles
        task={"name":name,"kind":"domain" if name.startswith("domain") else "template_holdout" if name.startswith("asa") else "source_holdout",
              "enabled":not reasons,"unsupported_reasons":reasons,"expected_classifier_fits":3 if not reasons else 0,
              "linked_rows_excluded":int(linked_excluded.sum()),
              "target_product":source_map.get(name),
              "role_content_sha256":hashlib.sha256(roles.astype("i1").tobytes()).hexdigest(),
              "evaluation_rows":int(test.sum())}
        tasks.append(task);supports[name]=support;repairs[name]={"moves":moves,"candidate_checks":checked}
        print(json.dumps({"stage":"protocol","task":name,"supported":not reasons,"rows":int(test.sum())}),flush=True)
    # Domain evaluation coverage and ASA family coverage are exhaustive within their stated populations.
    assert np.all(sum((columns[k]==3).astype(np.int8) for k in TASK_NAMES[:3])==1)
    asacover=sum((columns[k]==3).astype(np.int8) for k in TASK_NAMES[3:6])
    assert np.array_equal(asacover,asa.astype(np.int8))
    pq.write_table(pa.table(columns),root/"protocol_manifest.parquet",compression="zstd")
    base.save(root/"protocol_support.json",supports)
    base.save(root/"protocol_repairs.json",repairs)
    base.save(root/"asa_template_inventory.json",[
        {"name":name,"outer_fold":int(fouter[i]),"class_counts":dict(zip(base.LABELS,map(int,fcounts[i])))}
        for i,name in enumerate(family_names)])
    protocol={"version":"v331-protocol-1.0","seed":20260912,"roles":ROLE_NAMES,"tasks":tasks,
        "input_sha256":result["input_sha256"],"corpus_sha256":result["corpus_sha256"],
        "group_manifest_sha256":result["group_manifest_sha256"],
        "protocol_manifest_sha256":base.file_hash(root/"protocol_manifest.parquet"),
        "preparation_audit_sha256":base.file_hash(root/"preparation_audit.json"),
        "protocol_source_sha256":base.file_hash(__file__),"allocation_source_sha256":base.file_hash(Path(__file__).with_name("v32_split.py")),
        "planned_classifier_fits":sum(t["expected_classifier_fits"] for t in tasks),
        "all_registered_tasks_supported":all(t["enabled"] for t in tasks),
        "all_official_data_previously_seen_development":True,"target_domain_calibration":False,
        "labels_or_weights_changed":False,"model_trained":False,
        "elapsed_seconds":time.perf_counter()-started}
    base.save(root/"protocol.json",protocol)
    print(json.dumps(protocol,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-dir",required=True);build(p.parse_args())
