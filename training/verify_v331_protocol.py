"""Independent protocol checks and portable content receipt for cloud reproduction."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from v331_common import load,save,sha,portable_content


def verify_isolation(g,roles):
    ng=int(g.max())+1
    low=np.full(ng,127,dtype=np.int8);high=np.full(ng,-127,dtype=np.int8)
    active=roles>=0
    np.minimum.at(low,g[active],roles[active]);np.maximum.at(high,g[active],roles[active])
    if not np.all(low[high>=0]==high[high>=0]):
        raise AssertionError("Same group crosses learning/evaluation roles")
    if not np.isin(roles,[-1,0,1,2,3]).all():
        raise AssertionError("Invalid role")


def verify(root):
    root=Path(root);p=load(root/"protocol.json");r=load(root/"result.json")
    assert sha(root/"prepared_corpus.parquet")==r["corpus_sha256"]==p["corpus_sha256"]
    assert sha(root/"group_manifest.parquet")==r["group_manifest_sha256"]==p["group_manifest_sha256"]
    assert sha(root/"protocol_manifest.parquet")==p["protocol_manifest_sha256"]
    assert sha(root/"preparation_audit.json")==p["preparation_audit_sha256"]
    assert load(root/"preparation_audit.json")["all_checks_passed"]
    gm=pq.read_table(root/"group_manifest.parquet")
    g=gm["group_id"].to_numpy();y=gm["label_index"].to_numpy();info=gm["informative"].to_numpy()
    m=pq.read_table(root/"protocol_manifest.parquet")
    meta=pq.read_table(root/"prepared_corpus.parquet",columns=["product","repair_route","asa_template"])
    products=np.asarray(meta["product"].to_pylist(),dtype=object)
    templates=np.asarray(meta["asa_template"].to_pylist(),dtype=object)
    asa=np.asarray(meta["repair_route"].to_pylist(),dtype=object)=="asa"
    n=len(y);checks={};domain=np.zeros(n,np.int8);acover=np.zeros(n,np.int8)
    assert np.array_equal(m["row_position"].to_numpy(),np.arange(n))
    for task in p["tasks"]:
        name=task["name"];roles=m[name].to_numpy()
        assert hashlib.sha256(roles.astype("i1").tobytes()).hexdigest()==task["role_content_sha256"]
        verify_isolation(g,roles)
        test=roles==3;train=(roles>=0)&(roles<3)
        assert test.sum()==task["evaluation_rows"]
        held=np.unique(g[test])
        assert not np.any(np.isin(g,held)&train)
        if task["kind"]=="source_holdout":
            assert np.array_equal(test,products==task["target_product"])
        if task["kind"]=="template_holdout":
            assert not set(templates[test]).intersection(templates[train]);acover+=test.astype(np.int8)
        if task["kind"]=="domain":domain+=test.astype(np.int8)
        if task["enabled"]:
            for role in (0,1,2):
                for cls in (0,1,2):
                    mask=(roles==role)&(y==cls)&info
                    _,counts=np.unique(g[mask],return_counts=True)
                    assert len(counts)>=30,(name,role,cls)
                    if role in (1,2):assert counts.max()<=0.5*counts.sum()
        checks[name]=True
    assert (domain==1).all() and np.array_equal(acover,asa.astype(np.int8))
    content=portable_content(root)
    save(root/"portable_content.json",content)
    result={"all_checks_passed":True,"checks":checks,"domain_and_ASA_coverage_complete":True,
        "no_group_or_target_source_leakage":True,"protocol_manifest_sha256":p["protocol_manifest_sha256"],
        "portable_content":content,"source_sha256":sha(__file__),"model_trained":False}
    save(root/"protocol_verification.json",result)
    print(__import__("json").dumps(result,indent=2),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("--run-dir",required=True)
    verify(a.parse_args().run_dir)

