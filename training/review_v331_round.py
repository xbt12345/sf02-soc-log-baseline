"""Recompute results from all returned decisions, replaying every frozen model."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

from v331_common import (LABELS,BUDGETS,load,save,sha,load_data,unique_matrix,metrics,score_metrics,
                        risk_threshold,alerts,check_complete)


def manual_probabilities(x,model):
    logits=np.asarray(x@model.coef_.T)+model.intercept_
    logits-=logits.max(axis=1,keepdims=True)
    np.exp(logits,out=logits)
    logits/=logits.sum(axis=1,keepdims=True)
    return logits.astype(np.float64)


def review_task(data,roles,folder,task):
    report=load(folder/"report.json")
    joined=np.flatnonzero((roles==0)|(roles==1));y=data["y"]
    assert hashlib.sha256(joined.astype("<i4").tobytes()).hexdigest()==report["refit_positions_sha256"]
    assert hashlib.sha256(y[joined].astype("u1").tobytes()).hexdigest()==report["refit_labels_sha256"]
    selection=pq.read_table(folder/"selection_predictions.parquet")
    sel=np.flatnonzero(roles==1)
    assert np.array_equal(selection["row_position"].to_numpy(),sel)
    assert np.array_equal(selection["label_index"].to_numpy(),y[sel])
    scores=[]
    for phase,C in [("C_0_1",0.1),("C_1_0",1.0),("refit",report["selected_C"])]:
        fit=load(folder/(phase+"_fit.json"))
        assert fit["converged"] and fit["classes"]==[0,1,2] and fit["C"]==C
        assert fit["rows"]==int(((roles==0)|(roles==1)).sum() if phase=="refit" else (roles==0).sum())
        if phase!="refit":
            m=metrics(y[sel],selection[phase+"_pred"].to_numpy())
            assert m["confusion_matrix"]==load(folder/(phase+"_selection.json"))["confusion_matrix"]
            scores.append(m["macro_f1"])
    expected=0.1 if scores[0]>=scores[1]-0.001 else 1.0
    assert expected==report["selected_C"]
    assert sha(folder/"model.joblib")==report["model_sha256"]
    assert sha(folder/"vectorizer.joblib")==report["vectorizer_sha256"]
    model=joblib.load(folder/"model.joblib");v=joblib.load(folder/"vectorizer.joblib")
    assert model.C==expected and model.class_weight is None and model.solver=="saga"
    maxdelta=0.;views={}
    for filename,role in [("calibration_predictions.parquet",2),("outer_predictions.parquet",3)]:
        t=pq.read_table(folder/filename);pos=t["row_position"].to_numpy()
        assert np.array_equal(pos,np.flatnonzero(roles==role))
        assert np.array_equal(t["label_index"].to_numpy(),y[pos])
        stored=np.column_stack([t["R_p_"+c].to_numpy() for c in LABELS])
        assert np.isfinite(stored).all() and (stored>=0).all() and (stored<=1).all()
        assert np.allclose(stored.sum(1),1,atol=1e-6,rtol=0)
        x,inv,_=unique_matrix(data,pos,v)
        replay=manual_probabilities(x,model)[inv]
        delta=float(np.abs(replay-stored).max()) if len(pos) else 0.
        maxdelta=max(maxdelta,delta)
        assert delta<2e-6 and np.array_equal(replay.argmax(1),t["R_pred"].to_numpy())
        assert np.array_equal(stored.argmax(1),t["R_pred"].to_numpy())
        # Independently reconstruct N from fit+selection only.
        prior=np.bincount(y[joined],minlength=3)/len(joined)
        npred=np.empty_like(stored)
        for fmt in set(data["format"][pos]):
            subset=joined[data["format"][joined]==fmt]
            freq=np.bincount(y[subset],minlength=3)
            q=freq/freq.sum() if freq.sum() else prior
            npred[data["format"][pos]==fmt]=q
        actualn=np.column_stack([t["N_p_"+c].to_numpy() for c in LABELS])
        assert np.array_equal(actualn,npred)
        assert np.array_equal(actualn.argmax(1),t["N_pred"].to_numpy())
        if role==3:
            zeros=(np.diff(x.indptr)==0)[inv]
            assert np.array_equal(zeros,t["zero_vector"].to_numpy())
            review= zeros | data["no_observable_auth_facts"][pos]
            assert np.array_equal(review,t["no_evidence_or_review_suggested"].to_numpy())
        views[role]=(t,pos,stored,actualn)
        del x,replay;gc.collect()
    ct,cp,cal,_=views[2];et,ep,p,n=views[3]
    assert metrics(y[ep],p.argmax(1))["confusion_matrix"]==report["R"]["confusion_matrix"]
    assert metrics(y[ep],n.argmax(1))["confusion_matrix"]==report["N"]["confusion_matrix"]
    source=load(folder/"source_slices.json")
    for name,item in source.items():
        mask=data["product"][ep]==name
        assert metrics(y[ep][mask],p[mask].argmax(1))["confusion_matrix"]==item["R"]["confusion_matrix"]
    alarm_counts={}
    for threshold in report["risk_thresholds"]:
        budget=threshold["budget"];value=risk_threshold(1-cal[y[cp]==0,0],budget)
        assert value==threshold["R_threshold"]
        risk=1-p[:,0];alarm=risk>=value
        assert np.array_equal(alarm,et["R_alarm_"+str(budget)].to_numpy())
        assert alerts(y[ep],risk,value)["alert_rows"]==threshold["R_evaluation"]["alert_rows"]
        for name,item in source.items():
            mask=data["product"][ep]==name
            observed=next(v for v in item["alarms"] if v["budget"]==budget)
            assert alerts(y[ep][mask],risk[mask],value)["alert_rows"]==observed["alert_rows"]
        alarm_counts[str(budget)]=np.bincount(y[ep][alarm],minlength=3).tolist()
    assert load(folder/"model_stress.json")["all_invariance_checks_passed"]
    return {"task":task["name"],"all_checks_passed":True,"replayed_rows":len(cp)+len(ep),
        "max_probability_difference":maxdelta,"R":metrics(y[ep],p.argmax(1)),
        "alarm_counts":alarm_counts},views[3]


def review(root,out):
    root=Path(root);out=Path(out);protocol=load(root/"protocol.json");binding=load(out/"configuration.json")
    data=load_data(root);manifest=pq.read_table(root/"protocol_manifest.parquet")
    n=len(data["y"]);pred=np.full(n,-1,np.int8);n_pred=np.full(n,-1,np.int8)
    p_all=np.zeros((n,3));seen=np.zeros(n,np.int8)
    findings=[];alarm_counts={str(b):np.zeros(3,dtype=np.int64) for b in BUDGETS}
    for task in protocol["tasks"]:
        if not task["enabled"]:continue
        folder=check_complete(out/"tasks"/task["name"],binding)
        if folder is None:raise ValueError("Missing completed task: "+task["name"])
        checked,view=review_task(data,manifest[task["name"]].to_numpy(),folder,task)
        findings.append(checked)
        print(json.dumps({"stage":"independent_replay","task":task["name"],"rows":checked["replayed_rows"],"passed":True}),flush=True)
        if task["kind"]=="domain":
            table,pos,p,nn=view
            seen[pos]+=1;pred[pos]=p.argmax(1);n_pred[pos]=nn.argmax(1);p_all[pos]=p
            for b,c in checked["alarm_counts"].items():alarm_counts[b]+=c
        del view;gc.collect()
    assert (seen==1).all()
    y=data["y"];support=np.bincount(y,minlength=3)
    alarms={b:{"alerts_by_class":c.tolist(),"class_alert_rates":(c/support).tolist(),
                "false_alerts_per_10000_normal":float(c[0]/support[0]*10000)} for b,c in alarm_counts.items()}
    receipt={"status":"registered_batch_executed_and_independently_replayed",
        "all_execution_checks_passed":True,"all_registered_tasks_supported":protocol["all_registered_tasks_supported"],
        "rows":n,"completed_tasks":len(findings),"planned_classifier_fits":protocol["planned_classifier_fits"],
        "domain_R":score_metrics(y,p_all),"domain_N":metrics(y,n_pred),"domain_C0":metrics(y,np.zeros(n,np.int8)),
        "domain_alarm_operating_points":alarms,"tasks":findings,
        "internal_source_and_template_tests_executed":all(t["enabled"] for t in protocol["tasks"] if t["kind"]!="domain"),"external_transfer_validated":False,
        "remaining_quality_judgment":"Inspect worst sources, small support, template failures and result ablation; execution success is not quality acceptance.",
        "review_source_sha256":sha(__file__)}
    save(out/"independent_review.json",receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!="tasks"},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-dir",required=True);p.add_argument("--experiment-dir",required=True)
    a=p.parse_args();review(a.run_dir,a.experiment_dir)
