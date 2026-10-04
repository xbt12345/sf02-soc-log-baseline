"""Run the frozen v3.3.1 unweighted word baseline, with complete task-level resume."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import warnings

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import soc_v3_prepare as base
import v331_prepare as parser
from v331_common import (LABELS,BUDGETS,load,save,sha,load_data,row_texts,unique_matrix,metrics,
    score_metrics,group_metrics,risk_threshold,alerts,new_vectorizer,available_memory,
    probability_table,check_complete)

PARAMETERS={"C":[0.1,1.0],"tie_macro_f1":0.001,"solver":"saga","tol":0.001,"max_iter":1000,
            "random_state":20260912,"class_weight":None,"row_weight":1,
            "max_features":100000,"ngrams":[1,2],"dtype":"float32"}


class Heartbeat:
    def __init__(self,stage):self.stage=stage;self.stop=threading.Event()
    def __enter__(self):
        self.started=time.perf_counter()
        def report():
            while not self.stop.wait(20):
                print(json.dumps({"stage":self.stage,"status":"working","seconds":round(time.perf_counter()-self.started)}),flush=True)
        self.thread=threading.Thread(target=report,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()


def fit_model(x,y,C,folder,phase):
    if set(np.unique(y))!={0,1,2}:raise ValueError("Three fitted classes required")
    model=LogisticRegression(C=C,solver="saga",tol=0.001,max_iter=1000,random_state=20260912,class_weight=None)
    started=time.perf_counter()
    with Heartbeat(phase),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning);model.fit(x,y)
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    save(folder/(phase+"_fit.json"),{"C":C,"rows":len(y),"columns":x.shape[1],"nnz":x.nnz,
        "matrix_dtype":str(x.dtype),"converged":converged,"iterations":model.n_iter_.tolist(),
        "classes":model.classes_.tolist(),"seconds":time.perf_counter()-started,
        "warnings":[str(w.message) for w in caught]})
    if not converged or list(model.classes_)!=[0,1,2]:raise RuntimeError("Fit or class-order check failed")
    return model


def stress(folder,run,vectorizer,model):
    checks=[]
    pairs=load(run/"model_stress_fixtures.json")
    auth_file=run/"auth_field_review.json"
    if auth_file.exists():
        facts={json.dumps(r["facts"],sort_keys=True) for r in load(auth_file)}
        for i,serialized in enumerate(sorted(facts)):
            visible=json.loads(serialized)
            a=dict(visible,application={"key":"first"},user={"name":"first"},txid="first",
                   adaptive_trust_assessments={"trust_level":"NORMAL","model_version":"1"})
            b=dict(visible,application={"key":"second"},user={"name":"second"},txid="second",
                   adaptive_trust_assessments={"trust_level":"CRITICAL","model_version":"999"})
            pairs.append({"name":"auth_observed_facts_metadata_variation_"+str(i),"kind":"invariance",
                          "raw_a":json.dumps(a),"raw_b":json.dumps(b)})
    for pair in pairs:
        a=parser.prepare_record({"message_sanitized":pair["raw_a"]})["text"]
        b=parser.prepare_record({"message_sanitized":pair["raw_b"]})["text"]
        x=vectorizer.transform([a,b]);p=model.predict_proba(x)
        same=(x[0]-x[1]).nnz==0
        item={"name":pair["name"],"kind":pair["kind"],"texts_equal":a==b,"vectors_equal":same,
              "max_score_difference":float(np.abs(p[0]-p[1]).max()),"predictions":p.argmax(1).tolist()}
        if pair["kind"]=="invariance":
            assert a==b and same and item["max_score_difference"]<1e-12
        checks.append(item)
    save(folder/"model_stress.json",{"all_invariance_checks_passed":True,"pairs":checks,
        "scope":"Counterexamples; authentication metadata variants use official observed scalar facts with constructed wrappers. No new attack labels; semantic difference does not require a class change."})


def vector_collision_report(x,inverse,positions,data,roles):
    take=roles[positions]==3
    idx=inverse[take];labels=data["y"][positions[take]]
    counts=np.bincount(idx.astype(np.int64)*3+labels,minlength=x.shape[0]*3).reshape(-1,3)
    seen={};zero=0
    for i in np.flatnonzero(counts.sum(1)):
        a,b=x.indptr[i:i+2]
        if a==b:zero+=int(counts[i].sum());continue
        h=hashlib.sha256(x.indices[a:b].astype("<i4").tobytes()+x.data[a:b].astype("<f4").tobytes()).hexdigest()
        if h not in seen:seen[h]=[0,np.zeros(3,dtype=np.int64)]
        seen[h][0]+=1;seen[h][1]+=counts[i]
    collisions=[{"vector_sha256":h,"different_texts":v[0],"class_counts":v[1].tolist()}
                for h,v in seen.items() if v[0]>1]
    return {"zero_vector_rows":zero,"nonzero_vectors_shared_by_different_texts":len(collisions),
            "rows_in_collisions":sum(sum(v["class_counts"]) for v in collisions),"examples":collisions[:50],
            "scope":"Exact vectors on evaluation rows only, not incident or semantic equivalence."}


def controls(data,joined,positions):
    names,codes=np.unique(data["format"],return_inverse=True)
    y=data["y"];freq=np.bincount(codes[joined]*3+y[joined],minlength=len(names)*3).reshape(-1,3)
    prior=np.bincount(y[joined],minlength=3)/len(joined)
    table=np.tile(prior,(len(names),1));valid=freq.sum(1)>0
    table[valid]=freq[valid]/freq[valid].sum(1,keepdims=True)
    return table[codes[positions]],{str(k):v for k,v in zip(names,table.tolist())}


def run_task(data,roles,task,run,out,binding):
    taskroot=out/"tasks"/task["name"];taskroot.mkdir(parents=True,exist_ok=True)
    existing=check_complete(taskroot,binding)
    if existing:
        print(json.dumps({"task":task["name"],"status":"verified_complete_reused"}),flush=True)
        return load(existing/"report.json")
    attempts=[p for p in taskroot.glob("attempt_*") if p.is_dir()]
    folder=taskroot/("attempt_"+str(len(attempts)+1));folder.mkdir(exist_ok=False)
    started=time.perf_counter();timing={}
    save(folder/"configuration.json",{"binding":binding,"task":task,"parameters":PARAMETERS})
    fit=np.flatnonzero(roles==0);sel=np.flatnonzero(roles==1)
    cal=np.flatnonzero(roles==2);ev=np.flatnonzero(roles==3);joined=np.flatnonzero((roles==0)|(roles==1))
    y=data["y"]
    print(json.dumps({"stage":"task_start","task":task["name"],"fit_rows":len(fit),"evaluation_rows":len(ev)}),flush=True)
    t=time.perf_counter()
    vectorizer=new_vectorizer()
    with Heartbeat(task["name"]+":fit_vocabulary"):
        x=vectorizer.fit_transform(row_texts(data,fit,"fit_vocabulary"))
        xs,si,_=unique_matrix(data,sel,vectorizer)
    assert x.shape[0]==len(fit)
    timing["fit_vocabulary_and_selection_matrix"]=time.perf_counter()-t
    scores=[];selection={"row_position":sel.astype(np.int32),"label_index":y[sel]}
    for C,name in [(0.1,"C_0_1"),(1.0,"C_1_0")]:
        model=fit_model(x,y[fit],C,folder,name)
        pred=model.predict(xs)[si];measure=metrics(y[sel],pred)
        scores.append(measure["macro_f1"]);selection[name+"_pred"]=pred
        save(folder/(name+"_selection.json"),measure);del model
    pq.write_table(pa.table(selection),folder/"selection_predictions.parquet",compression="zstd")
    chosen=0.1 if scores[0]>=scores[1]-0.001 else 1.0
    del x,xs,vectorizer;gc.collect()
    t=time.perf_counter();vectorizer=new_vectorizer()
    with Heartbeat(task["name"]+":refit_vocabulary"):
        x=vectorizer.fit_transform(row_texts(data,joined,"refit_vocabulary"))
    timing["refit_vocabulary"]=time.perf_counter()-t
    assert x.shape[0]==len(joined)
    model=fit_model(x,y[joined],chosen,folder,"refit")
    del x;gc.collect()
    joblib.dump(vectorizer,folder/"vectorizer.joblib",compress=3)
    joblib.dump(model,folder/"model.joblib",compress=3)
    names=vectorizer.get_feature_names_out()
    save(folder/"coefficient_diagnostics.json",{"scope":"Linear associations, not causal or attack truth",
        "classes":{c:[[str(names[j]),float(model.coef_[i,j])] for j in np.argsort(model.coef_[i])[-30:][::-1]]
                   for i,c in enumerate(LABELS)}})
    t=time.perf_counter();positions=np.flatnonzero((roles==2)|(roles==3))
    with Heartbeat(task["name"]+":evaluate_unique_texts"):
        xp,inv,_=unique_matrix(data,positions,vectorizer)
        pu=np.asarray(model.predict_proba(xp),dtype=np.float64)
        rp=pu[inv]
        zero=(np.diff(xp.indptr)==0)[inv]
        coll=vector_collision_report(xp,inv,positions,data,roles)
    timing["unique_inference_and_collision"]=time.perf_counter()-t
    np_,ntable=controls(data,joined,positions)
    iscal=roles[positions]==2
    rcal,rout=rp[iscal],rp[~iscal];ncal,nout=np_[iscal],np_[~iscal]
    zero_out=zero[~iscal]
    assert np.array_equal(positions[iscal],cal) and np.array_equal(positions[~iscal],ev)
    save(folder/"vector_collisions.json",coll)
    stress(folder,run,vectorizer,model)
    report={"task":task,"selected_C":chosen,"selection_macro_f1":scores,
            "rows":{"fit":len(fit),"selection":len(sel),"calibration":len(cal),"evaluation":len(ev)},
            "refit_positions_sha256":hashlib.sha256(joined.astype("<i4").tobytes()).hexdigest(),
            "refit_labels_sha256":hashlib.sha256(y[joined].astype("u1").tobytes()).hexdigest(),
            "C0":metrics(y[ev],np.zeros(len(ev),dtype=np.int8)),
            "N":score_metrics(y[ev],nout),"R":score_metrics(y[ev],rout),
            "group_equal_R":group_metrics(y[ev],rout.argmax(1),data["g"][ev],data["info"][ev]),
            "group_equal_N":group_metrics(y[ev],nout.argmax(1),data["g"][ev],data["info"][ev]),
            "N_format_probabilities":ntable,"stage_seconds":timing}
    report["paired"]={"corrected_N_errors":int(((rout.argmax(1)==y[ev])&(nout.argmax(1)!=y[ev])).sum()),
                      "broke_N_correct":int(((rout.argmax(1)!=y[ev])&(nout.argmax(1)==y[ev])).sum())}
    risk=1-rout[:,0];crisk=1-rcal[:,0]
    review=zero_out|data["no_observable_auth_facts"][ev]
    extra={"zero_vector":zero_out,"no_evidence_or_review_suggested":review,
           "unknown_format":data["unknown_format"][ev],
           "authentication_result_unknown":data["authentication_result_unknown"][ev]}
    thresholds=[];slices={}
    for budget in BUDGETS:
        threshold=risk_threshold(crisk[y[cal]==0],budget)
        nthreshold=risk_threshold(1-ncal[y[cal]==0,0],budget)
        measured=alerts(y[cal],crisk,threshold)
        assert measured["class_alert_rates"]["benign"]<=budget+1e-12
        item={"budget":budget,"R_threshold":threshold,"N_threshold":nthreshold,
              "R_calibration":measured,"R_evaluation":alerts(y[ev],risk,threshold),
              "N_evaluation":alerts(y[ev],1-nout[:,0],nthreshold),
              "R_nonempty":alerts(y[ev][data["info"][ev]],risk[data["info"][ev]],threshold),
              "review_policy":{"suggested_rows":int(review.sum()),"automatic_coverage":float((~review).mean()),
                "scope":"Flag only; all records still classified and scored. Review is not a correct prediction.",
                "automatic_channel":alerts(y[ev][~review],risk[~review],threshold),
                "flagged_class_counts":np.bincount(y[ev][review],minlength=3).tolist()}}
        thresholds.append(item);extra["R_alarm_"+str(budget)]=risk>=threshold
    report["risk_thresholds"]=thresholds
    for source in sorted(set(data["product"][ev])):
        mask=data["product"][ev]==source
        slices[str(source)]={"R":metrics(y[ev][mask],rout[mask].argmax(1)),
          "N":metrics(y[ev][mask],nout[mask].argmax(1)),
          "group_equal_R":group_metrics(y[ev][mask],rout[mask].argmax(1),data["g"][ev][mask],data["info"][ev][mask]),
          "alarms":[{"budget":t["budget"],**alerts(y[ev][mask],risk[mask],t["R_threshold"])} for t in thresholds]}
    save(folder/"source_slices.json",slices)
    largest=[]
    for cls in range(3):
        mask=(y[ev]==cls)&data["info"][ev]
        ids,cnts=np.unique(data["g"][ev][mask],return_counts=True)
        if not len(ids):continue
        gid=int(ids[np.argmax(cnts)]);keep=data["g"][ev]!=gid
        largest.append({"class":LABELS[cls],"removed_group":gid,"class_rows_in_group":int(cnts.max()),
            "R":metrics(y[ev][keep],rout[keep].argmax(1)),"N":metrics(y[ev][keep],nout[keep].argmax(1))})
    save(folder/"largest_group_sensitivity.json",largest)
    duo=ev[data["repair_route"][ev]=="authentication"]
    if len(duo):
        original=[data["texts"][int(data["text_ids"][i])] for i in duo]
        masked=[re.sub(r"\bresult\s+\S+","",v).strip() for v in original]
        altered=model.predict_proba(vectorizer.transform(masked)).astype(np.float64)
        posidx=np.searchsorted(ev,duo)
        cols={"row_position":duo.astype(np.int32),"label_index":y[duo],
            "original_pred":rout[posidx].argmax(1),"masked_result_pred":altered.argmax(1)}
        cols.update({"masked_R_p_"+c:altered[:,i] for i,c in enumerate(LABELS)})
        pq.write_table(pa.table(cols),folder/"auth_result_ablation.parquet",compression="zstd")
        save(folder/"auth_result_ablation.json",{"scope":"Information-losing ablation, not a label-preserving transformation.",
            "rows":len(duo),"original":metrics(y[duo],rout[posidx].argmax(1)),
            "masked":metrics(y[duo],altered.argmax(1)),
            "class_flips":int((rout[posidx].argmax(1)!=altered.argmax(1)).sum())})
    pq.write_table(probability_table(cal,y[cal],rcal,ncal),folder/"calibration_predictions.parquet",compression="zstd")
    pq.write_table(probability_table(ev,y[ev],rout,nout,extra),folder/"outer_predictions.parquet",compression="zstd")
    report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=base.peak_memory_bytes(),
        model_sha256=sha(folder/"model.joblib"),vectorizer_sha256=sha(folder/"vectorizer.joblib"),
        model_trained=True,external_transfer_validated=False)
    save(folder/"report.json",report)
    files={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    save(taskroot/"complete.json",{"binding":binding,"attempt":folder.name,"files":files})
    print(json.dumps({"task":task["name"],"status":"trained_task_complete","R_macro_f1":report["R"]["macro_f1"],
        "seconds":round(report["elapsed_seconds"],1)}),flush=True)
    return report


def run(args):
    root=Path(args.run_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    protocol=load(root/"protocol.json")
    verified=load(root/"protocol_verification.json")
    if not verified["all_checks_passed"] or verified["protocol_manifest_sha256"]!=protocol["protocol_manifest_sha256"]:
        raise ValueError("Protocol verification missing")
    for name,key in [("prepared_corpus.parquet","corpus_sha256"),("group_manifest.parquet","group_manifest_sha256"),
                     ("protocol_manifest.parquet","protocol_manifest_sha256")]:
        if sha(root/name)!=protocol[key]:raise ValueError("Input identity changed: "+name)
    if sha(args.train)!=base.EXPECTED_SHA:raise ValueError("Official train identity failed")
    source={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")}
    packages={p:importlib.metadata.version(p) for p in ["numpy","scipy","scikit-learn","pyarrow","joblib"]}
    binding={"source_sha256":source,"corpus_sha256":protocol["corpus_sha256"],
             "protocol_sha256":sha(root/"protocol.json"),"packages":packages,"python":sys.version,"parameters":PARAMETERS}
    if (out/"configuration.json").exists():
        if load(out/"configuration.json")!=binding:raise ValueError("Existing output belongs to different code/data/runtime")
    else:save(out/"configuration.json",binding)
    memory=available_memory();required=4*1024**3
    save(out/"resource_preflight.json",{"available_bytes":memory,"minimum_required_bytes":required,
        "cpu_count_os":os.cpu_count(),"thread_cap_environment":os.environ.get("OMP_NUM_THREADS"),
        "note":"4 GiB minimum planning guard, not guaranteed peak; sequential tasks, CPU only"})
    if memory is None or memory<required:raise RuntimeError("Insufficient free memory for full unmodified-row batch")
    data=load_data(root);manifest=pq.read_table(root/"protocol_manifest.parquet")
    reports=[]
    for task in protocol["tasks"]:
        if not task["enabled"]:
            reports.append({"task":task,"status":"unsupported_before_training"});continue
        report=run_task(data,manifest[task["name"]].to_numpy(),task,root,out,binding)
        reports.append(report);gc.collect()
    result={"status":"registered_training_tasks_executed_pending_independent_review",
      "model_trained":True,"all_registered_tasks_supported":protocol["all_registered_tasks_supported"],
      "completed_tasks":sum("R" in r for r in reports),"planned_classifier_fits":protocol["planned_classifier_fits"],
      "external_transfer_validated":False,"tasks":[{"name":r["task"]["name"],"R":r.get("R"),"status":r.get("status","executed")} for r in reports]}
    save(out/"result.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("--train",required=True);a.add_argument("--run-dir",required=True);a.add_argument("--output-dir",required=True)
    args=a.parse_args()
    try:run(args)
    except Exception as e:
        out=Path(args.output_dir)
        if out.exists():save(out/"failure_"+str(time.time_ns())+".json",{"error":str(e),"traceback":traceback.format_exc()})
        raise
