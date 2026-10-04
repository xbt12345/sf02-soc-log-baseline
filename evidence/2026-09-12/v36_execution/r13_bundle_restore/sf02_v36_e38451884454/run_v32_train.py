"""Frozen v3.2 round: C0/N/C1-W, three folds, two C values. CPU only."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import sklearn
import joblib

import soc_v3_prepare as base
from v32_features import word_tokens
from v32_model_diagnostics import stress,vector_collisions

LABELS=["benign","malicious","suspicious"]
BUDGETS=[0.0001,0.001,0.01]


def metrics(y,pred):
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    return {"rows":len(y),"macro_f1":float(f.mean()) if (s>0).all() else None,
      "confusion_matrix":confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
      "classes":{name:{"support":int(s[i]),"precision":float(p[i]) if np.any(pred==i) else None,
       "recall":float(r[i]) if s[i] else None,"f1":float(f[i]) if s[i] else None} for i,name in enumerate(LABELS)}}


def risk_threshold(risk,budget):
    risk=np.asarray(risk,dtype=np.float64)
    if not len(risk):return None
    permitted=int(np.floor(len(risk)*budget))
    ordered=np.sort(risk)[::-1]
    return float(np.nextafter(ordered[min(permitted,len(risk)-1)],np.inf))


def alerts(y,risk,threshold):
    # Keep float64 comparison: float32 may round nextafter thresholds back onto ties.
    if threshold is None:return {"status":"no_normal_threshold_support"}
    alarm=np.asarray(risk,dtype=np.float64)>=threshold
    rates={k:float(alarm[y==i].mean()) if np.any(y==i) else None for i,k in enumerate(LABELS)}
    return {"threshold":threshold,"alert_rows":int(alarm.sum()),"class_alert_rates":rates,
      "false_alerts_per_10000_benign":rates["benign"]*10000 if rates["benign"] is not None else None,
      "alert_precision_nonbenign":float((y[alarm]!=0).mean()) if alarm.any() else None}


def available_memory():
    if sys.platform=="win32":
        import ctypes
        class Memory(ctypes.Structure):
            _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ["total","available","total_page","available_page","total_virtual","available_virtual","extended"]]
        m=Memory();m.length=ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):return int(m.available)
        return None
    values={}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k,v=line.split(":",1);values[k]=int(v.strip().split()[0])*1024
        free=values.get("MemAvailable",values.get("MemFree"))
        for limit_path,used_path in [("/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory.current"),
          ("/sys/fs/cgroup/memory/memory.limit_in_bytes","/sys/fs/cgroup/memory/memory.usage_in_bytes")]:
            if Path(limit_path).exists():
                value=Path(limit_path).read_text().strip()
                if value.isdigit() and int(value)<2**60:
                    remaining=max(0,int(value)-int(Path(used_path).read_text()))
                    free=min(free,remaining) if free is not None else remaining
        return free
    except (OSError,ValueError):return None


def iter_texts(run,mask):
    for b in pq.ParquetFile(run/"prepared_corpus.parquet").iter_batches(batch_size=4096,columns=["row_position","text"],use_threads=False):
        positions=b.column(0).to_numpy()
        keep=mask[positions]
        if not keep.any():continue
        for pos,text in zip(positions,b.column(1).to_pylist()):
            if mask[pos]:yield text


def probabilities(run,mask,vectorizer,model):
    pieces=[];zero_rows=[]
    for b in pq.ParquetFile(run/"prepared_corpus.parquet").iter_batches(batch_size=4096,columns=["row_position","text"],use_threads=False):
        pos=b.column(0).to_numpy(); selected=mask[pos]
        if not selected.any():continue
        strings=b.column(1).to_pylist()
        x=vectorizer.transform([t for t,k in zip(strings,selected) if k])
        pieces.append(np.asarray(model.predict_proba(x),dtype=np.float64))
        zero_rows.extend(pos[selected][np.diff(x.indptr)==0].tolist())
    return np.concatenate(pieces),zero_rows


def coded_column(table,name):
    col=table[name].combine_chunks()
    if not pa.types.is_dictionary(col.type):col=col.dictionary_encode()
    return col.indices.to_numpy(),col.dictionary.to_pylist()


def new_vectorizer():
    return TfidfVectorizer(tokenizer=word_tokens,token_pattern=None,lowercase=False,
       ngram_range=(1,2),min_df=1,max_features=100000,dtype=np.float32,norm="l2")


def fitted_model(x,y,C,folder,phase):
    if set(np.unique(y))!={0,1,2}:raise ValueError("All three fitted classes required")
    model=LogisticRegression(C=C,solver="saga",tol=0.001,max_iter=1000,random_state=20260912,class_weight=None)
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        model.fit(x,y)
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    receipt={"phase":phase,"C":C,"iterations":model.n_iter_.tolist(),"converged":converged,
      "seconds":time.perf_counter()-start,"input_rows":x.shape[0],"columns":x.shape[1],"nnz":x.nnz,
      "matrix_dtype":str(x.dtype),"coefficient_dtype":str(model.coef_.dtype),"classes":model.classes_.tolist(),
      "warnings":[str(w.message) for w in caught]}
    base.save(folder/(phase+"_fit.json"),receipt)
    if not converged:raise RuntimeError("Optimization did not converge; do not treat score as a fitted-model verdict")
    if list(model.classes_)!=[0,1,2]:raise AssertionError("Class order mismatch")
    return model


def paired_summary(y,model_pred,n_pred):
    return {"corrected_N_errors":int(((model_pred==y)&(n_pred!=y)).sum()),
      "broke_N_correct":int(((model_pred!=y)&(n_pred==y)).sum()),
      "both_wrong":int(((model_pred!=y)&(n_pred!=y)).sum()),
      "C1_W":metrics(y,model_pred),"N":metrics(y,n_pred)}


def run(args):
    run=Path(args.run_dir); out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    args._output_owned=True
    started=time.perf_counter()
    prep_result=json.loads((run/"result.json").read_text(encoding="utf-8"))
    split_result=json.loads((run/"split_result.json").read_text(encoding="utf-8"))
    semantic=json.loads((run/"semantic_review.json").read_text(encoding="utf-8"))
    verified=json.loads((run/"split_verification.json").read_text(encoding="utf-8"))
    if not semantic.get("eligible_for_domain_development") or not split_result.get("all_support_gates_passed") or not verified.get("all_checks_passed"):
        raise ValueError("Preparation and split gates required")
    if semantic.get("corpus_sha256")!=prep_result["corpus_sha256"] or verified.get("corpus_sha256")!=prep_result["corpus_sha256"] or verified.get("split_sha256")!=split_result["split_sha256"]:
        raise ValueError("Gate receipts belong to different content")
    if base.file_hash(args.train)!=prep_result["input_sha256"] or prep_result["input_sha256"]!=base.EXPECTED_SHA:
        raise ValueError("Official train identity mismatch")
    if base.file_hash(run/"prepared_corpus.parquet")!=prep_result["corpus_sha256"] or base.file_hash(run/"split_manifest.parquet")!=split_result["split_sha256"]:
        raise ValueError("Prepared corpus or split changed")
    if base.file_hash(run/"group_manifest.parquet")!=prep_result["group_manifest_sha256"]:
        raise ValueError("Group/label manifest changed")
    for dependency in ["soc_v3_prepare.py","soc_v32_prepare.py","v32_finalize.py"]:
        if base.file_hash(Path(__file__).with_name(dependency))!=prep_result["source_sha256"][dependency]:
            raise ValueError("Runtime representation differs from the reviewed prepared corpus: "+dependency)
    sources={}
    for name in ["run_v32_train.py","v32_features.py","soc_v3_prepare.py","soc_v32_prepare.py","v32_finalize.py","v32_model_diagnostics.py"]:
        shutil.copyfile(Path(__file__).with_name(name),out/name)
        sources[name]=base.file_hash(out/name)
    config={"version":"v32-word-round-1.0","mode":"controls_only" if args.controls_only else "C0_N_C1_W",
      "official_train_sha256":prep_result["input_sha256"],"corpus_sha256":prep_result["corpus_sha256"],
      "split_sha256":split_result["split_sha256"],"source_sha256":sources,
      "C":[0.1,1.0],"tie_macro_f1":0.001,"solver":"saga","tol":0.001,"max_iter":1000,
      "row_weights":1,"class_weight":None,"word_ngrams":[1,2],"max_features":100000,
      "dtype":"float32","calibration":"none; raw scores and separate empirical risk thresholds",
      "outer_fold_order":[0,1,2],"risk_budgets":BUDGETS,"sklearn":sklearn.__version__,"python":sys.version,
      "stress_fixtures_sha256":base.file_hash(run/"model_stress_fixtures.json"),
      "external_data":False,"all_official_training_rows_are_seen_development_material":True}
    base.save(out/"configuration.json",config)
    audit=json.loads((run/"independent_audit.json").read_text(encoding="utf-8"))
    nnz=audit["resource_sample"]["full_nnz_estimate_uncapped"]
    required=max(3*1024**3,int(nnz*0.67*8*2.5+512*1024**2))
    memory=available_memory()
    resource={"available_bytes":memory,"estimated_required_bytes":required,
      "method":"uncapped sampled word nnz * 0.67 * 8 * 2.5 plus 512 MiB, minimum 3 GiB; planning guard, not a peak guarantee"}
    base.save(out/"resource_preflight.json",resource)
    if not args.controls_only and (memory is None or memory<required):
        base.save(out/"result.json",{"status":"not_started_insufficient_available_memory","model_trained":False,"resource_preflight":resource})
        print(json.dumps({"status":"not_started_insufficient_available_memory",**resource}),flush=True)
        return 2
    gm=pq.read_table(run/"group_manifest.parquet")
    y=gm["label_index"].to_numpy(); g=gm["group_id"].to_numpy(); info=gm["informative"].to_numpy(); n=len(y)
    split=pq.read_table(run/"split_manifest.parquet"); outer=split["outer_fold"].to_numpy()
    meta=pq.read_table(run/"prepared_corpus.parquet",columns=["product","format","original_empty","filtered_empty","unknown_format","final_quarantined_fragments","invalid_string_values","unverified_field_count"],read_dictionary=["product","format"])
    formats,format_names=coded_column(meta,"format"); products,product_names=coded_column(meta,"product")
    all_pred=np.full(n,-1,dtype=np.int8); all_n=np.full(n,-1,dtype=np.int8)
    all_p=np.empty((n,3),dtype=np.float64); all_np=np.empty((n,3),dtype=np.float64)
    reports=[]; zero_positions=[]
    for fold in range(3):
        folder=out/("fold_"+str(fold));folder.mkdir()
        inner=split["inner_for_outer_"+str(fold)].to_numpy()
        fit=(inner>=0)&(inner<=2); select=inner==3;cal=inner==4;test=outer==fold
        joined=fit|select
        freq=np.bincount(formats[joined]*3+y[joined],minlength=len(format_names)*3).reshape(-1,3).astype(float)
        prior=np.bincount(y[joined],minlength=3)/joined.sum()
        n_table=np.tile(prior,(len(format_names),1));seen=freq.sum(axis=1)>0
        n_table[seen]=freq[seen]/freq[seen].sum(axis=1,keepdims=True)
        pn=n_table[formats[test]]; ncal=n_table[formats[cal]]
        npred=pn.argmax(axis=1).astype(np.int8); all_n[test]=npred;all_np[test]=pn
        report={"fold":fold,"rows":{"fit":int(fit.sum()),"selection":int(select.sum()),"calibration":int(cal.sum()),"outer":int(test.sum())},
            "C0":metrics(y[test],np.zeros(test.sum(),dtype=np.int8)),"N":metrics(y[test],npred),
            "N_format_probabilities":dict(zip(format_names,n_table.tolist()))}
        if not args.controls_only:
            print("Fold {}: fit word vocabulary and matrix on fit rows only".format(fold),flush=True)
            vectorizer=new_vectorizer()
            x=vectorizer.fit_transform(iter_texts(run,fit))
            xs=vectorizer.transform(iter_texts(run,select))
            if x.shape[0]!=fit.sum() or xs.shape[0]!=select.sum():raise AssertionError("Vectorization changed row frequency")
            scores=[]
            for C in [0.1,1.0]:
                name="C_"+str(C).replace(".","_")
                model=fitted_model(x,y[fit],C,folder,name)
                score=metrics(y[select],model.predict(xs));scores.append(score["macro_f1"])
                base.save(folder/(name+"_selection.json"),score)
                del model
            chosen=0.1 if scores[0]>=scores[1]-0.001 else 1.0
            report["selected_C"]=chosen;report["selection_macro_f1"]=scores
            del x,xs,vectorizer;gc.collect()
            # This refit must learn a new vocabulary/IDF using fit + selection.
            vectorizer=new_vectorizer();x=vectorizer.fit_transform(iter_texts(run,joined))
            if x.shape[0]!=joined.sum():raise AssertionError("Refit must retain every assigned row")
            model=fitted_model(x,y[joined],chosen,folder,"refit")
            del x;gc.collect()
            joblib.dump(vectorizer,folder/"vectorizer.joblib",compress=3)
            joblib.dump(model,folder/"model.joblib",compress=3)
            report["model_sha256"]=base.file_hash(folder/"model.joblib")
            report["vectorizer_sha256"]=base.file_hash(folder/"vectorizer.joblib")
            report["refit_row_positions_sha256"]=hashlib.sha256(np.flatnonzero(joined).astype("<i4").tobytes()).hexdigest()
            names=vectorizer.get_feature_names_out()
            base.save(folder/"coefficient_diagnostics.json",{"scope":"Linear associations only, not causal explanations or attack truth",
              "classes":{label:{"largest_positive":[[str(names[j]),float(model.coef_[c,j])] for j in np.argsort(model.coef_[c])[-30:][::-1]],
                                 "largest_negative":[[str(names[j]),float(model.coef_[c,j])] for j in np.argsort(model.coef_[c])[:30]]} for c,label in enumerate(LABELS)}})
            pcal,_=probabilities(run,cal,vectorizer,model)
            pout,zeros=probabilities(run,test,vectorizer,model)
            zero_positions.extend(zeros)
            pred=pout.argmax(axis=1).astype(np.int8);all_pred[test]=pred;all_p[test]=pout
            report["C1_W"]=metrics(y[test],pred)
            report["paired"]=paired_summary(y[test],pred,npred)
            report["zero_vector_outer_rows"]=len(zeros)
            stress_result=stress(run,vectorizer,model)
            base.save(folder/"model_stress.json",stress_result)
            if not stress_result["all_invariance_checks_passed"]:raise AssertionError("Frozen model invariant failed; stop the round")
            base.save(folder/"vector_collisions.json",vector_collisions(run,test,g,y,vectorizer))
            del model,vectorizer;gc.collect()
        else:
            pout=pn;pcal=ncal;pred=npred
        risk_reports=[]
        for budget in BUDGETS:
            nt=risk_threshold(1-ncal[y[cal]==0,0],budget)
            item={"requested_calibration_fpr_budget":budget,
              "N_calibration":alerts(y[cal],1-ncal[:,0],nt),"N_outer":alerts(y[test],1-pn[:,0],nt)}
            item["N_outer_nonempty"]=alerts(y[test][info[test]],(1-pn[:,0])[info[test]],nt)
            item["N_outer_empty"]=alerts(y[test][~info[test]],(1-pn[:,0])[~info[test]],nt)
            if not args.controls_only:
                threshold=risk_threshold(1-pcal[y[cal]==0,0],budget)
                item.update(C1_W_calibration=alerts(y[cal],1-pcal[:,0],threshold),C1_W_outer=alerts(y[test],1-pout[:,0],threshold))
                item["C1_W_outer_nonempty"]=alerts(y[test][info[test]],(1-pout[:,0])[info[test]],threshold)
                item["C1_W_outer_empty"]=alerts(y[test][~info[test]],(1-pout[:,0])[~info[test]],threshold)
                observed=item["C1_W_calibration"]["class_alert_rates"]["benign"]
                if observed>budget+1e-12:raise AssertionError("Calibration threshold tie handling violated the declared budget")
            risk_reports.append(item)
        report["risk_thresholds"]=risk_reports
        calibration_positions=np.flatnonzero(cal)
        pq.write_table(pa.table({"row_position":calibration_positions.astype(np.int32),"label_index":y[cal],
          "N_p_benign":ncal[:,0],"N_p_malicious":ncal[:,1],"N_p_suspicious":ncal[:,2],
          **({"C1_W_p_benign":pcal[:,0],"C1_W_p_malicious":pcal[:,1],"C1_W_p_suspicious":pcal[:,2]} if not args.controls_only else {})}),folder/"calibration_predictions.parquet",compression="zstd")
        report["calibration_prediction_sha256"]=base.file_hash(folder/"calibration_predictions.parquet")
        positions=np.flatnonzero(test)
        pq.write_table(pa.table({"row_position":positions.astype(np.int32),"label_index":y[test],"N_pred":npred,
          "N_p_benign":pn[:,0],"N_p_malicious":pn[:,1],"N_p_suspicious":pn[:,2],
          **({"C1_W_pred":pred,"C1_W_p_benign":pout[:,0],"C1_W_p_malicious":pout[:,1],"C1_W_p_suspicious":pout[:,2]} if not args.controls_only else {})}),folder/"outer_predictions.parquet",compression="zstd")
        report["outer_prediction_sha256"]=base.file_hash(folder/"outer_predictions.parquet")
        base.save(folder/"report.json",report);reports.append(report)
        print(json.dumps({"fold":fold,"N_macro_f1":report["N"]["macro_f1"],"C1_W_macro_f1":report.get("C1_W",{}).get("macro_f1")}),flush=True)
    result={"status":"controls_executed" if args.controls_only else "round_executed_not_transfer_validated",
      "rows":n,"model_trained":not args.controls_only,"C0":metrics(y,np.zeros(n,dtype=np.int8)),"N":metrics(y,all_n),"folds":reports,
      "elapsed_seconds":time.perf_counter()-started,"parent_peak_bytes":base.peak_memory_bytes(),"transfer_validated":False}
    control_slices={}
    for name,codes,names in [("format",formats,format_names),("product",products,product_names)]:
        control_slices[name]={label:metrics(y[codes==code],all_n[codes==code]) for code,label in enumerate(names)}
    control_slices["nonempty"]=metrics(y[info],all_n[info])
    control_slices["empty"]=metrics(y[~info],all_n[~info])
    base.save(out/"control_slices.json",control_slices)
    if not args.controls_only:
        if (all_pred<0).any():raise AssertionError("Outer coverage incomplete")
        result.update(C1_W=metrics(y,all_pred),paired=paired_summary(y,all_pred,all_n))
        slices={}
        for name,codes,names in [("format",formats,format_names),("product",products,product_names)]:
            slices[name]={}
            for code,label in enumerate(names):
                mask=codes==code
                slices[name][label]=paired_summary(y[mask],all_pred[mask],all_n[mask])
        for flag in ["original_empty","filtered_empty","unknown_format"]:
            mask=meta[flag].to_numpy()
            slices[flag]=paired_summary(y[mask],all_pred[mask],all_n[mask]) if mask.any() else {"rows":0}
        for field in ["final_quarantined_fragments","invalid_string_values","unverified_field_count"]:
            mask=meta[field].to_numpy()>0
            slices[field+"_present"]=paired_summary(y[mask],all_pred[mask],all_n[mask]) if mask.any() else {"rows":0}
        base.save(out/"slices.json",slices)
        base.save(out/"zero_vector_rows.json",{"rows":zero_positions,"note":"Includes original empty, filtered and out-of-vocabulary; keep these distinctions using corpus flags"})
    base.save(out/"result.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="folds"},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",required=True)
    p.add_argument("--run-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--controls-only",action="store_true")
    args=p.parse_args()
    try:
        exit_status=run(args)
        if exit_status:raise SystemExit(exit_status)
    except Exception as e:
        out=Path(args.output_dir)
        if getattr(args,"_output_owned",False) and out.exists() and not (out/"failure.json").exists():
            base.save(out/"failure.json",{"error":str(e),"traceback":traceback.format_exc()})
        raise
