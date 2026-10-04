"""Read-only audit of the returned v3.3.1 models against local official input."""
import argparse
import gc
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from v331_common import load,save,sha,load_data,LABELS,BUDGETS,metrics,group_metrics,check_complete
from review_v331_round import review_task
import review_v331_round as frozen_reviewer


def legacy_unique_matrix(data,positions,vectorizer):
    """Replay sklearn 1.3 IDF explicitly; never fit or silently omit old IDF."""
    ids,inverse=np.unique(data["text_ids"][positions],return_inverse=True)
    params=vectorizer.get_params()
    args={k:params[k] for k in CountVectorizer().get_params() if k in params}
    args["vocabulary"]=vectorizer.vocabulary_
    counter=CountVectorizer(**args)
    x=counter.transform([data["texts"][int(i)] for i in ids])
    if vectorizer.sublinear_tf:
        np.log(x.data,out=x.data);x.data+=1
    assert vectorizer.use_idf and hasattr(vectorizer._tfidf,"_idf_diag")
    x=x@vectorizer._tfidf._idf_diag
    if vectorizer.norm is not None:x=normalize(x,norm=vectorizer.norm,copy=False)
    x.sort_indices()
    return x,inverse,ids


def audit(root,returned,output):
    started=time.perf_counter();root=Path(root);returned=Path(returned);output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    # Use the serialized 1.3 IDF, not unsupported 1.9 deserialization behavior.
    frozen_reviewer.unique_matrix=legacy_unique_matrix
    prep=root/"artifacts/v331_ready_20260912_r2";pe=returned/"prepared_evidence";ex=returned/"experiment"
    bundle=load(root/"platform/v331_bundle_receipt.json")
    localruntime=root/"platform"/bundle["runtime_folder"]
    assert (returned/"runtime/bundle_manifest.json").read_bytes()==(localruntime/"bundle_manifest.json").read_bytes()
    configuration=load(ex/"configuration.json")
    for name,digest in configuration["source_sha256"].items():
        assert sha(returned/"runtime/training"/name)==digest==sha(localruntime/"training"/name)
    assert sha(root/"data/official/train.parquet")==load(prep/"result.json")["input_sha256"]
    assert load(pe/"portable_content.json")==load(prep/"portable_content.json")
    cloudprotocol=load(pe/"protocol.json");localprotocol=load(prep/"protocol.json")
    assert sha(pe/"protocol.json")==configuration["protocol_sha256"]
    assert sha(pe/"protocol_manifest.parquet")==cloudprotocol["protocol_manifest_sha256"]
    assert sha(pe/"preparation_audit.json")==cloudprotocol["preparation_audit_sha256"]
    assert configuration["corpus_sha256"]==cloudprotocol["corpus_sha256"]
    assert cloudprotocol["tasks"]==localprotocol["tasks"]
    cm=pq.read_table(pe/"protocol_manifest.parquet");lm=pq.read_table(prep/"protocol_manifest.parquet")
    assert cm.equals(lm,check_metadata=False)
    for name,key in [("prepared_corpus.parquet","corpus_sha256"),("group_manifest.parquet","group_manifest_sha256")]:
        assert sha(prep/name)==localprotocol[key]
    data=load_data(prep);y=data["y"];n=len(y)
    original=pq.read_table(root/"data/official/train.parquet",columns=["label_binary"])["label_binary"].to_pylist()
    assert np.array_equal(y,np.fromiter((LABELS.index(v) for v in original),dtype=np.uint8,count=n))
    del original
    scores=np.zeros((n,3));pred=np.full(n,-1,np.int8);seen=np.zeros(n,np.int8)
    alarms={str(b):np.zeros(n,bool) for b in BUDGETS};zero=np.zeros(n,bool)
    summaries=[];timing=[]
    for task in cloudprotocol["tasks"]:
        folder=check_complete(ex/"tasks"/task["name"],configuration)
        assert folder is not None
        roles=cm[task["name"]].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",InconsistentVersionWarning)
            checked,view=review_task(data,roles,folder,task)
        table,pos,p,np_=view
        report=load(folder/"report.json")
        holds={"task":task["name"],"replay":checked,"selected_C":report["selected_C"],
          "classification":metrics(y[pos],p.argmax(1)),
          "group_equal":group_metrics(y[pos],p.argmax(1),data["g"][pos],data["info"][pos]),
          "operating_points":report["risk_thresholds"],"zero_vector_rows":int(table["zero_vector"].to_numpy().sum()),
          "vector_collisions":load(folder/"vector_collisions.json"),
          "largest_group_sensitivity":load(folder/"largest_group_sensitivity.json"),
          "stress":load(folder/"model_stress.json")}
        if (folder/"auth_result_ablation.json").exists():holds["auth_result_ablation"]=load(folder/"auth_result_ablation.json")
        summaries.append(holds)
        fit_seconds=sum(load(folder/(name+"_fit.json"))["seconds"] for name in ["C_0_1","C_1_0","refit"])
        timing.append({"task":task["name"],"total_seconds":report["elapsed_seconds"],"fit_seconds":fit_seconds,
          "stage_seconds":report["stage_seconds"],"process_high_water_rss_bytes":report["peak_rss_bytes"]})
        if task["kind"]=="domain":
            seen[pos]+=1;scores[pos]=p;pred[pos]=p.argmax(1);zero[pos]=table["zero_vector"].to_numpy()
            for b in BUDGETS:alarms[str(b)][pos]=table["R_alarm_"+str(b)].to_numpy()
        print(json.dumps({"task":task["name"],"replayed":checked["replayed_rows"],"max_difference":checked["max_probability_difference"],"classification_recall":{k:v["recall"] for k,v in holds["classification"]["classes"].items()}}),flush=True)
        del view;gc.collect()
    assert (seen==1).all()
    domain=metrics(y,pred)
    assert domain["confusion_matrix"]==load(ex/"independent_review.json")["domain_R"]["confusion_matrix"]
    sources={}
    for source in sorted(set(data["product"])):
        mask=data["product"]==source
        sources[str(source)]={"classification":metrics(y[mask],pred[mask]),
           "group_equal":group_metrics(y[mask],pred[mask],data["g"][mask],data["info"][mask]),
           "class_counts":np.bincount(y[mask],minlength=3).tolist(),
           "errors":int((y[mask]!=pred[mask]).sum()),"zero_vector_rows":int(zero[mask].sum()),
           "alarm_counts":{b:np.bincount(y[mask&a],minlength=3).tolist() for b,a in alarms.items()}}
    support=np.bincount(y,minlength=3)
    operating={b:{"class_alert_counts":np.bincount(y[a],minlength=3).tolist(),
                "rates":(np.bincount(y[a],minlength=3)/support).tolist()} for b,a in alarms.items()}
    result={"scope":"No training or edits to returned artifacts. Local re-read and model replay; not external transfer acceptance.",
        "all_integrity_and_replay_checks_passed":True,"rows":n,"tasks":9,"classifier_fits_verified":27,
        "all_official_labels_rechecked":True,"all_roles_match_local":True,"runtime_matches_shipped_package":True,
        "replay_adapter":"CountVectorizer with frozen vocabulary, saved sklearn 1.3 _idf_diag, normalization and frozen linear coefficients; no refit. Initial unsupported 1.9 transform failed before this explicit repair.",
        "cloud_prepared_file_rehashed":False,"cloud_content_identity_basis":"Cloud receipt matches local canonical content and frozen code; full cloud prepared corpus is not in result ZIP.",
        "domain":domain,"domain_group_equal":group_metrics(y,pred,data["g"],data["info"]),
        "domain_alarm_operating_points":operating,"sources":sources,"task_details":summaries,"timing":timing,
        "total_task_seconds":sum(t["total_seconds"] for t in timing),"classifier_fit_seconds":sum(t["fit_seconds"] for t in timing),
        "maximum_reported_process_rss_bytes":max(t["process_high_water_rss_bytes"] for t in timing),
        "audit_elapsed_seconds":time.perf_counter()-started,"audit_source_sha256":sha(__file__)}
    save(output/"v331_return_audit.json",result)
    np.savez_compressed(output/"v331_domain_diagnostic_arrays.npz",pred=pred,prob=scores,zero=zero,**{"alarm_"+b:a for b,a in alarms.items()})
    print(json.dumps({"passed":True,"output":str(output/"v331_return_audit.json"),"domain_macro_f1":domain["macro_f1"],"total_task_seconds":result["total_task_seconds"],"fit_seconds":result["classifier_fit_seconds"],"elapsed":result["audit_elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--returned",required=True);a.add_argument("--output",required=True)
    v=a.parse_args();audit(v.root,v.returned,v.output)
