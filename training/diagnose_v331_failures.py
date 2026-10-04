"""Fixed-model diagnostics on already inspected official development results."""
import gc,json,re,sys,warnings
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score,average_precision_score
from v331_common import load,save,load_data,LABELS,BUDGETS,metrics,risk_threshold
from audit_v331_cloud_return import legacy_unique_matrix
from review_v331_round import manual_probabilities


def run(root):
    root=Path(root);prep=root/"artifacts/v331_ready_20260912_r2";out=root/"evidence/2026-09-12/v331_result_review"
    returned=root/"artifacts/v331_cloud_return_20260912T122349Z";d=load_data(prep);y=d["y"]
    m=pq.read_table(prep/"protocol_manifest.parquet");domain=np.load(out/"v331_domain_diagnostic_arrays.npz")
    details={};sample_positions=set()
    for task in ["source_ad","source_duo","source_waf","asa_template_0"]:
        roles=m[task].to_numpy();pos=np.flatnonzero(roles==3)
        folder=returned/"experiment/tasks"/task/"attempt_1"
        t=pq.read_table(folder/"outer_predictions.parquet");p=np.column_stack([t["R_p_"+c].to_numpy() for c in LABELS])
        assert np.array_equal(t["row_position"].to_numpy(),pos)
        target=y[pos]!=0;risk=1-p[:,0]
        detail={"source_rows":len(pos),"unique_texts":len(set(d["text_ids"][pos])),
          "nonempty_groups_by_class":[len(np.unique(d["g"][pos][(y[pos]==c)&d["info"][pos]])) for c in range(3)],
          "risk_ranking":{"roc_auc":float(roc_auc_score(target,risk)) if target.any() and (~target).any() else None,
                          "average_precision":float(average_precision_score(target,risk)) if target.any() and (~target).any() else None},
          "risk_ranges":{LABELS[c]:{"min":float(risk[y[pos]==c].min()),"max":float(risk[y[pos]==c].max())} for c in range(3) if (y[pos]==c).any()},
          "zero_vectors_by_class":np.bincount(y[pos][t["zero_vector"].to_numpy()],minlength=3).tolist(),
          "oracle_thresholds":[],"examples":[]}
        if (~target).any():
            for budget in BUDGETS:
                th=risk_threshold(risk[~target],budget);a=risk>=th
                detail["oracle_thresholds"].append({"scope":"Post-hoc target-label optimistic threshold diagnostic only; never a deployable or zero-shot threshold.","budget":budget,"threshold":th,"alerts_by_class":np.bincount(y[pos][a],minlength=3).tolist()})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore");v=joblib.load(folder/"vectorizer.joblib");model=joblib.load(folder/"model.joblib")
        # At most four distinct target texts per true class; all Duo texts fit.
        picks=[]
        for c in range(3):
            candidates=pos[y[pos]==c];seen=set()
            for row in candidates:
                text=d["texts"][d["text_ids"][row]]
                if text in seen:continue
                seen.add(text);picks.append(int(row))
                if len(seen)>=4:break
        take=np.array(picks);x,inv,_=legacy_unique_matrix(d,take,v);names=v.get_feature_names_out()
        for i,row in enumerate(take):
            a,b=x.indptr[inv[i]:inv[i]+2];columns=x.indices[a:b];values=x.data[a:b]
            text=d["texts"][d["text_ids"][row]];idx=np.searchsorted(pos,row)
            features=[{"feature":str(names[j]),"suspicious_minus_benign_contribution":float(val*(model.coef_[2,j]-model.coef_[0,j])),
                       "malicious_minus_benign_contribution":float(val*(model.coef_[1,j]-model.coef_[0,j]))} for j,val in zip(columns,values)]
            features.sort(key=lambda r:abs(r["suspicious_minus_benign_contribution"])+abs(r["malicious_minus_benign_contribution"]),reverse=True)
            detail["examples"].append({"row_position":int(row),"true_label":LABELS[int(y[row])],"text":text,"scores":p[idx].tolist(),
                "nonzero_features":len(columns),"top_features":features[:12]})
            sample_positions.add(int(row))
        if task=="source_duo":
            unique=[]
            for textid in sorted(set(d["text_ids"][pos])):
                rows=pos[d["text_ids"][pos]==textid]
                unique.append({"text":d["texts"][textid],"row_count":len(rows),"class_counts":np.bincount(y[rows],minlength=3).tolist(),
                    "domain_class_predictions":np.bincount(domain["pred"][rows],minlength=3).tolist(),
                    "domain_risk_ranges":[float((1-domain["prob"][rows,0]).min()),float((1-domain["prob"][rows,0]).max())]})
            detail["all_Duo_text_groups"]=unique
        details[task]=detail
        del model,v,x;gc.collect()
    # Count source/class support across all domain learning responsibilities.
    support={}
    for source in ["Duo","Windows Active Directory","Barracuda WAF"]:
        target=d["product"]==source;support[source]={}
        for fold in range(3):
            roles=m["domain_"+str(fold)].to_numpy()
            support[source][str(fold)]={str(role):{"rows_by_class":np.bincount(y[target&(roles==role)],minlength=3).tolist(),
              "nonempty_groups_by_class":[len(np.unique(d["g"][target&(roles==role)&(y==c)&d["info"]])) for c in range(3)]} for role in range(4)}
    # A bounded raw-text inspection, never used to issue new labels.
    raw=[];offset=0
    for batch in pq.ParquetFile(root/"data/official/train.parquet").iter_batches(batch_size=4096,columns=["message_sanitized","timestamp","src_ip","dst_ip","src_host","dst_host","username"],use_threads=False):
        wanted=[n for n in sample_positions if offset<=n<offset+len(batch)]
        if wanted:
            rows=batch.to_pylist()
            for n in sorted(wanted):raw.append({"row_position":n,**rows[n-offset]})
        offset+=len(batch)
    # JSON-safe times, not a claim of reliable event ordering.
    (out/"v331_raw_case_examples.json").write_text(json.dumps(raw,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    save(out/"v331_failure_diagnostics.json",{"scope":"Post-hoc fixed-model diagnosis; no new training, relabeling, or test claims.","tasks":details,"domain_source_role_support":support})
    print(json.dumps({k:{"risk":v["risk_ranking"],"zero":v["zero_vectors_by_class"],"groups":v["nonempty_groups_by_class"],"oracle":v["oracle_thresholds"]} for k,v in details.items()},indent=2))


if __name__=="__main__":run(Path(__file__).resolve().parents[1])
