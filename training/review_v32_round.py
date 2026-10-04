"""Recompute complete-fold decisions, paired gains, group sensitivity and intervals."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soc_v3_prepare as base
from run_v32_train import metrics,paired_summary


def macro(cm):
    true=cm.sum(axis=1);pred=cm.sum(axis=0)
    if not (true>0).all():return float("nan")
    return float(np.divide(2*np.diag(cm),true+pred,out=np.zeros(3,dtype=float),where=(true+pred)>0).mean())


def run(args):
    root=Path(args.run_dir); exp=Path(args.experiment_dir)
    result=json.loads((exp/"result.json").read_text(encoding="utf-8"))
    config=json.loads((exp/"configuration.json").read_text(encoding="utf-8"))
    if config["corpus_sha256"]!=base.file_hash(root/"prepared_corpus.parquet") or config["split_sha256"]!=base.file_hash(root/"split_manifest.parquet"):
        raise ValueError("Experiment and prepared data identities differ")
    if result["status"] not in ["controls_executed","round_executed_not_transfer_validated"]:
        raise ValueError("Only completed three-fold rounds are reviewable")
    gm=pq.read_table(root/"group_manifest.parquet")
    y=gm["label_index"].to_numpy();g=gm["group_id"].to_numpy();info=gm["informative"].to_numpy()
    split=pq.read_table(root/"split_manifest.parquet");outer=split["outer_fold"].to_numpy()
    n=len(y);counts=np.zeros(n,dtype=np.uint8);npred=np.full(n,-1,dtype=np.int8);pred=np.full(n,-1,dtype=np.int8)
    fitted=result["model_trained"]; checks={}; largest=[]
    for f in range(3):
        folder=exp/("fold_"+str(f));r=json.loads((folder/"report.json").read_text(encoding="utf-8"))
        path=folder/"outer_predictions.parquet"
        checks["fold_{}_prediction_file_identity".format(f)]=base.file_hash(path)==r["outer_prediction_sha256"]
        t=pq.read_table(path);pos=t["row_position"].to_numpy()
        counts[pos]+=1
        checks["fold_{}_roles_labels".format(f)]=bool(np.all(outer[pos]==f) and np.array_equal(t["label_index"].to_numpy(),y[pos]) and len(pos)==len(np.unique(pos)))
        probs=np.column_stack([t["N_p_"+k].to_numpy() for k in base.LABELS])
        npred[pos]=t["N_pred"].to_numpy()
        checks["fold_{}_N_probabilities_decisions".format(f)]=bool(np.isfinite(probs).all() and np.allclose(probs.sum(axis=1),1) and (probs>=0).all() and np.array_equal(probs.argmax(axis=1),npred[pos]))
        if "calibration_prediction_sha256" in r:
            cp=folder/"calibration_predictions.parquet";ct=pq.read_table(cp);cpos=ct["row_position"].to_numpy()
            inner=split["inner_for_outer_"+str(f)].to_numpy()
            checks["fold_{}_calibration_identity_roles".format(f)]=bool(base.file_hash(cp)==r["calibration_prediction_sha256"] and
              np.array_equal(cpos,np.flatnonzero(inner==4)) and np.array_equal(ct["label_index"].to_numpy(),y[cpos]))
            for model_name in (["N","C1_W"] if fitted else ["N"]):
                crisk=1-np.asarray(ct[model_name+"_p_benign"].to_numpy(),dtype=np.float64)
                orisk=1-np.asarray(t[model_name+"_p_benign"].to_numpy(),dtype=np.float64)
                for index,item in enumerate(r["risk_thresholds"]):
                    normal=crisk[y[cpos]==0];budget=item["requested_calibration_fpr_budget"]
                    # Ascending partition implements the same declared budget without calling the selector.
                    permitted=int(np.floor(len(normal)*budget));k=len(normal)-permitted-1
                    expected=float(np.nextafter(np.partition(normal,k)[k],np.inf))
                    actual=item[model_name+"_calibration"]["threshold"]
                    okay=expected==actual
                    for scope,positions,scores in [("calibration",cpos,crisk),("outer",pos,orisk)]:
                        target=item[model_name+"_"+scope]
                        alarm=scores>=actual
                        okay &= int(alarm.sum())==target["alert_rows"]
                        for c,label in enumerate(base.LABELS):
                            selected=y[positions]==c
                            observed=float(alarm[selected].mean()) if selected.any() else None
                            reported=target["class_alert_rates"][label]
                            okay &= (observed is None and reported is None) or (observed is not None and reported is not None and abs(observed-reported)<1e-12)
                    checks["fold_{}_{}_risk_budget_{}".format(f,model_name,index)]=bool(okay)
        if fitted:
            checks["fold_{}_frozen_model_files".format(f)]=base.file_hash(folder/"model.joblib")==r["model_sha256"] and base.file_hash(folder/"vectorizer.joblib")==r["vectorizer_sha256"]
            stress_report=json.loads((folder/"model_stress.json").read_text(encoding="utf-8"))
            checks["fold_{}_model_stress_invariants".format(f)]=stress_report["all_invariance_checks_passed"]
            p=np.column_stack([t["C1_W_p_"+k].to_numpy() for k in base.LABELS])
            pred[pos]=t["C1_W_pred"].to_numpy()
            checks["fold_{}_C1_probabilities_decisions".format(f)]=bool(np.isfinite(p).all() and np.allclose(p.sum(axis=1),1) and (p>=0).all() and np.array_equal(p.argmax(axis=1),pred[pos]))
            for c,label in enumerate(base.LABELS):
                ids,sizes=np.unique(g[pos[(y[pos]==c)&info[pos]]],return_counts=True)
                biggest=int(ids[np.argmax(sizes)])
                keep=pos[g[pos]!=biggest]
                largest.append({"fold":f,"class_selecting_largest_group":label,"group_id":biggest,
                  "removed_rows":int(len(pos)-len(keep)),"result":paired_summary(y[keep],pred[keep],npred[keep])})
    checks["all_rows_exactly_once"]=bool((counts==1).all())
    checks["N_confusion_matches_report"]=metrics(y,npred)["confusion_matrix"]==result["N"]["confusion_matrix"]
    report={"scope":"Completed-round replay and development-only diagnostics; no independent environment or retrained source/template holdout",
      "checks":checks,"rows":n,"N":metrics(y,npred),"model_trained":fitted,
      "probability_calibrated":False,"transfer_validated":False,"corpus_sha256":config["corpus_sha256"],"split_sha256":config["split_sha256"]}
    # Each nonempty approximate group has mass one; empty text is reported apart.
    counts_per_group=np.bincount(g[info]);weight=1.0/counts_per_group[g[info]]
    def group_equal(p):
        cm=np.bincount(y[info]*3+p[info],weights=weight,minlength=9).reshape(3,3)
        return {"weighted_confusion_matrix":cm.tolist(),"macro_f1":macro(cm)}
    report["nonempty_group_equal"]={"N":group_equal(npred),"groups":len(counts_per_group[counts_per_group>0]),
      "scope":"Each approximate nonempty group total weight one; mixed labels retained; not true independent incident sampling. Full-row scores remain primary."}
    if fitted:
        checks["C1_confusion_matches_report"]=metrics(y,pred)["confusion_matrix"]==result["C1_W"]["confusion_matrix"]
        report["paired"]=paired_summary(y,pred,npred)
        report["nonempty_group_equal"]["C1_W"]=group_equal(pred)
        report["without_largest_class_group"]=largest
        conflict=json.loads((root/"text_collisions.json").read_text(encoding="utf-8"))
        cmask=np.isin(g,[v["group_id"] for v in conflict["all_conflict_groups"]])
        report["conflict_slice"]=paired_summary(y[cmask],pred[cmask],npred[cmask]) if cmask.any() else {"rows":0}
        # Descriptive paired resampling of observed nonempty groups, retaining row mass.
        # Empty row units do not count as independently validated normal events.
        pos=np.flatnonzero(info);ids,inverse=np.unique(g[pos],return_inverse=True)
        a=np.bincount(inverse*9+y[pos]*3+pred[pos],minlength=len(ids)*9).reshape(-1,9)
        b=np.bincount(inverse*9+y[pos]*3+npred[pos],minlength=len(ids)*9).reshape(-1,9)
        rng=np.random.RandomState(20260912);samples=[]
        for _ in range(200):
            chosen=rng.randint(0,len(ids),size=len(ids))
            ma=macro(a[chosen].sum(axis=0).reshape(3,3));mb=macro(b[chosen].sum(axis=0).reshape(3,3))
            samples.append([ma,mb,ma-mb])
        arr=np.asarray(samples);good=np.isfinite(arr).all(axis=1)
        report["nonempty_group_bootstrap"]={"replicates":200,"valid_replicates":int(good.sum()),
          "groups":len(ids),"rows":len(pos),"observed":paired_summary(y[pos],pred[pos],npred[pos]),
          "percentile_95_intervals":dict(zip(["C1_macro_f1","N_macro_f1","paired_difference"],np.percentile(arr[good],[2.5,97.5],axis=0).T.tolist())),
          "limitations":"Descriptive conditional on fitted OOF predictions and these approximate groups. Ignores retraining variability, cross-fit dependence and unknown real incident dependence; not transfer or low-FPR assurance. Excludes empty text and does not replace full-row main scores."}
    report["all_checks_passed"]=all(checks.values())
    with (exp/"independent_round_review.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"all_checks_passed":report["all_checks_passed"],"model_trained":fitted,"rows":n},ensure_ascii=False),flush=True)
    if not report["all_checks_passed"]:raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True);p.add_argument("--experiment-dir",required=True)
    run(p.parse_args())
