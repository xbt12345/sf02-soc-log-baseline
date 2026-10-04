"""Independent saved-decision and official-truth audit for V127; no fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v126_frozen_audit import ROOT, PARENT, sha, read, save
from v125_evaluate import TRACE, OFFICIAL, ROWS, FOLDS
from v127_train import OUT, PLAN
from v127_experiment_review import require_run_seal, require_checkpoint, ReviewError
from experiment_review import check_bindings


def require(ok,message):
    if not ok:raise ValueError(message)


def per_class(y,p):
    result={}
    for c in (0,1,2):
        support=int((y==c).sum());called=int((p==c).sum());correct=int(((y==c)&(p==c)).sum())
        result[str(c)]={'support':support,'correct':correct,'missed':support-correct,
                        'false_called':called-correct,
                        'recall':correct/support if support else None,
                        'precision':correct/called if called else None,
                        'f1':2*correct/(support+called) if support and support+called else None}
    return result


def same_metrics(a,b):
    for c in ('0','1','2'):
        for key,value in a[c].items():
            other=b[c][key]
            if value is None:require(other is None,'Undefined metric replaced')
            elif isinstance(value,float):require(abs(value-other)<1e-12,'Metric mismatch '+str((c,key)))
            else:require(value==other,'Metric count mismatch '+str((c,key)))


def main():
    plan=require_run_seal(OUT/'run_seal.json',ROOT/'training/v127_train.py')
    check_bindings(read(PARENT/'run_seal.json')['source_sha256'])
    target=OUT/'verification.json'
    if target.exists():raise FileExistsError(target)
    report=read(OUT/'primary_evaluation.json')
    require(report['status']=='v127_nine_fit_source_closed_evaluated' and report['classifier_fits_new']==9
            and report['network_optimizer_steps']==17800,'Incomplete training report')
    original=pd.read_parquet(OFFICIAL,columns=['event_id','label_binary'])
    rows=pd.read_parquet(ROWS,columns=['row_position','event_id','label_index','route'])
    folds=pd.read_parquet(FOLDS,columns=['row_position','root','proposed_fold'])
    trace=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    expert=pd.read_parquet(OUT/'expert_ASA_predictions.parquet')
    full=pd.read_parquet(OUT/'full_prediction_ledger.parquet')
    n=len(original);asa=np.flatnonzero(rows.route.eq('asa'))
    truth=original.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(dtype=np.int8)
    require(n==2056871 and len(rows)==len(folds)==len(full)==n and len(asa)==len(trace)==len(expert)==112807,
            'Full or ASA population wrong')
    require(np.array_equal(rows.row_position,np.arange(n)) and np.array_equal(full.row_position,rows.row_position)
            and np.array_equal(original.event_id,rows.event_id)
            and np.array_equal(truth,rows.label_index.to_numpy(np.int8)),'Official event or truth changed')
    require(np.array_equal(asa,trace.row_position) and trace.row_position.equals(expert.row_position)
            and np.array_equal(truth[asa],trace.truth) and np.array_equal(folds.root.to_numpy()[asa],trace.root)
            and np.array_equal(folds.proposed_fold.to_numpy()[asa],trace.fold), 'Source or ASA identity changed')
    role=trace.fold.to_numpy();local=trace.local.to_numpy();y=truth[asa]
    arms=('A0','AH','K','W','P');pred={}
    baseline=pd.read_parquet(PARENT/'expert_ASA_predictions.parquet',columns=['row_position','expert_pred_A'])
    require(np.array_equal(baseline.row_position,trace.row_position),'Old baseline identity changed')
    for arm in arms:
        p=expert['pred_'+arm].to_numpy()
        require(np.isin(p,(0,1,2)).all() and np.array_equal(full['final_pred_'+arm].to_numpy()[asa],p),
                'ASA prediction ledger mismatch '+arm)
        pred[arm]=p
        same_metrics(per_class(y,p),report['metrics']['ASA'][arm])
        same_metrics(per_class(truth,full['final_pred_'+arm].to_numpy()),report['metrics']['full_task'][arm])
    require(np.array_equal(pred['A0'],baseline.expert_pred_A),'Hard reference changed')
    nonasa=~rows.route.eq('asa').to_numpy()
    require(int((full.teacher_prediction.to_numpy()[nonasa]!=truth[nonasa]).sum())==107,
            'Non-ASA reference changed')
    for arm in arms:
        require(np.array_equal(full['final_pred_'+arm].to_numpy()[nonasa],full.teacher_prediction.to_numpy()[nonasa]),
                'Collateral route changed '+arm)
    for fold in range(3):
        subset=role==fold
        for arm in arms:
            if arm=='A0':path=PARENT/f'fold{fold}_A/epoch25_prob.npy'
            elif arm=='AH':path=ROOT/'artifacts/v127_plan_review_20260929'/f'fold{fold}_A_header_probability.npy'
            elif arm=='K':path=OUT/f'fold{fold}_K/endpoint_prob.npy'
            else:path=OUT/f'fold{fold}_{arm}/epoch50_prob.npy'
            p=np.load(path)
            require(p.shape==(22546,3) and np.allclose(p.sum(1),1,atol=2e-6)
                    and np.array_equal(p[local[subset]].argmax(1),pred[arm][subset]),
                    'Saved probability decisions mismatch '+str((fold,arm)))
        for arm in ('W','P'):
            folder=OUT/f'fold{fold}_{arm}';fit=read(folder/'fit.json')
            require_checkpoint(plan,fit)
            require(fit['optimizer_steps']==plan['training']['expected_steps_per_arm'][fold],
                    'Network update count changed')
            with (folder/'steps.jsonl').open(encoding='utf-8') as stream:
                line_count=sum(1 for _ in stream)
            require(line_count==fit['optimizer_steps'],'Per-step log incomplete')
            progress=read(folder/'progress.json')
            require(len(progress)==50 and progress[-1]['steps']==fit['optimizer_steps'],
                    'Epoch log incomplete')
        k=read(OUT/f'fold{fold}_K/fit.json')
        require(k['status']=='fit_executed' and k['gradient_inf']<=1e-7,'K did not converge')
    counts={arm:{str(c):int(((y==c)&(pred[arm]!=c)).sum()) for c in (1,2)} for arm in arms}
    for arm in arms:
        require(sum(counts[arm].values())==report['ASA_total_errors'][arm], 'ASA wrong count differs')
    require(counts['A0']=={'1':318,'2':2074} and counts['AH']=={'1':318,'2':2114},
            'Registered hard/projection baseline changed')
    for arm in ('AH','K','W','P'):
        for c in (1,2):
            mask=y==c
            repair=int((mask&(pred['A0']!=c)&(pred[arm]==c)).sum())
            regress=int((mask&(pred['A0']==c)&(pred[arm]!=c)).sum())
            require({'repaired':repair,'regressed':regress}==report['paired_changes_vs_A0'][arm][str(c)],
                    'Paired transition mismatch')
    require(report['confirmation_allowed']==report['primary_quality_passed']==all(report['quality_gates'].values()),
            'Quality flag overrides gates')
    invalid=dict(status='fit_executed',completed_epochs=25,prediction_epoch=25)
    try:require_checkpoint(plan,invalid)
    except ReviewError:early_endpoint_rejected=True
    else:raise ValueError('Early endpoint passed V127 checker')
    all_benign=np.zeros_like(truth)
    require(per_class(truth,all_benign)['1']['recall']==0 and per_class(truth,all_benign)['2']['recall']==0,
            'All-normal adversarial baseline not exposed')
    files=[q for q in OUT.rglob('*') if q.is_file() and q!=target]
    files += [Path(__file__),ROOT/'training/v127_evaluate.py',ROOT/'training/v127_dynamics.py']
    hashes={q.relative_to(ROOT).as_posix():sha(q) for q in sorted(files)}
    result={'status':'V127_independent_full_row_audit_passed','latest_actual_training':'V127',
            'full_official_rows':n,'ASA_rows':len(trace),'frozen_non_ASA_wrong':107,
            'nine_fits_counted':9,'network_steps_counted':17800,
            'ASA_errors_independent':counts,'quality_gates_passed':report['primary_quality_passed'],
            'confirmation_allowed':report['confirmation_allowed'],
            'early_endpoint_rejected':early_endpoint_rejected,'all_normal_bad_class_recall_rejected':True,
            'model_promoted':False,'artifact_sha256':hashes,
            'scope':'Independent saved-decision, fit-identity and official-truth recount; model forward replays are in the evaluation. Local development folds are inspected, not a blind test.'}
    save(target,result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
