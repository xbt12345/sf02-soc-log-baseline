"""Independent full-population and targeted-slice recount after six V124 fits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v124_header_trial_20260929'
TRACE=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
LADDER=ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet'
OFFICIAL=ROOT/'data/official/train.parquet'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def metrics(y,p):
    out={}
    for c in (0,1,2):
        n=int((y==c).sum());tp=int(((y==c)&(p==c)).sum());fp=int(((y!=c)&(p==c)).sum())
        out[str(c)]={'support':n,'correct':tp,'missed':n-tp,'false_called':fp,
            'recall':tp/n if n else None,'precision':tp/(tp+fp) if tp+fp else None,
            'f1':2*tp/(n+tp+fp) if n+tp+fp else None}
    return out


def transitions(q,y,a,b):
    return {'rows':int(q.sum()),'A_errors':int(((a!=y)&q).sum()),
        'B_errors':int(((b!=y)&q).sum()),
        'repaired':int(((a!=y)&(b==y)&q).sum()),
        'regressed':int(((a==y)&(b!=y)&q).sum())}


def main():
    target=OUT/'independent_postfit_audit.json'
    if target.exists():raise FileExistsError(target)
    seal=read(OUT/'run_seal.json')
    for rel,h in seal['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    plan=read(ROOT/'training/review_policy/v124_header_trial.json')
    registration=read(OUT/'registration.json')
    evaluation=read(OUT/'primary_evaluation.json')
    assert sha(OUT/'run_seal.json')==registration['seal_sha256']
    assert evaluation['classifier_fits_new']==6 and evaluation['optimizer_steps']==8900
    expert=pd.read_parquet(OUT/'expert_ASA_predictions.parquet')
    full=pd.read_parquet(OUT/'full_prediction_ledger.parquet')
    trace=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    ladder=pd.read_parquet(LADDER,columns=['row_position','diagnostic_bucket','A_all_seven_wrong'])
    assert len(expert)==len(trace)==len(ladder)==112807
    for c in trace.columns:assert expert[c].equals(trace[c]),c
    assert expert.row_position.equals(ladder.row_position)
    official=pd.read_parquet(OFFICIAL,columns=['event_id','label_binary'])
    true=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(dtype=np.int8)
    assert len(official)==len(full)==2056871
    assert np.array_equal(full.row_position,np.arange(len(full)))
    fits=[]
    for fold in range(3):
        pair=[]
        for arm in ('A','B'):
            folder=OUT/f'fold{fold}_{arm}'
            q=read(folder/'fit.json');p=read(folder/'progress.json');check=read(folder/'checkpoints.json')
            assert q['completed_epochs']==q['prediction_epoch']==25 and len(p)==25 and len(check)==7
            assert q['optimizer_steps']==plan['expected_primary_schedule'][fold]['registered_optimizer_steps_per_arm']
            assert max(t['mass_reconstruction_max_error'] for t in p)<=1e-8
            assert sha(folder/'epoch25_model.pt')==q['model_sha256']
            assert sha(folder/'epoch25_prob.npy')==q['prob_sha256']
            assert [e['epoch'] for e in check]==[1,2,5,10,15,20,25]
            pair.append(q)
            fits.append({'fold':fold,'arm':arm,'optimizer_steps':q['optimizer_steps'],
                'epochs':q['completed_epochs'],'seconds':q['seconds'],
                'fit_M_correct':check[-1]['fit_by_class']['1']['correct'],
                'fit_S_correct':check[-1]['fit_by_class']['2']['correct'],
                'fit_M_ensemble_CE':check[-1]['fit_by_class']['1']['ensemble_CE'],
                'fit_S_ensemble_CE':check[-1]['fit_by_class']['2']['ensemble_CE']})
        assert pair[0]['initial_state_sha256']==pair[1]['initial_state_sha256']
    y=trace.truth.to_numpy(dtype=np.int8)
    a=expert.expert_pred_A.to_numpy(dtype=np.int8);b=expert.expert_pred_B.to_numpy(dtype=np.int8)
    assert np.array_equal(true[trace.row_position],y)
    fm={arm:metrics(true,full[f'final_pred_{arm}'].to_numpy(dtype=np.int8)) for arm in ('A','B')}
    am={arm:metrics(y,expert[f'expert_pred_{arm}'].to_numpy(dtype=np.int8)) for arm in ('A','B')}
    for arm in ('A','B'):
        for c in (0,1,2):
            got=fm[arm][str(c)];expected=evaluation['full_primary_recount']['full_task'][arm][str(c)]
            assert got['support']==expected['support'] and got['correct']==expected['correct']
        for c in (1,2):
            assert am[arm][str(c)]['correct']==evaluation['raw_expert_metrics'][arm][str(c)]['correct']
    gaprows=pd.read_parquet(OUT/'header_span_ledger.parquet',columns=['row_position'])
    gap=np.isin(expert.row_position,gaprows.row_position)
    oldpersistent=ladder.A_all_seven_wrong.to_numpy()&(y==2)
    slices={'header_682':gap,'remaining_ASA':~gap,'historical_persistent_S':oldpersistent,
        'S_outside_largest3_roots':(y==2)&~np.isin(trace.root,[21702,20849,29])}
    for name in ['known_parameter_two_roots_per_class','unknown_parameter_pooled_support',
                 'same_class_support_but_insufficient_dual_root_coverage','no_same_class_at_destination_resolution']:
        slices[name]=ladder.diagnostic_bucket.eq(name).to_numpy()
    for fold in range(3):
        for c in (1,2):slices[f'fold{fold}_class{c}']=(trace.fold.to_numpy()==fold)&(y==c)
    byslice={name:transitions(q,y,a,b) for name,q in slices.items()}
    group={}
    for arm,p in [('A',a),('B',b)]:
        s=pd.DataFrame({'root':trace.root[y==2],'correct':p[y==2]==2}).groupby('root').correct.agg(['sum','mean'])
        group[arm]={'roots':len(s),'mean_recall':float(s['mean'].mean()),
                    'zero_recall_roots':int(s['sum'].eq(0).sum())}
        assert group[arm]==evaluation['S_group_results'][arm]
    old=pd.read_parquet(ROOT/'artifacts/v121_paired_batch_training_20260929/expert_ASA_predictions.parquet',columns=['row_position','expert_pred_A'])
    assert np.array_equal(old.row_position,expert.row_position)
    oldp=old.expert_pred_A.to_numpy()
    drift={'old_A_errors':int((oldp!=y).sum()),'new_A_errors':int((a!=y).sum()),
           'class_M_old_new':[int(((oldp!=y)&(y==1)).sum()),int(((a!=y)&(y==1)).sum())],
           'class_S_old_new':[int(((oldp!=y)&(y==2)).sum()),int(((a!=y)&(y==2)).sum())]}
    preflight=read(OUT/'input_preflight.json')
    result={'status':'six_fits_independently_recounted','latest_actual_training':'V124',
        'classifier_fits_new':6,'optimizer_steps':8900,'calibration_fits':0,
        'quality_acceptance':False,'model_promoted':False,
        'primary_quality_passed':evaluation['primary_quality_passed'],
        'gates':evaluation['full_primary_recount']['gates'],
        'full_per_class':fm,'ASA_per_class':am,'slices':byslice,
        'S_group_results':group,'matched_baseline_drift':drift,
        'fits':fits,'input_contract':{k:preflight[k] for k in ['changed_rows','date_and_clock_invariance_checked_rows','event_body_bytes_changed','fact_metadata_nnz_changed','cross_fold_input_keys','empirical_collision_floor']},
        'qualification':{'eligible_capacity_factors':read(OUT/'capability_qualification.json')['eligible_capacity_factors'],
                         'priority_behavior_cards':read(OUT/'capability_qualification.json')['query_fold_behavior_cards']},
        'limitations':['Repeatedly inspected source-closed development folds; no independent external test.',
            'Historical error buckets are post hoc and not inference routes.',
            'Input invariance on registered variants does not prove label correctness.'],
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),OFFICIAL,TRACE,LADDER,
             OUT/'input_preflight.json',OUT/'capability_qualification.json',OUT/'registration.json',
             OUT/'primary_evaluation.json',OUT/'expert_ASA_predictions.parquet',OUT/'full_prediction_ledger.parquet',
             OUT/'run_seal.json']}}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'drift':drift,
        'ASA_M_errors':[am[z]['1']['missed'] for z in ('A','B')],
        'ASA_S_errors':[am[z]['2']['missed'] for z in ('A','B')],
        'gates':result['gates']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
