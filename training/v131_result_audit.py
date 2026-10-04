"""Post-selection source/support and matched-trajectory audit. Zero fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from v131_common import ROOT,OUT,REVIEW,OFFICIAL,read,save,sha,load_data,fit_context,require_run_seal
from v131_evaluate import load_reference
from experiment_review import class_counts


def count(frame,p):
    y=frame.truth.to_numpy()
    return {str(c):{'support':int((y==c).sum()),'errors':int(((y==c)&(p!=c)).sum())} for c in [1,2]}


def main():
    require_run_seal(ROOT/'training/v131_train.py')
    target=OUT/'result_audit'
    if target.exists():raise FileExistsError('Preserve prior audit')
    target.mkdir()
    binding={'source_sha256':sha(__file__),'delivery_sha256':sha(OUT/'delivery.json'),
        'run_seal_sha256':sha(OUT/'run_seal.json'),'new_fits':0,'new_updates':0}
    save(target/'binding.json',binding)
    x,d=load_data();ref=load_reference(d)
    predictions=pd.read_parquet(OUT/'ASA_prediction_ledger.parquet')
    if not predictions.row_position.equals(d.row_position) or not predictions.truth.equals(d.truth):raise ValueError('ASA alignment changed')
    ladder=pd.read_parquet(ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet')
    if not ladder.row_position.equals(d.row_position):raise ValueError('Support alignment changed')
    q=predictions.copy();q['canonical_key']=d.canonical_key
    columns=['facts_json','parameter_observed','full_fact_fit_M_rows','full_fact_fit_S_rows',
        'full_fact_fit_M_roots','full_fact_fit_S_roots','destination_M_rows','destination_S_rows',
        'destination_M_roots','destination_S_roots','behavior_M_rows','behavior_S_rows',
        'behavior_M_roots','behavior_S_roots','diagnostic_bucket','protocol']
    for col in columns:q[col]=ladder[col]
    q['canonical_TRAIN_M_rows']=0;q['canonical_TRAIN_S_rows']=0
    for f in range(3):
        train=d[d.fold!=f];mask=d.fold.eq(f)
        for c,name in [(1,'M'),(2,'S')]:
            masses=train[train.truth==c].groupby('canonical_key').size()
            q.loc[mask,f'canonical_TRAIN_{name}_rows']=d.loc[mask,'canonical_key'].map(masses).fillna(0).astype(int)
    y=q.truth.to_numpy();a=q.pred_A0.to_numpy();b=q.pred_R.to_numpy()
    q['error_status']=np.select([(a!=y)&(b==y),(a!=y)&(b!=y),(a==y)&(b!=y)],
        ['repaired','persistent','newly_wrong'],default='retained_correct')
    q.to_parquet(target/'ASA_error_changes_and_TRAIN_support.parquet',index=False)
    changes=[]
    for cl in [1,2]:
        z=q[q.truth==cl]
        for status in ['repaired','persistent','newly_wrong','retained_correct']:
            sub=z[z.error_status==status];name='M' if cl==1 else 'S'
            changes.append({'class':cl,'status':status,'rows':len(sub),'roots':int(sub.root.nunique()),
                'destination_same_class_zero':int(sub[f'destination_{name}_rows'].eq(0).sum()),
                'destination_same_class_single_root_or_none':int(sub[f'destination_{name}_roots'].lt(2).sum()),
                'full_fact_same_class_zero':int(sub[f'full_fact_fit_{name}_rows'].eq(0).sum()),
                'coarse_behavior_same_class_zero':int(sub[f'behavior_{name}_rows'].eq(0).sum()),
                'canonical_TRAIN_same_class_zero':int(sub[f'canonical_TRAIN_{name}_rows'].eq(0).sum())})
    roots=q.assign(wrong=b!=y,newly_wrong=q.error_status.eq('newly_wrong'),repaired=q.error_status.eq('repaired')).groupby(['root','truth']).agg(
        support=('row_position','size'),wrong=('wrong','sum'),newly_wrong=('newly_wrong','sum'),repaired=('repaired','sum')).reset_index()
    roots.to_csv(target/'root_error_changes.csv',index=False)
    trajectories=[]
    for ep in [25,50,75,100]:
        pred=np.zeros(len(d),np.int8)
        for f in range(3):
            cp=next(v for v in read(OUT/f'fold{f}_R/checkpoints.json') if v['epoch']==ep)
            file=OUT/f'fold{f}_R/epoch{ep}_sealed_all_prob.npy'
            if sha(file)!=cp['sealed_probability_sha256']:raise ValueError('Trajectory probability identity changed')
            p=np.load(file);mask=d.fold.eq(f).to_numpy();pred[mask]=p[d.loc[mask,'local'].to_numpy()].argmax(1)
        root2868=d.root.eq(2868).to_numpy()&(y==1)
        trajectories.append({'epoch':ep,'new_candidate_selected':False,'ASA':count(d,pred),
            'root2868_M_errors':int((pred[root2868]!=1).sum()),
            'scope':'Matched R same canonical input/seed/optimizer; saved intermediate predictions for diagnosis only, not independent intermediate-model replay.'})
    held_arms=[]
    mask=d.fold.eq(1).to_numpy()
    for arm in ['R','O','C','CO']:
        p=q['pred_'+arm].to_numpy()
        held_arms.append({'arm':arm,'held_fold':1,**count(d[mask],p[mask])})
    full=pd.read_parquet(OUT/'full_prediction_ledger.parquet')
    if not np.array_equal(full.row_position.to_numpy(),ref.row_position.to_numpy()):raise ValueError('Full scoring row mismatch')
    base=class_counts(ref.assign(pred=full.pred_A0.to_numpy()),'pred')
    current=class_counts(ref.assign(pred=full.pred_R.to_numpy()),'pred')
    headers=pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet').row_position
    h=q[q.row_position.isin(headers)];outside=q[q.truth.eq(2)&~q.root.isin([21702,20849,29])]
    unknown=q[q.truth.eq(2)&q.diagnostic_bucket.eq('unknown_parameter_pooled_support')]
    result={'status':'official_row_and_support_changes_recounted','binding':binding,
        'full_baseline':base,'full_candidate':current,'full_total_errors_A0':int((full.pred_A0.to_numpy()!=ref.truth.to_numpy()).sum()),
        'full_total_errors_R':int((full.pred_R.to_numpy()!=ref.truth.to_numpy()).sum()),
        'error_status_support':changes,'largest_new_errors':roots.sort_values('newly_wrong',ascending=False).head(16).to_dict('records'),
        'matched_R_transfer_trajectory':trajectories,'fold1_held_same_population_arms':held_arms,
        'header682':{'rows':len(h),'A0_errors':int(h.pred_A0.ne(h.truth).sum()),'R_errors':int(h.pred_R.ne(h.truth).sum()),
            'new_errors':int(h.error_status.eq('newly_wrong').sum())},
        'S_outside_top3':{'rows':len(outside),'A0_errors':int(outside.pred_A0.ne(2).sum()),'R_errors':int(outside.pred_R.ne(2).sum())},
        'unknown186_S':{'rows':len(unknown),'A0_errors':int(unknown.pred_A0.ne(2).sum()),'R_errors':int(unknown.pred_R.ne(2).sum())},
        'limitations':['Support is observational TRAIN label availability, not causal security truth.',
            'Alternative O/C/CO whole-population counts combine new fold1 with historical A0 elsewhere; compare alternatives only on common fold1 population.',
            'Intermediate checkpoints explain trajectories; cannot replace the fixed epoch100 candidate.']}
    save(target/'audit.json',result)
    save(target/'receipt.json',{'source_sha256':sha(__file__),'new_fits':0,'new_updates':0,
        'output_sha256':{p.name:sha(p) for p in target.iterdir() if p.is_file()}})
    print(json.dumps({'full_errors_A0':result['full_total_errors_A0'],'full_errors_R':result['full_total_errors_R'],
        'header682':result['header682'],'top_regression':result['largest_new_errors'][0]},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
