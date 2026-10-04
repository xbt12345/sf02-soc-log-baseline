"""Audit the proposed v102 folds and explanatory controls, without any fits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v102_root_review import ROOT, OLD, behavior, sha
from v102_verify_and_coverage import graph_roots

DEST = ROOT / 'artifacts/v103_plan_preflight_20260928'


def fold_of(root):
    return int(hashlib.sha256(f'10203:{int(root)}'.encode()).hexdigest()[:8],16)%3


def count_support(frame, train_mask, query, col, name):
    fits=frame[train_mask & frame[col].notna()]
    counts=fits.groupby([col,'label_index']).root.nunique().to_dict()
    return np.array([counts.get((b,int(c)),0) if b is not None else -1
                     for b,c in zip(query[col],query.label_index)],dtype=np.int32)


def main():
    DEST.mkdir(exist_ok=False)
    paths=[ROOT/'artifacts/v102_root_review_20260928/next_experiment_contract.json',
        ROOT/'artifacts/v75_four_arm_20260921_r2/rows.parquet',
        ROOT/'artifacts/v79_execution_20260927/row_feature_id.npy',
        OLD/'N1_ASA_group_map.parquet', OLD/'source_rows.parquet',
        ROOT/'artifacts/v75_four_arm_20260921_r2/projections.parquet',
        ROOT/'artifacts/v102_root_review_20260928/source_error_and_support_ledger.parquet',
        ROOT/'artifacts/v102_root_review_20260928/new_non_source_support_rows.parquet']
    plan=json.loads(paths[0].read_text());assert plan['split']['seed']==10203
    r=pd.read_parquet(paths[1],columns=['row_position','component','projection_id','route','label_index'])
    r['fid']=np.load(paths[2],mmap_mode='r')
    n1=pd.read_parquet(paths[3])
    a=r[r.route=='asa'].merge(n1[['row_position','N1_group']],on='row_position',validate='1:1')
    r['root']=graph_roots(r,a)
    units=np.unique(r.root);foldmap={int(c):fold_of(c) for c in units}
    r['proposed_fold']=r.root.map(foldmap).astype(np.int8)
    oldsrc=pd.read_parquet(paths[4],columns=['row_position'])
    r['old_source']=r.row_position.isin(oldsrc.row_position)
    for col in ['root','proposed_fold','old_source']:
        a[col]=r.set_index('row_position').loc[a.row_position,col].to_numpy()
    facts=pd.read_parquet(paths[5],columns=['facts'])
    keys={int(p):behavior(json.loads(facts.facts.iat[int(p)])) for p in a.projection_id.unique()}
    a['behavior']=a.projection_id.map(keys)
    def coarse(b):
        if b is None:return None
        j=json.loads(b)
        return json.dumps({k:v for k,v in j.items() if k not in ('dst_port_fixed','icmp_type','icmp_code')},sort_keys=True)
    a['coarse_behavior']=a.behavior.map(coarse)
    for column in ['small_exact_support','full_exact_support','small_coarse_support','full_coarse_support']:
        a[column]=-1
    populations=[]
    for fold in range(3):
        query=a[a.proposed_fold==fold]
        for name,mask in [('small',a.old_source & (a.proposed_fold!=fold)),('full',a.proposed_fold!=fold)]:
            for key,col in [('exact','behavior'),('coarse','coarse_behavior')]:
                a.loc[query.index,f'{name}_{key}_support']=count_support(a,mask,query,col,name)
        for scope,m in [('small_train',r.old_source&(r.proposed_fold!=fold)),('full_train',r.proposed_fold!=fold),('common_hold',r.proposed_fold==fold)]:
            rows=r[m];asa=rows[rows.route=='asa'];cnt=np.bincount(rows.label_index,minlength=3)
            counts=np.bincount(asa.label_index,minlength=3)
            populations.append({'fold':fold,'scope':scope,'all_rows':len(rows),'all_class_counts':cnt.tolist(),
                'all_class_proportions':(cnt/max(len(rows),1)).tolist(),'ASA_rows':len(asa),'ASA_class_counts':counts.tolist(),
                'ASA_M_S_ratio':float(counts[1]/counts[2]) if counts[2] else None,
                'ASA_S_roots':int(asa[asa.label_index==2].root.nunique()),
                'largest_S_root_fraction':float(asa[asa.label_index==2].groupby('root').size().max()/max(counts[2],1))})
    old=pd.read_parquet(paths[6]);more=pd.read_parquet(paths[7])
    focus=old[(old.label_index==2)&(old.R0_MAG_200!=2)].merge(more[more.new_non_source_same_class_roots>0][['row_position']],on='row_position',validate='1:1')
    assert len(focus)==113
    target=a[a.row_position.isin(focus.row_position)].copy()
    target[['row_position','root','proposed_fold','label_index','behavior','small_exact_support','full_exact_support']].to_parquet(DEST/'113_target_support_after_split.parquet',index=False)
    # These are attributes of OLD predictions, not new-model outcomes.
    olderrors=old[old.R0_MAG_200!=old.label_index]
    merged=olderrors[['row_position','label_index','support_bucket']].merge(a[['row_position','root','proposed_fold','small_exact_support','full_exact_support','small_coarse_support','full_coarse_support']],on='row_position',validate='1:1')
    merged.to_parquet(DEST/'historic_errors_new_split_support.parquet',index=False)
    prior=old.copy()
    prior['coarse_behavior']=prior.behavior.map(coarse)
    prior['coarse_train_support']=-1
    for fold in range(3):
        g=prior[(prior.fold!=fold)&prior.coarse_behavior.notna()]
        counts=g.groupby(['coarse_behavior','label_index']).merged_root.nunique().to_dict()
        take=prior.fold==fold
        prior.loc[take,'coarse_train_support']=[counts.get((b,int(c)),0) if b is not None else -1 for b,c in zip(prior.loc[take,'coarse_behavior'],prior.loc[take,'label_index'])]
    badzero=prior[(prior.support_bucket=='zero_same_class')&(prior.R0_MAG_200!=prior.label_index)]
    key_controls=[]
    for cl,g in badzero.groupby('label_index'):
        key_controls.append({'class':int(cl),'old_errors_strict_zero_support':len(g),
            'has_coarse_same_class_support':int((g.coarse_train_support>0).sum()),
            'coarse_zero_support':int((g.coarse_train_support==0).sum())})
    leakchecks={key:bool(r.groupby(key).proposed_fold.nunique().max()==1) for key in ['root','component','fid']}
    leakchecks['N1_group']=bool(a.groupby('N1_group').proposed_fold.nunique().max()==1)
    assert all(leakchecks.values())
    r[['row_position','root','proposed_fold','old_source']].to_parquet(DEST/'v102_proposed_full_population_folds.parquet',index=False)
    a[['row_position','root','proposed_fold','label_index','behavior','small_exact_support','full_exact_support','small_coarse_support','full_coarse_support']].to_parquet(DEST/'ASA_support_preflight.parquet',index=False)
    result={'status':'v102_plan_preflight_no_fit','classifier_fits':0,'calibration_fits':0,
        'all_rows':len(r),'ASA_rows':len(a),'global_units':len(units),'seed':10203,'split_leakage_checks':leakchecks,
        'populations':populations,'historic_113_targets':{
            'total':len(target),'has_full_fold_train_support':int((target.full_exact_support>0).sum()),
            'still_no_fold_train_support':int((target.full_exact_support==0).sum()),
            'newly_gains_support_over_matched_small':int(((target.small_exact_support==0)&(target.full_exact_support>0)).sum()),
            'already_has_matched_small_support':int((target.small_exact_support>0).sum()),
            'two_or_more_full_train_units':int((target.full_exact_support>=2).sum())},
        'support_key_sensitivity_old_predictions':key_controls,
        'all_current_ASA_classes':a.label_index.value_counts().sort_index().to_dict(),
        'limitations':['No new predictions or model gains measured.','Global labels used to describe previously observed development data, not assign folds.','Coarse behavior is not sufficient M/S evidence; zero exact-key support is not zero transferable evidence.'],
        'source_sha256':sha(__file__),'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
    (DEST/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
