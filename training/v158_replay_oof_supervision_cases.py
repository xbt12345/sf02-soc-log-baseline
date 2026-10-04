"""Recount legal OOF class sacrifice and fixed support counterexamples.

This is diagnosis of completed V158. No model, feature function, gradient,
new fit, class weight, or selection rule is evaluated or changed.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings

TRIAL=ROOT/'artifacts/v158_fusion_trial_20261001'
BANK=ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'
PARENT=ROOT/'artifacts/v158_independent_fusion_result_audit_v2_20261001'
FIXED=ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet'
OUT=ROOT/'artifacts/v158_oof_supervision_case_replay_20261001'
CASE=ROOT/'training/review_policy/v158_oof_supervision_and_support_cases.json'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def statistics(frame,prob):
    y=frame.truth.to_numpy();q=prob[frame.local.to_numpy()]
    assert np.isfinite(q).all() and np.allclose(q.sum(1),1,atol=1e-12,rtol=0)
    pred=q.argmax(1);ce=-np.log(np.maximum(q[np.arange(len(y)),y],np.finfo(np.float64).tiny))
    brier=((q-np.eye(3)[y])**2).sum(1)
    return dict(mean_CE=float(ce.mean()),classes={str(c):dict(support=int((y==c).sum()),errors=int(((y==c)&(pred!=c)).sum()),
        mean_CE=float(ce[y==c].mean()),mean_Brier=float(brier[y==c].mean())) for c in [1,2]})

def main():
    paths=[Path(__file__).resolve(),TRIAL/'final_delivery.json',TRIAL/'learning_qualification.json',
        PARENT/'audit.json',PARENT/'OOF_saved_checkpoint_learning.json',PARENT/'OOF_supervision_tradeoff_and_support_counterexamples.json',
        PARENT/'all_original_ASA_initial_endpoint_and_capacity.parquet',FIXED]
    for f in range(3):
        paths.append(BANK/f'fold{f}/legal_FIT_reference.parquet')
        for arm in ['A','B']:
            paths.extend([TRIAL/f'preflight{f}_{arm}/OOF_probability.npy',TRIAL/f'fold{f}_{arm}/endpoint_OOF_probability.npy',TRIAL/f'fold{f}_{arm}/fit.json'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    if OUT.exists():check_bindings(read(OUT/'pre_saved_array_bindings.json')['source_sha256'])
    else:OUT.mkdir();save(OUT/'pre_saved_array_bindings.json',dict(status='bound_before_original_OOF_class_recount',source_sha256=bindings))
    delivery=read(TRIAL/'final_delivery.json');assert delivery['classifier_fits']==60 and not delivery['quality_acceptance']
    assert all(delivery['TRAIN'][a]['all_roles_mastered'] for a in ['A','B'])
    parent=read(PARENT/'OOF_supervision_tradeoff_and_support_counterexamples.json')
    checkpoint=read(PARENT/'OOF_saved_checkpoint_learning.json')
    cases=[];complete=[];original_roles=0
    for f in range(3):
        frame=pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet')
        assert frame.fold.ne(f).all() and frame.training_role.eq(f).all();original_roles+=len(frame)
        for arm in ['A','B']:
            initial=statistics(frame,np.load(TRIAL/f'preflight{f}_{arm}/OOF_probability.npy'))
            terminal=statistics(frame,np.load(TRIAL/f'fold{f}_{arm}/endpoint_OOF_probability.npy'))
            fit=read(TRIAL/f'fold{f}_{arm}/fit.json')
            assert abs(initial['mean_CE']-fit['initial_OOF_CE'])<1e-12 and abs(terminal['mean_CE']-fit['final_OOF_CE'])<1e-12
            saved=[z for z in checkpoint if z['fold']==f and z['arm']==arm and z['update']==200]
            assert len(saved)==1 and abs(saved[0]['mean_CE']-terminal['mean_CE'])<1e-12
            for c in ['1','2']:
                assert saved[0]['classes'][c]['support']==terminal['classes'][c]['support']
                assert saved[0]['classes'][c]['errors']==terminal['classes'][c]['errors']
                assert abs(saved[0]['classes'][c]['mean_CE']-terminal['classes'][c]['mean_CE'])<1e-12
            result=dict(fold=f,arm=arm,total_CE_decreased=terminal['mean_CE']<initial['mean_CE'],
                M_error_delta=terminal['classes']['1']['errors']-initial['classes']['1']['errors'],
                S_error_delta=terminal['classes']['2']['errors']-initial['classes']['2']['errors'],
                M_CE_delta=terminal['classes']['1']['mean_CE']-initial['classes']['1']['mean_CE'],
                S_CE_delta=terminal['classes']['2']['mean_CE']-initial['classes']['2']['mean_CE'])
            cases.append(result);complete.append(dict(fold=f,arm=arm,initial=initial,terminal=terminal))
    assert original_roles==225614
    assert len(parent['class_tradeoff_cases'])==6
    for computed,reported in zip(cases,parent['class_tradeoff_cases']):
        for key,value in computed.items():
            if isinstance(value,float):assert abs(value-reported[key])<1e-12
            else:assert value==reported[key]
    # Two failing roles do not license hiding the improved role.
    b=[z for z in cases if z['arm']=='B']
    assert [z['S_error_delta'] for z in b]==[50,30,-38]
    assert all(z['total_CE_decreased'] for z in cases)
    assert b[0]['S_CE_delta']>0 and b[1]['S_CE_delta']>0
    assert b[2]['M_error_delta']<0 and b[2]['S_error_delta']<0 and b[2]['M_CE_delta']<0 and b[2]['S_CE_delta']<0
    fixed=pd.read_parquet(FIXED);capacity=pd.read_parquet(PARENT/'all_original_ASA_initial_endpoint_and_capacity.parquet')
    assert np.array_equal(fixed.row_position,capacity.row_position) and np.array_equal(fixed.truth,capacity.truth)
    masks=dict(blocked_S=capacity.truth.eq(2)&capacity.strict_convex_wrong_margin,
        remaining_S_with_correct_expert=capacity.truth.eq(2)&capacity.pred_B.ne(2)&capacity.some_expert_correct,
        hard578=fixed.known_578_cohort)
    support={}
    for name,mask in masks.items():
        p=fixed[mask];support[name]=dict(original_rows=len(p),
            exact_same_class_support_rows=int(p.historical_support_exact_S_rows.gt(0).sum()),
            destination_same_class_support_rows=int(p.historical_support_destination_S_rows.gt(0).sum()),
            coarse_behavior_same_class_support_rows=int(p.historical_support_behavior_S_rows.gt(0).sum()),source_roots=int(p.root.nunique()))
    assert support==parent['fixed_support_diagnostics']
    assert support['blocked_S']['original_rows']==support['blocked_S']['coarse_behavior_same_class_support_rows']==979
    assert support['blocked_S']['exact_same_class_support_rows']==36 and support['blocked_S']['destination_same_class_support_rows']==126
    assert support['hard578']['exact_same_class_support_rows']==0 and support['hard578']['coarse_behavior_same_class_support_rows']==578
    check_bindings(bindings)
    policy=dict(version='V158_actual_OOF_class_supervision_and_support_cases',source_sha256=bindings,
        original_OOF_role_rows=225614,cases=cases,fixed_support_diagnostics=support,
        actions=['Keep OOF supervision and deployment FIT protection as distinct populations.',
            'Do not infer per-class supervision success from lower overall CE or legal FIT guard success.',
            'Preserve the role2 B two-class improvement alongside role0/1 S regressions.',
            'Treat exact/destination/coarse support and source coverage as separate diagnostics; no causal claim.',
            'Do not alter completed V158 weights, class rules, endpoints, or budgets; new training requires separate qualification.'],
        new_formal_fit_qualification=False)
    if CASE.exists():assert read(CASE)==policy
    else:save(CASE,policy)
    receipt=dict(status='actual_full_original_OOF_class_sacrifice_and_support_granularity_replayed',
        original_OOF_role_rows=225614,six_completed_fusions=6,class_tradeoff_cases=cases,fixed_support_diagnostics=support,
        all_six_total_CE_decreased=True,deployment_FIT_retained=True,OOF_all_classes_mastered=False,
        own_classifier_calls=0,own_feature_calls=0,own_gradients=0,own_fits=0,own_updates=0,
        quality_acceptance=False,model_promoted=False,new_formal_fit_qualification=False,
        source_bindings_sha256=sha(OUT/'pre_saved_array_bindings.json'),case_sha256=sha(CASE))
    if (OUT/'replay.json').exists():assert read(OUT/'replay.json')==receipt
    else:save(OUT/'original_class_risk_endpoints.json',complete);save(OUT/'replay.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='class_tradeoff_cases'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
