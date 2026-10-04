"""Recount actual failed V158 classification cases, never execute models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v138_closeout import pair

OUT=ROOT/'artifacts/v158_fusion_trial_20261001'
CASE=ROOT/'training/review_policy/v158_observed_fusion_cases.json'
REPLAY=ROOT/'artifacts/v158_observed_fusion_case_replay_20261001'

def main():
    case=read(CASE);check_bindings(case['source_sha256'])
    d=read(OUT/'final_delivery.json');q=read(OUT/'quality.json');v=read(OUT/'verification.json')
    a=pd.read_parquet(OUT/'ASA_prediction_ledger.parquet');full=pd.read_parquet(OUT/'full_prediction_ledger.parquet')
    assert len(full)==2056871 and full.row_position.is_unique and len(a)==112807 and a.row_position.is_unique
    assert full.truth.value_counts().sort_index().tolist()==[1899723,111728,45420]
    assert pair(a,'pred_A','pred_B')==d['original_pairs']['B_vs_A']
    assert pair(a,'pred_A0','pred_B')==d['original_pairs']['B_vs_A0']
    assert pair(a,'pred_V146_A','pred_B')==d['original_pairs']['B_vs_V146_A']
    assert d['classifier_fits']==45+9+6==60 and d['fusion_full_gradients']==d['fusion_proposals']==d['fusion_updates']==1200
    assert v['actual_evaluation_classifier_calls']==480 and v['new_gradients']==0
    assert all(d['TRAIN'][arm]['all_roles_mastered'] for arm in ['A','B'])
    assert not q['matched_effect_passed'] and not q['task_acceptance'] and not d['model_promoted']
    assert q['matched_gates']['M_no_increase'] and not q['matched_gates']['S_no_increase']
    assert not q['matched_gates']['two_folds_improve']
    assert sum(a.pred_A.ne(a.truth))==sum(a.pred_B.ne(a.truth))==2286
    assert d['original_pairs']['B_vs_A']['1']['repairs']==10 and d['original_pairs']['B_vs_A']['2']['new_errors']==10
    flipped=a[a.pred_A.ne(a.pred_B)]
    assert len(flipped)==20 and set(flipped.loc[flipped.truth.eq(2),'root'])=={29}
    assert flipped.loc[flipped.truth.eq(2),'pred_B'].eq(1).all()
    cohorts=read(OUT/'all_fixed_cohort_quality.json');cohort={z['cohort']:z for z in cohorts['cohorts']}
    assert cohort['hard578']['original_rows']==578 and cohort['hard578']['paired']['2']['after_errors']==578
    assert cohort['strict51']['original_rows']==51 and cohort['strict51']['paired']['2']['after_errors']==49
    assert cohort['all17_strict_S_less_M']['original_rows']==188
    blocked=pd.read_parquet(OUT/'fixed_cohort_all17_strict_S_less_M_all_original_rows.parquet')
    bankdir=ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'
    for f in range(3):
        part=blocked[blocked.fold.eq(f)];bank=np.load(bankdir/f'fold{f}/deployment_probabilities.npy')
        margin=bank[part.local,:,2]-bank[part.local,:,1]
        assert np.all(margin<0) and not part[['pred_A','pred_B']].eq(2).any().any()
    zeros=read(ROOT/'artifacts/v158_zero_update_stage_identity_audit_20261001/audit.json')
    assert zeros['new_fits']==0
    failed=[k for k,b in q['B']['gates'].items() if not b]
    assert failed==case['failed_task_gates']
    # Lower original-frequency OOF loss is real, but does not license class
    # regression or replace the fixed endpoint with a selected earlier state.
    for f in range(3):
        for arm in ['A','B']:
            r=read(OUT/f'fold{f}_{arm}/fit.json')
            assert r['final_OOF_CE']<r['initial_OOF_CE'] and r['accepted_updates']==200
            assert len(r['last5'])==5 and len({z['parameter_sha256'] for z in r['last5']})==5
            assert all(z['stats']['mastered'] and z['stats']['new_errors_vs_start']==0 for z in r['last5'])
    check_bindings(case['source_sha256']);assert not REPLAY.exists();REPLAY.mkdir()
    result=dict(status='actual_V158_full_population_class_tradeoff_and_convex_limits_replayed',
        official_classifier_calls=0,new_features=0,new_gradients=0,new_fits=0,new_updates=0,
        complete_rows=2056871,ASA_rows=112807,failed_task_gates=failed,matched_gates=q['matched_gates'],
        trained_fits=60,quality_acceptance=False,model_promoted=False,
        class_tradeoff=dict(M_repairs=10,S_regressions=10,S_regression_roots=[29]),
        hard578_errors=578,strict51_errors=49,convex_unrepairable_S_rows=188,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),CASE]})
    (REPLAY/'replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
