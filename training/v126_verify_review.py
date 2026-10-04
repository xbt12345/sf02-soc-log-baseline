"""Read-only evidence/plan replay; --record creates a review receipt, never a training seal."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v126_frozen_audit import ROOT, PARENT, OUT, read, sha, save
from v125_evaluate import TRACE, OFFICIAL
from v124_header import HEADER, old_text

PLAN = ROOT / 'training/review_policy/v126_next_training_plan.json'
REPORT = ROOT / 'docs/V126_BRANCH_COLLAPSE_REVIEW_AND_TRAINING_PLAN.md'
RISKS = ROOT / 'training/review_policy/v126_risk_actions.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def review_plan(p):
    require(p['status'] == 'design_only_runtime_not_implemented_or_sealed', 'Do not misstate implementation status')
    require(p['latest_actual_training'] == 'V125', 'Plan is not actual training')
    require(set(p['arms']) == {'A0', 'K', 'F', 'W', 'P'}, 'Missing registered control')
    b = p['base']; t = p['training']; q = p['primary_quality']
    require(b['frozen_parameters'] and b['frozen_buffers'] and b['shared_exact_base_across_arms'] and not b['optimizer_contains_base'], 'Base must remain identical and frozen')
    require('per-member' in b['cache'] and 'mean-member' in t['loss'], 'Do not collapse member logits before the registered objective')
    require(t['epochs'] == 50 and t['selector'].startswith('fixed epoch50'), 'Wrong endpoint')
    require(not any(t[k] for k in ['class_or_group_reweighting', 'base_unfreezing', 'adaptive_outer_selection', 'threshold_tuning', 'candidate_substitution', 'automatic_epoch_extension']), 'Unregistered adaptation')
    require(q['candidate'] == 'P' and 'sole' in p['arms']['P']['role'], 'Control cannot become candidate')
    require(p['arms']['K']['role'].startswith('nonpromotable'), 'Bias control is not a winner')
    require(len(t['fit_order']) == 12 and len({tuple(v) for v in t['fit_order']}) == 12, 'Incomplete or repeated fits')
    require(set(map(tuple,t['fit_order'])) == {(f,a) for f in (0,1,2) for a in 'KFWP'}, 'Missing fold/control')
    require(t['expected_steps_per_arm'] == [3650,1600,3650] and t['expected_primary_steps'] == 35600, 'Wrong update budget')
    require(q['ASA_M_errors_max'] == 318 and q['ASA_S_errors_max'] == 2074 and q['ASA_total_errors_max'] == 2170, 'Posthoc quality relaxation')
    require(q['zero_prediction_flips_registered_harmless_header_variants'] and q['additional_classification_benefit_over_K_required'], 'Protection omitted')
    require(p['data']['full_scoring_rows'] == 2056871 and p['data']['ASA_original_rows'] == 112807, 'Population omitted')
    require(p['data']['labels_unchanged'] and p['data']['original_row_frequency_unchanged'] and not p['data']['unknown_parameter_imputation'], 'Data/label evidence changed')
    require(not p['branch']['sequence_order_benefit_claim_allowed'], 'No new shuffle control, no order-specific claim')
    require(p['confirmation']['only_after_all_primary_quality_and_integrity_gates'], 'Confirmation escaped its gate')


def main(record=False):
    receipt = OUT / 'verification.json'
    if record and receipt.exists():
        raise FileExistsError(receipt)
    parent = read(PARENT / 'delivery.json')
    bound = {**parent['artifact_sha256'], **parent['source_sha256'],
             parent['report']: parent['report_sha256'], parent['risk_addendum']: parent['risk_addendum_sha256']}
    for rel, h in bound.items():
        require(sha(ROOT/rel) == h, 'Historical file changed: ' + rel)
    d = pd.read_parquet(PARENT / 'expert_ASA_predictions.parquet')
    trace = pd.read_parquet(TRACE, columns=['row_position','local','truth','raw_message'])
    require(d.row_position.equals(trace.row_position) and not d.row_position.duplicated().any(), 'Row identities changed')
    official = pd.read_parquet(OFFICIAL, columns=['label_binary'])
    y = official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    require(len(y) == 2056871 and np.array_equal(y[d.row_position], d.truth), 'Official truth mismatch')
    local=d.local.to_numpy(); folds=d.fold.to_numpy(); truth=d.truth.to_numpy()
    dynamics=read(OUT/'branch_dynamics.json'); trajectories=read(OUT/'checkpoint_trajectory.json')
    require(len(dynamics)==42 and len(trajectories)==63, 'Missing diagnostic checkpoint')
    for r in dynamics:
        z=np.load(OUT/f"fold{r['fold']}_{r['arm']}_epoch{r['epoch']}_branch_logits.npy")
        delta=z[:,2]-z[:,1]
        require(np.isclose(delta.astype('float64').std(), r['all_unique_MS_residual']['std'], rtol=1e-8,atol=1e-12), 'Spread cannot be replayed')
    for r in trajectories:
        f=r['fold']; a=r['arm']; e=r['epoch']; mask=folds==f
        p=np.load(PARENT/f'fold{f}_{a}'/f'epoch{e}_prob.npy')[local[mask]].argmax(1)
        for c in (1,2):
            require(int(((truth[mask]==c)&(p!=c)).sum())==r['held_errors'][str(c)], 'Trajectory count differs')
    for r in read(OUT/'constant_branch_control.json'):
        f=r['fold'];a=r['arm'];mask=folds==f
        z=np.load(OUT/f'fold{f}_{a}_epoch25_branch_logits.npy')
        mass=np.bincount(local[~mask],minlength=len(z))
        mean=(z.astype('float64')*mass[:,None]).sum(0)/mass.sum()
        require(np.allclose(mean,r['fit_mean_logits'],rtol=0,atol=1e-12), 'Mean used wrong population')
        p=np.load(PARENT/f'fold{f}_{a}'/'epoch25_prob.npy')[local[mask]]
        pc=np.load(OUT/f'fold{f}_{a}_constant_probability.npy')[local[mask]]
        require(int((p.argmax(1)!=pc.argmax(1)).sum())==r['held_constant_vs_actual_prediction_flips'], 'Constant control changed')
        for c in (1,2):
            require(int(((truth[mask]==c)&(pc.argmax(1)!=c)).sum())==r['held_constant_errors'][str(c)], 'Constant error count differs')
        if f==1:
            require(r['held_constant_vs_actual_prediction_flips']==0 and int(mask.sum())==71850, 'Key constant counterexample changed')
    target=d.root.eq(29)&d.truth.eq(2)&d.expert_pred_A.eq(2)&d.expert_pred_B.eq(1)
    require(int(target.sum())==2046, 'Wrong new-error population')
    gap=pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet',columns=['row_position'])
    require(not d.loc[target,'row_position'].isin(gap.row_position).any(), 'Header gap overlap changed')
    for raw in trace.loc[target,'raw_message']:
        m=HEADER.match(raw)
        require(m is not None, 'Unrecognized header')
        require(old_text(raw)==old_text('<164>Jan 01 2000 00:00:00: '+raw[m.end():]), 'Date exposure differs')
    r=read(OUT/'row_review.json')
    for rel,h in r['sources'].items():
        require(sha(ROOT/rel)==h,'Row-review source changed')
    ladder_rel='artifacts/v123_targeted_plan_20260929/support_ladder.parquet'
    old=read(ROOT/'artifacts/v123_targeted_plan_20260929/verification.json')
    require(sha(ROOT/ladder_rel)==old['artifact_sha256'][ladder_rel], 'Verified support table changed')
    require(sha(OUT/'row_decision_review.parquet')==r['row_ledger_sha256'], 'Row ledger changed')
    rows=pd.read_parquet(OUT/'row_decision_review.parquet')
    require(rows.row_position.equals(d.row_position), 'Incomplete row review')
    require(rows.loc[target,'S_support_bucket'].value_counts().to_dict()=={
        'no_destination_same_class_support':1500,'unknown_parameter_pooled':420,
        'thin_same_class_support':96,'known_parameter_dual_multi_root':30}, 'Support partition changed')
    require(bool((rows.loc[target,'behavior_S_rows']>0).all()) and bool((rows.loc[target,'behavior_M_rows']>0).all()), 'Coarse support counterexample changed')
    p=read(PLAN);review_plan(p)
    mutations=[('remove_constant',lambda v:v['arms'].pop('K')),
               ('base_trainable',lambda v:v['base'].update(optimizer_contains_base=True)),
               ('control_as_candidate',lambda v:v['primary_quality'].update(candidate='W')),
               ('selected_early',lambda v:v['training'].update(epochs=25)),
               ('allow_extension',lambda v:v['training'].update(automatic_epoch_extension=True)),
               ('ignore_header',lambda v:v['primary_quality'].update(zero_prediction_flips_registered_harmless_header_variants=False)),
               ('collapse_members',lambda v:v['base'].update(cache='mean probability only')),
               ('promote_order',lambda v:v['branch'].update(sequence_order_benefit_claim_allowed=True))]
    rejected=[]
    for name,mutate in mutations:
        v=copy.deepcopy(p);mutate(v)
        try:review_plan(v)
        except ValueError:rejected.append(name)
        else:raise ValueError('Bad plan passed: '+name)
    risks=read(RISKS)
    for case in risks['cases']:
        require((ROOT/case['evidence']).is_file(), 'Missing risk evidence')
    files=[v for v in OUT.iterdir() if v.is_file() and v.name!='verification.json']
    files += [PLAN,REPORT,RISKS,Path(__file__),ROOT/'training/v126_frozen_audit.py',ROOT/'training/v126_row_review.py']
    hashes={v.relative_to(ROOT).as_posix():sha(v) for v in files}
    if receipt.exists():
        require(read(receipt)['artifact_sha256']==hashes,'Review receipt no longer matches')
    result={'status':'V126_review_evidence_and_plan_verified_not_training_runtime',
            'latest_actual_training':'V125','new_classifier_fits':0,'optimizer_steps':0,
            'quality_acceptance':False,'model_promoted':False,'training_runtime_implemented':False,
            'historical_bound_files_verified':len(bound),'official_rows':len(y),'ASA_rows':len(d),
            'checkpoints_recounted':len(trajectories),'branch_checkpoints_replayed':len(dynamics),
            'constant_controls_verified':6,'negative_plan_cases_rejected':rejected,
            'artifact_sha256':hashes,
            'limits':'Saved-array/count and plan verification; first audit executed model replays. This receipt is not a prefit execution seal or independent blind test.'}
    if record:save(receipt,result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--record',action='store_true')
    main(parser.parse_args().record)
