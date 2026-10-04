"""Independent source/evidence review; no official model or gradient evaluation."""
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_review import ROOT, check_bindings, read, sha
from v168_decision_floor_execution_review import prospective_plan, review_plan
from v168_decision_floor_joint_restoration import decision_floors, shortfall

OUT = ROOT / 'artifacts/v168_independent_preseal_review_20261002'


def main():
    assert not OUT.exists()
    assert not (ROOT/'training/review_policy/v168_decision_floor_contract.json').exists()
    assert not (ROOT/'artifacts/v168_decision_floor_diagnostic_20261002').exists()
    bundle_path = ROOT/'artifacts/v168_decision_floor_qualification_bundle_20261002/bundle.json'
    bundle = read(bundle_path)
    check_bindings(bundle['source_sha256'])
    plan = prospective_plan()
    review_plan(plan)
    assert bundle['new_caps'] == plan['new_caps']
    assert bundle['execution_authority'] is False
    qualifiers = {}
    paths = {Path(__file__).resolve(), bundle_path,
             ROOT/'training/v168_seal_decision_floor_diagnostic.py',
             ROOT/'training/v168_independent_actual_decision_floor_review.py'}
    paths.update(ROOT/p for p in bundle['source_sha256'])
    for item in bundle['qualifications']:
        path = ROOT/item['path']
        assert sha(path) == item['sha256']
        q = read(path)
        check_bindings(q['source_sha256'])
        assert q['status'] == item['status']
        assert all(q[k] == 0 for k in ['official_heads','official_features',
                                     'official_derivatives','fits','permanent_updates'])
        assert q['execution_authority'] is False
        qualifiers[path.parent.name] = q
        paths.add(path)
        paths.update(ROOT/p for p in q['source_sha256'])
    core = qualifiers['v168_decision_floor_core_qualification_20261002']
    assert core['actual_CPU_QP_solves'] == 5 and core['full_parameter_count'] == 1060832
    assert len(core['saved_zero_floor_regressions']) == 2
    assert core['CPU_fixture_not_actual_SOC_nonlinear_gain']
    identity = qualifiers['v168_decision_floor_identity_qualification_v2_20261002']
    assert identity['complete_functions'] == 25
    assert identity['old_Jacobian_parameter_sha256_is_different'] != plan['actual_failed_trial_parameter_sha256']
    lifecycle = qualifiers['v168_decision_floor_lifecycle_qualification_v2_20261002']
    modes = {c['mode']: c for c in lifecycle['lifecycle_fixtures']}
    assert len(modes) == 7
    assert modes['single_candidate_guard_oracle_pass']['oracle_guard_accepted']
    assert all(c['full_CPU_checkpoint_exactly_restored'] and c['not_actual_model_quality_evidence']
               and c['finite_proposals'] <= 1 and c['mock_QP_events'] <= 1 for c in modes.values())
    assert not modes['exact_S_tie_still_refused']['oracle_guard_accepted']
    assert not modes['four_M_candidate_repairs_lost_refused']['oracle_guard_accepted']
    adversarial = qualifiers['v168_decision_floor_adversarial_qualification_20261002']
    assert len(adversarial['cases']) == 5
    assert all(c['refused'] and c['complete_original_CPU_state_restored'] for c in adversarial['cases'])

    old = ROOT/'artifacts/v167_trial_point_restoration_diagnostic_20261002'
    prior = ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
    root_audit = ROOT/'artifacts/v167_independent_actual_trial_point_review_v2_20261002'
    check_bindings(read(root_audit/'pre_review_bindings.json')['source_sha256'])
    paths.update(ROOT/p for p in read(root_audit/'pre_review_bindings.json')['source_sha256'])
    paths.add(root_audit/'review.json')
    trial = ROOT/plan['actual_failed_trial_path']
    trial_rows = pd.read_parquet(trial/'OOF_original_rows.parquet').set_index('row_position',verify_integrity=True)
    origin_rows = pd.read_parquet(prior/'role1/endpoint/OOF_original_rows.parquet').set_index('row_position',verify_integrity=True)
    ties = pd.read_parquet(trial/'actual_blocking_original_rows.parquet')
    assert len(ties) == 2 and ties.truth.eq(2).all() and ties.local.eq(21985).all()
    logs = np.load(trial/'OOF_logq.npy')
    q = np.load(trial/'OOF_q.npy')
    assert logs[21985,1] == logs[21985,2] and q[21985,1] == q[21985,2]
    assert int(q[21985].argmax()) == 1
    refs = read(old/'role1/correction0/active_normal_references.json')
    records = {key:value['metadata'] for key,value in refs.items()}
    trial_logs = {scope:np.load(trial/f'{scope}_logq.npy') for scope in ['OOF','deployment']}
    actual = decision_floors(records,trial_logs)
    expected = []
    actual_c = []
    for meta in records.values():
        t, r = meta['truth'], meta['rival']
        row = trial_logs[meta['scope']][meta['local']]
        expected.append(16*np.finfo(np.float64).eps*max(1.,abs(row[t]),abs(row[r])) if t>r else 0.)
        actual_c.append(row[t]-row[r])
    assert len(expected) == 25 and np.array_equal(actual,np.asarray(expected))
    assert float(np.max(np.maximum(-np.asarray(actual_c),0))) == 0.
    assert float(np.max(shortfall(actual_c,actual))) > 0.
    goal_path = ROOT/plan['additional_candidate_repair_goal']
    goal = pd.read_parquet(goal_path)
    idx = goal.row_position.to_numpy()
    assert len(idx) == len(set(idx)) == 4 and goal.truth.eq(1).all()
    assert np.array_equal(goal[['local','truth','pred']].to_numpy(), trial_rows.loc[idx,['local','truth','pred']].to_numpy())
    assert trial_rows.loc[idx].pred.eq(1).all() and origin_rows.loc[idx].pred.ne(1).all()
    assert not origin_rows.loc[idx].protected_correct.any()
    paths.add(goal_path)
    for role in [0,2]:
        saved = ROOT/plan['actual_safe_role_references'][str(role)]
        proof = read(saved/'probe.json')
        control = ROOT/f'artifacts/v166_coverage_first_diagnostic_20261002/role{role}/treatment'
        assert proof['accepted'] and proof['probe_parameter_sha256'] == read(control/'probe.json')['probe_parameter_sha256']
        assert np.array_equal(np.load(saved/'direction.npy'),np.load(control/'direction.npy'))

    negative_cases = []
    for key, value in [('execute_only_actual_failed_role',0),('complete_functions',26),('new_fit_permission',True)]:
        bad = copy.deepcopy(plan); bad[key] = value
        try: review_plan(bad)
        except AssertionError: negative_cases.append(key)
        else: raise AssertionError(f'Incorrect plan accepted: {key}')
    # Failed legacy sources are immutable evidence, not executable entrypoints.
    active_sources = [Path(__file__).resolve(), ROOT/bundle['entry'],
                      ROOT/'training/v168_seal_decision_floor_diagnostic.py',
                      ROOT/'training/v168_independent_actual_decision_floor_review.py',
                      ROOT/'training/v168_decision_floor_execution_review.py',
                      ROOT/'training/v168_decision_floor_joint_restoration.py']
    active_sources.extend(ROOT/'training'/f'{name.removesuffix("_20261002")}.py' for name in qualifiers)
    for p in active_sources: ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings = {(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(paths)}
    check_bindings(bindings)
    OUT.mkdir()
    result = dict(status='V168_independent_preseal_source_and_saved_evidence_review_passed',
                  supports_physical_seal=True,execution_authority=False,
                  official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,
                  saved_exact_tie_and_positive_shortfall_verified=True,
                  separate_four_M_goals_not_origin_correct_mask=True,
                  prospective_caps=plan['new_caps'],
                  actual_qualification_scope='5 CPU QPs; cached/synthetic model-output oracles; no new SOC candidate',
                  negative_plan_cases_refused=negative_cases,
                  source_review_findings=[
                      'New Jacobians are measured at actual final rejected V167 parameter; class gradients stay at V164 origin.',
                      'Local tau shifts c in the unchanged original-unit core; actual argmax/risk/protection acceptance is unchanged.',
                      'Four M repairs and two S tie rows are checked separately and combined with the complete base guard.',
                      'One correction and one proposal only; original full tensor/q/logq restore on pass, refusal and exception.',
                      'Safe roles0/2 are frozen references with no new calls or gains.'
                  ],
                  actual_candidate_still_unverified=True,new_training_permission=False,
                  first_issue_mastery=False,full_quality_acceptance=False,source_sha256=bindings)
    (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False))


if __name__ == '__main__': main()
