"""Independent original-gold and saved trial-point floor audit, CPU only."""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz

from v160_independent_fixed_diagnostic_review import read, sha, rows_review
from v161_independent_all_finite_results_review import close, parameter_hash, gradient_repeat, target_risk
from v164_independent_actual_short_trajectory_review import proposal_review, tables
from v166_independent_actual_coverage_review import class_progress
from v166_coverage_joint_restoration import _joint_propose
from v167_independent_actual_trial_point_review_v2 import function_identity, margin_values
from v159_float64_repeat_policy_v2 import finite_step_review

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'artifacts/v168_decision_floor_diagnostic_20261002'
PRIOR = ROOT / 'artifacts/v164_short_supervised_trajectory_20261002'
OLD = ROOT / 'artifacts/v167_trial_point_restoration_diagnostic_20261002'
COHORT = ROOT / 'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OUT = ROOT / 'artifacts/v168_independent_actual_decision_floor_review_20261002'
EPS = float(np.finfo(np.float64).eps)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def same_finite_review(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if key in ['actual_drop', 'resolution', 'Armijo_bounds', 'Armijo_slack']:
            close(value, actual[key])
        else:
            assert actual[key] == value


def gradients_at_origin(role, spec, state):
    point = PRIOR / f"role{role}/parameter_point{spec['parameter_point']}"
    result = []
    for cls in [1, 2]:
        pair = [np.load(point / f'class{cls}_repeat{k}/complete_fixed_error_target_gradient.npy') for k in [0, 1]]
        gradient_repeat(*pair)
        assert all(read(point / f'class{cls}_repeat{k}/parameter_point.json')['parameter_sha256'] == parameter_hash(state) for k in [0, 1])
        result.append(pair[0])
    return result


def main():
    folder = TRIAL / 'role1'
    assert not OUT.exists() and (folder / 'diagnostic.json').exists()
    planpath = ROOT / 'training/review_policy/v168_decision_floor_contract.json'
    plan, seal, diag = read(planpath), read(TRIAL / 'run_seal.json'), read(folder / 'diagnostic.json')
    assert plan['execution_authority'] and sha(planpath) == seal['plan_sha256']
    assert diag['exception'] is None and diag['new_fits'] == diag['permanent_updates'] == 0
    old_audit_path = ROOT / 'artifacts/v167_independent_actual_trial_point_review_v2_20261002'
    old_bindings = read(old_audit_path / 'pre_review_bindings.json')['source_sha256']
    assert all(sha(ROOT / p) == v for p, v in old_bindings.items())
    goldpath = ROOT / 'data/official/train.parquet'
    gold = pd.read_parquet(goldpath, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert np.bincount(gold, minlength=3).tolist() == [1899723, 111728, 45420]
    xfile = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    paths = {Path(__file__).resolve(), planpath, goldpath, xfile, old_audit_path / 'pre_review_bindings.json', old_audit_path / 'review.json'}
    paths |= {ROOT / p for p in old_bindings}
    paths |= {p for p in TRIAL.rglob('*') if p.is_file()}
    paths.add(ROOT / plan['additional_candidate_repair_goal'])
    paths |= {ROOT / 'training' / name for name in [
        'v160_independent_fixed_diagnostic_review.py', 'v161_independent_all_finite_results_review.py',
        'v164_independent_actual_short_trajectory_review.py', 'v166_independent_actual_coverage_review.py',
        'v167_independent_actual_trial_point_review_v2.py', 'v166_coverage_joint_restoration.py',
        'v168_decision_floor_joint_restoration.py', 'v168_decision_floor_diagnostic_v2.py',
        'v168_decision_floor_execution_review.py', 'v167_trial_point_measurement.py', 'v159_float64_repeat_policy_v2.py']}
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
    OUT.mkdir()
    save(OUT / 'pre_review_bindings.json', dict(source_sha256=bindings, official_calls=0))
    old_plan = read(ROOT / 'training/review_policy/v167_trial_point_restoration_contract.json')
    safe = []
    for role in [0, 2]:
        state = torch.load(PRIOR / f'role{role}/endpoint.pt', weights_only=True, map_location='cpu')['state']
        baseline = tables(PRIOR / f'role{role}/endpoint')
        targets = pd.read_parquet(COHORT / f'role{role}/fixed_pure_error_targets.parquet').row_position
        gs = gradients_at_origin(role, old_plan['roles'][role], state)
        path = ROOT / plan['actual_safe_role_references'][str(role)]
        assert path == OLD / f'role{role}/probe0'
        review, frames = proposal_review(path, baseline, targets, gs, state, gold)
        assert review['accepted']
        safe.append(dict(role=role, review=review, old_candidate_progress_vs_V164=class_progress(baseline['OOF'], frames['OOF']),
                         official_calls=0, not_new_V168_classification_gain=True))
    state = torch.load(PRIOR / 'role1/endpoint.pt', weights_only=True, map_location='cpu')['state']
    origin = parameter_hash(state)
    assert origin == plan['origin_parameter_sha256'] == diag['origin_parameter_sha256']
    baseline, reference = tables(folder / 'baseline'), tables(PRIOR / 'role1/endpoint')
    for scope in baseline:
        frame, _, _, _ = rows_review(folder / 'baseline', scope, reference[scope], gold)
        close(frame[['p0', 'p1', 'p2']], reference[scope][['p0', 'p1', 'p2']])
        close(frame[['logp0', 'logp1', 'logp2']], reference[scope][['logp0', 'logp1', 'logp2']])
    targets = pd.read_parquet(COHORT / 'role1/fixed_pure_error_targets.parquet').row_position
    gs = gradients_at_origin(1, plan['role_spec'], state)
    trial_path = ROOT / plan['actual_failed_trial_path']
    assert trial_path == OLD / 'role1/probe1'
    u = np.load(trial_path / 'direction.npy')
    trial = parameter_hash(state, u, 1.)
    assert trial == plan['actual_failed_trial_parameter_sha256'] == read(trial_path / 'probe.json')['probe_parameter_sha256'] != origin
    joint = folder / 'correction0'
    refs = read(joint / 'active_normal_references.json')
    original_refs = read(ROOT / 'artifacts/v166_coverage_first_diagnostic_20261002/role1/joint_restoration/active_normal_references.json')
    assert list(refs) == list(original_refs) and len(refs) == 25
    original = {identity: ref['metadata'] for identity, ref in original_refs.items()}
    assert read(joint / 'parameter_point_roles.json') == dict(origin_parameter_sha256=origin,
        class_gradient_parameter_sha256=origin, margin_Jacobian_parameter_sha256=trial, complete_functions=25)
    ids_oof = np.sort(baseline['OOF'].local.unique())
    x = load_npz(xfile).tocsr()
    x.sort_indices()
    opinions = {s: np.asarray(np.load(ROOT / f'artifacts/v158_legal_fusion_bank_v2_20261001/fold1/{s}_probabilities.npy', mmap_mode='r')[:, :16], np.float64) for s in baseline}
    origin_logs = {s: np.load(folder / f'baseline/{s}_logq.npy') for s in baseline}
    trial_logs = {s: np.load(trial_path / f'{s}_logq.npy') for s in baseline}
    normals, floors = [], []
    for identity, ref in refs.items():
        meta, old = ref['metadata'], original[identity]
        for key in ['scope', 'local', 'truth', 'rival', 'query', 'role', 'input_identity']:
            assert meta[key] == old[key]
        assert meta['base_parameter_sha256'] == meta['linearization_parameter_sha256'] == trial
        assert meta['origin_parameter_sha256'] == origin and meta['correction_stage'] == 0
        actual_identity, chunk = function_identity(meta, 1, ids_oof, x, opinions)
        assert actual_identity == identity
        gf = ROOT / ref['gradient']
        assert gf == folder / f'trial_point0/normals/{identity}/repeat0_gradient.npy'
        assert read(gf.parent / 'input_binding.json') == meta
        assert np.array_equal(np.load(gf.parent / 'chunk_local_ids.npy'), chunk)
        pair = [np.load(gf.parent / f'repeat{k}_gradient.npy') for k in [0, 1]]
        gradient_repeat(*pair)
        normals.append(pair[0])
        scope, local, truth, rival = [meta[k] for k in ['scope', 'local', 'truth', 'rival']]
        for k in [0, 1]:
            close(np.load(gf.parent / f'repeat{k}_q.npy'), np.load(trial_path / f'{scope}_q.npy')[chunk])
            close(np.load(gf.parent / f'repeat{k}_logq.npy'), trial_logs[scope][chunk])
            close(np.load(gf.parent / f'repeat{k}_margin.npy'), trial_logs[scope][local, truth]-trial_logs[scope][local, rival])
        scale = max(1., abs(float(trial_logs[scope][local, truth])), abs(float(trial_logs[scope][local, rival])))
        floors.append(16*EPS*scale if truth > rival else 0.)
    b, c, tau = margin_values(original, origin_logs), margin_values(original, trial_logs), np.asarray(floors, np.float64)
    assert np.array_equal(u, np.load(joint / 'current_displacement.npy'))
    assert np.array_equal(b, np.load(joint / 'actual_origin_margins.npy'))
    assert np.array_equal(c, np.load(joint / 'actual_trial_margins.npy'))
    assert np.array_equal(tau, np.load(joint / 'prospective_decision_floors.npy'))
    assert np.array_equal(np.maximum(tau-c, 0.), np.load(joint / 'actual_trial_floor_shortfalls.npy'))
    assert np.max(np.maximum(tau-c, 0.)) > 0 and max(0., -float(c.min())) == 0.
    math = _joint_propose(u, np.stack(normals), np.zeros_like(b), c-tau, *gs)
    math.update(local_protection_target='prospective_existing_argmax_decision_floor',
        actual_origin_margins_for_identity_only=b.tolist(), actual_trial_margins=c.tolist(),
        prospective_decision_floors=tau.tolist(), actual_trial_floor_shortfalls=np.maximum(tau-c, 0.).tolist(),
        truth_index_losing_tie_is_only_positive_floor_trigger=True, local_floor_not_actual_acceptance_tolerance=True,
        actual_argmax_all_original_rows_and_retention_guards_still_required=True,
        confidence_restoration_not_required_by_this_local_policy=True,
        finite_step_authority=False, no_new_fitting_permission=True)
    for key in ['displacement', 'correction']:
        if key in math:
            assert np.array_equal(math[key], np.load(joint / f'{key}.npy'))
    assert {k: v for k, v in math.items() if k not in ['displacement', 'correction']} == read(joint / 'original_unit_restoration_review.json')
    eligible = math['status'] == 'one_sided_joint_restoration_requires_full_actual_finite_guard'
    assert diag['finite_proposals'] == int(eligible)
    final, progress, extra = None, None, None
    if eligible:
        path = folder / 'probe0'
        base_review, frames = proposal_review(path, baseline, targets, gs, state, gold)
        assert np.array_equal(np.load(path / 'direction.npy'), math['displacement'])
        proof = read(path / 'v168_complete_probe_review.json')
        assert proof['base_original_guard_review'] == read(path / 'probe.json')
        goal = pd.read_parquet(ROOT / plan['additional_candidate_repair_goal'])
        original_wrong = reference['OOF'].pred.ne(reference['OOF'].truth)
        control = tables(trial_path)['OOF']
        assert np.array_equal(control.row_position, reference['OOF'].row_position)
        correct_M = original_wrong & control.pred.eq(control.truth) & control.truth.eq(1)
        assert len(goal) == 4 and np.array_equal(goal.row_position, control.loc[correct_M, 'row_position'])
        assert np.array_equal(goal.truth, gold[goal.row_position])
        assert not reference['OOF'].loc[reference['OOF'].row_position.isin(goal.row_position), 'protected_correct'].any()
        ties = pd.read_parquet(trial_path / 'actual_blocking_original_rows.parquet')
        assert len(ties) == 2 and np.array_equal(ties.truth, gold[ties.row_position])
        indexed = frames['OOF'].set_index('row_position', verify_integrity=True)
        goal_rows, tie_rows = indexed.loc[goal.row_position], indexed.loc[ties.row_position]
        assert np.array_equal(goal_rows[['local', 'truth']].to_numpy(), goal[['local', 'truth']].to_numpy())
        assert np.array_equal(tie_rows[['local', 'truth']].to_numpy(), ties[['local', 'truth']].to_numpy())
        errors = {name: int((indexed.truth.eq(cls) & indexed.pred.ne(cls)).sum()) for name, cls in [('M', 1), ('S', 2)]}
        repairs_kept = bool(goal_rows.pred.eq(goal_rows.truth).all())
        ties_fixed = bool(tie_rows.pred.eq(tie_rows.truth).all())
        caps_passed = all(errors[k] <= v for k, v in plan['prospective_full_class_error_caps'].items())
        extra = dict(passed=bool(repairs_kept and ties_fixed and caps_passed),
            all_four_observed_M_candidate_repairs_retained=repairs_kept,
            all_two_actual_S_tie_rows_now_correct=ties_fixed, original_full_class_error_caps_passed=caps_passed,
            actual_original_class_errors=errors, prospective_class_error_caps=plan['prospective_full_class_error_caps'],
            additional_goal_not_merged_into_origin_mask=True)
        assert extra == proof['additional_prospective_guard_review']
        combined = bool(read(path / 'probe.json')['classification_guard'] and extra['passed'])
        assert combined == proof['classification_guard']
        before = target_risk(baseline['OOF'], targets)
        after = target_risk(frames['OOF'], targets)
        mass = np.bincount(baseline['OOF'].truth, minlength=3)
        slopes = [r['linear_change'] for r in math['class_reviews']]
        finite = finite_step_review(before, after, slopes, *mass[1:], 'B', 1., combined)
        same_finite_review(proof['finite_error_target_review'], finite)
        accepted = bool(finite['accepted'] and read(path / 'probe.json')['actual_parameter_change'])
        assert accepted == proof['accepted'] == diag['actual_finite_accepted']
        assert proof == diag['final_candidate']
        progress = dict(vs_V164=class_progress(baseline['OOF'], frames['OOF']), vs_V167_final_tie=class_progress(control, frames['OOF']))
        final = dict(accepted=accepted, base_original_finite_review=base_review, extra_guard=extra,
                     original_gold_finite_review=finite, original_class_recall={name: 1.-errors[name]/int(mass[c]) for name, c in [('M', 1), ('S', 2)]})
    else:
        assert diag['final_candidate'] is None and not diag['actual_finite_accepted']
    restored = torch.load(folder / 'endpoint.pt', weights_only=True, map_location='cpu')['state']
    assert state.keys() == restored.keys() and all(torch.equal(state[k], restored[k]) for k in state)
    assert parameter_hash(restored) == origin == diag['restored_parameter_sha256']
    for scope in baseline:
        frame, _, _, _ = rows_review(folder / 'endpoint', scope, baseline[scope], gold)
        close(frame[['p0', 'p1', 'p2']], baseline[scope][['p0', 'p1', 'p2']])
        close(frame[['logp0', 'logp1', 'logp2']], baseline[scope][['logp0', 'logp1', 'logp2']])
    heads = (2+int(eligible))*16+50
    counts = dict(head_attempts=heads, head_completed=heads, feature_attempts=heads, feature_completed=heads,
                  gradient_attempts=0, gradient_completed=0, margin_attempts=50, margin_completed=50)
    assert diag['counts'] == counts and heads <= 98
    events = [json.loads(line) for line in (folder / 'calls.jsonl').read_text().splitlines()]
    for kind, n in [('head', heads), ('feature', heads), ('full_parameter_margin_gradient', 50)]:
        for event in ['attempt', 'completed']:
            selected = [v for v in events if v['kind'] == kind and v['event'] == event]
            assert [v['ordinal'] for v in selected] == list(range(1, n+1))
            if kind == 'full_parameter_margin_gradient':
                assert all(v['stage'] == 0 and v['origin_parameter_sha256'] == origin and v['linearization_parameter_sha256'] == trial for v in selected)
                assert {v['input_identity'] for v in selected} == set(refs)
                assert all(sum(v['input_identity'] == identity for v in selected) == 2 for identity in refs)
    qps = [v for v in events if v['kind'] == 'restoration_QP']
    expected_qps = int('optimizer_iterations' in math)
    assert [v['event'] for v in qps] == ['attempt', 'returned']*expected_qps
    assert diag['actual_QP_solves'] == expected_qps <= 1
    assert diag['optimizer_iterations'] == sum(v['optimizer_iterations'] for v in qps if v['event'] == 'returned') == math.get('optimizer_iterations', 0)
    proposals = [v for v in events if v['kind'] == 'finite_decision_floor_proposal']
    assert [v['event'] for v in proposals] == ['attempt', 'completed']*int(eligible)
    assert not any(v['kind'] in ['permanent_update', 'supervised_classifier_fit', 'fixed_error_target_gradient', 'full_class_gradient'] for v in events)
    assert all(sha(ROOT / p) == v for p, v in bindings.items())
    report = dict(status='V168_actual_final_tie_Jacobians_decision_floors_original_gold_extra_guards_costs_and_restore_verified',
        unchanged_safe_references=safe, role1_status=diag['status'], local_math_status=math['status'],
        original_point_class_gradients_and_final_tie_margin_Jacobians_verified=True,
        all25_complete_input_function_identities_and50_pair_gradients_verified=True,
        independent_CPU_saved_vector_QP_replays=expected_qps, final_candidate=final, paired_classification=progress,
        all_three_actual_finite_candidates_safe=bool(final and final['accepted']),
        supports_new_short_training_registration=bool(final and final['accepted']),
        new_heads=heads, new_complete_margin_derivatives=50, actual_QP_solves=expected_qps,
        official_calls_by_this_review=0, new_fits=0, permanent_updates=0,
        training_issue_mastered=False, quality_acceptance=False,
        scope='Overlapping supervised development roles; no independent transfer, five-state mastery, or complete-task confirmation.')
    save(OUT / 'review.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['unchanged_safe_references', 'final_candidate', 'paired_classification']}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT / 'failure.json', dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), official_calls=0))
        raise
