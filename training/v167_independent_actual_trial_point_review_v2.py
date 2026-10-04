"""Original-gold, complete saved-gradient and temporary-point audit, CPU only."""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz

from v160_independent_fixed_diagnostic_review import read, sha, rows_review
from v160_margin_normal import input_identity
from v161_independent_all_finite_results_review import close, parameter_hash, gradient_repeat, target_risk
from v164_independent_actual_short_trajectory_review import proposal_review, tables
from v166_independent_actual_coverage_review import class_progress
from v166_coverage_joint_restoration import propose
from v159_float64_repeat_policy_v2 import finite_step_review

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'artifacts/v167_trial_point_restoration_diagnostic_20261002'
PRIOR = ROOT / 'artifacts/v164_short_supervised_trajectory_20261002'
CONTROL = ROOT / 'artifacts/v166_coverage_first_diagnostic_20261002'
COHORT = ROOT / 'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OUT = ROOT / 'artifacts/v167_independent_actual_trial_point_review_v2_20261002'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def function_identity(meta, role, ids_oof, x, opinions):
    scope, local, truth, rival = (meta[k] for k in ['scope', 'local', 'truth', 'rival'])
    ids = ids_oof if scope == 'OOF' else np.arange(22546)
    pos = int(np.searchsorted(ids, local))
    assert ids[pos] == local
    chunk = ids[pos//2048*2048:pos//2048*2048+2048]
    identity = input_identity(role, scope, chunk, x[chunk], opinions[scope][chunk], pos % 2048, truth, rival)
    return identity, chunk


def blockers_identity(path, role, ids_oof, x, opinions):
    rows = pd.read_parquet(path / 'actual_blocking_original_rows.parquet')
    result = set()
    for values, group in rows.groupby(['scope', 'local', 'truth', 'rival'], sort=True):
        meta = dict(zip(['scope', 'local', 'truth', 'rival'], values))
        result.add(function_identity(meta, role, ids_oof, x, opinions)[0])
    return rows, result


def margin_values(refs, logs):
    return np.asarray([logs[m['scope']][m['local'], m['truth']] - logs[m['scope']][m['local'], m['rival']]
                       for m in refs.values()], np.float64)


def main():
    assert not OUT.exists() and all((TRIAL / f'role{r}/diagnostic.json').exists() for r in range(3))
    planpath = ROOT / 'training/review_policy/v167_trial_point_restoration_contract.json'
    plan, seal = read(planpath), read(TRIAL / 'run_seal.json')
    assert seal['plan_sha256'] == sha(planpath) and plan['execution_authority']
    assert plan['new_caps']['fits'] == plan['new_caps']['permanent_updates'] == 0
    goldpath = ROOT / 'data/official/train.parquet'
    gold = pd.read_parquet(goldpath, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert np.bincount(gold, minlength=3).tolist() == [1899723, 111728, 45420]
    xfile = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    paths = {Path(__file__).resolve(), ROOT / 'training/v167_independent_actual_trial_point_review.py', planpath, goldpath, xfile}
    paths |= {p for base in [TRIAL, CONTROL] for p in base.rglob('*') if p.is_file()}
    paths |= {ROOT / 'training' / name for name in [
        'v160_independent_fixed_diagnostic_review.py', 'v160_margin_normal.py',
        'v161_independent_all_finite_results_review.py', 'v164_independent_actual_short_trajectory_review.py',
        'v166_independent_actual_coverage_review.py', 'v166_coverage_joint_restoration.py',
        'v167_trial_point_restoration_diagnostic.py', 'v167_trial_point_measurement.py',
        'v167_trial_point_execution_review.py', 'v159_float64_repeat_policy_v2.py']}
    for spec in plan['roles']:
        r = spec['role']
        paths.add(PRIOR / f'role{r}/endpoint.pt')
        paths.add(COHORT / f'role{r}/fixed_pure_error_targets.parquet')
        paths |= {p for p in (PRIOR / f'role{r}/endpoint').rglob('*') if p.is_file()}
        point = PRIOR / f"role{r}/parameter_point{spec['parameter_point']}"
        for c in [1, 2]:
            for k in [0, 1]:
                paths.add(point / f'class{c}_repeat{k}/complete_fixed_error_target_gradient.npy')
                paths.add(point / f'class{c}_repeat{k}/parameter_point.json')
        for s in ['OOF', 'deployment']:
            paths.add(ROOT / f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{r}/{s}_probabilities.npy')
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
    OUT.mkdir()
    save(OUT / 'pre_review_bindings.json', dict(source_sha256=bindings, official_calls=0))
    x = load_npz(xfile).tocsr()
    x.sort_indices()
    roles, replays = [], 0
    for spec in plan['roles']:
        role = spec['role']
        folder = TRIAL / f'role{role}'
        diag = read(folder / 'diagnostic.json')
        assert diag['exception'] is None and diag['new_fits'] == diag['permanent_updates'] == 0
        state = torch.load(PRIOR / f'role{role}/endpoint.pt', weights_only=True, map_location='cpu')['state']
        origin = parameter_hash(state)
        assert origin == spec['endpoint_parameter_sha256'] == diag['origin_parameter_sha256']
        reference, baseline = tables(PRIOR / f'role{role}/endpoint'), tables(folder / 'baseline')
        for scope in baseline:
            frame, _, _, _ = rows_review(folder / 'baseline', scope, reference[scope], gold)
            close(frame[['p0', 'p1', 'p2']], reference[scope][['p0', 'p1', 'p2']])
            close(frame[['logp0', 'logp1', 'logp2']], reference[scope][['logp0', 'logp1', 'logp2']])
        ids_oof = np.sort(baseline['OOF'].local.unique())
        opinions = {s: np.asarray(np.load(ROOT / f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{s}_probabilities.npy',
                    mmap_mode='r')[:, :16], np.float64) for s in baseline}
        targets = pd.read_parquet(COHORT / f'role{role}/fixed_pure_error_targets.parquet').row_position
        before = target_risk(baseline['OOF'], targets)
        mass = np.bincount(baseline['OOF'].truth, minlength=3)
        point = PRIOR / f"role{role}/parameter_point{spec['parameter_point']}"
        gradients = []
        for c in [1, 2]:
            pair = [np.load(point / f'class{c}_repeat{k}/complete_fixed_error_target_gradient.npy') for k in [0, 1]]
            gradient_repeat(*pair)
            assert all(read(point / f'class{c}_repeat{k}/parameter_point.json')['parameter_sha256'] == origin for k in [0, 1])
            gradients.append(pair[0])
        control_path = CONTROL / f'role{role}/treatment'
        control_review, control_rows = proposal_review(control_path, baseline, targets, gradients, state, gold)
        original = {identity: ref['metadata'] for identity, ref in read(CONTROL / f'role{role}/joint_restoration/active_normal_references.json').items()}
        assert len(original) == spec['complete_functions'] <= 25
        for identity, meta in original.items():
            assert meta['base_parameter_sha256'] == origin and function_identity(meta, role, ids_oof, x, opinions)[0] == identity
        origin_logs = {s: np.load(folder / f'baseline/{s}_logq.npy') for s in baseline}
        corrections, proposals = [], []
        proposal_paths = sorted(folder.glob('probe*'), key=lambda p: int(p.name.replace('probe', '')))
        for path in proposal_paths:
            report, frames = proposal_review(path, baseline, targets, gradients, state, gold)
            assert read(path / 'probe.json')['step'] == 1.
            proposals.append(dict(review=report, progress_vs_V164=class_progress(baseline['OOF'], frames['OOF']),
                                  progress_vs_V166=class_progress(control_rows['OOF'], frames['OOF'])))
        if control_review['accepted']:
            assert spec['guard_condition'] == 'replay_V166_actual_safe_candidate_unchanged'
            assert len(proposals) == 1 and proposals[0]['review']['accepted']
            assert diag['joint_restoration_attempts'] == diag['actual_QP_solves'] == diag['optimizer_iterations'] == 0
            assert np.array_equal(np.load(proposal_paths[0] / 'direction.npy'), np.load(control_path / 'direction.npy'))
            for scope in baseline:
                for suffix in ['q', 'logq']:
                    ids = ids_oof if scope == 'OOF' else np.arange(22546)
                    close(np.load(proposal_paths[0] / f'{scope}_{suffix}.npy')[ids], np.load(control_path / f'{scope}_{suffix}.npy')[ids])
        else:
            assert spec['guard_condition'] == 'repair_V166_actual_covered_only_finite_failure'
            control_blockers, identities = blockers_identity(control_path, role, ids_oof, x, opinions)
            assert len(control_blockers) > 0 and identities <= set(original)
            current = np.load(control_path / 'direction.npy')
            trial_path = control_path
            correction_paths = sorted(folder.glob('correction*'), key=lambda p: int(p.name.replace('correction', '')))
            assert 1 <= len(correction_paths) <= 2
            finite_index = 0
            for stage, path in enumerate(correction_paths):
                trial = parameter_hash(state, current, 1.)
                assert trial != origin == read(path / 'parameter_point_roles.json')['origin_parameter_sha256']
                assert read(path / 'parameter_point_roles.json') == dict(origin_parameter_sha256=origin,
                    class_gradient_parameter_sha256=origin, margin_Jacobian_parameter_sha256=trial, stage=stage,
                    complete_functions=len(original))
                refs = read(path / 'active_normal_references.json')
                assert list(refs) == list(original) and len(refs) == 25
                trial_logs = {s: np.load(trial_path / f'{s}_logq.npy') for s in baseline}
                normals = []
                for identity, ref in refs.items():
                    meta = ref['metadata']
                    old = original[identity]
                    for key in ['scope', 'local', 'truth', 'rival', 'query', 'role', 'input_identity']:
                        assert meta[key] == old[key]
                    assert meta['base_parameter_sha256'] == meta['linearization_parameter_sha256'] == trial
                    assert meta['origin_parameter_sha256'] == origin and meta['correction_stage'] == stage
                    measured = ROOT / ref['gradient']
                    expected = folder / f'trial_point{stage}/normals/{identity}/repeat0_gradient.npy'
                    assert measured == expected and read(measured.parent / 'input_binding.json') == meta
                    actual_identity, chunk = function_identity(meta, role, ids_oof, x, opinions)
                    assert actual_identity == identity and np.array_equal(np.load(measured.parent / 'chunk_local_ids.npy'), chunk)
                    pair = [np.load(measured.parent / f'repeat{k}_gradient.npy') for k in [0, 1]]
                    gradient_repeat(*pair)
                    normals.append(pair[0])
                    for k in [0, 1]:
                        close(np.load(measured.parent / f'repeat{k}_q.npy'), np.load(trial_path / f"{meta['scope']}_q.npy")[chunk])
                        close(np.load(measured.parent / f'repeat{k}_logq.npy'), trial_logs[meta['scope']][chunk])
                        close(np.load(measured.parent / f'repeat{k}_margin.npy'),
                              trial_logs[meta['scope']][meta['local'], meta['truth']] - trial_logs[meta['scope']][meta['local'], meta['rival']])
                b, c = margin_values(original, origin_logs), margin_values(original, trial_logs)
                close(b, np.load(path / 'actual_origin_margins.npy'))
                close(c, np.load(path / 'actual_trial_margins.npy'))
                assert np.array_equal(current, np.load(path / 'current_displacement.npy'))
                math = propose(current, np.stack(normals), b, c, *gradients)
                replays += int('optimizer_iterations' in math)
                assert {k: v for k, v in math.items() if k not in ['displacement', 'correction']} == read(path / 'original_unit_restoration_review.json')
                for key in ['displacement', 'correction']:
                    if key in math:
                        assert np.array_equal(math[key], np.load(path / f'{key}.npy'))
                info = dict(stage=stage, linearization_parameter_sha256=trial, math_status=math['status'],
                            optimizer_iterations=math.get('optimizer_iterations', 0), full_functions=25)
                if math['status'] != 'one_sided_joint_restoration_requires_full_actual_finite_guard':
                    assert stage == len(correction_paths)-1
                    corrections.append(info)
                    break
                proposal_path = proposal_paths[finite_index]
                assert np.array_equal(np.load(proposal_path / 'direction.npy'), math['displacement'])
                proof = read(proposal_path / 'probe.json')
                finite_index += 1
                if proof['accepted']:
                    assert stage == len(correction_paths)-1
                else:
                    logs = {s: np.load(proposal_path / f'{s}_logq.npy') for s in baseline}
                    next_c = margin_values(original, logs)
                    blockers, identities = blockers_identity(proposal_path, role, ids_oof, x, opinions)
                    covered = bool(len(blockers) > 0 and identities and identities <= set(original))
                    target = finite_step_review(before, np.load(proposal_path / 'fixed_error_risk.npy'),
                                               proof['class_slopes'], *mass[1:], 'B', 1., True)
                    previous = max(0., -float(np.min(c)))
                    now = max(0., -float(np.min(next_c)))
                    decrease = previous > 0 and now < .99*previous
                    expected_gate = dict(continue_second_correction=bool(covered and target['accepted'] and decrease),
                        covered_only=covered, blocking_function_identities=sorted(identities),
                        uncovered_function_identities=sorted(identities-set(original)), fixed_target_drop_and_Armijo_only=target,
                        classification_acceptance_not_overridden=True, previous_max_negative_margin=previous,
                        current_max_negative_margin=now, prospective_max_negative_margin_factor=.99,
                        strict_residual_decrease_passed=decrease)
                    actual_gate = read(path / 'second_correction_gate.json')
                    assert actual_gate.keys() == expected_gate.keys()
                    # Original-row fsum and official chunk reduction can differ
                    # by an ulp. Audit decisions exactly, diagnostic arithmetic
                    # within the already registered repeat envelope; do not
                    # add any tolerance to actual classification acceptance.
                    for key in expected_gate:
                        if key != 'fixed_target_drop_and_Armijo_only':
                            assert actual_gate[key] == expected_gate[key]
                    actual_target = actual_gate['fixed_target_drop_and_Armijo_only']
                    assert actual_target.keys() == target.keys()
                    for key, value in target.items():
                        if key in ['actual_drop', 'resolution', 'Armijo_bounds', 'Armijo_slack']:
                            close(value, actual_target[key])
                        else:
                            assert value == actual_target[key]
                    if stage == 0:
                        assert (len(correction_paths) == 2) == expected_gate['continue_second_correction']
                    if stage == 1:
                        assert diag['status'] == 'bounded_second_correction_actual_failed_stop'
                    info['continuation_gate'] = expected_gate
                corrections.append(info)
                current, trial_path = math['displacement'], proposal_path
            assert finite_index == len(proposals)
        assert len(proposal_paths) == diag['finite_proposals'] <= spec['finite_proposal_cap']
        assert len(corrections) == diag['joint_restoration_attempts'] <= spec['QP_cap']
        qps = sum('optimizer_iterations' in read(p / 'original_unit_restoration_review.json') for p in folder.glob('correction*'))
        nit = sum(r['optimizer_iterations'] for r in corrections)
        assert diag['actual_QP_solves'] == qps and diag['optimizer_iterations'] == nit
        expected_margins = 50*len(corrections)
        expected_heads = (2+len(proposals))*(spec['OOF_chunks']+12)+expected_margins
        assert expected_heads <= spec['head_cap'] and expected_margins <= spec['margin_gradient_cap']
        counts = dict(head_attempts=expected_heads, head_completed=expected_heads, feature_attempts=expected_heads,
                      feature_completed=expected_heads, gradient_attempts=0, gradient_completed=0,
                      margin_attempts=expected_margins, margin_completed=expected_margins)
        assert counts == diag['counts']
        events = [json.loads(line) for line in (folder / 'calls.jsonl').read_text().splitlines()]
        for kind, n in [('head', expected_heads), ('feature', expected_heads), ('full_parameter_margin_gradient', expected_margins)]:
            for event in ['attempt', 'completed']:
                selected = [v for v in events if v['kind'] == kind and v['event'] == event]
                assert [v['ordinal'] for v in selected] == list(range(1, n+1))
                if kind == 'full_parameter_margin_gradient':
                    for stage in range(len(corrections)):
                        block = [v for v in selected if v['stage'] == stage]
                        assert len(block) == 50 and {v['input_identity'] for v in block} == set(original)
                        assert all(v['origin_parameter_sha256'] == origin and v['linearization_parameter_sha256'] == corrections[stage]['linearization_parameter_sha256'] for v in block)
        calls = [v for v in events if v['kind'] == 'restoration_QP']
        assert [v['event'] for v in calls] == ['attempt', 'returned']*qps
        assert sum(v['optimizer_iterations'] for v in calls if v['event'] == 'returned') == nit
        finite_events = [v for v in events if v['kind'] == 'finite_trial_point_proposal']
        assert [v['event'] for v in finite_events] == ['attempt', 'completed']*len(proposals)
        assert not any(v['kind'] in ['full_class_gradient', 'fixed_error_target_gradient', 'permanent_update', 'supervised_classifier_fit'] for v in events)
        final = read(proposal_paths[-1] / 'probe.json') if proposals else None
        assert diag['final_candidate'] == final
        assert diag['actual_finite_accepted'] == bool(final and final['accepted'])
        restored = torch.load(folder / 'endpoint.pt', weights_only=True, map_location='cpu')['state']
        assert state.keys() == restored.keys() and all(torch.equal(state[k], restored[k]) for k in state)
        assert parameter_hash(restored) == origin == diag['restored_parameter_sha256']
        for scope in baseline:
            frame, _, _, _ = rows_review(folder / 'endpoint', scope, baseline[scope], gold)
            close(frame[['p0', 'p1', 'p2']], baseline[scope][['p0', 'p1', 'p2']])
            close(frame[['logp0', 'logp1', 'logp2']], baseline[scope][['logp0', 'logp1', 'logp2']])
        roles.append(dict(role=role, status=diag['status'], actual_finite_accepted=diag['actual_finite_accepted'],
                          safe_V166_control_unchanged=control_review['accepted'], heads=expected_heads,
                          complete_margin_derivatives=expected_margins, actual_QP_solves=qps,
                          optimizer_iterations=nit, corrections=corrections, proposals=proposals,
                          all_original_gold_parameters_predictions_costs_verified=True))
    assert all(sha(ROOT / p) == value for p, value in bindings.items())
    safe = all(r['actual_finite_accepted'] for r in roles)
    gain = any(r['proposals'] and any(v['pure_errors_after'] < v['pure_errors_before']
               for v in r['proposals'][-1]['progress_vs_V164'].values()) for r in roles)
    report = dict(status='V167_actual_trial_points_original_gold_full_gradients_finite_guards_costs_and_restoration_independently_verified',
        auditor_revision='v2_independent_fsum_repeat_envelope_diagnostics_exact_decisions', roles=roles, new_heads=sum(r['heads'] for r in roles), new_complete_margin_derivatives=sum(r['complete_margin_derivatives'] for r in roles),
        actual_QP_solves=sum(r['actual_QP_solves'] for r in roles), CPU_saved_vector_QP_replays_by_this_review=replays,
        all_three_actual_finite_guards_passed=safe, any_pure_classification_gain_vs_V164=gain,
        supports_new_short_training_registration=safe and gain, official_calls_by_this_review=0,
        new_fits=0, permanent_updates=0, training_issue_mastered=False, quality_acceptance=False,
        scope='Overlapping supervised development roles; gains not added across roles. Not independent transfer or full task scoring.')
    save(OUT / 'review.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'roles'}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT / 'failure.json', dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), official_calls=0))
        raise
