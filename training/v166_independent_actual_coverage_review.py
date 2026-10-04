"""Independent original-gold and saved-vector audit; no official model calls."""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz

from v160_independent_fixed_diagnostic_review import read, sha, rows_review
from v160_margin_normal import input_identity
from v161_independent_all_finite_results_review import close, parameter_hash, gradient_repeat
from v164_independent_actual_short_trajectory_review import proposal_review, tables
from v166_coverage_joint_restoration import propose

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'artifacts/v166_coverage_first_diagnostic_20261002'
PRIOR = ROOT / 'artifacts/v164_short_supervised_trajectory_20261002'
CONTROL = ROOT / 'artifacts/v165_fixed_endpoint_decision_floor_diagnostic_20261002'
COVER = ROOT / 'artifacts/v165_independent_blocker_coverage_review_20261002'
COHORT = ROOT / 'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OUT = ROOT / 'artifacts/v166_independent_actual_coverage_review_20261002'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def class_progress(before, after):
    result = {}
    old_wrong = before.pred.ne(before.truth)
    new_wrong = after.pred.ne(after.truth)
    for cls, name in [(1, 'M'), (2, 'S')]:
        population = before.truth.eq(cls)
        pure = before.pure_current_input
        repair = population & old_wrong & ~new_wrong
        regress = population & ~old_wrong & new_wrong
        result[name] = dict(original_rows=int(population.sum()),
            errors_before=int((population & old_wrong).sum()),
            errors_after=int((population & new_wrong).sum()),
            pure_errors_before=int((population & old_wrong & pure).sum()),
            pure_errors_after=int((population & new_wrong & pure).sum()),
            repairs_vs_V164_endpoint=int(repair.sum()),
            new_errors_vs_V164_endpoint_all_original_rows=int(regress.sum()),
            pure_repairs_vs_V164_endpoint=int((repair & pure).sum()),
            new_pure_errors_vs_V164_endpoint=int((regress & pure).sum()),
            new_mixed_errors_vs_V164_endpoint=int((regress & ~pure).sum()),
            registered_cumulative_protection_regressions=int((population & new_wrong & before.protected_correct).sum()))
    return result


def main():
    assert not OUT.exists()
    assert all((TRIAL / f'role{r}/diagnostic.json').exists() for r in range(3))
    planpath = ROOT / 'training/review_policy/v166_coverage_first_diagnostic_contract.json'
    plan, seal = read(planpath), read(TRIAL / 'run_seal.json')
    assert sha(planpath) == seal['plan_sha256']
    assert plan['new_caps']['fits'] == plan['new_caps']['permanent_updates'] == 0
    goldpath = ROOT / 'data/official/train.parquet'
    gold = pd.read_parquet(goldpath, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert np.bincount(gold, minlength=3).tolist() == [1899723, 111728, 45420]
    xfile = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    sources = {p for p in TRIAL.rglob('*') if p.is_file()} | {Path(__file__).resolve(), planpath, goldpath, xfile}
    sources |= {ROOT / 'training' / name for name in [
        'v160_independent_fixed_diagnostic_review.py', 'v160_margin_normal.py',
        'v161_independent_all_finite_results_review.py', 'v164_independent_actual_short_trajectory_review.py',
        'v166_coverage_joint_restoration.py', 'v166_coverage_first_diagnostic.py',
        'v166_observed_function_measurement.py', 'v166_solver_trace.py']}
    for spec in plan['roles']:
        role = spec['role']
        sources.add(PRIOR / f'role{role}/endpoint.pt')
        sources.add(COHORT / f'role{role}/fixed_pure_error_targets.parquet')
        sources.add(COVER / f'role{role}_function_coverage.json')
        sources |= {p for p in (CONTROL / f'role{role}/treatment').rglob('*') if p.is_file()}
        point = PRIOR / f"role{role}/parameter_point{spec['parameter_point']}"
        for cls in [1, 2]:
            for repeat in [0, 1]:
                sources.add(point / f'class{cls}_repeat{repeat}/complete_fixed_error_target_gradient.npy')
                sources.add(point / f'class{cls}_repeat{repeat}/parameter_point.json')
        for ref in read(TRIAL / f'role{role}/joint_restoration/active_normal_references.json').values():
            sources.add(ROOT / ref['gradient'])
        for scope in ['OOF', 'deployment']:
            sources.add(ROOT / f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{scope}_probabilities.npy')
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)}
    OUT.mkdir()
    save(OUT / 'pre_review_bindings.json', dict(source_sha256=bindings, official_calls=0))
    x = load_npz(xfile).tocsr()
    x.sort_indices()
    results = []
    for spec in plan['roles']:
        role = spec['role']
        folder = TRIAL / f'role{role}'
        diag = read(folder / 'diagnostic.json')
        assert diag['exception'] is None
        assert diag['new_fits'] == diag['permanent_updates'] == 0
        state = torch.load(PRIOR / f'role{role}/endpoint.pt', map_location='cpu', weights_only=True)['state']
        origin = parameter_hash(state)
        assert origin == spec['endpoint_parameter_sha256'] == diag['initial_parameter_sha256']
        reference, baseline = tables(PRIOR / f'role{role}/endpoint'), tables(folder / 'baseline')
        for scope in baseline:
            actual, _, _, _ = rows_review(folder / 'baseline', scope, reference[scope], gold)
            close(actual[['p0', 'p1', 'p2']], reference[scope][['p0', 'p1', 'p2']])
            close(actual[['logp0', 'logp1', 'logp2']], reference[scope][['logp0', 'logp1', 'logp2']])
        point = PRIOR / f"role{role}/parameter_point{spec['parameter_point']}"
        gradients = []
        for cls in [1, 2]:
            pair = [np.load(point / f'class{cls}_repeat{k}/complete_fixed_error_target_gradient.npy') for k in [0, 1]]
            gradient_repeat(*pair)
            assert all(read(point / f'class{cls}_repeat{k}/parameter_point.json')['parameter_sha256'] == origin for k in [0, 1])
            gradients.append(pair[0])
        joint = folder / 'joint_restoration'
        refs = read(joint / 'active_normal_references.json')
        known = read(point / 'restoration0/active_normal_references.json')
        coverage = read(COVER / f'role{role}_function_coverage.json')
        fresh = set(coverage['fresh_function_ids'])
        assert set(refs) == set(known) | fresh and not set(known) & fresh
        assert len(refs) == spec['joint_functions'] <= 25
        assert set(p.name for p in (folder / 'fresh_normals').iterdir()) == fresh
        ids_oof = np.sort(baseline['OOF'].local.unique())
        opinions = {s: np.asarray(np.load(ROOT / f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{s}_probabilities.npy', mmap_mode='r')[:, :16], np.float64) for s in baseline}
        normals, origin_margins, current_margins = [], [], []
        normal_reviews = []
        for identity, ref in refs.items():
            meta = ref['metadata']
            assert meta['input_identity'] == identity and meta['base_parameter_sha256'] == origin
            scope, local, truth, rival = meta['scope'], meta['local'], meta['truth'], meta['rival']
            ids = ids_oof if scope == 'OOF' else np.arange(22546)
            pos = int(np.searchsorted(ids, local))
            assert ids[pos] == local
            chunk = ids[pos//2048*2048:pos//2048*2048+2048]
            assert input_identity(role, scope, chunk, x[chunk], opinions[scope][chunk], pos % 2048, truth, rival) == identity
            gradient_path = ROOT / ref['gradient']
            gradient = np.load(gradient_path)
            gradient_repeat(gradient, np.load(gradient_path.parent / 'repeat1_gradient.npy'))
            normals.append(gradient)
            origin_logs = np.load(folder / f'baseline/{scope}_logq.npy')
            control_logs = np.load(CONTROL / f'role{role}/treatment/{scope}_logq.npy')
            b = origin_logs[local, truth] - origin_logs[local, rival]
            c = control_logs[local, truth] - control_logs[local, rival]
            assert b >= 0
            origin_margins.append(b)
            current_margins.append(c)
            if identity in fresh:
                assert gradient_path == folder / 'fresh_normals' / identity / 'repeat0_gradient.npy'
                binding = read(gradient_path.parent / 'input_binding.json')
                assert binding == meta
                assert np.array_equal(np.load(gradient_path.parent / 'chunk_local_ids.npy'), chunk)
                blocker_rows = pd.read_parquet(gradient_path.parent / 'blocking_original_rows.parquet')
                assert len(blocker_rows) == coverage['blocking_functions'][identity]['original_rows']
                assert np.array_equal(blocker_rows.truth, gold[blocker_rows.row_position])
                for repetition in [0, 1]:
                    prefix = gradient_path.parent / f'repeat{repetition}'
                    close(np.load(str(prefix)+'_q.npy'), np.load(folder / f'baseline/{scope}_q.npy')[chunk])
                    close(np.load(str(prefix)+'_logq.npy'), origin_logs[chunk])
                    close(np.load(str(prefix)+'_margin.npy'), b)
            else:
                assert ref == known[identity]
            normal_reviews.append(dict(identity=identity, fresh=identity in fresh, origin_margin=float(b), V165_trial_margin=float(c)))
        u = np.load(joint / 'current_displacement.npy')
        assert np.array_equal(u, np.load(CONTROL / f'role{role}/treatment/direction.npy'))
        close(np.array(origin_margins), np.load(joint / 'actual_origin_margins.npy'))
        close(np.array(current_margins), np.load(joint / 'actual_current_margins.npy'))
        math = propose(u, np.stack(normals), np.array(origin_margins), np.array(current_margins), *gradients)
        for key in ['displacement', 'correction']:
            if key in math:
                assert np.array_equal(math[key], np.load(joint / f'{key}.npy'))
        assert {k: v for k, v in math.items() if k not in ['displacement', 'correction']} == read(joint / 'original_unit_restoration_review.json')
        eligible = math['status'] == 'one_sided_joint_restoration_requires_full_actual_finite_guard'
        assert diag['finite_proposals'] == int(eligible)
        targets = pd.read_parquet(COHORT / f'role{role}/fixed_pure_error_targets.parquet').row_position
        control_review, _ = proposal_review(CONTROL / f'role{role}/treatment', baseline, targets, gradients, state, gold)
        treatment = paired = None
        if eligible:
            treatment, frames = proposal_review(folder / 'treatment', baseline, targets, gradients, state, gold)
            assert np.array_equal(np.load(folder / 'treatment/direction.npy'), math['displacement'])
            assert read(folder / 'treatment/probe.json')['step'] == 1.
            paired = class_progress(baseline['OOF'], frames['OOF'])
            assert paired == read(folder / 'paired_treatment_vs_V164_endpoint.json')
            assert treatment['accepted'] == diag['candidate']['accepted']
        restored = torch.load(folder / 'endpoint.pt', map_location='cpu', weights_only=True)['state']
        assert state.keys() == restored.keys() and all(torch.equal(state[k], restored[k]) for k in state)
        assert parameter_hash(restored) == origin == diag['restored_parameter_sha256']
        for scope in baseline:
            actual, _, _, _ = rows_review(folder / 'endpoint', scope, baseline[scope], gold)
            close(actual[['p0', 'p1', 'p2']], baseline[scope][['p0', 'p1', 'p2']])
            close(actual[['logp0', 'logp1', 'logp2']], baseline[scope][['logp0', 'logp1', 'logp2']])
        events = [json.loads(line) for line in (folder / 'calls.jsonl').read_text().splitlines()]
        expected_heads = (2 + int(eligible))*(spec['OOF_chunks'] + 12) + spec['fresh_margin_gradient_cap']
        for kind, number in [('head', expected_heads), ('feature', expected_heads), ('full_parameter_margin_gradient', spec['fresh_margin_gradient_cap'])]:
            for event in ['attempt', 'completed']:
                selected = [r for r in events if r['kind'] == kind and r['event'] == event]
                assert [r['ordinal'] for r in selected] == list(range(1, number+1))
                if kind == 'full_parameter_margin_gradient':
                    assert all(r['parameter_sha256'] == origin for r in selected)
                    assert {r['input_identity'] for r in selected} == fresh
        qp = [r for r in events if r['kind'] == 'restoration_QP']
        assert [r['event'] for r in qp] == ['attempt', 'returned']
        assert qp[-1]['optimizer_iterations'] == math['optimizer_iterations'] == diag['optimizer_iterations']
        assert diag['actual_QP_solves'] == diag['joint_restoration_attempts'] == 1
        finite = [r for r in events if r['kind'] == 'finite_coverage_proposal']
        assert [r['event'] for r in finite] == (['attempt', 'completed'] if eligible else [])
        assert not any(r['kind'] in ['full_class_gradient', 'permanent_update', 'supervised_classifier_fit'] for r in events)
        counts = diag['counts']
        assert counts == dict(head_attempts=expected_heads, head_completed=expected_heads, feature_attempts=expected_heads,
            feature_completed=expected_heads, gradient_attempts=0, gradient_completed=0,
            margin_attempts=spec['fresh_margin_gradient_cap'], margin_completed=spec['fresh_margin_gradient_cap'])
        save(OUT / f'role{role}_normal_identities.json', dict(role=role, normals=normal_reviews))
        results.append(dict(role=role, math_status=math['status'], full_original_unit_solver_reproduced=True,
            joint_functions=len(refs), fresh_functions=len(fresh), fresh_complete_derivatives=spec['fresh_margin_gradient_cap'],
            heads=expected_heads, actual_QP_solves=1, optimizer_iterations=math['optimizer_iterations'],
            control=control_review, treatment=treatment, paired_classification=paired,
            full_parameters_and_original_predictions_restored=True))
    assert all(sha(ROOT / p) == value for p, value in bindings.items())
    safe = all(r['treatment'] is not None and r['treatment']['accepted'] for r in results)
    gain = any(r['paired_classification'] and sum(v['pure_errors_after'] for v in r['paired_classification'].values()) <
        sum(v['pure_errors_before'] for v in r['paired_classification'].values()) for r in results)
    report = dict(status='all_three_actual_coverage_math_fresh_gradients_original_gold_costs_and_restoration_verified',
        roles=results, new_heads=sum(r['heads'] for r in results), new_complete_margin_derivatives=66,
        actual_QP_solves=3, CPU_saved_vector_QP_replays_by_this_review=3, new_fits=0, permanent_updates=0,
        all_three_actual_finite_guards_passed=safe, any_actual_pure_classification_gain=gain,
        supports_new_short_training_registration=safe and gain, official_calls_by_this_review=0,
        training_issue_mastered=False, quality_acceptance=False,
        scope='Supervised development rows; no independent transfer or complete-task confirmation.')
    save(OUT / 'review.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'roles'}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT / 'failure.json', dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), official_calls=0))
        raise
