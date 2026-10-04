"""Independent saved-output review: original gold, every probe and costs.

No classifier, input features, official gradient, fit or parameter update.
CPU checkpoint tensors are read only to reconstruct recorded candidate hashes.
"""
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v160_independent_fixed_diagnostic_review import rows_review, read, sha
from v160_independent_saved_direction_certificate import certificate

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'artifacts/v161_fixed_error_endpoint_diagnostic_20261002'
TAIL = ROOT / 'artifacts/v161_cached_direction_backtrack_tail_20261002/role2'
OLD = ROOT / 'artifacts/v160_fixed_endpoint_diagnostic_20261002'
COHORT = ROOT / 'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OUT = ROOT / 'artifacts/v161_independent_all_finite_results_review_20261002'
EPS = float(np.finfo(np.float64).eps)
SEGMENTS = [(0, 1060592), (1060592, 1060768), (1060768, 1060784), (1060784, 1060832)]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def close(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.all(np.abs(a - b) <= 8 * EPS * np.maximum(1., np.maximum(np.abs(a), np.abs(b))))


def target_risk(frame, target_positions):
    selected = frame.row_position.isin(target_positions).to_numpy()
    return np.array([math.fsum(frame.loc[selected & frame.truth.eq(c), 'stable_CE']) /
                     int(frame.truth.eq(c).sum()) for c in [1, 2]])


def parameter_hash(state, direction=None, step=None):
    h, offset = hashlib.sha256(), 0
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous().numpy()
        if direction is not None:
            # Official assign(): v + step*torch.as_tensor(direction slice).
            value = value + step * direction[offset:offset + value.size].reshape(value.shape)
        h.update(name.encode())
        h.update(value.tobytes())
        offset += value.size
    assert offset == 1060832
    return h.hexdigest()


def gradient_repeat(a, b):
    assert a.shape == b.shape == (1060832,) and np.isfinite(a).all() and np.isfinite(b).all()
    reports = []
    for start, end in SEGMENTS:
        x, y = a[start:end], b[start:end]
        scale = max(float(np.abs(x).max()), float(np.abs(y).max()))
        n = max(float(np.linalg.norm(x)), float(np.linalg.norm(y)))
        gap = x-y
        assert float(np.abs(gap).max()) <= 8*EPS*scale
        assert float(np.linalg.norm(gap)) <= 8*EPS*n
        significant = np.maximum(np.abs(x), np.abs(y)) > 8*EPS*scale
        assert not np.any(np.sign(x[significant]) != np.sign(y[significant]))
        reports.append(dict(start=start, end=end, scale=scale, exact_repeat=bool(np.array_equal(x, y))))
    return reports


def margin_progress(frame, baseline):
    oldwrong = baseline.pred.ne(baseline.truth).to_numpy()
    truth = baseline.truth.to_numpy()[oldwrong]
    rival = baseline.pred.to_numpy()[oldwrong]
    old = baseline[['logp0', 'logp1', 'logp2']].to_numpy()[oldwrong]
    new = frame[['logp0', 'logp1', 'logp2']].to_numpy()[oldwrong]
    ix = np.arange(len(truth))
    before = old[ix, truth] - old[ix, rival]
    delta = new[ix, truth] - new[ix, rival] - before
    return dict(old_wrong_rows=int(oldwrong.sum()), margin_improved=int((delta > 0).sum()),
                margin_worsened=int((delta < 0).sum()), median_before=float(np.median(before)),
                median_delta=float(np.median(delta)))


def check_calls(folder, diagnostic, tail=False):
    events = [json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
    result = {}
    kinds = [('head', 'head'), ('feature', 'feature'),
             ('full_class_gradient' if tail else 'fixed_error_target_gradient', 'gradient')]
    if not tail:
        kinds.append(('full_parameter_margin_gradient', 'margin'))
    for kind, prefix in kinds:
        for event, suffix in [('attempt', 'attempts'), ('completed', 'completed')]:
            matching = [x for x in events if x['kind'] == kind and x['event'] == event]
            assert [x['ordinal'] for x in matching] == list(range(1, len(matching)+1))
            result[f'{prefix}_{suffix}'] = len(matching)
    assert result == diagnostic['counts']
    assert not any(x['kind'] == 'full_class_gradient' for x in events)
    if not tail:
        for c in [1, 2]:
            assert sum(x['event'] == 'attempt' and x['kind'] == 'fixed_error_target_gradient'
                       and x['class_id'] == c for x in events) == 2
    return result


def review_probes(folder, baseline, gold, targets, state, direction, gradients, paths):
    before = target_risk(baseline['OOF'], targets)
    slopes = np.array([math.fsum(g*direction) for g in gradients])
    probes = []
    for path in paths:
        report = read(path)
        j = int(path.parent.name.removeprefix('probe'))
        assert report['step'] == 2.**(-j)
        if (path.parent/'direction.npy').exists():
            assert np.array_equal(np.load(path.parent/'direction.npy'), direction)
        assert report['probe_parameter_sha256'] == parameter_hash(state, direction, report['step'])
        assert report['actual_parameter_change']
        frames, metrics = {}, {}
        for scope in ['OOF', 'deployment']:
            f, risks, _, met = rows_review(path.parent, scope, baseline[scope], gold)
            close(np.exp(f[['logp0', 'logp1', 'logp2']].to_numpy()),
                  f[['p0', 'p1', 'p2']].to_numpy())
            frames[scope], metrics[scope] = f, met
            for key in ['original_rows', 'M_errors', 'S_errors', 'pure_errors', 'total_errors',
                        'protected_regressions', 'new_errors_vs_initial', 'repairs_vs_initial']:
                assert met[key] == report[f'{scope}_stats'][key]
            if scope == 'OOF':
                after = target_risk(f, targets)
                close(after, np.load(path.parent/'fixed_error_risk.npy'))
                close(risks, np.load(path.parent/'full_original_class_risk.npy'))
                full_CE_change = (risks-np.array([math.fsum(baseline[scope].loc[
                    baseline[scope].truth.eq(c), 'stable_CE'])/int(baseline[scope].truth.eq(c).sum())
                    for c in [1, 2]])).tolist()
        old = baseline['OOF']
        count_guard = all(metrics['OOF'][key] <= int((old.pred.ne(old.truth) & old.truth.eq(c)).sum())
                          for key, c in [('M_errors', 1), ('S_errors', 2)])
        guard = (count_guard and metrics['OOF']['protected_regressions'] == 0 and
                 metrics['deployment']['new_errors_vs_initial'] == 0 and
                 report['deployment_stats']['mastered'] and report['joint_TRAIN_retention']['passed'])
        assert guard == report['classification_guard']
        assert count_guard == report['full_original_M_S_error_count_guard']
        assert np.all(np.abs(slopes-report['class_slopes']) <= 16*EPS*np.maximum(1., np.abs(slopes)))
        resolution = 16*EPS*np.maximum(1., np.maximum(np.abs(before), np.abs(after)))
        drop = before-after
        slack = before+1e-4*report['step']*slopes-after
        accepted = bool(guard and np.all(slopes < 0) and np.all(drop > resolution) and np.all(slack > resolution))
        assert accepted == report['accepted'] == report['finite_error_target_review']['accepted']
        actual_blockers = pd.read_parquet(path.parent/'actual_blocking_original_rows.parquet')
        expected = []
        for scope, f in frames.items():
            protection = f.protected_correct if scope == 'OOF' else f.initial_correct
            expected += [(scope, int(row.row_position), int(row.local), int(row.truth), int(row.pred))
                         for row in f[protection & f.pred.ne(f.truth)].itertuples()]
        observed = [(str(row.scope), int(row.row_position), int(row.local), int(row.truth), int(row.rival))
                    for row in actual_blockers.itertuples()]
        assert sorted(expected) == sorted(observed)
        progress = read(path.parent/'fixed_endpoint_progress.json')
        for scope in metrics:
            assert progress[scope]['classification_changes_vs_fixed_endpoint'] == metrics[scope]['classifications_changed_vs_fixed_endpoint']
            assert progress[scope]['repairs_vs_fixed_endpoint'] == metrics[scope]['repairs_vs_fixed_endpoint']
        probes.append(dict(path=path.relative_to(ROOT).as_posix(), probe=j, step=report['step'],
                           accepted=accepted, fixed_error_target_drop=drop.tolist(), full_CE_change=full_CE_change,
                           metrics=metrics, progress={s: margin_progress(frames[s], baseline[s]) for s in frames},
                           protected_blocking_original_rows=len(observed)))
    return probes


def main():
    assert not OUT.exists()
    OUT.mkdir()
    gold = pd.read_parquet(ROOT/'data/official/train.parquet', columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert len(gold) == 2056871 and np.isfinite(gold).all()
    assert np.bincount(gold, minlength=3).tolist() == [1899723, 111728, 45420]
    plan_path = ROOT/'training/review_policy/v161_fixed_error_endpoint_diagnostic_contract.json'
    plan = read(plan_path)
    files = {p for folder in [TRIAL, TAIL, COHORT] for p in folder.rglob('*') if p.is_file()}
    files |= {Path(__file__).resolve(), plan_path, ROOT/'data/official/train.parquet',
              ROOT/'training/v160_independent_fixed_diagnostic_review.py',
              ROOT/'training/v160_independent_saved_direction_certificate.py'}
    for role in range(3):
        files.add(ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt')
        for n in plan['roles'][role]['cached_normals']:
            files |= {ROOT/n['metadata'], ROOT/n['gradient'],
                      (ROOT/n['gradient']).with_name('repeat1_gradient.npy')}
    binding = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    save(OUT/'pre_review_bindings.json', dict(source_sha256=binding, official_calls=0))
    all_reports = []
    contexts = []
    for role in range(3):
        folder, spec = TRIAL/f'role{role}', plan['roles'][role]
        diag = read(folder/'diagnostic.json')
        state = torch.load(ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt',
                           map_location='cpu', weights_only=True)['state']
        assert parameter_hash(state) == spec['endpoint_parameter_sha256'] == diag['initial_parameter_sha256'] == diag['restored_parameter_sha256']
        assert diag['exception'] is None and diag['new_fits'] == diag['permanent_updates'] == 0
        baseline = {}
        for scope, old_name, new_name in [('OOF', 'baseline_class1', 'baseline_error_class1_repeat0'),
                                          ('deployment', 'baseline_deployment', 'baseline_deployment')]:
            old = pd.read_parquet(OLD/f'role{role}'/old_name/f'{scope}_original_rows.parquet')
            if scope == 'OOF':
                old = old.copy()
                old['protected_correct'] = old.protected_correct | (old.pure_current_input & old.pred.eq(old.truth))
            f, _, mass, _ = rows_review(folder/new_name, scope, old, gold)
            close(f[['p0', 'p1', 'p2']].to_numpy(), old[['p0', 'p1', 'p2']].to_numpy())
            baseline[scope] = f
            if scope == 'OOF':
                assert mass.tolist() == spec['complete_original_mass']
        b = baseline['OOF']
        targets = pd.read_parquet(COHORT/f'role{role}/fixed_pure_error_targets.parquet')
        assert np.array_equal(targets.row_position, b.loc[b.pure_current_input & b.pred.ne(b.truth), 'row_position'])
        assert len(targets) == spec['fixed_error_original_rows']
        observed_counts = np.bincount(targets.local.to_numpy(np.int64)*3+targets.truth.to_numpy(np.int64),
                                     minlength=22546*3).reshape(22546, 3)
        assert np.array_equal(observed_counts, np.load(folder/'fixed_target_original_counts.npy'))
        gradients, repeat_reports = [], []
        for c in [1, 2]:
            gs = [np.load(folder/f'baseline_error_class{c}_repeat{k}/complete_fixed_error_target_gradient.npy') for k in [0, 1]]
            repeat_reports.append(gradient_repeat(*gs))
            gradients.append(gs[0])
            for k in [0, 1]:
                fp = folder/f'baseline_error_class{c}_repeat{k}'
                f, risk, _, _ = rows_review(fp, 'OOF', b, gold)
                close(target_risk(f, targets.row_position), np.load(fp/'fixed_pure_error_contribution.npy'))
                close(risk, np.load(fp/'full_original_class_CE.npy'))
        rounded = folder/'round0'
        direction = np.load(rounded/'polished_direction.npy')
        normals = np.load(rounded/'raw_margin_normals.npy')
        records = read(rounded/'normal_records.json')
        assert len(records) == len(normals) == len(spec['cached_normals'])
        for i, cached in enumerate(spec['cached_normals']):
            metadata = read(ROOT/cached['metadata'])
            assert metadata['base_parameter_sha256'] == parameter_hash(state)
            assert metadata == records[metadata['input_identity']]
            ga = np.load(ROOT/cached['gradient'])
            gb = np.load((ROOT/cached['gradient']).with_name('repeat1_gradient.npy'))
            gradient_repeat(ga, gb)
            assert np.array_equal(normals[i], ga)
        cert = certificate(*gradients, normals, direction)
        assert cert == read(rounded/'polished_certificate.json')['independent_saved_vector_certificate']
        assert cert['eligible_for_finite_trial_only']
        paths = sorted(folder.glob('round*/probe*/probe.json'), key=lambda p: int(p.parent.name[5:]))
        probes = review_probes(folder, baseline, gold, targets.row_position, state, direction, gradients, paths)
        assert [p['probe'] for p in probes] == list(range(len(probes)))
        counts = check_calls(folder, diag)
        k = spec['OOF_chunks']
        assert counts['head_completed'] == counts['feature_completed'] == 5*k+24+len(probes)*(k+12)
        assert counts['gradient_completed'] == 4 and counts['margin_completed'] == 0
        assert counts['head_attempts'] <= spec['head_cap'] and len(probes) == diag['finite_proposals']
        assert sum(p['accepted'] for p in probes) == int(diag['finite_error_target_pass'])
        for scope in baseline:
            f, _, _, _ = rows_review(folder/'restored', scope, baseline[scope], gold)
            close(f[['p0', 'p1', 'p2']].to_numpy(), baseline[scope][['p0', 'p1', 'p2']].to_numpy())
        assert diag['restored_joint_TRAIN_retention']['passed']
        contexts.append((baseline, targets.row_position, state, direction, gradients, normals))
        all_reports.append(dict(role=role, status=diag['status'], actual_counts=counts, QP_solves=diag['QP_solves'],
                                gradient_repeats=repeat_reports, numerical_direction_certificate=cert, probes=probes,
                                source_parameter_and_restoration_verified=True))
    baseline, targets, state, direction, gradients, normals = contexts[2]
    diag = read(TAIL/'diagnostic.json')
    assert diag['initial_parameter_sha256'] == diag['restored_parameter_sha256'] == parameter_hash(state)
    assert all(diag[k] == 0 for k in ['new_fits', 'permanent_updates', 'new_fixed_error_target_gradients',
                                    'new_full_original_class_gradients', 'new_margin_gradients', 'QP_solves'])
    for scope in baseline:
        for name in ['baseline', 'restored']:
            f, _, _, _ = rows_review(TAIL/name, scope, baseline[scope], gold)
            close(f[['p0', 'p1', 'p2']].to_numpy(), baseline[scope][['p0', 'p1', 'p2']].to_numpy())
    tail_probes = review_probes(TAIL, baseline, gold, targets, state, direction, gradients,
                                sorted(TAIL.glob('probe*/probe.json')))
    assert len(tail_probes) == diag['finite_proposals'] == 1 and tail_probes[0]['probe'] == 20 and tail_probes[0]['accepted']
    counts = check_calls(TAIL, diag, tail=True)
    assert counts['head_completed'] == counts['feature_completed'] == 44+22*len(tail_probes) == 66
    assert diag['restored_joint_TRAIN_retention']['passed']
    all_reports.append(dict(role=2, tail=True, actual_counts=counts, probes=tail_probes, parameter_restored=True))
    # Independent observed curvature of the actual two-row protected function.
    b = baseline['OOF']
    row = b.loc[b.row_position.eq(1720534)].iloc[0]
    assert row.local == 21050 and row.truth == row.pred == 2 and row.protected_correct
    m0 = float(row.logp2-row.logp1)
    slope = math.fsum(normals[0]*direction)
    curvature = []
    for p in all_reports[2]['probes'] + tail_probes:
        if p['probe'] not in [4, 8, 12, 16, 19, 20]:
            continue
        fp = ROOT/Path(p['path']).parent
        lp = np.load(fp/'OOF_logq.npy')
        margin = float(lp[21050, 2]-lp[21050, 1])
        residual = margin-m0-p['step']*slope
        curvature.append(dict(probe=p['probe'], step=p['step'], actual_margin=margin,
                              nonlinear_remainder=residual, remainder_per_step_squared=residual/p['step']**2))
    successful = [next(p for p in all_reports[r]['probes'] if p['accepted']) for r in [0, 1]] + tail_probes
    assert all(p['metrics']['OOF']['classifications_changed_vs_fixed_endpoint'] == 0 for p in successful)
    assert all(p['metrics']['deployment']['classifications_changed_vs_fixed_endpoint'] == 0 for p in successful)
    new_heads = sum(r['actual_counts']['head_completed'] for r in all_reports)
    target_grads = sum(r['actual_counts']['gradient_completed'] for r in all_reports)
    assert new_heads == 1180 and target_grads == 12
    for path, digest in binding.items():
        assert sha(ROOT/path) == digest, path
    result = dict(status='independent_all_saved_V161_probes_original_gold_costs_and_candidate_hashes_passed',
                  roles=all_reports, safe_candidates=successful,
                  protected_role2_curvature=dict(original_rows=[1720534, 1720544], initial_margin=m0,
                                                   cached_first_order_slope=slope, measured_points=curvature),
                  new_heads_and_features=new_heads, fixed_target_gradients=target_grads,
                  cumulative_heads_and_features=15728+new_heads, cumulative_full_original_class_gradients=368,
                  cumulative_fixed_target_gradients=12, cumulative_margin_gradients=10,
                  official_calls_by_this_audit=0, new_fits=0, permanent_updates=0,
                  classification_repair_achieved=False, quality_acceptance=False,
                  limitation='Saved-function and row evidence only; not new classifier execution, full quality or cross-source generalization.',
                  source_sha256=sha(Path(__file__)))
    save(OUT/'audit.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'new_heads_and_features', 'cumulative_heads_and_features',
                                            'classification_repair_achieved', 'official_calls_by_this_audit']}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            save(OUT/'failure.json', dict(error_type=type(error).__name__, error=str(error),
                                         traceback=traceback.format_exc(), official_calls=0))
        raise
