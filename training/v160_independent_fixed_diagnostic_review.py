"""Read saved V160 vectors/rows only; never call an official model or gradient."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from v160_independent_saved_direction_certificate import certificate

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'artifacts/v160_fixed_endpoint_diagnostic_20261002'
OUT = ROOT / 'artifacts/v160_independent_fixed_diagnostic_review_20261002'
EPS = float(np.finfo(float).eps)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rows_review(folder, scope, baseline, gold):
    frame = pd.read_parquet(folder / f'{scope}_original_rows.parquet')
    q = np.load(folder / f'{scope}_q.npy')
    lp = np.load(folder / f'{scope}_logq.npy')
    assert q.shape == lp.shape == (22546, 3)
    ids = frame.local.to_numpy()
    truth = frame.truth.to_numpy()
    assert np.isfinite(q[ids]).all() and np.isfinite(lp[ids]).all()
    assert np.array_equal(truth, gold[frame.row_position.to_numpy()])
    for name in ['row_position', 'local', 'truth', 'root', 'initial_correct',
                 'pure_current_input', 'protected_correct']:
        assert np.array_equal(frame[name], baseline[name])
    pred = q[ids].argmax(1)
    assert np.array_equal(pred, frame.pred)
    assert np.array_equal(q[ids], frame[['p0', 'p1', 'p2']].to_numpy())
    assert np.array_equal(lp[ids], frame[['logp0', 'logp1', 'logp2']].to_numpy())
    ce = -lp[ids, truth]
    assert np.array_equal(ce, frame.stable_CE)
    wrong = pred != truth
    mass = np.bincount(truth, minlength=3)
    risks = np.array([math.fsum(ce[truth == c]) / int(mass[c]) for c in [1, 2]])
    metrics = dict(original_rows=len(frame),
                   M_errors=int(np.count_nonzero(wrong & (truth == 1))),
                   S_errors=int(np.count_nonzero(wrong & (truth == 2))),
                   total_errors=int(wrong.sum()),
                   pure_errors=int(np.count_nonzero(wrong & frame.pure_current_input)),
                   protected_regressions=int(np.count_nonzero(wrong & frame.protected_correct)),
                   new_errors_vs_initial=int(np.count_nonzero(wrong & frame.initial_correct)),
                   repairs_vs_initial=int(np.count_nonzero(~wrong & ~frame.initial_correct)),
                   classifications_changed_vs_fixed_endpoint=int(np.count_nonzero(pred != baseline.pred)),
                   repairs_vs_fixed_endpoint=int(np.count_nonzero(~wrong & baseline.pred.ne(baseline.truth))),
                   new_errors_vs_fixed_endpoint=int(np.count_nonzero(wrong & baseline.pred.eq(baseline.truth))))
    return frame, risks, mass, metrics


def review_role(role, gold, plan):
    folder = INPUT / f'role{role}'
    diagnostic = read(folder / 'diagnostic.json')
    baseline = {scope: pd.read_parquet(folder / name / f'{scope}_original_rows.parquet')
                for scope, name in [('OOF', 'baseline_class1'), ('deployment', 'baseline_deployment')]}
    _, initial_risk, mass, _ = rows_review(folder / 'baseline_class1', 'OOF', baseline['OOF'], gold)
    gradients = [np.load(folder / f'baseline_class{c}/gradient.npy') for c in [1, 2]]
    assert all(g.shape == (1060832,) and np.isfinite(g).all() for g in gradients)
    qp_reports = []
    for target in sorted(folder.glob('round*')):
        recorded = read(target / 'QP_certificate.json')
        direction = np.load(target / 'direction.npy')
        normals = np.load(target / 'raw_margin_normals.npy')
        records = read(target / 'normal_records.json')
        assert len(records) == len(normals)
        for index, (identity, info) in enumerate(records.items()):
            nf = folder / 'normals' / identity
            assert read(nf / 'measurement_repeat_review.json')['gradient']['passed']
            assert np.array_equal(normals[index], np.load(nf / 'repeat0_gradient.npy'))
            assert info['base_parameter_sha256'] == diagnostic['initial_parameter_sha256']
        computed = certificate(*gradients, normals, direction)
        assert computed == recorded['independent_saved_vector_certificate']
        if recorded['status'] == 'local_QP_certified_requires_actual_finite_guard':
            assert computed['eligible_for_finite_trial_only']
        qp_reports.append(dict(round=target.name, status=recorded['status'],
                               certificate=computed, normal_count=len(normals)))
    probes = []
    for path in sorted(folder.rglob('probe.json')):
        target = path.parent
        report = read(path)
        frames, measured_risk, m, metrics = {}, None, None, {}
        for scope in ['OOF', 'deployment']:
            frame, risk, masses, met = rows_review(target, scope, baseline[scope], gold)
            frames[scope], metrics[scope] = frame, met
            for key in ['original_rows', 'M_errors', 'S_errors', 'pure_errors',
                        'total_errors', 'protected_regressions', 'new_errors_vs_initial',
                        'repairs_vs_initial']:
                assert met[key] == report[f'{scope}_stats'][key]
            if scope == 'OOF':
                measured_risk, m = risk, masses
        saved_risk = np.load(target / 'risk.npy')
        assert np.all(np.abs(saved_risk - measured_risk) <= 8 * EPS * np.maximum(1., np.abs(saved_risk)))
        guard = (metrics['OOF']['protected_regressions'] == 0 and
                 metrics['deployment']['new_errors_vs_initial'] == 0 and
                 report['deployment_stats']['mastered'] and
                 report['joint_TRAIN_retention']['passed'])
        assert guard == report['classification_guard']
        slopes = np.array([math.fsum(g * np.load(target / 'direction.npy')) for g in gradients])
        assert np.all(np.abs(slopes - report['class_slopes']) <= 16 * EPS * np.maximum(1., np.abs(slopes)))
        before = np.load(folder / 'baseline_class1/risk.npy')
        resolution = 16 * EPS * np.maximum(1., np.maximum(np.abs(before), np.abs(saved_risk)))
        actual_drop = before - saved_risk
        slack = before + 1e-4 * report['step'] * np.asarray(report['class_slopes']) - saved_risk
        accepted = bool(guard and report['actual_parameter_change'] and
                        np.all(slopes < 0) and np.all(actual_drop > resolution) and np.all(slack > resolution))
        assert accepted == report['accepted']
        actual_blockers = pd.read_parquet(target / 'actual_blocking_original_rows.parquet')
        expected = []
        for scope, f in frames.items():
            protection = f.protected_correct if scope == 'OOF' else f.initial_correct
            expected += [(scope, int(row.row_position), int(row.local), int(row.truth), int(row.pred))
                         for row in f[protection & f.pred.ne(f.truth)].itertuples()]
        observed = [(str(row.scope), int(row.row_position), int(row.local), int(row.truth), int(row.rival))
                    for row in actual_blockers.itertuples()]
        assert sorted(expected) == sorted(observed)
        probes.append(dict(path=target.relative_to(ROOT).as_posix(), step=report['step'],
                           accepted=accepted, class_risk_drop=actual_drop.tolist(),
                           metrics=metrics, actual_blocking_original_rows=len(observed)))
    events = [json.loads(x) for x in (folder / 'calls.jsonl').read_text().splitlines()]
    counts = diagnostic['counts']
    for kind, prefix, cap in [('head', 'head', plan['head_cap']), ('feature', 'feature', plan['feature_cap']),
                              ('full_class_gradient', 'gradient', plan['class_gradient_cap']),
                              ('full_parameter_margin_gradient', 'margin', plan['margin_gradient_cap'])]:
        for event, suffix in [('attempt', 'attempts'), ('completed', 'completed')]:
            actual = sum(x['event'] == event and x['kind'] == kind for x in events)
            assert actual == counts[f'{prefix}_{suffix}'] <= cap
    assert len(probes) == diagnostic['finite_proposals'] <= plan['finite_proposal_cap']
    expected_heads = (3 * plan['OOF_chunks'] + 24 + counts['margin_completed'] +
                      len(probes) * (plan['OOF_chunks'] + 12))
    assert counts['head_completed'] == counts['feature_completed'] == expected_heads
    assert diagnostic['initial_parameter_sha256'] == diagnostic['restored_parameter_sha256'] == plan['endpoint_parameter_sha256']
    assert diagnostic['permanent_updates'] == diagnostic['official_fits'] == 0
    for scope in ['OOF', 'deployment']:
        f, _, _, _ = rows_review(folder / 'restored_replay', scope, baseline[scope], gold)
        assert np.array_equal(f.pred, baseline[scope].pred)
    assert diagnostic['finite_probe_pass'] == any(p['accepted'] for p in probes)
    return dict(role=role, status=diagnostic['status'], counts=counts,
                finite_probe_pass=diagnostic['finite_probe_pass'], qp=qp_reports, probes=probes,
                fixed_endpoints_restored=True, no_official_calls_by_this_audit=True)


def main():
    assert not OUT.exists()
    OUT.mkdir()
    source_hash = sha(Path(__file__))
    plan = read(ROOT / 'training/review_policy/v160_fixed_endpoint_diagnostic_contract.json')
    gold = pd.read_parquet(ROOT / 'data/official/train.parquet', columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert len(gold) == 2056871 and np.isfinite(gold).all()
    roles = [review_role(role, gold, plan['roles'][role]) for role in [0, 1, 2]]
    output = dict(status='saved_vectors_original_gold_all_probes_and_costs_independently_passed',
                  roles=roles, source_sha256=source_hash, official_heads=0, official_features=0,
                  official_gradients=0, official_fits=0, permanent_updates=0, quality_acceptance=False,
                  limits='Uses saved output arrays and measured vectors. No new model call, no gradient repeat beyond saved measurements, no nonlinear safety extrapolation or task acceptance.')
    (OUT / 'audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: output[k] for k in ['status', 'official_heads', 'official_fits', 'quality_acceptance']}))


if __name__ == '__main__':
    main()
