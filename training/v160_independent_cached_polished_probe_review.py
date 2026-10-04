"""Independently check saved finite probes; no model, gradient or fit call."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from v160_independent_fixed_diagnostic_review import rows_review, read, sha, EPS
from v160_independent_saved_direction_certificate import certificate

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'artifacts/v160_cached_polished_direction_finite_probe_20261002'
OLD = ROOT / 'artifacts/v160_fixed_endpoint_diagnostic_20261002'
OUT = ROOT / 'artifacts/v160_independent_cached_polished_probe_review_20261002'


def role_review(spec, gold):
    role = spec['role']
    f = INPUT / f'role{role}'
    summary = read(f / 'diagnostic.json')
    old = {scope: pd.read_parquet(OLD / f'role{role}' / parent / f'{scope}_original_rows.parquet')
           for scope, parent in [('OOF', 'baseline_class1'), ('deployment', 'baseline_deployment')]}
    baseline = {scope: pd.read_parquet(f / 'baseline' / f'{scope}_original_rows.parquet')
                for scope in old}
    expected_mask = old['OOF'].protected_correct | (old['OOF'].pred.eq(old['OOF'].truth) & old['OOF'].pure_current_input)
    assert np.array_equal(baseline['OOF'].protected_correct, expected_mask)
    protected = pd.read_parquet(f / 'fixed_endpoint_correct_protection_rows.parquet')
    assert np.array_equal(protected.row_position, baseline['OOF'].loc[expected_mask, 'row_position'])
    assert protected.pure_current_input.all()
    fixed_repairs = old['OOF'].pred.eq(old['OOF'].truth) & ~old['OOF'].initial_correct & old['OOF'].pure_current_input
    assert expected_mask[fixed_repairs].all()
    zero_review = read(f / 'zero_step_repeat_review.json')
    assert all(item['passed'] for item in zero_review.values())
    for scope in old:
        checked, _, _, _ = rows_review(f / 'baseline', scope, baseline[scope], gold)
        assert np.array_equal(checked.pred, old[scope].pred)
        assert np.array_equal(checked[['row_position', 'local', 'truth', 'root']], old[scope][['row_position', 'local', 'truth', 'root']])
    gradients = [np.load(OLD / f'role{role}/baseline_class{c}/gradient.npy') for c in [1, 2]]
    normals = np.load(OLD / f'role{role}/round{spec["failed_original_round"]}/raw_margin_normals.npy')
    d = np.load(ROOT / spec['qualified_direction'])
    qualified = read(ROOT / spec['qualified_certificate'])
    signs = certificate(*gradients, normals, d)
    assert signs == qualified['independent_saved_vector_certificate'] and signs['eligible_for_finite_trial_only']
    before = np.load(f / 'baseline_risk.npy')
    probes = []
    for index, p in enumerate(sorted(f.glob('probe*/probe.json'), key=lambda x: int(x.parent.name[5:]))):
        target = p.parent
        report = read(p)
        assert target.name == f'probe{index}' and report['step'] == 2.**(-index)
        assert np.array_equal(np.load(target / 'direction.npy'), d)
        actual_risk, metrics, frames = None, {}, {}
        for scope in ['OOF', 'deployment']:
            frame, risk, _, met = rows_review(target, scope, baseline[scope], gold)
            frames[scope], metrics[scope] = frame, met
            for name in ['original_rows', 'M_errors', 'S_errors', 'pure_errors', 'total_errors',
                         'new_errors_vs_initial', 'protected_regressions', 'repairs_vs_initial']:
                assert met[name] == report[f'{scope}_stats'][name]
            if scope == 'OOF':
                actual_risk = risk
        saved = np.load(target / 'risk.npy')
        assert np.all(np.abs(actual_risk - saved) <= 8 * EPS * np.maximum(1., np.abs(saved)))
        slopes = np.array([math.fsum(g * d) for g in gradients])
        assert np.all(np.abs(slopes - report['class_slopes']) <= 16 * EPS * np.maximum(1., np.abs(slopes)))
        guard = (metrics['OOF']['protected_regressions'] == 0 and
                 metrics['deployment']['new_errors_vs_initial'] == 0 and
                 report['deployment_stats']['mastered'] and report['joint_TRAIN_retention']['passed'])
        assert guard == report['classification_guard']
        resolution = 16 * EPS * np.maximum(1., np.maximum(np.abs(before), np.abs(saved)))
        drop = before - saved
        slack = before + 1e-4 * report['step'] * np.array(report['class_slopes']) - saved
        accepted = bool(guard and report['actual_parameter_change'] and np.all(slopes < 0) and
                        np.all(drop > resolution) and np.all(slack > resolution))
        assert accepted == report['accepted']
        blockers = pd.read_parquet(target / 'actual_blocking_original_rows.parquet')
        expected = []
        for scope, frame in frames.items():
            mask = frame.protected_correct if scope == 'OOF' else frame.initial_correct
            expected += [(scope, int(row.row_position), int(row.local), int(row.truth), int(row.pred))
                         for row in frame[mask & frame.pred.ne(frame.truth)].itertuples()]
        observed = [(row.scope, int(row.row_position), int(row.local), int(row.truth), int(row.rival))
                    for row in blockers.itertuples()]
        assert sorted(expected) == sorted(observed)
        progress = read(target / 'fixed_endpoint_progress.json')
        for scope, frame in frames.items():
            b = old[scope]
            wrong, initial_wrong = frame.pred.ne(frame.truth), b.pred.ne(b.truth)
            assert progress[scope]['repairs_vs_fixed_endpoint'] == int((initial_wrong & ~wrong).sum())
            assert progress[scope]['new_errors_vs_fixed_endpoint'] == int((~initial_wrong & wrong).sum())
            assert progress[scope]['pure_new_errors_vs_fixed_endpoint'] == int((~initial_wrong & wrong & b.pure_current_input).sum())
            if accepted:
                assert progress[scope]['pure_new_errors_vs_fixed_endpoint'] == 0
        probes.append(dict(index=index, accepted=accepted, step=report['step'],
                           class_risk_drop=drop.tolist(), metrics=metrics, progress=progress,
                           blocking_original_rows=len(expected)))
    assert len(probes) == summary['finite_proposals'] <= spec['finite_proposal_cap']
    assert summary['finite_probe_pass'] == any(p['accepted'] for p in probes)
    assert sum(p['accepted'] for p in probes) <= 1
    events = [json.loads(line) for line in (f / 'calls.jsonl').read_text().splitlines()]
    counts = summary['counts']
    for kind in ['head', 'feature']:
        for event, suffix in [('attempt', 'attempts'), ('completed', 'completed')]:
            assert sum(x['event'] == event and x['kind'] == kind for x in events) == counts[f'{kind}_{suffix}']
    assert not any('gradient' in x['kind'] for x in events)
    assert counts['gradient_attempts'] == counts['gradient_completed'] == 0
    expected_heads = (2 + len(probes)) * (spec['OOF_chunks'] + spec['deployment_chunks'])
    assert counts['head_attempts'] == counts['head_completed'] == counts['feature_attempts'] == counts['feature_completed'] == expected_heads <= spec['head_cap']
    assert summary['initial_parameter_sha256'] == summary['restored_parameter_sha256'] == spec['endpoint_parameter_sha256']
    assert summary['new_class_gradients'] == summary['new_margin_gradients'] == summary['new_fits'] == summary['permanent_updates'] == 0
    for scope in old:
        restored, _, _, _ = rows_review(f / 'restored', scope, baseline[scope], gold)
        assert np.array_equal(restored.pred, baseline[scope].pred)
    return dict(role=role, status=summary['status'], counts=counts, protected_original_rows=int(expected_mask.sum()),
                fixed_endpoint_pure_repairs_protected=int(fixed_repairs.sum()), probes=probes,
                source_vector_signs_verified=True, parameter_hash_and_restored_classifications_match=True)


def main():
    assert not OUT.exists()
    OUT.mkdir()
    plan = read(ROOT / 'training/review_policy/v160_cached_polished_direction_finite_probe_contract.json')
    gold = pd.read_parquet(ROOT / 'data/official/train.parquet', columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert len(gold) == 2056871 and np.isfinite(gold).all()
    roles = [role_review(spec, gold) for spec in plan['roles']]
    report = dict(status='cached_polished_actual_probes_original_gold_protection_and_costs_independently_passed',
                  roles=roles, source_sha256=sha(Path(__file__)), official_heads=0, official_features=0,
                  official_gradients=0, official_fits=0, permanent_updates=0, quality_acceptance=False,
                  scope='Recomputes saved actual finite outputs and legal original labels; no new model call or wider trajectory safety assertion.')
    (OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], official_calls=0)))


if __name__ == '__main__':
    main()
