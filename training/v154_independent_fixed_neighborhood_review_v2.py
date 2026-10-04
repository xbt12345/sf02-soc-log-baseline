"""Read-only recount of fixed diagnostic receipts; no model/gradient execution."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'artifacts/v154_directional_neighborhood_v2_20261001'
OUT = ROOT / 'artifacts/v154_independent_fixed_neighborhood_v2_20261001'
REF = ROOT / 'artifacts/v153_independent_training_transfer_gap_20261001'
OLD = ROOT / 'artifacts/v146_guarded_pair_training_20261001'
POINTS = ['base', 'plus_TRAIN_gradient', 'minus_TRAIN_gradient', 'fixed_random']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def flat(values):
    return torch.cat([v.reshape(-1) for v in values.values()])


def counts(y, pred, base, mask):
    wrong, oldwrong = pred != y, base != y
    return {'rows': int(mask.sum()),
            'class_support': [int((mask & (y == c)).sum()) for c in range(3)],
            'class_errors': [int((mask & wrong & (y == c)).sum()) for c in range(3)],
            'repairs': int((mask & ~wrong & oldwrong).sum()),
            'new_errors': int((mask & wrong & ~oldwrong).sum()),
            'decision_changes': int((mask & (pred != base)).sum())}


def main():
    assert not OUT.exists(), 'Preserve completed independent receipts'
    audit = read(SRC / 'audit.json')
    seal = read(SRC / 'run_seal.json')
    plan = read(ROOT / 'training/review_policy/v154_directional_neighborhood_plan_v2.json')
    assert seal['plan_sha256'] == sha(ROOT / 'training/review_policy/v154_directional_neighborhood_plan_v2.json')
    assert plan['full_gradient_evaluations'] == 18 and plan['relative_L2_radius'] == .001
    assert not plan['held_labels_used_for_direction'] and not plan['model_selection']
    assert not plan['automatic_training']
    assert audit['full_classifier_gradient_evaluations'] == 18
    assert audit['classifier_fits'] == audit['persistent_parameter_updates'] == 0
    bound = {}
    for name, expected in seal['source_sha256'].items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        assert path.is_file() and sha(path) == expected, name
    originals = pd.read_parquet(REF / 'all_original_classifier_gap_and_control_ledger.parquet')
    trainref = pd.read_parquet(REF / 'both_arms_legal_TRAIN_outer_control.parquet')
    assert len(originals) == 112807 and originals.row_position.is_unique
    assert len(trainref) == 451228
    y = originals.truth.to_numpy()
    masks = {'all_ASA': np.ones(len(originals), dtype=bool),
             'known_578': originals.known_578_cohort.to_numpy(),
             'strict_correct_51': originals.same_family_and_outer_fold_control_S.to_numpy()}
    assert masks['known_578'].sum() == 578 and masks['strict_correct_51'].sum() == 51
    identities = ['row_position', 'local', 'root', 'fold', 'truth', 'canonical_key', 'pure_TRAIN_input']
    receipts, summaries, changes, trainstats = [], [], [], []
    geometry_checks = probability_ledger_checks = 0
    for arm in ['A', 'B']:
        qouter = {p: np.empty((len(originals), 3), dtype=np.float64) for p in POINTS}
        trainparts = {p: [] for p in POINTS}
        for fold in range(3):
            folder = SRC / f'fold{fold}_{arm}'
            probe = read(folder / 'probe.json')
            fit = read(OLD / f'fold{fold}_{arm}/fit.json')
            assert sha(OLD / f'fold{fold}_{arm}/endpoint.pt') == fit['model_sha256']
            assert sha(OLD / f'fold{fold}_{arm}/sealed_all_prob.npy') == fit['probability_sha256']
            assert probe['before_model_tensor_sha256'] == probe['after_model_tensor_sha256']
            assert probe['seal_sha256'] == sha(SRC / 'run_seal.json')
            assert probe['full_classifier_gradient_evaluations'] == 3
            assert probe['classifier_fits'] == probe['persistent_parameter_updates'] == 0
            assert sha(folder / 'directions.pt') == probe['directions_sha256']
            assert sha(folder / 'gradients.pt') == probe['gradients_sha256']
            gradients = torch.load(folder / 'gradients.pt', map_location='cpu', weights_only=True)
            directions = torch.load(folder / 'directions.pt', map_location='cpu', weights_only=True)
            state = torch.load(OLD / f'fold{fold}_{arm}/endpoint.pt', map_location='cpu', weights_only=True)['state']
            expected_names = ['weight', 'r', 's', 'bias']
            assert list(gradients['base_gradient']) == expected_names
            assert set(state) == set(expected_names + ['reference_weight', 'reference_r', 'reference_s', 'reference_bias', 'head_weight', 'head_bias', 'head_facts'])
            state = {k: state[k] for k in expected_names}
            norm = float(flat(gradients['base_gradient']).norm())
            parameter_norm = float(flat(state).norm())
            assert flat(state).numel() == 22528
            assert np.isclose(parameter_norm, probe['parameter_L2'], rtol=1e-12)
            rho = .001 * parameter_norm
            assert np.isclose(norm, probe['base_gradient_L2'], rtol=1e-12)
            for k, value in gradients['base_gradient'].items():
                assert torch.equal(value, gradients['repeated_base_gradient'][k])
                assert torch.equal(directions['plus_TRAIN_gradient'][k], value * (rho / norm))
                assert torch.equal(directions['minus_TRAIN_gradient'][k], -directions['plus_TRAIN_gradient'][k])
            for direction in directions.values():
                assert np.isclose(float(flat(direction).norm()), rho, rtol=1e-12)
            geometry_checks += 1
            expected = trainref[(trainref.arm == arm) & (trainref.training_role == fold)].reset_index(drop=True)
            assert expected.fold.ne(fold).all() and expected.row_position.is_unique
            baseline = np.load(OLD / f'fold{fold}_{arm}/sealed_all_prob.npy')
            outside = originals.fold.eq(fold).to_numpy()
            for item in probe['points']:
                point = item['point']
                qpath, tpath = folder / f'{point}_all_prob.npy', folder / f'{point}_TRAIN_rows.parquet'
                assert sha(qpath) == item['probability_sha256'] and sha(tpath) == item['TRAIN_rows_sha256']
                q, rows = np.load(qpath), pd.read_parquet(tpath).reset_index(drop=True)
                assert np.isfinite(q).all() and q.shape == baseline.shape
                assert np.allclose(q.sum(axis=1), 1., rtol=0, atol=1e-12)
                assert rows[identities].equals(expected[identities])
                assert rows.training_role.eq(fold).all()
                pred = q[rows.local].argmax(axis=1)
                assert np.array_equal(pred, rows.pred)
                assert np.array_equal(rows.pred_base, expected.pred)
                assert np.array_equal(rows[['p0', 'p1', 'p2']].to_numpy(), q[rows.local])
                if point == 'base':
                    assert np.array_equal(q, baseline)
                    assert np.array_equal(rows[['p0', 'p1', 'p2']].to_numpy(), expected[['p0', 'p1', 'p2']].to_numpy())
                wrong = rows.pred.ne(rows.truth)
                assert item['pure_TRAIN_errors'] == int((wrong & rows.pure_TRAIN_input).sum())
                assert item['TRAIN_class_errors'] == [int((wrong & rows.truth.eq(c)).sum()) for c in range(3)]
                assert item['all_base_correct_TRAIN_regressions'] == int((wrong & rows.pred_base.eq(rows.truth)).sum())
                qouter[point][outside] = q[originals.loc[outside, 'local']]
                trainparts[point].append(rows)
                probability_ledger_checks += 1
                bound[str(qpath.relative_to(ROOT))] = sha(qpath)
                bound[str(tpath.relative_to(ROOT))] = sha(tpath)
            receipts.append({'arm': arm, 'fold': fold, 'gradient_norm': norm,
                             'parameter_norm': parameter_norm, 'radius': rho,
                             'base_member_CE': probe['original_member_CE_gradient'],
                             'plus_member_CE': probe['plus_member_CE_gradient'],
                             'geometry_and_saved_repeated_gradients_exact': True,
                             'checkpoint_matches_actual_V146': True})
        base = qouter['base'].argmax(axis=1)
        assert np.array_equal(base, originals[f'outer_{arm}_pred'])
        assert np.array_equal(qouter['base'], originals[[f'outer_{arm}_p{c}' for c in range(3)]].to_numpy())
        for point in POINTS:
            pred = qouter[point].argmax(axis=1)
            for cohort, mask in masks.items():
                record = {'arm': arm, 'point': point, 'cohort': cohort, **counts(y, pred, base, mask),
                          'max_abs_probability_delta': float(np.abs(qouter[point][mask] - qouter['base'][mask]).max())}
                summaries.append(record)
                reported = next(s for s in audit['outer_summaries'] if s['arm'] == arm and s['point'] == point and s['cohort'] == cohort)
                assert record['class_errors'] == reported['outer_class_errors']
                assert record['repairs'] == reported['outer_repairs'] and record['new_errors'] == reported['outer_new_errors']
            rows = pd.concat(trainparts[point], ignore_index=True)
            assert len(rows) == 225614 and not rows.duplicated(['row_position', 'training_role']).any()
            wrong = rows.pred.ne(rows.truth)
            trainstats.append({'arm': arm, 'point': point, 'rows': len(rows),
                               'pure_rows': int(rows.pure_TRAIN_input.sum()),
                               'pure_errors': int((wrong & rows.pure_TRAIN_input).sum()),
                               'all_errors': int(wrong.sum()),
                               'old_correct_new_errors': int((wrong & rows.pred_base.eq(rows.truth)).sum())})
            changed = pred != base
            z = originals.loc[changed, ['row_position', 'root', 'local', 'fold', 'truth', 'canonical_key']].copy()
            z['arm'], z['point'] = arm, point
            z['base_pred'], z['stress_pred'] = base[changed], pred[changed]
            z['known_578'] = masks['known_578'][changed]
            z['strict_correct_51'] = masks['strict_correct_51'][changed]
            changes.append(z)
    assert geometry_checks == 6 and probability_ledger_checks == 24
    for path in [Path(__file__), SRC / 'audit.json', SRC / 'run_seal.json',
                 REF / 'all_original_classifier_gap_and_control_ledger.parquet',
                 REF / 'both_arms_legal_TRAIN_outer_control.parquet']:
        bound[str(path.relative_to(ROOT))] = sha(path)
    OUT.mkdir()
    pd.DataFrame(summaries).to_parquet(OUT / 'recounted_fixed_point_summary.parquet', index=False)
    pd.concat(changes, ignore_index=True).to_parquet(OUT / 'all_changed_original_rows.parquet', index=False)
    result = {'status': 'actual_fixed_neighborhood_receipts_independently_recounted',
              'own_model_forwards': 0, 'own_gradients': 0, 'own_fits': 0, 'own_updates': 0,
              'executed_worker_gradients': 18, 'executed_worker_forward_chunks': audit['classifier_forward_chunk_calls'],
              'executed_worker_classifier_fits': 0, 'original_ASA_rows': 112807,
              'sealed_dependency_hashes_checked': len(seal['source_sha256']),
              'six_endpoint_tensor_geometries_checked': geometry_checks,
              'probability_original_TRAIN_bindings_checked': probability_ledger_checks,
              'endpoint_receipts': receipts, 'TRAIN': trainstats, 'outer': summaries,
              'no_probe_selected_or_promoted': True,
              'scope': 'Saved tensors/probabilities and original identities recounted. No independent model forward, maximized sharpness, causal transfer proof, SAM fit or new full-task quality evaluation.',
              'source_sha256': bound, 'python': sys.version}
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k in ['status', 'TRAIN', 'outer']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
