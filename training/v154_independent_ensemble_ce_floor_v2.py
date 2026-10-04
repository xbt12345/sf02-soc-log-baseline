"""Separate saved ensemble CE, conflict entropy and Jensen gap; zero model calls."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'artifacts/v154_directional_neighborhood_v2_20261001'
VERIFIED = ROOT / 'artifacts/v154_independent_fixed_neighborhood_v3_20261001/audit.json'
OUT = ROOT / 'artifacts/v154_independent_ensemble_ce_floor_v2_20261001'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    assert not OUT.exists()
    verified = read(VERIFIED)
    assert verified['probability_original_TRAIN_bindings_checked'] == 24
    for name, expected in verified['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    result, conflict_groups = [], []
    for arm in ['A', 'B']:
        for fold in range(3):
            folder = SRC / f'fold{fold}_{arm}'
            probe = read(folder / 'probe.json')
            for point in probe['points']:
                rows = pd.read_parquet(folder / (point['point'] + '_TRAIN_rows.parquet'))
                q = rows[['p0', 'p1', 'p2']].to_numpy()
                y, pure = rows.truth.to_numpy(), rows.pure_TRAIN_input.to_numpy()
                assert rows.fold.ne(fold).all()
                py = q[np.arange(len(rows)), y]
                assert (py > 0).all() and np.isfinite(py).all()
                losses = -np.log(py)
                weights = rows[~pure].groupby(['canonical_key', 'truth']).size().unstack(fill_value=0)
                masses = weights.to_numpy()
                proportions = masses / masses.sum(1, keepdims=True)
                entropy = -(np.where(masses > 0, masses * np.log(np.maximum(proportions, 1e-300)), 0)).sum() / len(rows)
                canonical_entropy = float(entropy)
                local_weights = rows[~pure].groupby(['local', 'truth']).size().unstack(fill_value=0).to_numpy()
                local_proportions = local_weights / local_weights.sum(1, keepdims=True)
                entropy = -(np.where(local_weights > 0, local_weights * np.log(np.maximum(local_proportions, 1e-300)), 0)).sum() / len(rows)
                # Actual local-to-input mapping is fixed; canonical aliases need not have exact equal outputs.
                for key, group in rows[~pure].groupby('local'):
                    unique = np.unique(group[['p0', 'p1', 'p2']].to_numpy(), axis=0)
                    assert len(unique) == 1, 'Do not treat unequal output aliases as one entropy-bound group'
                    conflict_groups.append({'arm': arm, 'fold': fold, 'point': point['point'],
                                            'local': int(key), 'rows': len(group),
                                            'M': int(group.truth.eq(1).sum()), 'S': int(group.truth.eq(2).sum()),
                                            'unique_actual_output_vectors': len(unique)})
                ensemble = float(losses.mean())
                pure_contribution = float(losses[pure].sum() / len(rows))
                mixed_contribution = float(losses[~pure].sum() / len(rows))
                member = point['TRAIN_member_CE']
                assert np.isclose(ensemble, pure_contribution + mixed_contribution, rtol=1e-12)
                assert member + 1e-12 >= ensemble >= entropy - 1e-12
                result.append({'arm': arm, 'fold': fold, 'point': point['point'], 'TRAIN_rows': len(rows),
                               'mixed_rows': int((~pure).sum()), 'member_CE': member, 'ensemble_CE': ensemble,
                               'pure_ensemble_CE_contribution': pure_contribution,
                               'mixed_ensemble_CE_contribution': mixed_contribution,
                               'local_group_empirical_entropy_contribution': float(entropy),
                               'canonical_reference_entropy_not_full_input_floor': canonical_entropy,
                               'mixed_ensemble_CE_above_entropy': mixed_contribution - float(entropy),
                               'member_minus_ensemble_Jensen_gap': member - ensemble,
                               'group_entropy_fraction_of_member_CE': float(entropy) / member})
    assert len(result) == 24
    OUT.mkdir()
    pd.DataFrame(result).to_parquet(OUT / 'all_fixed_point_TRAIN_loss_decomposition.parquet', index=False)
    pd.DataFrame(conflict_groups).to_parquet(OUT / 'actual_same_output_conflict_groups.parquet', index=False)
    audit = {'status': 'saved_TRAIN_loss_and_same_output_entropy_actual_recount',
             'own_model_forwards': 0, 'own_gradients': 0, 'own_fits': 0, 'own_updates': 0,
             'TRAIN_points_recounted': 24, 'results': result,
             'limits': ['Ensemble CE is not member CE or its gradient. Jensen gap is not a causal attribution.',
                        'Local-group entropy is a lower bound for the fixed local-to-input mapping, not for raw-input models. Canonical reference entropy is separately descriptive; floating aliases are not merged for this bound.',
                        'TRAIN labels only are used here; no weights, loss or selected point is produced.',
                        'This decomposition neither qualifies a deployed classifier nor automatically qualifies SAM.'],
             'source_sha256': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                               str(VERIFIED.relative_to(ROOT)): sha(VERIFIED)}}
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': audit['status'], 'TRAIN_points': 24,
                      'base_entropy_fraction': [r['group_entropy_fraction_of_member_CE'] for r in result if r['point'] == 'base'],
                      'own_model_forwards': 0, 'own_gradients': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
