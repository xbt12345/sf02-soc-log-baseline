"""Bind the proposed fixed error targets from saved rows; no model calls."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OLD = ROOT / 'artifacts/v160_fixed_endpoint_diagnostic_20261002'
BANK = ROOT / 'artifacts/v158_legal_fusion_bank_v2_20261001'
PREP = ROOT / 'artifacts/v159_boundary_input_preparation_20261002'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir()
    gold_path = ROOT / 'data/official/train.parquet'
    gold = pd.read_parquet(gold_path, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert len(gold) == 2056871 and np.isfinite(gold).all()
    sources = {str(gold_path.relative_to(ROOT)): sha(gold_path)}
    result, target_positions, all_positions = [], [], []
    for role in range(3):
        baseline_path = OLD / f'role{role}/baseline_class1/OOF_original_rows.parquet'
        legal_path = BANK / f'fold{role}/legal_FIT_reference.parquet'
        prep_path = PREP / f'fold{role}/OOF_visible_input_rows.parquet'
        baseline, legal, prep = [pd.read_parquet(p) for p in [baseline_path, legal_path, prep_path]]
        for path in [baseline_path, legal_path, prep_path]:
            sources[str(path.relative_to(ROOT))] = sha(path)
        assert baseline.row_position.is_unique
        for reference in [legal, prep]:
            for name in ['row_position', 'local', 'truth']:
                assert np.array_equal(baseline[name], reference[name])
        assert np.array_equal(baseline.truth, gold[baseline.row_position.to_numpy()])
        assert np.array_equal(baseline.pure_current_input, prep.pure_current_input)
        assert baseline.truth.isin([1, 2]).all()
        pure = baseline.pure_current_input.to_numpy()
        wrong = baseline.pred.ne(baseline.truth).to_numpy()
        target = baseline.loc[pure & wrong,
                              ['row_position', 'local', 'truth', 'root', 'pred', 'stable_CE']].copy()
        target.rename(columns={'pred': 'fixed_endpoint_rival'}, inplace=True)
        protected = baseline.loc[pure & ~wrong,
                                 ['row_position', 'local', 'truth', 'root']].copy()
        assert len(target) + len(protected) + int((~pure).sum()) == len(baseline)
        assert not set(target.row_position).intersection(protected.row_position)
        # Proposed target counts retain each original row exactly once. They
        # must not be replaced by one count per local numeric input.
        target_counts = np.bincount(target.local.to_numpy(np.int64) * 3 +
                                    target.truth.to_numpy(np.int64),
                                    minlength=22546 * 3).reshape(22546, 3)
        assert int(target_counts.sum()) == len(target)
        for local, group in target.groupby('local'):
            assert group.truth.nunique() == 1
        role_out = OUT / f'role{role}'
        role_out.mkdir()
        target.to_parquet(role_out / 'fixed_pure_error_targets.parquet', index=False)
        protected.to_parquet(role_out / 'fixed_pure_correct_protection.parquet', index=False)
        np.save(role_out / 'target_original_counts.npy', target_counts)
        classes = []
        for cls in [1, 2]:
            mask = baseline.truth.eq(cls).to_numpy()
            error_mask = mask & pure & wrong
            denominator = int(mask.sum())
            classes.append(dict(class_id=cls, complete_original_class_mass=denominator,
                                fixed_pure_error_original_rows=int(error_mask.sum()),
                                fixed_pure_error_numeric_locals=int(baseline.loc[error_mask, 'local'].nunique()),
                                fixed_pure_error_source_roots=int(baseline.loc[error_mask, 'root'].nunique()),
                                fixed_error_CE_original_frequency_contribution=
                                math.fsum(baseline.loc[error_mask, 'stable_CE']) / denominator,
                                all_original_class_CE=math.fsum(baseline.loc[mask, 'stable_CE']) / denominator,
                                mixed_original_rows=int((mask & ~pure).sum()),
                                mixed_original_errors=int((mask & ~pure & wrong).sum()),
                                protected_pure_original_rows=int((mask & pure & ~wrong).sum())))
        target_positions.extend(target.row_position.tolist())
        all_positions.extend(baseline.row_position.tolist())
        result.append(dict(role=role, original_rows=len(baseline), fixed_targets=len(target),
                           protected_pure_correct_rows=len(protected), mixed_rows=int((~pure).sum()),
                           classes=classes, generated_sha256={p.name: sha(p) for p in role_out.iterdir()}))
    assert [r['fixed_targets'] for r in result] == [3278, 1960, 1586]
    report = dict(status='saved_legal_original_error_cohorts_bound_without_model_calls', roles=result,
                  target_role_entries=len(target_positions), unique_target_original_rows=len(set(target_positions)),
                  all_role_entries=len(all_positions), unique_legal_original_rows=len(set(all_positions)),
                  official_gold_population=len(gold), official_gold_class_counts=np.bincount(gold, minlength=3).tolist(),
                  input_sha256=sources, source_sha256=sha(Path(__file__)),
                  official_heads=0, official_features=0, official_gradients=0, official_fits=0,
                  permanent_updates=0, quality_acceptance=False,
                  scope='Binds saved fixed pure flags to prequalified visible input rows, legal original labels and frequency. Does not recalculate full visible-input identities, prove new gradients, or authorize a training entry.')
    (OUT / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'target_role_entries', 'unique_target_original_rows',
                                          'all_role_entries', 'unique_legal_original_rows']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
