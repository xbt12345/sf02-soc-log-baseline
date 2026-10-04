"""A classifier-decision counterexample for an initial-only correct-row mask."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v161_synthetic_cumulative_correct_guard_review_20261002'


def main():
    assert not OUT.exists()
    OUT.mkdir()
    truth = np.array([1, 1, 2, 2])
    pure = np.ones(4, bool)
    initial_pred = np.array([1, 2, 2, 1])
    first_pred = np.array([1, 1, 2, 2])
    second_pred = np.array([1, 2, 2, 2])
    initial_guard = pure & (initial_pred == truth)
    fixed_target = pure & (initial_pred != truth)
    cumulative_guard = initial_guard | (pure & (first_pred == truth))
    initial_only_new_errors = int(np.count_nonzero(initial_guard & (second_pred != truth)))
    cumulative_new_errors = int(np.count_nonzero(cumulative_guard & (second_pred != truth)))
    assert initial_only_new_errors == 0 and cumulative_new_errors == 1
    assert int(np.count_nonzero(fixed_target)) == 2
    # Targets remain fixed even after both were repaired. Only the guard grows.
    assert np.array_equal(fixed_target, pure & (initial_pred != truth))
    assert int(np.count_nonzero(first_pred != truth)) == 0
    report = dict(status='initial_only_guard_misses_newly_repaired_pure_classification_regression',
                  truth=truth.tolist(), initial_pred=initial_pred.tolist(), first_accepted_pred=first_pred.tolist(),
                  subsequent_pred=second_pred.tolist(), fixed_target_mask=fixed_target.tolist(),
                  initial_guard_mask=initial_guard.tolist(), cumulative_guard_mask=cumulative_guard.tolist(),
                  initial_only_guard_regressions=initial_only_new_errors,
                  cumulative_guard_regressions=cumulative_new_errors,
                  first_accepted_total_errors=0, subsequent_total_errors=1,
                  official_heads=0, official_features=0, official_gradients=0, official_fits=0,
                  permanent_updates=0, quality_acceptance=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Exact synthetic decision-mask counterexample only; no actual new learner or protected SOC trajectory has been executed.')
    (OUT / 'review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'initial_only_guard_regressions', 'cumulative_guard_regressions']}))


if __name__ == '__main__':
    main()
