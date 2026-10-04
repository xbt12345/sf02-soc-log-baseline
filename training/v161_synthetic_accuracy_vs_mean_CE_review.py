"""Counterexample: preserving correct decisions does not require mean CE descent."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v161_synthetic_accuracy_vs_mean_CE_review_20261002'


def state(theta):
    # Each M/S class has 100 initially correct original rows and one wrong
    # original row. The same one-dimensional parameter controls both margins.
    correct_margin = 2 + theta
    wrong_margin = -1 - theta
    correct_loss = float(np.logaddexp(0., -correct_margin))
    wrong_loss = float(np.logaddexp(0., -wrong_margin))
    return dict(theta=theta, correct_margin=correct_margin, old_wrong_margin=wrong_margin,
                original_mean_class_CE=(100 * correct_loss + wrong_loss) / 101,
                original_frequency_old_error_CE_contribution=wrong_loss / 101,
                old_correct_regressions_per_class=100 * int(correct_margin <= 0),
                remaining_errors_per_class=100 * int(correct_margin <= 0) + int(wrong_margin <= 0),
                recall_per_class=(100 * int(correct_margin > 0) + int(wrong_margin > 0)) / 101)


def main():
    assert not OUT.exists()
    baseline, whole_loss_step, error_step = [state(v) for v in [0., .5, -1.5]]
    assert whole_loss_step['original_mean_class_CE'] < baseline['original_mean_class_CE']
    assert whole_loss_step['original_frequency_old_error_CE_contribution'] > baseline['original_frequency_old_error_CE_contribution']
    assert whole_loss_step['old_wrong_margin'] < baseline['old_wrong_margin']
    assert error_step['old_correct_regressions_per_class'] == error_step['remaining_errors_per_class'] == 0
    assert error_step['original_mean_class_CE'] > baseline['original_mean_class_CE']
    assert error_step['original_frequency_old_error_CE_contribution'] < baseline['original_frequency_old_error_CE_contribution']
    correct_probability = 1 / (1 + np.exp(-2.))
    wrong_probability = 1 / (1 + np.exp(1.))
    full_gradient = (-100 * (1 - correct_probability) + (1 - wrong_probability)) / 101
    error_gradient = (1 - wrong_probability) / 101
    assert full_gradient < 0 < error_gradient
    report = dict(status='synthetic_exact_classification_repair_excluded_by_whole_mean_CE_gate',
                  baseline=baseline, whole_mean_CE_descent_step=whole_loss_step,
                  old_error_objective_classification_repair=error_step,
                  complete_class_gradient_at_zero=float(full_gradient),
                  original_frequency_old_error_gradient_at_zero=float(error_gradient),
                  original_rows=202, classes=['M', 'S'], original_counts_changed=False,
                  labels_changed=False, old_correct_decisions_preserved=True,
                  official_heads=0, official_features=0, official_gradients=0,
                  official_fits=0, permanent_updates=0, quality_acceptance=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limits='Analytic two-class linear-margin counterexample only. Does not establish a feasible SOC error-target direction or authorize changing a sealed execution policy.')
    OUT.mkdir()
    (OUT / 'review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
