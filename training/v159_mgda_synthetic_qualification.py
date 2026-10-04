"""Synthetic mathematics only: no project classifier, gold or gradient calls.

The two-class common-descent calculation is adapted from the two-objective
minimum-norm formulation, not a copy of an archived framework implementation.
This script does not authorize or execute an official-data fit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def exact_two_gradient_direction(g_m, g_s):
    gm = np.asarray(g_m, dtype=np.float64)
    gs = np.asarray(g_s, dtype=np.float64)
    if gm.shape != gs.shape or gm.size == 0:
        raise ValueError("Gradient shapes must match and be nonempty")
    if not (np.isfinite(gm).all() and np.isfinite(gs).all()):
        raise ValueError("Gradients must be finite")
    # Common scaling preserves the optimum and avoids squared overflow.
    scale = max(float(np.max(np.abs(gm))), float(np.max(np.abs(gs))))
    if scale == 0:
        return 0.5, np.zeros_like(gm)
    a_m, a_s = gm / scale, gs / scale
    mm = float(np.vdot(a_m, a_m))
    ms = float(np.vdot(a_m, a_s))
    ss = float(np.vdot(a_s, a_s))
    if ms >= mm:
        alpha = 1.0
    elif ms >= ss:
        alpha = 0.0
    else:
        alpha = (ss - ms) / ((mm - ms) + (ss - ms))
    return alpha, -(alpha * gm + (1 - alpha) * gs)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts/v159_mgda_synthetic_qualification_20261001"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(15901)
    checked = 0
    max_violation = 0.0
    for dimension in (1, 2, 7, 257):
        for _ in range(250):
            scale = 10.0 ** rng.uniform(-8, 8)
            gm, gs = rng.normal(size=(2, dimension)) * scale
            alpha, direction = exact_two_gradient_direction(gm, gs)
            assert 0 <= alpha <= 1
            bound = -float(np.vdot(direction, direction))
            normalizer = max(float(np.vdot(gm, gm)), float(np.vdot(gs, gs)))
            violation = max(float(np.vdot(gm, direction)) - bound,
                            float(np.vdot(gs, direction)) - bound) / normalizer
            max_violation = max(max_violation, violation)
            assert violation <= 1e-12
            # Independent one-dimensional convex minimization check.
            candidates = np.linspace(0, 1, 2001)
            mm, ms, ss = (float(np.vdot(gm, gm)), float(np.vdot(gm, gs)),
                          float(np.vdot(gs, gs)))
            grid_costs = candidates ** 2 * mm + 2 * candidates * (1 - candidates) * ms + (1 - candidates) ** 2 * ss
            assert -bound <= float(grid_costs.min()) + normalizer * 1e-12
            checked += 1

    gm, gs = np.array([1., 0.]), np.array([-1., 2.])
    ordinary_direction = -(0.95 * gm + 0.05 * gs)
    alpha, common_direction = exact_two_gradient_direction(gm, gs)
    ordinary_slope_s = float(gs @ ordinary_direction)
    common_slopes = [float(g @ common_direction) for g in (gm, gs)]
    assert ordinary_slope_s > 0 and max(common_slopes) < 0

    endpoint_alpha, endpoint_direction = exact_two_gradient_direction([1., 0.], [2., 1.])
    assert endpoint_alpha == 1.0 and np.array_equal(endpoint_direction, [-1., 0.])
    _, zero_direction = exact_two_gradient_direction([0., 0.], [2., 1.])
    assert np.array_equal(zero_direction, [0., 0.])
    _, opposite_direction = exact_two_gradient_direction([1., 0.], [-2., 0.])
    assert np.linalg.norm(opposite_direction) < 1e-15
    _, equal_direction = exact_two_gradient_direction([1., 2.], [1., 2.])
    assert np.array_equal(equal_direction, [-1., -2.])

    def actual_loss_changes(step):
        dx = step * common_direction
        return [float(gm @ dx + .5 * (dx @ dx)),
                float(gs @ dx + .5 * (dx @ np.diag([1., 40.]) @ dx))]

    oversized_changes = actual_loss_changes(1.0)
    assert oversized_changes[1] > 0
    accepted = None
    for exponent in range(12):
        step = 2.0 ** -exponent
        changes = actual_loss_changes(step)
        if max(changes) < 0:
            accepted = {"step": step, "actual_loss_changes": changes}
            break
    assert accepted is not None

    initial_true_probs = np.array([.51, .01])
    final_true_probs = np.array([.49, .02])
    ce_initial = float(-np.log(initial_true_probs).mean())
    ce_final = float(-np.log(final_true_probs).mean())
    assert ce_final < ce_initial
    correct_initial, correct_final = (int((p > .5).sum()) for p in (initial_true_probs, final_true_probs))
    assert correct_final < correct_initial

    report = {
        "status": "synthetic_mathematical_qualification_passed_not_SOC_quality",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "random_gradient_pairs": checked,
        "maximum_normalized_common_descent_violation": max_violation,
        "ordinary_mean_gradient_can_raise_S": {"S_directional_derivative": ordinary_slope_s},
        "exact_two_objective_direction": {"M_weight": alpha, "class_directional_derivatives": common_slopes},
        "finite_step_counterexample": {"oversized_step_loss_changes": oversized_changes,
                                         "synthetic_backtracking_result": accepted},
        "classification_counterexample": {"S_mean_CE_initial": ce_initial, "S_mean_CE_final": ce_final,
                                             "S_correct_initial": correct_initial, "S_correct_final": correct_final},
        "boundaries": [
            "Common first-order decrease does not ensure finite-step decrease; evaluate both actual risks.",
            "Per-class CE decrease does not ensure per-class classification retention.",
            "Opposing or zero gradients can have no strict common-descent direction; report infeasibility.",
            "Applying Adam to this vector would change the proved direction; no Adam guarantee is made.",
            "No implication of cross-source transfer or causal M/S label identification."
        ],
        "official_classifier_calls": 0, "official_feature_calls": 0,
        "official_gradient_calls": 0, "official_fits": 0, "official_parameter_updates": 0,
        "references": ["https://arxiv.org/abs/1810.04650",
                       "https://github.com/isl-org/MultiObjectiveOptimization/blob/master/multi_task/min_norm_solvers_numpy.py"]
    }
    (out / "qualification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
