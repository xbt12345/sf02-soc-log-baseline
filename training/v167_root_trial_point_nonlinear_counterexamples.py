"""Actual full-width CPU solver counterexamples, no official data or model."""
import json
from pathlib import Path

import numpy as np

from v160_independent_fixed_diagnostic_review import sha
from v166_coverage_joint_restoration import propose

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v167_root_trial_point_nonlinear_counterexamples_20261002'


def main():
    assert not OUT.exists()
    width = 1060832
    gradient = np.zeros(width)
    gradient[0] = -1.
    cases = []
    for name, origin, trial, value, derivative, maximum in [
        ('convex_boundary_trial_Jacobian_repairs_actual_constraint', -1.1, -0.9,
            lambda z: z*z-1., lambda z: 2*z, 1),
        ('concave_boundary_two_local_passes_still_actual_unsafe', .9, 1.1,
            lambda z: 1.-z*z, lambda z: -2*z, 2),
    ]:
        u = np.zeros(width)
        u[0] = trial-origin
        steps = []
        for index in range(maximum):
            at = origin+u[0]
            c = value(at)
            assert c < 0
            normal = np.zeros((1, width))
            normal[0, 0] = derivative(at)
            result = propose(u, normal, np.array([value(origin)]), np.array([c]), gradient, 2*gradient)
            assert result['status'] == 'one_sided_joint_restoration_requires_full_actual_finite_guard'
            d = result['displacement']
            correction = result['correction']
            predicted = c+derivative(at)*correction[0]
            actual = value(origin+d[0])
            steps.append(dict(stage=index+1, linearization_point=float(at), input_margin=float(c),
                actual_trial_Jacobian=float(derivative(at)), predicted_margin=float(predicted), actual_margin=float(actual),
                original_unit_inequalities_passed=all(r['passed'] for r in result['inequality_reviews']),
                both_origin_class_slopes_resolved_negative=all(r['resolved_negative'] for r in result['class_reviews']),
                actual_finite_constraint_passed=bool(actual >= 0), actual_CPU_QP_iterations=result['optimizer_iterations'],
                nonlinear_violation_ratio=abs(min(actual, 0))/abs(c), nonzero_displacement_coordinates=int(np.count_nonzero(d))))
            u = d
        cases.append(dict(case=name, parameters=width, original_safe_margin=float(value(origin)), stages=steps))
    assert cases[0]['stages'][0]['actual_finite_constraint_passed']
    assert all(not s['actual_finite_constraint_passed'] for s in cases[1]['stages'])
    assert all(s['nonlinear_violation_ratio'] < .99 for s in cases[1]['stages'])
    OUT.mkdir()
    sources = [Path(__file__).resolve(), ROOT/'training/v166_coverage_joint_restoration.py',
        ROOT/'training/v160_independent_saved_direction_certificate.py', ROOT/'training/v159_float64_repeat_policy_v2.py']
    report = dict(status='full_width_trial_point_solver_has_actual_success_and_two_stage_finite_failure_counterexamples',
        cases=cases, actual_CPU_synthetic_QP_solves=3, official_heads=0, official_features=0,
        official_derivatives=0, fits=0, permanent_updates=0,
        required_action='Keep actual finite classification gates and stop after the registered two corrections even if nonlinear violation shrinks and every local certificate passes',
        scope='Synthetic smooth constrained optimization fixture, not a SOC classifier or official quality test',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    (OUT/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases', 'source_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
