"""Actual multi-normal function, independent toy classification and rank cases.

CPU vector arithmetic only, no official model, features or gradients.
"""
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np

from v163_one_sided_joint_restoration import propose
from v159_float64_repeat_policy_v2 import finite_step_review

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v163_one_sided_nonlinear_fixture_qualification_20261002'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(width):
    def embed(values):
        result = np.zeros(width)
        result[-3:] = values
        return result
    def classifier(point):
        x, y, z = point[-3:]
        logits = np.array([[-50., -1+1.4*x-.05*y-.03*z, 0.],
                           [-50., 0., -2+2*x+.1*y+.02*z],
                           [-50., 0., 3.8535841184739184e-13+y-.27018*x*x],
                           [-50., 0., 7.707168236947837e-13+z-.13*x*x]])
        shifted = logits-logits.max(1, keepdims=True)
        lp = shifted-np.log(np.exp(shifted).sum(1, keepdims=True))
        q = np.exp(lp)
        truth = np.array([1, 2, 2, 2])
        ce = -lp[np.arange(len(truth)), truth]
        return truth, q.argmax(1), lp, ce
    point, displacement = np.zeros(width), embed([1., 0., 0.])
    truth, pred0, lp0, ce0 = classifier(point)
    _, pred1, lp1, _ = classifier(point+displacement)
    base_margins = lp0[2:, 2]-lp0[2:, 1]
    current_margins = lp1[2:, 2]-lp1[2:, 1]
    normals = np.stack([embed([0., 1., 0.]), embed([0., 0., 1.])])
    sigmoid = lambda v: 1./(1.+math.exp(-v))
    gm = embed([-1.4, .05, .03])*(1.-sigmoid(-1.))
    # One fixed S error / full original S mass of three.
    gs = embed([-2., -.1, -.02])*(1.-sigmoid(-2.))/3.
    copied = [v.copy() for v in [displacement, normals, base_margins, current_margins, gm, gs]]
    result = propose(displacement, normals, base_margins, current_margins, gm, gs)
    assert result['status'] == 'one_sided_joint_restoration_requires_full_actual_finite_guard'
    assert not result['finite_step_authority'] and not result['QP_optimality_claim']
    assert all(np.array_equal(v, old) for v, old in zip(
        [displacement, normals, base_margins, current_margins, gm, gs], copied))
    corrected = result['displacement']
    _, pred, lp, ce = classifier(point+corrected)
    assert np.array_equal(pred, truth)
    guard = bool(np.all(pred[pred0 == truth] == truth[pred0 == truth]))
    slopes = [math.fsum(g*corrected) for g in [gm, gs]]
    gate = finite_step_review([ce0[0], ce0[1]/3.], [ce[0], ce[1]/3.], slopes, 1, 3, 'B', 1., guard)
    assert gate['accepted']
    # A one-boundary restore remains unsafe for the second old correct row.
    one = propose(displacement, normals[:1], base_margins[:1], current_margins[:1], gm, gs)
    _, single_pred, _, single_ce = classifier(point+one['displacement'])
    single_guard = bool(np.all(single_pred[pred0 == truth] == truth[pred0 == truth]))
    assert not single_guard and single_pred[2] == truth[2] and single_pred[3] != truth[3]
    single_gate = finite_step_review([ce0[0], ce0[1]/3.], [single_ce[0], single_ce[1]/3.],
        [math.fsum(g*one['displacement']) for g in [gm, gs]], 1, 3, 'B', 1., single_guard)
    assert not single_gate['accepted'] and single_gate['reason'] == 'classification_guard'
    opposing = np.stack([normals[0], -normals[0]])
    conflict = propose(displacement, opposing, base_margins, current_margins, gm, gs)
    assert conflict['status'] == 'local_inequality_or_common_descent_unqualified_stop'
    assert conflict['no_global_infeasibility_claim']
    cases = []
    for units, permutation in [([1e-24, 1e24], [0, 1]), ([1e12, 1e-12], [1, 0])]:
        scale = np.asarray(units)
        ix = np.asarray(permutation)
        r = propose(displacement, (normals*scale[:, None])[ix], (base_margins*scale)[ix],
                    (current_margins*scale)[ix], gm, gs)
        assert r['status'] == result['status']
        gap = r['displacement']-corrected
        assert float(np.abs(gap).max()) <= 8*np.finfo(float).eps*float(np.abs(corrected).max())
        assert float(np.linalg.norm(gap)) <= 8*np.finfo(float).eps*float(np.linalg.norm(corrected))
        _, unit_pred, _, _ = classifier(point+r['displacement'])
        assert np.array_equal(unit_pred, truth)
        cases.append(dict(units=units, row_permutation=permutation, actual_fixture_classes_preserved=True))
    # New v2 must emit ordinary JSON without hiding a failed qualification.
    serialized = json.dumps({k: v for k, v in result.items() if k not in ['displacement', 'correction']})
    assert json.loads(serialized)['status'] == result['status']
    return dict(width=width, two_protected_functions_restored=True,
                two_target_classifications_repaired_in_fixture=True,
                target_original_frequency_denominator_S=3,
                finite_guard_and_16eps_Armijo=gate,
                one_function_collateral_rejected=single_gate,
                opposing_constraints_explicitly_rejected=True,
                row_units_and_order_cases=cases, inputs_not_mutated=True,
                actual_SOC_safety_or_quality_proven=False)


def main():
    assert not OUT.exists()
    OUT.mkdir()
    deps = [Path(__file__).resolve(), ROOT/'training/v163_one_sided_joint_restoration.py',
            ROOT/'training/v160_independent_saved_direction_certificate.py',
            ROOT/'training/v159_float64_repeat_policy_v2.py']
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in deps}
    save(OUT/'pre_bindings.json', dict(source_sha256=bindings, official_calls=0))
    cases = [fixture(width) for width in [3, 1060832]]
    for path, digest in bindings.items():
        assert sha(ROOT/path) == digest
    result = dict(status='one_sided_joint_restoration_new_function_collateral_and_two_boundary_CPU_fixture_cases_passed',
                  cases=cases, official_calls=0, official_gradients=0, fits=0,
                  permanent_updates=0, source_sha256=bindings, quality_acceptance=False,
                  limitation='CPU nonlinear fixture only; does not replace actual SOC full-row finite guards, legal gradient identities or training/generalization results.')
    save(OUT/'review.json', result)
    print(json.dumps(dict(status=result['status'], widths=len(cases), official_calls=0)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            save(OUT/'failure.json', dict(error_type=type(error).__name__, error=str(error),
                                         traceback=traceback.format_exc(), official_calls=0))
        raise
