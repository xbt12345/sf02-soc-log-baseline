"""Root review of actual restoration code and collateral classification.

Calls the new vector arithmetic function with CPU fixtures only. No official
classifier, feature, gradient, fit or parameter update is performed.
"""
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np

from v162_single_function_finite_restoration import propose
from v159_float64_repeat_policy_v2 import finite_step_review

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v162_independent_restoration_function_review_20261002'
DIAG = ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role2'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def outputs(point, collateral=False):
    x, y = point[-2:]
    m0 = 3.8535841184739184e-13
    logits = [[-50., -1+1.4*x-.05*y, 0.], [-50., 0., -2+2*x+.1*y],
              [-50., 0., m0+y-.27018*x*x]]
    truth = [1, 2, 2]
    if collateral:
        logits.append([-50., m0-y-.27018*x*x, 0.])
        truth.append(1)
    z = np.asarray(logits)
    shifted = z-z.max(1, keepdims=True)
    lp = shifted-np.log(np.exp(shifted).sum(1, keepdims=True))
    pred = np.exp(lp).argmax(1)
    return np.asarray(truth), pred, lp, -lp[np.arange(len(truth)), truth]


def fixture(width, collateral):
    def embed(x, y):
        v = np.zeros(width)
        v[-2:] = x, y
        return v
    point = np.zeros(width)
    truth, old_pred, old_lp, ce = outputs(point, collateral)
    u, a = embed(1., 0.), embed(0., 1.)
    _, bad_pred, bad_lp, _ = outputs(point+u, collateral)
    m0 = float(old_lp[2, 2]-old_lp[2, 1])
    failed = float(bad_lp[2, 2]-bad_lp[2, 1])
    sigmoid = lambda v: 1./(1.+math.exp(-v))
    gm = embed(-1.4, .05)*(1.-sigmoid(-1.))
    gs = embed(-2., -.1)*(1.-sigmoid(-2.))
    originals = [v.copy() for v in [u, a, gm, gs]]
    result = propose(u, a, m0, failed, gm, gs)
    assert result['status'] == 'linear_restoration_candidate_requires_full_actual_finite_guard'
    assert not result['finite_step_authority'] and not result['QP_optimality_claim']
    assert all(np.array_equal(v, saved) for v, saved in zip([u, a, gm, gs], originals))
    displacement = result['displacement']
    assert np.array_equal(displacement, u+result['correction'])
    _, pred, _, new_ce = outputs(point+displacement, collateral)
    protected = old_pred == truth
    guard = bool(np.all(pred[protected] == truth[protected]))
    slopes = [math.fsum(g*displacement) for g in [gm, gs]]
    review = finite_step_review(ce[:2], new_ce[:2], slopes, 1, 1, 'B', 1., guard)
    assert not np.all(bad_pred[protected] == truth[protected])
    assert review['accepted'] == (not collateral)
    assert np.array_equal(pred[:2], truth[:2])
    if collateral:
        assert pred[2] == truth[2] and pred[3] != truth[3]
        assert review['reason'] == 'classification_guard'
    else:
        assert np.array_equal(pred, truth)
    # The unchanged 16eps finite gate must also reject a drop below resolution.
    tiny = finite_step_review(np.ones(2), np.nextafter(np.ones(2), 0.),
                              [-1., -1.], 1, 1, 'B', 1e-16, True)
    assert not tiny['accepted'] and tiny['reason'] == 'below_numeric_resolution'
    return dict(width=width, additional_protected_boundary=collateral,
                proposal_arithmetic_qualified=True, old_linear_tangent_candidate_unsafe=True,
                original_target_classifications_repaired=True,
                protected_regressions=int(np.count_nonzero(pred[protected] != truth[protected])),
                actual_fixture_acceptance=review, tiny_drop_rejected=True,
                input_vectors_not_mutated=True, official_classifier_calls=0)


def main():
    assert not OUT.exists()
    OUT.mkdir()
    deps = [Path(__file__).resolve(), ROOT/'training/v162_single_function_finite_restoration.py',
            ROOT/'training/v160_independent_saved_direction_certificate.py',
            ROOT/'training/v159_float64_repeat_policy_v2.py']
    deps += [DIAG/'round0/polished_direction.npy', DIAG/'round0/raw_margin_normals.npy',
             DIAG/'baseline_error_class1_repeat0/OOF_logq.npy', DIAG/'round0/probe4/OOF_logq.npy']
    deps += [DIAG/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1, 2]]
    binding = {p.relative_to(ROOT).as_posix(): sha(p) for p in deps}
    save(OUT/'pre_bindings.json', dict(source_sha256=binding, official_calls=0))
    fixtures = [fixture(w, extra) for w in [2, 1060832] for extra in [False, True]]
    d = np.load(DIAG/'round0/polished_direction.npy')
    a = np.load(DIAG/'round0/raw_margin_normals.npy')[0]
    lp = np.load(DIAG/'baseline_error_class1_repeat0/OOF_logq.npy')
    failed_lp = np.load(DIAG/'round0/probe4/OOF_logq.npy')
    gm, gs = [np.load(DIAG/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1, 2]]
    m0 = float(lp[21050, 2]-lp[21050, 1])
    failed = float(failed_lp[21050, 2]-failed_lp[21050, 1])
    reviewed = propose(d/16., a, m0, failed, gm, gs)
    assert reviewed['status'] == 'linear_restoration_candidate_requires_full_actual_finite_guard'
    expected = (m0-failed)/math.fsum(a*a)*a
    assert np.array_equal(reviewed['correction'], expected)
    assert reviewed['relative_correction_norm'] < .004
    for c in reviewed['class_reviews']:
        assert c['resolved_negative']
    for path, digest in binding.items():
        assert sha(ROOT/path) == digest
    result = dict(status='actual_restoration_function_independent_curvature_collateral_and_resolution_cases_passed',
                  fixtures=fixtures, saved_SOC_candidate_arithmetic_matched=True,
                  source_sha256=binding, official_calls=0, official_gradients=0, fits=0,
                  permanent_updates=0, actual_corrected_SOC_safety_or_quality_proven=False,
                  limitation='Single protected-function proposal; full classifier evaluation must reject all collateral regressions. Toy classifier and saved-vector arithmetic do not prove SOC classification gains.')
    save(OUT/'review.json', result)
    print(json.dumps(dict(status=result['status'], fixture_cases=len(fixtures), official_calls=0, fits=0)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            save(OUT/'failure.json', dict(error_type=type(error).__name__, error=str(error),
                                         traceback=traceback.format_exc(), official_calls=0))
        raise
