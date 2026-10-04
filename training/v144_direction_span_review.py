"""Analytical TRAIN-only coefficient feasibility; no search, fits or gradients."""
import json
from pathlib import Path

from v144_independent_decision_review import ROOT, sha, read

OUT = ROOT / 'artifacts/v144_direction_span_review_20261001'


def interval(a, b):
    """All r >= 0 for which a + r*b <= 0, from an existing local derivative."""
    if b > 0:
        return None if a > 0 else [0.0, -a / b]
    if b < 0:
        return [max(0.0, -a / b), None]
    return None if a > 0 else [0.0, None]


def intersect(values):
    if any(v is None for v in values):
        return None
    lower = max(v[0] for v in values)
    upper = min((v[1] for v in values if v[1] is not None), default=float('inf'))
    return None if lower > upper else [lower, None if upper == float('inf') else upper]


def main():
    if OUT.exists():
        raise FileExistsError('Preserve the completed analytical evidence')
    original = ROOT / 'artifacts/v144_aux_gradient_evidence_20261001/probe.json'
    review_path = ROOT / 'artifacts/v144_independent_decision_review_20261001/review.json'
    p, reviewed = read(original), read(review_path)
    assert reviewed['source_sha256'][original.relative_to(ROOT).as_posix()] == sha(original)
    assert reviewed['new_fits'] == reviewed['new_gradients'] == reviewed['new_updates'] == 0
    data = []
    for role in p['folds']:
        f = role['fold']
        values = []
        for c in reviewed['matched_CE_only_recovered_algebraically']:
            if c['training_role'] != f:
                continue
            a = c['CE_only_class_member_CE_derivative']
            b = c['auxiliary_increment_class_member_CE_derivative'] / p['auxiliary_gradient_ratio']
            feasible = interval(a, b)
            values.append(feasible)
            # Verify the registered 0.1 point algebraically, not as a new trial.
            assert abs(a + 0.1*b - c['combined_class_member_CE_derivative_at_equal_CE_displacement']) < 1e-14
            data.append(dict(training_role=f, truth=c['truth'], CE_only_slope=a,
                             unit_auxiliary_slope=b, nonnegative_ratio_local_CE_nonincrease_interval=feasible,
                             pure_mean_margin_CE_only=c['pure_original_margin_derivatives']['CE_only_unit_margin_derivative']['mean'],
                             pure_mean_margin_unit_auxiliary=(
                                 c['pure_original_margin_derivatives']['increment_from_auxiliary']['mean']/0.1)))
        common = intersect(values)
        data.append(dict(training_role=f, simultaneous_M_S_local_CE_nonincrease_interval=common))
    # Role 2 M has both positive loss slopes: no positive combination repairs this risk.
    m2 = next(x for x in data if x.get('training_role') == 2 and x.get('truth') == 1)
    assert m2['CE_only_slope'] > 0 and m2['unit_auxiliary_slope'] > 0
    assert m2['nonnegative_ratio_local_CE_nonincrease_interval'] is None
    assert m2['pure_mean_margin_CE_only'] < 0 and m2['pure_mean_margin_unit_auxiliary'] < 0
    OUT.mkdir()
    result = dict(status='analytical_local_two_direction_span_review', new_fits=0, new_gradients=0,
                  new_updates=0, coefficient_trials=0, analytical_intervals=data,
                  fixed_registered_ratio=p['auxiliary_gradient_ratio'],
                  all_roles_common_ratio_for_both_class_CE_nonincrease_exists=False,
                  conclusion='Changing only a nonnegative CE/auxiliary norm ratio cannot remove role-2 M local CE/mean-margin risk at the fixed endpoint.',
                  limitations=[
                      'Intervals describe first-order TRAIN loss slopes, not classification error rates.',
                      'They do not prohibit a bounded finite-update trial with real prediction guards.',
                      'They do not show that either gradient remains unchanged later in training.',
                      'No coefficient was selected, scanned, fitted or chosen with HELD labels.',
                      'No claim that all contrastive methods or other parameter directions are infeasible.'
                  ], evidence_sha256={v.relative_to(ROOT).as_posix(): sha(v)
                                      for v in [Path(__file__), original, review_path,
                                                ROOT/'training/v144_independent_decision_review.py']})
    (OUT/'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
