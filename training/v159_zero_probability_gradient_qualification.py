"""Synthetic zero-probability/autograd review; no official model/data calls."""
import hashlib
import json
from pathlib import Path

import torch


def synthetic_case(q, target, delta_values, detached_zero_branch=False):
    q = torch.tensor([q], dtype=torch.float64)
    delta = torch.tensor([delta_values], dtype=torch.float64, requires_grad=True)
    logits = q.clamp_min(1e-12).log() + delta
    if detached_zero_branch:
        probability = torch.where(delta.eq(0), q, logits.softmax(-1))
        loss = -probability[0, target].clamp_min(1e-12).log()
    else:
        probability = logits.softmax(-1)
        loss = -logits.log_softmax(-1)[0, target]
    gradient, = torch.autograd.grad(loss, delta)
    return dict(probability=probability.detach().tolist()[0],
                loss=float(loss.detach()), gradient=gradient.tolist()[0],
                finite_loss_gradient=bool(torch.isfinite(loss) and torch.isfinite(gradient).all()))


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'artifacts/v159_zero_probability_gradient_qualification_20261001'
    output.mkdir(parents=True, exist_ok=True)
    saturated = synthetic_case([0., 1., 0.], 2, [0., 0., 0.])
    broken_branch = synthetic_case([0., 1., 0.], 2, [0., 0., 0.], True)
    assert saturated['finite_loss_gradient']
    assert saturated['gradient'][2] < -.99 and saturated['gradient'][1] > .99
    assert broken_branch['gradient'] == [0., 0., 0.]
    extreme = synthetic_case([0., 1., 0.], 2, [0., 1000., -1000.])
    assert extreme['finite_loss_gradient'] and extreme['gradient'][2] < -.99
    # An unconstrained class logit can cross a consensus-wrong expert mixture.
    crossed = synthetic_case([.01, .98, .01], 2, [0., 0., 8.])
    assert crossed['probability'].index(max(crossed['probability'])) == 2
    retained = synthetic_case([.1, .4, .5], 2, [0., 0., 0.])
    assert max(abs(a - b) for a, b in zip(retained['probability'], [.1, .4, .5])) < 1e-15
    invalid = []
    for q in ([-.1, .6, .5], [float('nan'), .5, .5], [0., 0., 0.]):
        t = torch.tensor(q)
        rejected = bool(not torch.isfinite(t).all() or (t < 0).any() or abs(float(t.sum()) - 1.) > 1e-12)
        assert rejected
        invalid.append(rejected)
    report = dict(status='synthetic_zero_probability_and_gradient_checks_passed_not_official_head',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  fixed_diagnostic_floor=1e-12,
                  zero_true_probability_usable_gradient=saturated,
                  exact_probability_zero_branch_dead_gradient_counterexample=broken_branch,
                  extreme_logits_stable_log_loss=extreme,
                  consensus_wrong_class_can_be_changed=crossed,
                  nonsaturated_zero_correction_retention=retained,
                  invalid_probabilities_rejected=invalid,
                  official_classifier_calls=0, official_feature_calls=0,
                  official_gradient_calls=0, official_fits=0, official_updates=0,
                  scope=['Only synthetic logits/probabilities and synthetic autograd are executed.',
                         'Log floor changes initial tiny probabilities; the actual-data initialization still requires a registered tolerance and independent identity checks.',
                         'The generic class-logit transform does not prescribe a new network or guarantee retention of N probabilities.',
                         'Crossing expert consensus is an expressivity witness, not evidence of learned SOC discrimination.',
                         'Stable training log-loss and clipped saved-probability CE must be distinguished.'])
    (output / 'qualification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
