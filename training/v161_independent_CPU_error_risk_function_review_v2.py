"""Independent CPU-only synthetic review of the new error-risk implementation."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from v161_fixed_pure_error_risk import error_risk
from v159_float64_repeat_policy_v2 import finite_step_review, repeat_gradient, repeat_values

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v161_independent_CPU_error_risk_function_review_v2_20261002'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Counter:
    def __init__(self):
        self.attempts, self.completed = [], []

    def gradient_before(self, cls, mass):
        self.attempts.append((cls, list(mass)))

    def gradient_after(self, cls, mass):
        self.completed.append((cls, list(mass)))


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[.1, -.2], [-.7, .3], [.8, -.4]], dtype=torch.float64))
        self.calls = 0

    def forward(self, x, unused):
        self.calls += 1
        z = x @ self.weight.T
        return z.softmax(-1), z.log_softmax(-1), z


class MarginFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, ids, unused):
        t = self.theta
        margins = torch.stack([2 + t, -1 - t, 2 + t, -1 - t])[ids]
        labels = torch.tensor([1, 1, 2, 2])[ids]
        row = torch.arange(len(ids))
        z = torch.full((len(ids), 3), -40., dtype=torch.float64)
        z[row, 3 - labels] = 0.
        z[row, labels] = margins
        return z.softmax(-1), z.log_softmax(-1), z


def main():
    assert not OUT.exists()
    OUT.mkdir()
    torch.set_num_threads(2)
    function_path = ROOT / 'training/v161_fixed_pure_error_risk.py'
    initial_hash = sha(function_path)
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., -1.], [-1., -1.], [-1., 1.]], dtype=torch.float64)
    c = np.zeros((22546, 3), np.int64)
    c[0, 1:] = [3, 1]
    c[1, 1], c[2, 1], c[3, 2], c[4, 2], c[5, 2] = 4, 7, 2, 5, 3
    e = np.zeros_like(c)
    e[2, 1], e[4, 2] = 7, 5
    ctx = dict(counts=c, target_counts=e, mass=c.sum(0))
    model, results = TinyClassifier(), []
    weight_before = model.weight.detach().clone()

    def builder(ctx, scope, ids):
        assert scope == 'OOF'
        return x[ids.copy()], None

    for cls in [1, 2]:
        counter = Counter()
        runs = [error_risk(model, ctx, ids, counter, cls, builder, batch_size=batch)
                for ids, batch in [(np.arange(6), 2), (np.arange(6)[::-1], 3)]]
        risk_value, q, lp, gradient = runs[0]
        assert np.isfinite(lp[:6]).all()
        assert q[2].argmax() != 1 and q[4].argmax() != 2
        assert repeat_gradient(gradient, runs[1][3])['passed']
        for target_name, counts in [('fixed_pure_error_contribution', e), ('full_original_class_CE', c)]:
            expected = [math.fsum(-lp[:6, j] * counts[:6, j]) / int(ctx['mass'][j]) for j in [1, 2]]
            assert repeat_values(risk_value[target_name], expected, 'risk')['passed']
        # Independent expansion to literal original rows, without local-count
        # weighting in the loss expression being audited.
        local_ids = np.repeat(np.arange(6), e[:6, cls])
        model.zero_grad(set_to_none=True)
        _, direct_lp, _ = model(x[local_ids], None)
        direct_loss = -direct_lp[:, cls].sum() / int(ctx['mass'][cls])
        direct_loss.backward()
        direct_gradient = model.weight.grad.detach().numpy().ravel().copy()
        assert repeat_gradient(gradient, direct_gradient)['passed']
        wrong_denominator_loss = -direct_lp[:, cls].mean().detach().item()
        assert not repeat_values(risk_value['fixed_pure_error_contribution'][cls - 1], wrong_denominator_loss, 'risk')['passed']
        assert counter.attempts == counter.completed == [(cls, ctx['mass'].tolist())] * 2
        results.append(dict(class_id=cls, target_original_rows=int(e[:, cls].sum()), complete_original_mass=int(ctx['mass'][cls]),
                            literal_original_row_gradient_matches=True, partition_and_order_invariant=True,
                            subset_mass_denominator_rejected=True))
    assert torch.equal(model.weight, weight_before)
    c2 = np.zeros((22546, 3), np.int64)
    c2[[0, 1], 1] = [100, 1]
    c2[[2, 3], 2] = [100, 1]
    e2 = np.zeros_like(c2)
    e2[1, 1] = e2[3, 2] = 1
    fixture = MarginFixture()
    ctx2 = dict(counts=c2, target_counts=e2, mass=c2.sum(0))

    def margin_builder(ctx, scope, ids):
        return torch.tensor(ids, dtype=torch.int64), None

    before, qb, _, g = error_risk(fixture, ctx2, np.arange(4), Counter(), 1, margin_builder, batch_size=2)
    with torch.no_grad():
        fixture.theta.fill_(-1.5)
    after, qa, _, _ = error_risk(fixture, ctx2, np.arange(4), inputs_builder=margin_builder, batch_size=2)
    truth = np.array([1, 1, 2, 2])
    assert (qb[:4].argmax(-1) != truth).sum() == 2
    assert np.array_equal(qa[:4].argmax(-1), truth)
    assert np.all(after['full_original_class_CE'] > before['full_original_class_CE'])
    finite = finite_step_review(before['fixed_pure_error_contribution'], after['fixed_pure_error_contribution'],
                                [-float(g[0]), -float(g[0])], 101, 101, 'B', 1.5, True)
    assert finite['accepted']
    assert sha(function_path) == initial_hash
    report = dict(status='new_error_risk_function_independent_CPU_original_row_gradient_and_acceptance_fixtures_passed',
                  target_classes=results, complete_CE_increase_classification_repair_fixture=finite,
                  source_sha256={str(function_path.relative_to(ROOT)): initial_hash,
                                 str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
                  synthetic_only=True, device='cpu', official_heads=0, official_features=0,
                  official_gradients=0, official_fits=0, permanent_official_updates=0,
                  quality_acceptance=False,
                  scope='Synthetic tiny classifiers execute the actual new risk function. Does not prove official head, new source seal, legal fixed-cohort selection, GPU repeatability, or SOC finite-step eligibility.')
    (OUT / 'review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], official_calls=0)))


if __name__ == '__main__':
    main()

