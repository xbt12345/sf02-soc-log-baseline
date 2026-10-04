"""One free three-class head; synthetic qualification only until sealed entry.

No data loaders, teacher calls, optimizer, or official activation live here.
The logits must come from the 16 current, role-legal saved NN members.
"""
from __future__ import annotations
import math
import torch
from torch import nn


class CurrentInputBoundary(nn.Module):
    def __init__(self, width=66287, hidden=16, seed=15901):
        super().__init__()
        if width <= 0 or hidden <= 0:
            raise ValueError('Positive input and hidden widths required')
        self.width, self.hidden = width, hidden
        rng = torch.Generator(device='cpu').manual_seed(seed)
        self.observation_weight = nn.Parameter(torch.randn(width, hidden, generator=rng, dtype=torch.float64) / math.sqrt(width))
        self.opinion_weight = nn.Parameter(torch.randn(11, hidden, generator=rng, dtype=torch.float64) / math.sqrt(11))
        self.bias = nn.Parameter(torch.zeros(hidden, dtype=torch.float64))
        self.output_weight = nn.Parameter(torch.zeros(hidden, 3, dtype=torch.float64))

    def forward(self, x, logits):
        if logits.ndim != 3 or logits.shape[1:] != (16, 3):
            raise ValueError('Exactly 16 current members and three outputs required')
        if x.ndim != 2 or x.shape != (len(logits), self.width):
            raise ValueError('Complete current input width/row identity required')
        if x.dtype != torch.float64 or logits.dtype != torch.float64:
            raise ValueError('Explicit float64 cast required; no fitted normalization')
        values = x if x.layout == torch.strided else x.values()
        if not torch.isfinite(values).all() or not torch.isfinite(logits).all():
            raise ValueError('Invalid inputs cannot be repaired by probability flooring')
        lp = torch.log_softmax(logits, dim=-1)
        p = torch.softmax(logits, dim=-1)
        entropy = -(p * lp).sum(-1, keepdim=True)
        common = torch.cat([p.mean(1), p.var(1, unbiased=False), entropy.mean(1)], -1)
        features = torch.cat([p, common[:, None, :].expand(-1, 16, -1), entropy], -1)
        obs = x @ self.observation_weight if x.layout == torch.strided else torch.sparse.mm(x, self.observation_weight)
        hidden = torch.tanh(obs[:, None, :] + features @ self.opinion_weight + self.bias)
        delta = (hidden @ self.output_weight).mean(1)
        shifted = logits + delta[:, None, :]
        # No delta==0 branch: output gradients must survive zero initialization.
        q = torch.softmax(shifted, dim=-1).mean(1)
        logq = torch.logsumexp(torch.log_softmax(shifted, dim=-1), dim=1) - math.log(16)
        if not torch.isfinite(q).all() or not torch.isfinite(logq).all():
            raise ValueError('Nonfinite model output')
        return q, logq, delta


def original_class_risk(logq, counts, cls):
    """Stable ensemble CE with complete original-row class frequency."""
    if cls not in (0, 1, 2) or counts.shape != logq.shape or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Invalid original class counts')
    mass = counts[:, cls]
    total = mass.sum()
    if total <= 0:
        raise ValueError('Absent class cannot have a descent certificate')
    return -(mass * logq[:, cls]).sum() / total


def clipped_probability_class_ce(q, counts, cls):
    """Separate historical diagnostic, never the training objective."""
    return original_class_risk(torch.log(q.clamp_min(1e-300)), counts, cls)


def flattened_class_gradient(risk, model):
    gradients = torch.autograd.grad(risk, tuple(model.parameters()), retain_graph=True)
    if not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError('Nonfinite complete-head gradient')
    return torch.cat([g.reshape(-1) for g in gradients])
