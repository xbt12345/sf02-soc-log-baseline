"""V137 round-one original readout; frozen caches, no alternative architecture."""
import torch
from torch import nn


class Readout(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.weight = nn.Parameter(state['head.weight'].detach().clone().double())
        self.bias = nn.Parameter(state['head.bias'].detach().clone().double())
        self.facts = nn.Parameter(state['facts_direct.weight'].detach().clone().double())

    def forward(self, hidden, facts):
        return (hidden.transpose(0, 1) @ self.weight).transpose(0, 1) + self.bias + (facts @ self.facts.T)[:, None, :]


def full_gradient(model, hidden, facts, counts, ids, chunk=2048):
    """Each original label occurrence exactly once, independent-member CE."""
    model.zero_grad(set_to_none=True)
    total = counts[ids].sum()
    loss_value = 0.
    seen = torch.zeros(3, dtype=torch.float64, device=counts.device)
    for start in range(0, len(ids), chunk):
        take = ids[start:start+chunk]
        mass = counts[take]
        logits = model(hidden[take], facts[take])
        loss = -(torch.log_softmax(logits, -1).mean(1) * mass).sum() / total
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite original-row risk')
        loss.backward()
        loss_value += float(loss.detach())
        seen += mass.sum(0)
    if not torch.equal(seen, counts[ids].sum(0)):
        raise ValueError('Full-role original exposure mismatch')
    if any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite readout gradient')
    return loss_value, seen.cpu().tolist()


@torch.no_grad()
def probabilities(model, hidden, facts, ids, chunk=2048):
    result = []
    for start in range(0, len(ids), chunk):
        take = ids[start:start+chunk]
        result.append(torch.softmax(model(hidden[take], facts[take]), -1).mean(1).cpu())
    return torch.cat(result).numpy()


class GradientBudgetExhausted(RuntimeError):
    pass


def bounded_lbfgs_step(optimizer, model, closure):
    """A line-search evaluation is not an accepted parameter update.

    If a strong-Wolfe trial crosses the hard global closure limit, restore the
    previously accepted parameters. The fit terminates, never resumes this
    partially mutated optimizer. The failed trial cannot become the endpoint.
    """
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        optimizer.step(closure)
    except GradientBudgetExhausted:
        model.load_state_dict(before)
        return 'gradient_budget_trial_rolled_back'
    changed = any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())
    return 'accepted_changed_state' if changed else 'numerical_no_change'
