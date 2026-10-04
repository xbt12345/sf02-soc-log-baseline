"""Original-frequency member versus actual probability-mixture NLL."""
import math
import torch


def log_risk(logits, arm):
    lp = torch.log_softmax(logits, -1)
    if arm == 'C':
        return lp.mean(1)
    if arm == 'E':
        return torch.logsumexp(lp, 1) - math.log(logits.shape[1])
    raise ValueError('Unregistered objective')


def full_gradient(model, hidden, facts, counts, ids, arm, chunk=2048):
    model.zero_grad(set_to_none=True)
    total = counts[ids].sum()
    value = 0.
    seen = torch.zeros(3, device=counts.device, dtype=torch.float64)
    for start in range(0, len(ids), chunk):
        take = ids[start:start+chunk]
        mass = counts[take]
        loss = -(log_risk(model(hidden[take], facts[take]), arm)*mass).sum()/total
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite original-row objective')
        loss.backward()
        value += float(loss.detach())
        seen += mass.sum(0)
    if not torch.equal(seen, counts[ids].sum(0)):
        raise ValueError('Original class mass not preserved')
    if any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite gradient')
    return value, seen.cpu().tolist()


@torch.no_grad()
def losses(model, hidden, facts, counts, ids):
    total = counts[ids].sum()
    result = {'C': 0., 'E': 0.}
    for start in range(0, len(ids), 2048):
        take = ids[start:start+2048]
        z = model(hidden[take], facts[take])
        for arm in result:
            result[arm] -= float((log_risk(z, arm)*counts[take]).sum()/total)
    return {'original_member_CE': result['C'], 'original_ensemble_CE': result['E']}
