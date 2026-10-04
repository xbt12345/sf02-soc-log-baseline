"""Exact small active-set projection onto M, S and objective descent halfspaces."""
import itertools
import numpy as np
import torch
from v138_readout import full_gradient

DESCENT_SLACK=0.001
FIXED_DIAGNOSTIC_STEP=0.01


def class_gradients(model,h,facts,counts,ids):
    parameters=list(model.parameters());risks=[];gradients=[];seen=[]
    for c in [1,2]:
        mass=counts.clone();mass[:,3-c]=0
        value,observed=full_gradient(model,h,facts,mass,ids)
        risks.append(value);gradients.append(tuple(v.grad.detach().clone() for v in parameters));seen.append(observed)
    total=float(counts.sum());weights=[float(counts[:,c].sum())/total for c in [1,2]]
    original=tuple(weights[0]*a+weights[1]*b for a,b in zip(*gradients))
    return risks,gradients,original,seen


def project_vector(vector,constraints,slack=DESCENT_SLACK):
    """Nearest vector subject to normalized constraint dot products >= slack."""
    norm=float(torch.linalg.vector_norm(vector))
    if norm<=1e-15:raise ValueError('Zero objective direction')
    unit=vector/norm
    vectors=torch.stack([g/torch.linalg.vector_norm(g) for g in constraints]+[unit])
    gram=(vectors@vectors.T).detach().cpu().numpy();dots=(vectors@unit).detach().cpu().numpy()
    bound=np.full(len(vectors),slack);candidates=[]
    # Strict convexity makes a feasible KKT active set the unique Euclidean projection.
    for size in range(len(vectors)+1):
        for active in itertools.combinations(range(len(vectors)),size):
            multipliers=np.zeros(len(vectors))
            if active:
                ii=np.array(active);matrix=gram[np.ix_(ii,ii)];rhs=bound[ii]-dots[ii]
                values=np.linalg.lstsq(matrix,rhs,rcond=1e-12)[0]
                if np.max(np.abs(matrix@values-rhs))>1e-9 or (values<-1e-9).any():continue
                multipliers[ii]=values
            result_dots=dots+gram@multipliers
            if (result_dots<bound-1e-9).any():continue
            delta=torch.as_tensor(multipliers,device=vector.device,dtype=vector.dtype)@vectors
            candidates.append((float(delta.square().sum()),unit+delta,active,result_dots))
    if not candidates:raise ValueError('No simultaneous strict descent direction; no silent relaxation')
    _,chosen,active,dots=min(candidates,key=lambda a:a[0]);length=float(chosen.norm());chosen=chosen/length
    actual=(vectors@chosen).detach().cpu().numpy()
    if (actual<=0).any() or not torch.isfinite(chosen).all():raise ValueError('Projected direction is not finite strict class/objective descent')
    return chosen,dict(active_constraints=list(active),projection_distance_L2=float((chosen-unit).norm()),
        class_M_cosine=float(actual[0]),class_S_cosine=float(actual[1]),objective_cosine=float(actual[2]),
        strict_slack_before_unit_normalization=slack,pre_normalization_norm=length)


def flatten(values):return torch.cat([v.reshape(-1) for v in values])


def unflatten(vector,parameters):
    values=[];start=0
    for p in parameters:
        values.append(vector[start:start+p.numel()].reshape(p.shape));start+=p.numel()
    if start!=len(vector):raise ValueError('Parameter direction dimension mismatch')
    return tuple(values)
