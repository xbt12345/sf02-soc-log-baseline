"""Full original member CE, first-order SAM and frozen-adversary proposal risk."""
import numpy as np
import torch
from torch.func import functional_call

CHUNK=2048

def full_gradient_at(model,hidden,facts,counts,ids,shift=None,pure=None):
    params=dict(model.named_parameters())
    if shift is not None:params={k:(v.detach()+shift[k]).requires_grad_(True) for k,v in params.items()}
    total=counts[ids].sum();seen=torch.zeros_like(counts.sum(0));grad={k:torch.zeros_like(v) for k,v in params.items()};risk=0.
    class_cost=torch.zeros_like(seen);pure_cost=mixed_cost=0.
    for start in range(0,len(ids),CHUNK):
        take=ids[start:start+CHUNK];mass=counts[take]
        logits=functional_call(model,params,(hidden[take],facts[take]))
        contributions=-(torch.log_softmax(logits,-1).mean(1)*mass)
        loss=contributions.sum()/total
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite full member CE')
        computed=torch.autograd.grad(loss,tuple(params.values()))
        for (k,_),v in zip(params.items(),computed):grad[k].add_(v.detach())
        seen+=mass.sum(0);risk+=float(loss.detach())
        cost=contributions.detach();class_cost+=cost.sum(0)
        if pure is not None:
            p=torch.as_tensor(pure[take],device=cost.device,dtype=cost.dtype)
            pure_cost+=float((cost.sum(1)*p).sum());mixed_cost+=float((cost.sum(1)*(1-p)).sum())
    if not torch.equal(seen,counts[ids].sum(0)) or any(not bool(torch.isfinite(v).all()) for v in grad.values()):raise ValueError('Full gradient exposure/finite failure')
    return dict(member_CE=risk,original_class_mass_seen=seen.cpu().tolist(),gradient_norm=float(flat(grad).norm()),gradient=grad,
        class_member_CE=[None if n==0 else float(v/n) for v,n in zip(class_cost,seen)],
        pure_member_CE_contribution=None if pure is None else pure_cost/float(total),
        mixed_member_CE_contribution=None if pure is None else mixed_cost/float(total))

def flat(values):return torch.cat([v.reshape(-1) for v in values.values()])

def fixed_adversary(gradient,radius):
    norm=float(flat(gradient).norm())
    return {k:torch.zeros_like(v) if norm==0 else v*(radius/norm) for k,v in gradient.items()}

def objective_gradient(model,h,ff,counts,ids,arm,radius,pure=None):
    base=full_gradient_at(model,h,ff,counts,ids,pure=pure)
    shift=fixed_adversary(base['gradient'],radius) if arm=='B' else {k:torch.zeros_like(v) for k,v in base['gradient'].items()}
    outer=full_gradient_at(model,h,ff,counts,ids,shift,pure) if arm=='B' else base
    return dict(base=base,proxy=outer,shift=shift,full_gradients=2 if arm=='B' else 1,
        direction_is_first_order_ignores_shift_derivative=arm=='B')

@torch.no_grad()
def values_at(model,h,ff,counts,ids,pure,shift=None,need_probabilities=True):
    params=dict(model.named_parameters())
    if shift is not None:params={k:v+shift[k] for k,v in params.items()}
    mass_sum=counts[ids].sum();class_sum=counts[ids].sum(0);byclass=torch.zeros_like(class_sum)
    pure_total=mixed_total=0.;q=np.full((len(h),3),np.nan,dtype=np.float64)
    for start in range(0,len(ids),CHUNK):
        take=ids[start:start+CHUNK];mass=counts[take]
        logits=functional_call(model,params,(h[take],ff[take]));contribution=-(torch.log_softmax(logits,-1).mean(1)*mass)
        byclass+=contribution.sum(0);cost=contribution.sum(1)
        p=torch.as_tensor(pure[take],device=cost.device,dtype=cost.dtype)
        pure_total+=float((cost*p).sum());mixed_total+=float((cost*(1-p)).sum())
        if need_probabilities:q[take]=torch.softmax(logits,-1).mean(1).cpu().numpy()
    risk=float(byclass.sum()/mass_sum)
    if not np.isfinite(risk):raise FloatingPointError('Nonfinite proposal risk')
    info=dict(member_CE=risk,class_member_CE=[None if n==0 else float(v/n) for v,n in zip(byclass,class_sum)],
        pure_member_CE_contribution=pure_total/float(mass_sum),mixed_member_CE_contribution=mixed_total/float(mass_sum),
        original_class_mass_seen=class_sum.cpu().tolist())
    return info,q

def acceptance(base_proxy,trial_proxy,step,norm,guard,armijo):
    return bool(guard and trial_proxy<base_proxy and trial_proxy<=base_proxy-armijo*step*norm)
