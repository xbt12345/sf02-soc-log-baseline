"""Fixed legal pure-error CE contribution with complete-class denominators.

All original rows remain in forward/full-risk/quality reporting. Only the
declared learning target changes; no arbitrary error multiplier is used.
"""
import numpy as np
import torch
from v159_boundary_train_v4 import inputs

def error_risk(model,ctx,ids,counter=None,gradient_class=None,inputs_builder=inputs,batch_size=2048):
    ids=np.asarray(ids,np.int64);full=ctx['counts'];target=ctx['target_counts'];mass=ctx['mass']
    if full.shape!=target.shape or full.shape!=(22546,3) or np.any(target<0) or np.any(target>full) or target[:,0].any() or mass[1]<=0 or mass[2]<=0:raise ValueError('Full original masses and legal fixed target counts required')
    if gradient_class is not None:
        assert gradient_class in [1,2] and counter is not None
        counter.gradient_before(gradient_class,mass.tolist());model.zero_grad(set_to_none=True)
    q=np.full((22546,3),np.nan,np.float64);lp=np.full_like(q,np.nan)
    full_numerators=np.zeros(3);error_numerators=np.zeros(3);seen_full=np.zeros(3,np.int64);seen_error=np.zeros(3,np.int64)
    with torch.enable_grad() if gradient_class is not None else torch.no_grad():
        for start in range(0,len(ids),batch_size):
            ii=ids[start:start+batch_size];value,logq,_=model(*inputs_builder(ctx,'OOF',ii))
            c=torch.tensor(full[ii],dtype=torch.float64,device=logq.device);e=torch.tensor(target[ii],dtype=torch.float64,device=logq.device)
            if gradient_class is not None:
                loss=-(e[:,gradient_class]*logq[:,gradient_class]).sum()/float(mass[gradient_class])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pure-error target contribution')
                loss.backward()
            full_numerators+=(-(c*logq).sum(0)).detach().cpu().numpy();error_numerators+=(-(e*logq).sum(0)).detach().cpu().numpy()
            seen_full+=full[ii].sum(0);seen_error+=target[ii].sum(0);q[ii]=value.detach().cpu().numpy();lp[ii]=logq.detach().cpu().numpy()
    assert np.array_equal(seen_full,mass) and np.array_equal(seen_error,target.sum(0))
    risks=dict(fixed_pure_error_contribution=error_numerators[1:]/mass[1:],full_original_class_CE=full_numerators[1:]/mass[1:])
    assert all(np.isfinite(r).all() for r in risks.values())
    gradient=None
    if gradient_class is not None:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        gradient=np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in model.parameters()]);counter.gradient_after(gradient_class,mass.tolist())
    return risks,q,lp,gradient
