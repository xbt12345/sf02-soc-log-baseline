"""Prospective nested-source and full-pipeline cost contract, no models."""
import hashlib,math
import numpy as np

STAGES=('V135_R_decay_full_network','V138_H_L_readout','V140_C_readout','V142_S2_second','V146_A_second')

def inner_fold(outer,root):
    # Preserve V128 existing label-independent split; never reroll for support.
    key=f'V128|split=12801|outer={outer}|root={root}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%3

def counts(frame,n=22546):
    out=np.bincount(frame.local.to_numpy(np.int64)*3+frame.truth.to_numpy(np.int64),minlength=n*3).reshape(n,3)
    assert int(out.sum())==len(frame) and not out[:,0].any()
    return out

def role(frame,outer,excluded):
    legal=frame[frame.fold.ne(outer)].copy()
    legal['inner_fold']=legal.root.map(lambda r:inner_fold(outer,r))
    fit=legal[legal.inner_fold.ne(excluded)].copy();query=legal[legal.inner_fold.eq(excluded)].copy()
    held=frame[frame.fold.eq(outer)]
    assert not set(fit.root)&set(query.root) and not set(legal.root)&set(held.root)
    assert legal.groupby('local').inner_fold.nunique().max()==1
    assert fit.groupby('root').inner_fold.nunique().max()==1
    assert all(fit.truth.eq(cl).any() and query.truth.eq(cl).any() for cl in [1,2])
    return fit,query,held

def validate_stage_roots(fit_roots,query_roots,outer_roots,stages):
    assert set(stages)==set(STAGES)
    for name in STAGES:
        roots=set(stages[name])
        if roots!=set(fit_roots) or roots&set(query_roots) or roots&set(outer_roots):
            raise ValueError('Supervised source exposure violates full-pipeline OOF: '+name)
    return True

def pipeline_cost(used_locals):
    batches=math.ceil(used_locals/256)
    return dict(pipeline_fits=5,full_network_epochs=100,full_network_batches=batches,
        full_network_gradient_updates=100*batches,readout_fits=2,second_layer_fits=2,
        dense_stage_full_gradient_cap=800,dense_stage_accepted_update_cap=800,
        final_second_stage_proposal_cap=600,
        note='Dense stages use actual global budgeted gradients; closure/proposal/cache/evaluation forwards need separate prospective implementation counts before activation.')
