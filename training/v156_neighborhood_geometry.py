"""Conditional root-excluded geometry; never vote, route or synthesize labels."""
import json
import numpy as np
from v148_observed_relation import FIELDS,observed

TOLERANCE=1e-12

def field_state(f,k):
    if observed(f,k):return 'observed'
    if k not in f:return 'absent'
    protocol=f.get('transport_protocol')
    if (k.startswith('icmp_') and protocol in ['tcp','udp']) or ('port' in k and protocol=='icmp'):
        return 'not_applicable'
    return 'unobserved:'+json.dumps(f[k],sort_keys=True,separators=(',',':'))

def condition(f):
    return json.dumps([f.get('transport_protocol','<absent>')]+[field_state(f,k) for k in FIELDS],separators=(',',':'))

def normalize_dense(a):
    a=np.asarray(a,dtype=np.float64).reshape(len(a),-1)
    n=np.linalg.norm(a,axis=1)
    return a/np.where(n==0,1,n)[:,None],n==0

def nearest(similarity,candidates,remaining_mass,root_sets,query_roots):
    """Exact best class support after removing own root, ties keep all mass."""
    sim=np.asarray(similarity,dtype=np.float64)
    assert sim.shape==remaining_mass.shape and sim.shape[0]==len(query_roots)
    assert np.isfinite(sim).all() and remaining_mass.min(initial=0)>=0
    assert np.all(sim>=-1-1e-9) and np.all(sim<=1+1e-9)
    sim=np.clip(sim,-1,1);available=remaining_mass>0
    masked=np.where(available,sim,-np.inf);any_support=available.any(1)
    maximum=masked.max(1) if len(candidates) else np.full(len(query_roots),-np.inf)
    ties=available&(np.abs(sim-maximum[:,None])<=TOLERANCE)
    selected=np.argmax(ties,axis=1) if len(candidates) else np.zeros(len(query_roots),dtype=int)
    local=np.where(any_support,np.asarray(candidates)[selected],-1) if len(candidates) else np.full(len(query_roots),-1)
    roots=[]
    for row,own in zip(ties,query_roots):
        all_roots=set()
        for k in np.flatnonzero(row):all_roots.update(root_sets[int(candidates[k])])
        all_roots.discard(int(own));roots.append(len(all_roots))
    return dict(cosine=np.where(any_support,maximum,np.nan),local=local,
        tied_locals=ties.sum(1),tied_original_rows=(ties*remaining_mass).sum(1),tied_roots=np.asarray(roots),
        available_original_rows=remaining_mass.sum(1),available_locals=available.sum(1))
