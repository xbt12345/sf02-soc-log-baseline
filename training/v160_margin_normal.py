"""Full-parameter log-probability margin and exact function-input identity."""
import hashlib,json
import numpy as np
import torch

def input_identity(role,scope,ids,x,p,query,truth,rival):
    ids=np.asarray(ids,dtype=np.int64);p=np.asarray(p)
    if role not in [0,1,2] or scope not in ['OOF','deployment']:raise ValueError('Legal role and scope required')
    if ids.ndim!=1 or len(ids)==0 or not 0<=query<len(ids):raise ValueError('Matched complete chunk required')
    if truth not in [1,2] or rival not in [0,1,2] or truth==rival:raise ValueError('Legal distinct training truth/rival required')
    if x.shape!=(len(ids),66287) or p.shape!=(len(ids),16,3):raise ValueError('Complete current input and ordered probabilities required')
    if x.dtype!=np.float32 or p.dtype!=np.float64 or not x.has_canonical_format:raise ValueError('Preserve raw canonical CSR float32 and saved float64 opinions')
    if not np.isfinite(x.data).all() or not np.isfinite(p).all():raise ValueError('Nonfinite input')
    h=hashlib.sha256()
    h.update(json.dumps(dict(role=int(role),scope=scope,query=int(query),truth=int(truth),rival=int(rival),shape=list(x.shape)),sort_keys=True).encode())
    for arr in [ids,x.indptr,x.indices,x.data,p]:
        a=np.ascontiguousarray(arr);h.update(a.dtype.str.encode());h.update(np.asarray(a.shape,dtype='<i8').tobytes());h.update(a.tobytes())
    return h.hexdigest()

def measure(model,x,p,query,truth,rival):
    if truth not in [1,2] or rival not in [0,1,2] or truth==rival or not 0<=query<len(p):raise ValueError('Distinct legal training classes and query required')
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        q,lp,_=model(x,p);margin=lp[query,truth]-lp[query,rival]
        if not torch.isfinite(margin):raise FloatingPointError('Nonfinite margin')
        margin.backward()
    parameters=list(model.parameters())
    if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters):raise FloatingPointError('Incomplete finite margin normal')
    gradient=np.concatenate([v.grad.detach().cpu().numpy().ravel() for v in parameters])
    return dict(margin=float(margin.detach().cpu()),gradient=gradient,q=q.detach().cpu().numpy(),logq=lp.detach().cpu().numpy())
