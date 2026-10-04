"""Explicit CPU transport; unchanged complete original objectives and counts."""
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from v161_fixed_pure_error_risk import error_risk as original_error_risk

_BLAS_LIMIT = None


def configure():
    global _BLAS_LIMIT
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    _BLAS_LIMIT = threadpool_limits(limits=24, user_api='blas')


def inputs(ctx, scope, ids):
    x = ctx['x'][ids]
    xx = torch.sparse_csr_tensor(
        torch.tensor(x.indptr, dtype=torch.int64, device='cpu'),
        torch.tensor(x.indices, dtype=torch.int64, device='cpu'),
        torch.tensor(x.data, dtype=torch.float64, device='cpu'),
        size=x.shape, device='cpu')
    p = torch.tensor(np.array(ctx[scope][ids]), dtype=torch.float64, device='cpu')
    return xx, p


def probabilities(model, ctx, scope, ids, return_log=False):
    # Same allocation, chunk size, finite check and return as V159 original.
    q = np.full((22546, 3), np.nan, np.float64)
    lp = np.full_like(q, np.nan)
    with torch.no_grad():
        for start in range(0, len(ids), 2048):
            ii = ids[start:start+2048]
            value, logq, _ = model(*inputs(ctx, scope, ii))
            q[ii] = value.cpu().numpy()
            lp[ii] = logq.cpu().numpy()
    assert np.isfinite(q[ids]).all()
    return (q, lp) if return_log else q


def error_risk(model, ctx, ids, counter=None, gradient_class=None):
    return original_error_risk(
        model, ctx, ids, counter, gradient_class,
        inputs_builder=inputs, batch_size=2048)
