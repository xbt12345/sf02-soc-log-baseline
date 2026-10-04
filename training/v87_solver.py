"""Factorized per-example margin Jacobian and actual-step dual QP projection.

No sample-by-17-million-parameter Jacobian is materialized. The factorization
is specific to the frozen 256/64 GELU residual architecture and is independently
checked against autograd before registration.
"""
import numpy as np
import osqp
from scipy import sparse
import torch
from v85_protection import csr_tensor


def gelu_derivative(x):
    a = (2 / np.pi) ** .5
    t = torch.tanh(a * (x + .044715 * x.pow(3)))
    return .5 * (1 + t) + .5 * x * (1 - t.square()) * a * (1 + .134145 * x.square())


class MarginJacobian:
    def __init__(self, model, x, true, other, device='cuda'):
        self.x = x.tocsr()
        self.xt = csr_tensor(self.x, device)
        with torch.no_grad():
            pre1 = torch.sparse.mm(self.xt, model.first.weight.T)
            self.h1 = torch.nn.functional.gelu(pre1, approximate='tanh')
            pre2 = model.second(self.h1)
            self.h2 = torch.nn.functional.gelu(pre2, approximate='tanh')
            self.d3 = torch.nn.functional.one_hot(torch.as_tensor(true, device=device), 3).float()
            self.d3 -= torch.nn.functional.one_hot(torch.as_tensor(other, device=device), 3).float()
            self.d2 = (self.d3 @ model.last.weight) * gelu_derivative(pre2)
            self.d1 = (self.d2 @ model.second.weight) * gelu_derivative(pre1)

    def gram(self):
        # Float64 accumulation prevents a nearly rank-one head from making a
        # rounded float32 Gram matrix spuriously indefinite.
        def product(a):
            return (a.double() @ a.double().T).cpu().numpy()
        xx = (self.x.astype(np.float64) @ self.x.astype(np.float64).T).toarray()
        return (xx * product(self.d1) + product(self.h1) * product(self.d2)
                + product(self.d2) + product(self.h2) * product(self.d3) + product(self.d3))

    def dot(self, direction):
        first, second, second_bias, last, last_bias = direction
        with torch.no_grad():
            a = torch.sparse.mm(self.xt, first.T)
            b = self.h1 @ second.T + second_bias
            c = self.h2 @ last.T + last_bias
            return ((a.double() * self.d1.double()).sum(1)
                    + (b.double() * self.d2.double()).sum(1)
                    + (c.double() * self.d3.double()).sum(1)).cpu().numpy()

    def transpose(self, weights):
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.d1.device)[:, None]
        with torch.no_grad():
            return [torch.sparse.mm(self.xt.transpose(0, 1), w * self.d1).T,
                    (w * self.d2).T @ self.h1, (w * self.d2).sum(0),
                    (w * self.d3).T @ self.h2, (w * self.d3).sum(0)]


def project_step(jacobian, direction, bound, tolerance=1e-6, max_iter=20000):
    """min ||d-d0||²/2 subject to Gd>=bound, solved in the small dual.

    A solver return code alone does not establish feasibility. Recompute KKT
    statistics with the represented, float32 parameter step and then perform
    full nonlinear protection in the caller.
    """
    gram = jacobian.gram()
    norm = np.sqrt(np.diag(gram))
    if not np.all(np.isfinite(norm)) or np.any(norm < 1e-12):
        raise FloatingPointError('Degenerate/nonfinite active margin Jacobian')
    kernel = gram / norm[:, None] / norm[None, :]
    kernel = (kernel + kernel.T) / 2
    q = (jacobian.dot(direction) - np.asarray(bound)) / norm
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(kernel), q=q, A=sparse.eye(len(q), format='csc'),
                 l=np.zeros(len(q)), u=np.full(len(q), np.inf),
                 eps_abs=tolerance / 10, eps_rel=tolerance / 10,
                 max_iter=max_iter, polishing=True, verbose=False)
    result = solver.solve(raise_error=False)
    record = {'status':result.info.status, 'status_val':int(result.info.status_val),
              'iterations':int(result.info.iter), 'primal_residual':float(result.info.prim_res),
              'dual_residual':float(result.info.dual_res), 'active_constraints':len(q)}
    if result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
        return None, record
    multipliers = np.maximum(result.x, 0)
    correction = jacobian.transpose(multipliers / norm)
    projected = [a + b for a,b in zip(direction, correction)]
    slack = (jacobian.dot(projected) - bound) / norm
    dual_gradient = kernel @ multipliers + q
    record.update(
        actual_scaled_primal_violation=float(np.maximum(-slack, 0).max()),
        actual_complementarity=float(np.abs(multipliers * slack).max()),
        dual_stationarity=float(np.maximum(-dual_gradient, 0).max()),
        negative_multiplier=float(np.maximum(-multipliers, 0).max()),
        active_multipliers=multipliers.tolist(),
    )
    accepted = (record['actual_scaled_primal_violation'] <= tolerance
                and record['dual_stationarity'] <= tolerance
                and record['actual_complementarity'] <= tolerance * max(1., float(multipliers.max())))
    record['kkt_passed'] = bool(accepted)
    return (projected if accepted else None), record
