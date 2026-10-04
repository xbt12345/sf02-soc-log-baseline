"""Load-bearing mathematical checks, not a classifier quality claim."""
import unittest
import numpy as np
from scipy import sparse
import torch
from v85_protection import Residual, csr_tensor
from v87_solver import MarginJacobian, project_step


class SolverChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8701)
        self.model = Residual(width=11,h1=7,h2=5)
        torch.nn.init.normal_(self.model.last.weight, std=.3)
        self.x = sparse.csr_matrix(np.random.default_rng(8701).normal(size=(4,11)).astype('float32'))
        self.true = np.array([0,1,2,1]); self.other = np.array([1,2,0,0])
        self.j = MarginJacobian(self.model,self.x,self.true,self.other,'cpu')
        self.direction = [torch.randn_like(p)*.02 for p in self.model.parameters()]

    def test_factorized_jacobian_matches_independent_autograd(self):
        z=self.model(csr_tensor(self.x,'cpu'))
        gradients=[]
        for i in range(4):
            g=torch.autograd.grad(z[i,self.true[i]]-z[i,self.other[i]],tuple(self.model.parameters()),retain_graph=True)
            gradients.append(torch.cat([a.flatten() for a in g]))
        dense=torch.stack(gradients).double().numpy()
        d=torch.cat([a.flatten() for a in self.direction]).double().numpy()
        np.testing.assert_allclose(self.j.gram(),dense@dense.T,rtol=2e-6,atol=2e-7)
        np.testing.assert_allclose(self.j.dot(self.direction),dense@d,rtol=2e-6,atol=2e-7)
        weights=np.array([.4,-.2,.8,.1])
        assembled=torch.cat([a.flatten() for a in self.j.transpose(weights)]).numpy()
        np.testing.assert_allclose(assembled,dense.T@weights,rtol=2e-5,atol=2e-7)

    def test_actual_parameter_step_satisfies_both_competitors(self):
        # Full six possible boundaries for two inputs, including both rivals.
        ix=np.repeat([0,1],2);true=np.repeat([0,1],2);other=np.array([1,2,0,2])
        j=MarginJacobian(self.model,self.x[ix],true,other,'cpu')
        bound=np.zeros(4)
        proposal=[-a for a in j.transpose(np.ones(4))]
        step,info=project_step(j,proposal,bound)
        self.assertIsNotNone(step,info);self.assertTrue(info['kkt_passed'])
        self.assertLessEqual(float(np.maximum(-j.dot(step),0).max()),2e-6)

    def test_feasible_direction_is_not_changed(self):
        bound=self.j.dot(self.direction)-.1
        step,info=project_step(self.j,self.direction,bound)
        self.assertIsNotNone(step,info)
        for actual,expected in zip(step,self.direction):
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)

    def test_contradictory_active_constraints_fail_closed(self):
        j=MarginJacobian(self.model,self.x[[0,0]],np.array([0,1]),np.array([1,0]),'cpu')
        step,info=project_step(j,self.direction,np.ones(2))
        self.assertIsNone(step);self.assertNotEqual(info['status_val'],1)


if __name__=='__main__':unittest.main()
