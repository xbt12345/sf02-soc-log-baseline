"""Load-bearing solver checks against independent small mathematical problems."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from v89_solver import Problem,self_check


class SolverTests(unittest.TestCase):
    def test_objective_gradient_and_infeasibility(self):
        result=self_check();self.assertLess(result['objective_gradient_error'],1e-7)

    def test_cut_qp_keeps_the_original_objective(self):
        n=501;v=np.c_[np.zeros(n),np.ones(n)];v[400,0]=1;v[500,0]=1
        z=np.zeros((n,3));z[:500,0]=.1;z[400,0]=1.2;z[500,1]=-1
        eps=np.ones(n)*.001;p=Problem(v,z,np.arange(500),eps,[500],[1],np.array([1.]),'test')
        p.initial(True);H=np.eye(p.n);g=np.array([3.,0.,-3.,0.,0.,0.]);center=np.zeros(p.n)
        with patch('v89_solver.ledger_start',return_value=0),patch('v89_solver.ledger_end'),patch('v89_solver.emit'):
            actual,record=p.qp(H,g,center,True,'test')
        self.assertIsNotNone(actual,record);self.assertGreaterEqual(len(record['trace']),2)
        D,b=p.matrix()
        independent=minimize(lambda u:(.5*u@u+g@u,u+g),np.zeros(p.n),jac=True,method='SLSQP',
            constraints=[LinearConstraint(D,b,np.inf)],options={'ftol':1e-12,'maxiter':1000})
        self.assertTrue(independent.success,independent.message)
        np.testing.assert_allclose(actual,independent.x,atol=1e-5,rtol=0)

    def test_unknown_resource_status_is_not_infeasibility(self):
        from types import SimpleNamespace
        p=Problem(np.ones((1,1)),np.array([[0.,1.,0.]]),[],np.array([0.]),[0],[2],np.array([1.]),'test')
        with patch('v89_solver.lp',return_value=SimpleNamespace(status=1,success=False,message='time limit')):
            solution,record=p.linear(False)
        self.assertIsNone(solution);self.assertEqual(record['status'],'unresolved_solver_limit')


if __name__=='__main__':unittest.main()
