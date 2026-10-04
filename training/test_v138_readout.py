"""Risk-bearing invariants: original mass, gradients, and line-search budget."""
import unittest
import numpy as np
import torch
from v138_readout import Readout,full_gradient,probabilities,GradientBudgetExhausted,bounded_lbfgs_step
from v138_runtime import endpoint
from v138_feasibility import matrix,packed,logits


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13801);torch.set_num_threads(2)
        self.state={'head.weight':torch.randn(16,128,3)*.05,'head.bias':torch.randn(16,3)*.05,'facts_direct.weight':torch.randn(3,495)*.05}
        self.h=torch.randn(4,16,128,dtype=torch.float64);self.f=torch.zeros(4,495,dtype=torch.float64);self.f[:,7]=1

    def test_original_mass_matches_expanded_rows_and_gradient(self):
        a=Readout(self.state);b=Readout(self.state)
        c=torch.tensor([[0,72,6],[0,0,2],[0,3,0],[0,0,1]],dtype=torch.float64)
        loss,seen=full_gradient(a,self.h,self.f,c,np.arange(4),chunk=2)
        expanded=[(i,j) for i in range(4) for j in range(3) for _ in range(int(c[i,j]))]
        ii=torch.tensor([z[0] for z in expanded]);yy=torch.tensor([z[1] for z in expanded]);z=b(self.h[ii],self.f[ii]);risk=-torch.log_softmax(z,-1)[torch.arange(len(ii))[:,None],torch.arange(16)[None,:],yy[:,None]].mean();risk.backward()
        self.assertAlmostEqual(loss,float(risk),places=12);self.assertEqual(seen,[0.,75.,9.])
        for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-12,rtol=1e-12))

    def test_budget_failure_rolls_back_trial(self):
        m=Readout(self.state);before={k:v.clone() for k,v in m.state_dict().items()};o=torch.optim.LBFGS(m.parameters(),max_iter=1,line_search_fn='strong_wolfe')
        calls=0
        def closure():
            nonlocal calls
            if calls==1:raise GradientBudgetExhausted()
            calls+=1;o.zero_grad();l=sum(p.square().sum() for p in m.parameters());l.backward();return l
        self.assertEqual(bounded_lbfgs_step(o,m,closure),'gradient_budget_trial_rolled_back')
        self.assertEqual(calls,1)
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[k]))

    def test_numerical_noop_not_a_learning_state(self):
        m=Readout(self.state);o=torch.optim.LBFGS(m.parameters(),max_iter=1,line_search_fn='strong_wolfe')
        def closure():
            o.zero_grad();l=sum((p*0).sum() for p in m.parameters());l.backward();return l
        self.assertEqual(bounded_lbfgs_step(o,m,closure),'numerical_no_change')

    def test_early_adam_endpoint_rejected(self):
        r={'status':'fit_executed','arm':'H_A','candidate_selected_by_score':False,'full_gradient_evaluations':199,'accepted_updates':199,'termination':'fixed_update_budget'}
        with self.assertRaises(ValueError):endpoint(r)

    def test_lbfgs_trial_count_separate_from_update_count(self):
        r={'status':'fit_executed','arm':'H_L','candidate_selected_by_score':False,'full_gradient_evaluations':200,'accepted_updates':61,'termination':'gradient_budget_trial_rolled_back'}
        self.assertTrue(endpoint(r));r['full_gradient_evaluations']=201
        with self.assertRaises(ValueError):endpoint(r)

    def test_probe_constraints_equal_original_readout(self):
        m=Readout(self.state);theta=packed(m.state_dict());z=logits(theta,self.h.numpy(),self.f.numpy())
        self.assertTrue(np.allclose(z,m(self.h,self.f).detach().numpy(),atol=1e-12))
        rows=[(0,0,2),(1,7,0),(3,15,1)];y=np.array([1,2,1,2]);a=matrix(rows,self.h.numpy(),self.f.numpy(),y)
        want=np.array([z[i,k,j]-z[i,k,y[i]] for i,k,j in rows])
        self.assertTrue(np.allclose(a@theta,want,atol=1e-12))


if __name__=='__main__':unittest.main()
