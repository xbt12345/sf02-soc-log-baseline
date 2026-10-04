"""Synthetic gradient and endpoint contract tests, no official data/model call."""
import unittest
import numpy as np
import torch
from v158_fusion_prototype_v2 import SharedProbabilityScorer
from v158_fusion_runtime import endpoint

class ExecutionTests(unittest.TestCase):
    def test_mean_probability_CE_autograd_finite_difference(self):
        torch.manual_seed(15801);m=SharedProbabilityScorer(2,16)
        with torch.no_grad():m.output.weight.copy_(torch.randn_like(m.output.weight)*.08)
        probabilities=torch.tensor([[[.1,.7,.2],[.15,.25,.6]],[[.1,.5,.4],[.3,.2,.5]]],dtype=torch.float64)
        condition=torch.tensor([[.2,-.1],[.4,.5]],dtype=torch.float64);prior=torch.tensor([.8,.2],dtype=torch.float64)
        mass=torch.tensor([[0.,4.,2.],[0.,1.,3.]],dtype=torch.float64)
        def loss():
            q,_=m(probabilities,condition,prior);return -(mass*q.log()).sum()/mass.sum()
        loss().backward();analytic=float(m.hidden.weight.grad[0,0]);old=float(m.hidden.weight[0,0]);delta=1e-6
        with torch.no_grad():
            m.hidden.weight[0,0]=old+delta;plus=float(loss());m.hidden.weight[0,0]=old-delta;minus=float(loss());m.hidden.weight[0,0]=old
        self.assertAlmostEqual(analytic,(plus-minus)/(2*delta),places=8)

    def test_zero_init_hidden_zero_and_output_nonzero_gradient(self):
        torch.manual_seed(15801);m=SharedProbabilityScorer(0,16)
        p=torch.tensor([[[.1,.8,.1],[.1,.2,.7]]],dtype=torch.float64);prior=torch.tensor([.8,.2],dtype=torch.float64)
        q,_=m(p,prior=prior);(-q[0,2].log()).backward()
        self.assertEqual(float(m.hidden.weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.output.weight.grad.norm()),0.)
        np.testing.assert_allclose(q.detach().numpy(),[[.1,.68,.22]],atol=1e-15,rtol=0)

    def test_registered_valid_and_invalid_resource_endpoints(self):
        good=dict(status='fusion_fit_executed',fold=0,arm='B',selected_by_score=False,accepted_updates=3,full_gradients=4,proposal_evaluations=8,termination='no_feasible_step')
        self.assertTrue(endpoint(good))
        for key,value in [('selected_by_score',True),('full_gradients',201),('proposal_evaluations',601),('accepted_updates',5),('termination','best_outer_checkpoint')]:
            with self.subTest(field=key),self.assertRaises(ValueError):endpoint(dict(good,**{key:value}))

if __name__=='__main__':unittest.main()
