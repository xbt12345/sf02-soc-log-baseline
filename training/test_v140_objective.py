"""Objective correctness, mass, and registered authority boundary."""
import copy
import unittest
import numpy as np
import torch
from v138_readout import Readout
from v140_objective import full_gradient,log_risk
from v140_runtime import PLAN,read,validate,adversaries,endpoint


class Tests(unittest.TestCase):
    def test_mixture_original_mass_and_gradient_equal_expanded_rows(self):
        torch.manual_seed(14001);torch.set_num_threads(2)
        s={'head.weight':torch.randn(16,128,3)*.05,'head.bias':torch.randn(16,3)*.05,'facts_direct.weight':torch.randn(3,495)*.05}
        a=Readout(s);b=Readout(s);h=torch.randn(3,16,128,dtype=torch.float64);f=torch.randn(3,495,dtype=torch.float64)
        c=torch.tensor([[0,72,6],[0,0,2],[0,3,0]],dtype=torch.float64)
        value,seen=full_gradient(a,h,f,c,np.arange(3),'E',chunk=2)
        ii=[];yy=[]
        for i in range(3):
            for y in range(3): ii.extend([i]*int(c[i,y]));yy.extend([y]*int(c[i,y]))
        z=b(h[ii],f[ii]);q=torch.softmax(z,-1).mean(1)
        loss=-q[torch.arange(len(ii)),yy].log().mean();loss.backward()
        self.assertAlmostEqual(value,float(loss.detach()),places=12);self.assertEqual(seen,[0.,75.,8.])
        for x,y in zip(a.parameters(),b.parameters()): self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-12,rtol=1e-12))

    def test_correct_ensemble_does_not_require_all_members_correct(self):
        q=torch.tensor([[[.01,.89,.10],[.01,.30,.69]]],dtype=torch.float64)
        z=q.log();self.assertEqual(int(q.mean(1).argmax(-1)),1)
        self.assertNotEqual(int(q[0,1].argmax()),1)
        self.assertTrue(torch.allclose(log_risk(z,'E'),q.mean(1).log()))
        self.assertTrue(bool((log_risk(z,'C')<=log_risk(z,'E')+1e-12).all()))

    def test_plan_and_mutations(self):
        p=read(PLAN);self.assertTrue(validate(p));self.assertEqual(len(adversaries(p)),12)
        q=copy.deepcopy(p);q['task_adoption_quality']['ASA_M_errors_max']=2006
        with self.assertRaises(ValueError):validate(q)
        q=copy.deepcopy(p);q['risk_actions'].pop(next(iter(q['risk_actions'])))
        with self.assertRaises(ValueError):validate(q)

    def test_incomplete_or_overbudget_endpoint_rejected(self):
        r={'status':'fit_executed','arm':'E','fold':0,'candidate_selected_by_score':False,'accepted_updates':90,'full_gradient_evaluations':200,'termination':'gradient_budget'}
        self.assertTrue(endpoint(r));r['full_gradient_evaluations']=199
        with self.assertRaises(ValueError):endpoint(r)
        r['full_gradient_evaluations']=201
        with self.assertRaises(ValueError):endpoint(r)


if __name__=='__main__':unittest.main()
