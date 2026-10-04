"""Zero replay, frozen head, second-layer original mass gradient and review."""
import unittest
import numpy as np
import torch
from v141_second_model import SecondRepresentation
from v138_readout import Readout,full_gradient
from v142_runtime import PLAN,read,validate,adversaries,endpoint


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14201);torch.set_num_threads(2)
        self.s={'second.weight':torch.randn(128,128)*.05,'second.r':torch.randn(16,128),'second.s':torch.ones(16,128),'second.bias':torch.randn(16,128)*.05}
        self.head={'weight':torch.randn(16,128,3,dtype=torch.float64)*.05,'bias':torch.randn(16,3,dtype=torch.float64)*.05,'facts':torch.randn(3,495,dtype=torch.float64)*.05}
        self.h=torch.randn(3,16,256,dtype=torch.float64);self.f=torch.randn(3,495,dtype=torch.float64)

    def test_exact_zero_step_and_only_existing_second_parameters(self):
        m=SecondRepresentation(self.s,self.head)
        s={'head.weight':self.head['weight'],'head.bias':self.head['bias'],'facts_direct.weight':self.head['facts']}
        r=Readout(s)
        self.assertTrue(torch.equal(m(self.h,self.f),r(self.h[:,:,128:],self.f)))
        self.assertEqual(sum(p.numel() for p in m.parameters()),22528)
        self.assertEqual(set(dict(m.named_parameters())),{'weight','r','s','bias'})

    def test_full_mass_gradient_matches_original_expanded_rows(self):
        a=SecondRepresentation(self.s,self.head);b=SecondRepresentation(self.s,self.head)
        c=torch.tensor([[0,72,6],[0,0,2],[0,3,0]],dtype=torch.float64)
        loss,seen=full_gradient(a,self.h,self.f,c,np.arange(3),chunk=2)
        ii=[];yy=[]
        for i in range(3):
            for y in range(3):ii.extend([i]*int(c[i,y]));yy.extend([y]*int(c[i,y]))
        lp=torch.log_softmax(b(self.h[ii],self.f[ii]),-1)
        v=-lp[torch.arange(len(ii))[:,None],torch.arange(16)[None,:],torch.tensor(yy)[:,None]].mean();v.backward()
        self.assertAlmostEqual(loss,float(v.detach()),places=12);self.assertEqual(seen,[0.,75.,8.])
        for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-11,rtol=1e-11))
        self.assertTrue(all(buf.grad is None for _,buf in a.named_buffers()))

    def test_design_counterexamples_and_fixed_endpoint(self):
        p=read(PLAN);self.assertTrue(validate(p));self.assertEqual(len(adversaries(p)),13)
        r={'status':'fit_executed','arm':'S2','fold':0,'candidate_selected_by_score':False,'accepted_updates':80,'full_gradient_evaluations':200,'termination':'gradient_budget'}
        self.assertTrue(endpoint(r));r['full_gradient_evaluations']=199
        with self.assertRaises(ValueError):endpoint(r)


if __name__=='__main__':unittest.main()
