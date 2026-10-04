"""Meaningful first-order/frozen-adversary numerical checks, no project fit."""
import unittest
import numpy as np
import torch
from torch import nn
from v155_sam_objective import full_gradient_at,fixed_adversary,objective_gradient,values_at,acceptance

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor([[.3,-.4,.1],[-.2,.7,.4]],dtype=torch.float64))
    def forward(self,h,ff):return (h@self.weight)[:,None,:].expand(-1,2,-1)

class MethodChecks(unittest.TestCase):
    def test_full_original_mass_and_first_order_sam(self):
        m=Tiny();h=torch.tensor([[1.,2.],[-1.,.5],[.2,.3]],dtype=torch.float64);ff=h
        c=torch.tensor([[0.,3.,1.],[0.,2.,0.],[0.,0.,5.]],dtype=torch.float64);ids=np.arange(3)
        before=m.weight.detach().clone();r=objective_gradient(m,h,ff,c,ids,'B',.01)
        self.assertEqual(r['full_gradients'],2);self.assertEqual(r['base']['original_class_mass_seen'],[0.,5.,6.])
        self.assertAlmostEqual(float(r['shift']['weight'].norm()),.01,places=14)
        eps=r['shift']['weight'];w=(before+eps).requires_grad_(True)
        l=-(torch.log_softmax(h@w,-1)*c).sum()/c.sum();expected=torch.autograd.grad(l,w)[0]
        self.assertTrue(torch.equal(expected,r['proxy']['gradient']['weight']))
        self.assertTrue(torch.equal(before,m.weight));self.assertIsNone(m.weight.grad)
        # Finite differences of fixed-epsilon proxy agree, not a claim about
        # differentiating the parameter-dependent epsilon construction.
        index=(0,1);eta=1e-6;loss=[]
        with torch.no_grad():
            for sign in [-1,1]:
                m.weight.copy_(before);m.weight[index]+=sign*eta
                value,_=values_at(m,h,ff,c,ids,np.ones(3),r['shift'],False);loss.append(value['member_CE'])
            m.weight.copy_(before)
        self.assertAlmostEqual((loss[1]-loss[0])/(2*eta),float(expected[index]),places=8)

    def test_proxy_guard_and_zero_adversary(self):
        self.assertFalse(acceptance(.5,.4,.01,1.,False,1e-4))
        self.assertFalse(acceptance(.5,.5001,.01,1.,True,1e-4))
        self.assertTrue(acceptance(.5,.49,.01,1.,True,1e-4))
        eps=fixed_adversary({'x':torch.zeros(2,dtype=torch.float64)},.1)
        self.assertTrue(torch.equal(eps['x'],torch.zeros(2,dtype=torch.float64)))

    def test_first_order_is_not_moving_epsilon_total_derivative(self):
        m=Tiny();h=torch.tensor([[1.,2.],[-1.,.5],[.2,.3]],dtype=torch.float64)
        c=torch.tensor([[0.,3.,1.],[0.,2.,0.],[0.,0.,5.]],dtype=torch.float64)
        base=-(torch.log_softmax(h@m.weight,-1)*c).sum()/c.sum()
        g=torch.autograd.grad(base,m.weight,create_graph=True)[0];eps=.1*g/g.norm()
        moving=-(torch.log_softmax(h@(m.weight+eps),-1)*c).sum()/c.sum()
        exact=torch.autograd.grad(moving,m.weight)[0]
        first=full_gradient_at(m,h,h,c,np.arange(3),{'weight':eps.detach()})['gradient']['weight']
        self.assertGreater(float((exact-first).abs().max()),1e-8)

if __name__=='__main__':unittest.main()
