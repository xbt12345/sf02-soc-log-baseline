"""Load-bearing V131 algebra, objective and acceptance counterexamples. No fits."""
import unittest
from copy import deepcopy

import numpy as np
from scipy import sparse
import torch

from v131_model import Classifier,ActiveFirst,extend,objective
from v104_phase_b import SparseFirstBatchEnsemble,csr_tensor
from v131_common import require_checkpoint,learning_gates,ReviewError


class V131Tests(unittest.TestCase):
    def test_active_sparse_forward_and_gradient_equal_full_reference(self):
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for dev in devices:
            torch.manual_seed(10)
            old=SparseFirstBatchEnsemble(17,5,3).to(dev)
            new=ActiveFirst(17,5,3).to(dev);new.load_state_dict(old.state_dict())
            dense=np.zeros((6,17),np.float32)
            dense[:,[1,3,11]]=np.random.default_rng(40).normal(size=(6,3))
            x=csr_tensor(sparse.csr_matrix(dense),dev)
            a,b=old(x),new(x)
            self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-6))
            a.square().mean().backward();b.square().mean().backward()
            for p,q in zip(old.parameters(),new.parameters()):
                self.assertTrue(torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-6))
            self.assertTrue((new.weight.grad[:,0]==0).all())

    def test_expansion_preserves_function_without_dead_extra_weights(self):
        torch.manual_seed(4);small=Classifier(4,din=17,facts=3,members=3)
        wide=extend(small,Classifier(8,din=17,facts=3,members=3))
        dense=np.random.default_rng(6).normal(size=(7,17)).astype('f4')
        x=csr_tensor(sparse.csr_matrix(dense),'cpu');facts=torch.zeros(7,3)
        self.assertTrue(torch.allclose(small(x,facts),wide(x,facts),atol=1e-6,rtol=1e-6))
        self.assertGreater(float(wide.first.weight[4:].abs().sum()),0)
        self.assertTrue((wide.head.weight[:,4:]==0).all())
        self.assertTrue((wide.second.weight[:4,4:]==0).all())

    def test_fixed_denominators_preserve_objective_across_batches(self):
        z=torch.randn(5,3,3,dtype=torch.float64,requires_grad=True)
        c=torch.tensor([[0,72,6],[0,4,0],[0,0,7],[0,8,0],[0,0,3]],dtype=torch.float64)
        pure=torch.tensor([0,1,1,1,1],dtype=torch.float64)
        totals=(c*pure[:,None]).sum(0);n=float(c.sum())
        all_loss,*_=objective(z,c,pure,n,totals,1,True)
        split=[]
        for ids in [[0,1],[2,3,4]]:
            val,*_=objective(z[ids],c[ids],pure[ids],n,totals,2,True);split.append(val)
        self.assertAlmostEqual(float(all_loss.detach()),float(((split[0]+split[1])/2).detach()),12)
        # A mixed input receives ONLY the original relative 72:6 supervision.
        focused,*_=objective(z[:1],c[:1],pure[:1],n,totals,1,True)
        ordinary,*_=objective(z[:1],c[:1],pure[:1],n,totals,1,False)
        self.assertAlmostEqual(float(focused.detach()),.5*float(ordinary.detach()),12)
        blind_s=(6/2071)/(72/38886+6/2071)
        self.assertGreater(blind_s,.5)

    def test_registered_endpoint_and_kind_cannot_be_replaced(self):
        good={'status':'fit_executed','kind':'primary','endpoint':100,'optimizer_steps':3200,'expected_steps':3200}
        require_checkpoint(good,'primary')
        for field,value in [('endpoint',25),('optimizer_steps',3199),('kind','probe')]:
            bad=deepcopy(good);bad[field]=value
            with self.assertRaises(ReviewError):require_checkpoint(bad,'primary')

    def test_joint_classification_gate_accepts_positive_and_rejects_regression(self):
        good={'M_errors':0,'S_errors':32,'pure_S_errors':26,'hard_S_errors':26,
            'matched_M_errors':0,'old_correct_S_regressions':0,'hard_error_roots':29}
        self.assertTrue(all(learning_gates(good).values()))
        for key in ['M_errors','old_correct_S_regressions','matched_M_errors']:
            bad=dict(good);bad[key]=1
            self.assertFalse(all(learning_gates(bad).values()))
        bad=dict(good);bad['hard_error_roots']=30
        self.assertFalse(all(learning_gates(bad).values()))


if __name__=='__main__':unittest.main()
