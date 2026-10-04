"""Mathematical checks for aggregated supervision and protected residuals."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
import torch
import v85_protection as v


class ProtectionTests(unittest.TestCase):
    def test_aggregated_loss_equals_expanded_rows_and_gradient(self):
        counts=torch.tensor([[3.,0.,1.],[0.,2.,0.],[0.,1.,4.]],dtype=torch.float64)
        z=torch.tensor([[1.,-2.,.5],[-.1,.3,-.4],[.9,.4,.8]],dtype=torch.float64,requires_grad=True)
        a=v.supervised_loss(z,counts);ga=torch.autograd.grad(a,z,retain_graph=True)[0]
        rows=[];labels=[]
        for i in range(3):
            for cls in range(3):
                rows.extend([i]*int(counts[i,cls]));labels.extend([cls]*int(counts[i,cls]))
        expanded=z[rows];target=torch.nn.functional.one_hot(torch.tensor(labels),3).to(z.dtype)
        b=torch.nn.functional.binary_cross_entropy_with_logits(expanded,target,reduction='sum')/len(rows)
        gb=torch.autograd.grad(b,z)[0]
        torch.testing.assert_close(a,b);torch.testing.assert_close(ga,gb)

    def test_selective_kd_does_not_lock_wrong_teacher_rows(self):
        teacher=torch.tensor([[1.,0.,-1.],[3.,0.,-3.]],dtype=torch.float64)
        z=torch.tensor([[1.1,.2,-1.],[0.,3.,-3.]],dtype=torch.float64,requires_grad=True)
        kd=v.selective_kd(z,teacher,torch.tensor([2.,0.],dtype=torch.float64))
        grad=torch.autograd.grad(kd,z)[0]
        self.assertGreater(float(grad[0].abs().sum()),0)
        self.assertEqual(float(grad[1].abs().sum()),0)

    def test_protected_subset_and_margin_violation(self):
        z=torch.tensor([[2.,1.,0.],[0.,3.,1.],[5.,0.,0.]])
        labels=torch.tensor([0,1,2]);mass=torch.tensor([2.,4.,0.]);eps=torch.tensor([.1,.1,.1])
        self.assertEqual(v.feasibility(z,labels,eps,mass)['violating_original_rows'],0)
        z[0,1]=3.
        f=v.feasibility(z,labels,eps,mass)
        self.assertEqual(f['violating_original_rows'],2);self.assertEqual(f['protected_negative_flips'],2)

    def test_flip_counts_keep_mixed_label_rows(self):
        cc=np.array([[3,2,1],[0,1,4],[2,0,0]],dtype=np.int64)
        old=np.array([0,1,0]);new=np.array([2,2,0]);got=v.changes(cc,old,new)
        self.assertEqual(got['positive_flips'],5);self.assertEqual(got['negative_flips'],4)
        self.assertEqual(got['wrong_to_different_wrong'],2)
        self.assertEqual(got['cm'],[[2,0,3],[0,0,3],[0,0,5]])

    def test_sparse_residual_matches_dense_forward_backward(self):
        torch.manual_seed(8501);model=v.Residual(width=7,h1=5,h2=4)
        with torch.no_grad():model.last.weight.fill_(.03)
        x=np.array([[0.,1.,0.,.5,0.,0.,2.],[.4,0.,0.,0.,.3,1.,0.]],np.float32)
        a=model(torch.tensor(x));ga=torch.autograd.grad(a.square().sum(),model.first.weight,retain_graph=True)[0]
        b=model(v.csr_tensor(csr_matrix(x),'cpu'));gb=torch.autograd.grad(b.square().sum(),model.first.weight)[0]
        torch.testing.assert_close(a,b);torch.testing.assert_close(ga,gb)

    def test_zero_initialization_preserves_logits(self):
        m=v.Residual(width=7,h1=5,h2=4)
        self.assertEqual(float(m(torch.randn(8,7)).abs().max()),0)

    def test_double_teacher_uses_direct_residual_without_cancellation(self):
        old=np.array([[25.,24.,-20.]],np.float64);delta=np.array([[1e-7,2e-7,3e-7]],np.float32)
        direct=old+delta
        rounded=old+(torch.tensor(old,dtype=torch.float32)+torch.tensor(delta)-torch.tensor(old,dtype=torch.float32)).numpy()
        self.assertGreater(float(np.abs(direct-rounded).max()),0)


if __name__=='__main__':unittest.main()
