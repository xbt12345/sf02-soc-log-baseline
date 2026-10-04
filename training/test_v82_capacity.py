import unittest
import numpy as np
from scipy import sparse
from v82_capacity import objective, decode, predictions, counts_for, support_bucket


class CapacityTests(unittest.TestCase):
    def test_gradient_with_nonlinear_branch(self):
        rng=np.random.default_rng(4);x=sparse.csr_matrix(rng.normal(size=(7,5)))
        k=rng.normal(size=(7,3));counts=rng.integers(0,5,(7,3));w=rng.normal(size=27)
        loss,g=objective(w,x,k,counts,int(counts.sum()));direction=rng.normal(size=27);direction/=np.linalg.norm(direction)
        eps=1e-5;numeric=(objective(w+eps*direction,x,k,counts,int(counts.sum()))[0]-objective(w-eps*direction,x,k,counts,int(counts.sum()))[0])/(2*eps)
        self.assertAlmostEqual(numeric,float(g@direction),places=7)

    def test_grouped_loss_is_original_row_loss_including_conflicts(self):
        x=sparse.csr_matrix([[1.,2.],[3.,4.]])
        fid=np.array([0,0,0,1]);y=np.array([1,1,2,0]);selected=np.ones(4,bool)
        cc=counts_for(fid,y,selected,2);w=np.arange(9)/9
        grouped=objective(w,x,None,cc,4)
        expanded=objective(w,x[fid],None,np.eye(3)[y],4)
        self.assertAlmostEqual(grouped[0],expanded[0],places=12)
        np.testing.assert_allclose(grouped[1],expanded[1],atol=1e-12)
        np.testing.assert_array_equal(cc,[[0,2,1],[1,0,0]])

    def test_zero_added_coefficients_reproduce_original_linear_logits(self):
        x=sparse.csr_matrix([[1.,2.],[3.,4.]])
        model={'coef':np.arange(6).reshape(2,3),'kernel_coef':np.zeros((4,3)),'intercept':np.array([2,1,0])}
        np.testing.assert_array_equal(predictions(x,np.ones((2,4)),model),(x@model['coef']+model['intercept']).argmax(1))

    def test_support_uses_selected_training_only(self):
        ids=np.array([0,0,1,2]);labels=np.array([1,2,2,0]);selected=np.array([1,0,1,0],bool)
        np.testing.assert_array_equal(support_bucket(ids,labels,selected,3),[2,1,2,0])

    def test_linear_nonlinear_parameter_roundtrip(self):
        for width,d in [(5,0),(5,3)]:
            w=np.arange((width+d+1)*3);model=decode(w,width,d)
            np.testing.assert_array_equal(np.concatenate([model['coef'].ravel(),model['kernel_coef'].ravel(),model['intercept']]),w)


if __name__=='__main__':unittest.main()
