import unittest
import numpy as np
from scipy import sparse
from scipy.special import softmax
from v51_residual import applicable,loss_gradient,predict_matrix

class ToyBase:
    def decision_function(self,x):return np.asarray(x.dot(np.array([[.3,-.1,.7],[-.8,.5,.2]])))
    def predict_proba(self,x):return softmax(self.decision_function(x),axis=1)

class ResidualTests(unittest.TestCase):
    def test_gate_has_no_source_or_prediction_dependency(self):
        facts=[{'auth_result':'failure','outcome':'failure','product_name':'x','route':'authentication'},
               {'auth_result':'failure','outcome':'failure','product_name':None,'route':'asa'},
               {'policy_decision':'allowed'}, {'auth_result':'success','outcome':'failure'}]
        np.testing.assert_array_equal(applicable(facts),[True,True,False,False])

    def test_analytic_gradient_with_fixed_offset_matches_finite_difference(self):
        rng=np.random.default_rng(51);x=sparse.csr_matrix(rng.normal(size=(5,4)));o=rng.normal(size=(5,3));y=np.array([0,1,2,0,2]);weights=np.array([.5,1,2,.5,1]);w=rng.normal(size=12)*.2
        _,g=loss_gradient(w,x,o,y,weights,.1);numeric=[]
        for k in range(len(w)):
            e=np.zeros(len(w));e[k]=1e-6
            numeric.append((loss_gradient(w+e,x,o,y,weights,.1)[0]-loss_gradient(w-e,x,o,y,weights,.1)[0])/2e-6)
        np.testing.assert_allclose(g,numeric,atol=2e-8,rtol=0)

    def test_aggregation_preserves_mixed_labels_and_loss(self):
        x=sparse.csr_matrix([[1,0],[1,0],[1,0],[0,1]]);o=np.zeros((4,3));y=np.array([0,0,2,1]);w=np.array([.5,.5,1,1]);coef=np.arange(6)*.1
        a,ga=loss_gradient(coef,x,o,y,w,.1)
        b,gb=loss_gradient(coef,x[[0,2,3]],o[:3],y[[0,2,3]],np.ones(3),.1)
        self.assertAlmostEqual(a,b);np.testing.assert_allclose(ga,gb,atol=1e-14)

    def test_arbitrary_residual_cannot_change_nonapplicable_predictions(self):
        x=sparse.csr_matrix([[1,2],[9,-3],[1,0]]);b=ToyBase();r={'columns':np.array([0,1]),'coef':np.arange(9).reshape(3,3)*1000.}
        q=predict_matrix(b,r,x,np.array([False,True,False]));np.testing.assert_array_equal(q[[0,2]],b.predict_proba(x)[[0,2]])
        np.testing.assert_allclose(q.sum(axis=1),1)

    def test_zero_residual_reproduces_base_and_keeps_three_classes(self):
        x=sparse.csr_matrix([[1,2],[9,-3]]);b=ToyBase();r={'columns':np.array([0,1]),'coef':np.zeros((3,3))}
        np.testing.assert_allclose(predict_matrix(b,r,x,np.ones(2,dtype=bool)),b.predict_proba(x),atol=1e-15)

if __name__=='__main__':unittest.main()
