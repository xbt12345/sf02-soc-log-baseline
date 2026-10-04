import unittest
import numpy as np
import torch
from v158_fusion_contract import STAGES,validate_stage_roots,pipeline_cost,inner_fold
from v158_fusion_prototype_v2 import SharedProbabilityScorer
from v158_bank_initialization import safe_legacy_prior

class FusionContracts(unittest.TestCase):
    def test_every_supervised_stage_must_exclude_query_and_outer(self):
        stages={k:{1,2} for k in STAGES};self.assertTrue(validate_stage_roots({1,2},{3},{4},stages))
        for stage in STAGES:
            corrupt=dict(stages);corrupt[stage]={1,2,3}
            with self.assertRaises(ValueError):validate_stage_roots({1,2},{3},{4},corrupt)

    def test_full_pipeline_cost_not_one_fit(self):
        self.assertEqual(pipeline_cost(257)['full_network_gradient_updates'],200)
        self.assertEqual(pipeline_cost(257)['pipeline_fits'],5)
        self.assertEqual(9*pipeline_cost(257)['pipeline_fits']+6,51)
        self.assertEqual(inner_fold(0,123),inner_fold(0,123))

    def test_zero_initialization_and_shared_member_permutation(self):
        torch.manual_seed(15801);p=torch.softmax(torch.randn(5,16,3,dtype=torch.float64),-1);c=torch.randn(5,2,dtype=torch.float64)
        for dim in [0,2]:
            m=SharedProbabilityScorer(dim);q,w=m(p,c if dim else None)
            self.assertTrue(torch.equal(q,p.mean(1)))
            with torch.no_grad():m.output.weight.fill_(.3)
            perm=torch.randperm(16);q,w=m(p,c if dim else None);reordered,rw=m(p[:,perm],c if dim else None)
            self.assertTrue(torch.allclose(q,reordered,atol=1e-15,rtol=0))
            # Floating reductions across permuted members may round differently.
            self.assertTrue(torch.allclose(w[:,perm],rw,atol=1e-15,rtol=0));self.assertTrue((w>=0).all())
            self.assertTrue(torch.allclose(w.sum(1),torch.ones(5,dtype=torch.float64)))

    def test_condition_can_change_weights_but_common_failure_stays(self):
        p=torch.tensor([[[.1,.8,.1],[.2,.6,.2]]],dtype=torch.float64)
        m=SharedProbabilityScorer(1,hidden=1)
        with torch.no_grad():
            m.hidden.weight.zero_();m.hidden.bias.zero_();m.hidden.weight[0,1]=2.;m.hidden.weight[0,-1]=3.;m.output.weight.fill_(1.)
        q0,w0=m(p,torch.zeros(1,1,dtype=torch.float64));q1,w1=m(p,torch.ones(1,1,dtype=torch.float64))
        self.assertFalse(torch.allclose(w0,w1));self.assertEqual(int(q0.argmax(1)),1);self.assertEqual(int(q1.argmax(1)),1)
        with self.assertRaises(ValueError):SharedProbabilityScorer()(p,torch.zeros(1,1,dtype=torch.float64))

    def test_prior_uses_only_legal_fit_and_moves_with_permutation(self):
        current=np.array([[.1,.7,.2],[.1,.6,.3],[.1,.8,.1]])
        legacy=np.array([[.1,.2,.7],[.1,.7,.2],[.1,.1,.8]])
        y=np.array([1,1,1]);fit=np.array([True,True,False]);a=safe_legacy_prior(current,legacy,y,fit)
        changed=legacy.copy();changed[2]=[.99,.005,.005]
        self.assertEqual(a,safe_legacy_prior(current,changed,np.array([1,1,2]),fit))
        self.assertAlmostEqual(a['legacy_prior'],.125)
        p=torch.tensor(legacy[None,:,:],dtype=torch.float64);prior=torch.tensor([.2,.3,.5],dtype=torch.float64)
        m=SharedProbabilityScorer();q,w=m(p,prior=prior);perm=torch.tensor([2,0,1]);other,ow=m(p[:,perm],prior=prior[perm])
        self.assertTrue(torch.allclose(q,other,atol=1e-15,rtol=0));self.assertTrue(torch.allclose(w[:,perm],ow,atol=1e-15,rtol=0))

if __name__=='__main__':unittest.main()
