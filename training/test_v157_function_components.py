import unittest
from types import SimpleNamespace
import torch
from v157_function_components import components,ensemble_summary

class FunctionalTests(unittest.TestCase):
    def test_exact_full_function_and_field_partition_not_independent_predictions(self):
        torch.manual_seed(157)
        h=torch.randn(3,2,4,dtype=torch.float64);f=torch.randn(3,5,dtype=torch.float64)
        m=SimpleNamespace(head_weight=torch.randn(2,4,3,dtype=torch.float64),head_bias=torch.randn(2,3,dtype=torch.float64),head_facts=torch.randn(3,5,dtype=torch.float64))
        b,df,bi,ff=components(h,f,m,{'first':[0,1],'second':[2,3,4]})
        z=(h.transpose(0,1)@m.head_weight).transpose(0,1)+m.head_bias+(f@m.head_facts.T)[:,None,:]
        self.assertTrue(torch.equal(z,b+bi+df[:,None,:]))
        torch.testing.assert_close(sum(ff.values()),df,atol=1e-14,rtol=1e-14)

    def test_mean_logits_and_mean_probabilities_have_real_opposite_decisions(self):
        z=torch.tensor([[[-100.,0.,10.],[-100.,2.,0.],[-100.,2.,0.]]],dtype=torch.float64)
        p,s=ensemble_summary(z,z,torch.zeros(1,3,dtype=torch.float64),torch.zeros(3,3,dtype=torch.float64))
        self.assertEqual(s['mean_logit_pred'][0].item(),2)
        self.assertEqual(s['probability_mean_pred'][0].item(),1)
        self.assertTrue(p[0,1]>p[0,2])

    def test_common_class_offset_changes_component_raw_values_but_not_function(self):
        z=torch.tensor([[[-10.,2.,3.],[-10.,4.,3.]]],dtype=torch.float64)
        p=torch.softmax(z,-1).mean(1);offset=torch.tensor([[[40.],[40.]]],dtype=torch.float64)
        torch.testing.assert_close(torch.softmax(z+offset,-1).mean(1),p,atol=1e-15,rtol=1e-15)
        torch.testing.assert_close((z+offset)[:,:,2]-(z+offset)[:,:,1],z[:,:,2]-z[:,:,1])

if __name__=='__main__':unittest.main()
