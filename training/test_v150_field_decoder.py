import unittest
import numpy as np
import torch
from v150_field_decoder import solve,predict,decode,inner_held

class FieldDecoderTests(unittest.TestCase):
    def test_original_frequency_equivalent_to_repeated_rows(self):
        x=torch.tensor([[1.,2.],[2.,4.],[3.,1.]],dtype=torch.float64)
        y=torch.tensor([[0.,1.],[1.,0.],[1.,1.]],dtype=torch.float64);w=torch.tensor([3.,2.,1.],dtype=torch.float64)
        a,r=solve(x,y,w);ix=torch.tensor([0,0,0,1,1,2]);b,s=solve(x[ix],y[ix],torch.ones(6,dtype=torch.float64))
        np.testing.assert_allclose(predict(x,a),predict(x,b),atol=1e-12)
        self.assertEqual(r['original_fit_mass'],6);self.assertLess(r['normal_equation_relative_residual'],1e-8)

    def test_constant_feature_and_target_unknown_states(self):
        x=torch.tensor([[1.,1.],[2.,1.],[3.,1.]],dtype=torch.float64);y=torch.ones((3,1),dtype=torch.float64)
        a,r=solve(x,y,torch.ones(3,dtype=torch.float64));self.assertEqual(r['active_feature_dimensions'],1)
        np.testing.assert_allclose(predict(x,a),np.ones((3,1)),atol=1e-12)
        entries=[dict(field='src_port_fixed',start=0,stop=17,values=None),dict(field='src_role',start=17,stop=20,values=['inside','outside','dmz'])]
        z=np.zeros((3,20));z[1,16]=1;z[2,0]=1;z[2,18]=1;d=decode(z,entries)
        np.testing.assert_array_equal(d['src_port_fixed'],[0,65536,1]);self.assertEqual(d['src_role'].tolist(),[None,None,'outside'])
        self.assertEqual(inner_held(123),inner_held(123))

if __name__=='__main__':unittest.main()
