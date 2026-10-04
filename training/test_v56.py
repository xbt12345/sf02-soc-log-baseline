import unittest
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.special import softmax
import run_v54 as v
import v56_pooling as m

class Tests(unittest.TestCase):
 def data(self):
  x=pd.DataFrame(v.MISSING,index=range(6),columns=v.FIELDS,dtype=object);x.transport_protocol='tcp';x.src_role=['outside']*4+['inside']*2;x.dst_role='dmz';x.src_port_fixed=['80','80','80','80','443','443'];return x,np.array([1,1,2,1,0,2]),np.array(['a','a','b','a','c','d'])
 def test_analytic_gradient_and_exact_row_compression(self):
  x,y,g=self.data();b=m.fit_design(x,g);a=m.design(b,x);xx,c,ids=m.compress(x,y);aa=m.design(b,xx);offset=np.tile([-.1,.2,.7],(len(xx),1));p=m.penalty(b,'support',1.);w=np.random.default_rng(7).normal(0,.1,aa.shape[1]*3)
  f=lambda t:m.objective(t,aa,offset,c,p)[0];j=lambda t:m.objective(t,aa,offset,c,p)[1];self.assertLess(check_grad(f,j,w),1e-6)
  raw=np.eye(3)[y];r=m.objective(w,a,offset[ids],raw,p);np.testing.assert_allclose(r[0],f(w),atol=1e-12);np.testing.assert_allclose(r[1],j(w),atol=1e-12);np.testing.assert_array_equal(c.sum(0),np.bincount(y,minlength=3))
 def test_support_not_row_multiplicity_and_global_replication(self):
  x,y,g=self.data();b=m.fit_design(x,g);rr=m.fit_design(pd.concat([x]*3,ignore_index=True),np.tile(g,3));np.testing.assert_array_equal(b['bodies'],rr['bodies']);np.testing.assert_allclose(m.penalty(b,'support',1),m.penalty(rr,'support',1));self.assertTrue((m.penalty(b,'support',1)<=m.penalty(b,'uniform',1)).all())
 def test_new_combinations_get_no_untrained_effect(self):
  x,y,g=self.data();b=m.fit_design(x,g);z=x.iloc[:1].copy();z.transport_protocol='unseen_protocol';self.assertEqual(m.design(b,z).nnz,0)
 def test_support_does_not_use_labels_or_evaluation(self):
  x,y,g=self.data();b=m.fit_design(x,g);x['label_index']=2;x['timestamp']='2099';x['product_name']='vendor';x['src_ip']='1.2.3.4';bb=m.fit_design(x,g);self.assertEqual(b['maps'],bb['maps']);np.testing.assert_array_equal(b['bodies'],bb['bodies'])
 def test_rare_support_implies_stronger_relative_shrinkage(self):
  b={'rows':np.array([100.,100.]),'fit_rows':200,'bodies':np.array([1.,16.])};p=m.penalty(b,'support',1);self.assertEqual(p[0]/p[1],4.)
if __name__=='__main__':unittest.main()
