import unittest
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
import run_v54 as v

class Tests(unittest.TestCase):
 def frame(self):
  x=pd.DataFrame(v.MISSING,index=range(6),columns=v.FIELDS,dtype=object)
  x['transport_protocol']=['tcp','tcp','udp','udp','tcp','tcp'];x['src_port_fixed']=['80','443','80','443','80','80'];return x
 def test_unknown_is_not_missing_and_has_constant_norm(self):
  x=self.frame();enc=v.fit_encoder(x);z=x.iloc[:1].copy();z['src_port_fixed']='65001';a=v.encode(enc,z);z['src_port_fixed']=v.MISSING;b=v.encode(enc,z)
  self.assertEqual(a.multiply(a).sum(),13);self.assertEqual(b.multiply(b).sum(),13);self.assertEqual((a-b).multiply(a-b).sum(),2)
  self.assertNotIn('65001',enc['seen'][5])
 def test_rbf_is_hamming_kernel(self):
  x=self.frame();a=v.encode(v.fit_encoder(x),x).toarray();d=cdist(a,a,'sqeuclidean');h=np.sum(x.to_numpy()[:,None,:]!=x.to_numpy()[None,:,:],axis=2)
  np.testing.assert_allclose(d,2*h,atol=1e-12)
 def test_compression_preserves_every_class_and_conflict(self):
  x=self.frame();y=np.array([0,0,1,1,2,1]);xx,yy,w=v.compress(x,y)
  self.assertEqual(w.sum(),6)
  k=v.frame_keys(xx);targetkey=v.frame_keys(x.iloc[[0]])[0];self.assertEqual(set(yy[k==targetkey]),{0,1,2})
  scores=np.random.default_rng(4).normal(size=(6,3));lookup={q:scores[i] for i,q in enumerate(v.frame_keys(x))}
  raw=sum(max(0,1-lookup[q][t]) for q,t in zip(v.frame_keys(x),y));compressed=sum(wi*max(0,1-lookup[q][t]) for q,t,wi in zip(k,yy,w));self.assertAlmostEqual(raw,compressed)
 def test_actual_margin_fit_compressed_vs_original(self):
  x=self.frame();y=np.array([0,0,1,1,2,1]);enc=v.fit_encoder(x);a=v.encode(enc,x);xx,yy,w=v.compress(x,y)
  one=SVC(C=1,kernel='rbf',gamma=.125,tol=1e-9);two=SVC(C=1,kernel='rbf',gamma=.125,tol=1e-9)
  one.fit(a,y);two.fit(v.encode(enc,xx),yy,sample_weight=w)
  np.testing.assert_array_equal(one.predict(a),two.predict(a));np.testing.assert_allclose(one.decision_function(a),two.decision_function(a),atol=1e-7)
if __name__=='__main__':unittest.main()
