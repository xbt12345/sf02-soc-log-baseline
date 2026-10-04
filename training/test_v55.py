import unittest
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.neural_network import MLPClassifier
import run_v54 as v
import run_v55 as n
from probe_v55_calibration import choose_threshold

class Tests(unittest.TestCase):
 def setup_data(self):
  x=pd.DataFrame(v.MISSING,index=range(5),columns=v.FIELDS,dtype=object);x['transport_protocol']=['tcp']*4+['udp'];x['src_role']='outside';x['dst_role']='dmz';return x,np.array([1,1,1,2,0])
 def test_weighted_loss_exact_group_objective(self):
  x,y=self.setup_data();xx,yy,vi,gi,k,counts,gv,prior=n.expanded(x,y,True);loss=np.arange(len(xx))/7
  q=n.distribution('balanced',np.zeros(len(k)),None,gv,prior);means=n.group_means(loss,gi,counts)
  for mode in ['balanced','dro']:
   w=n.risk_weights(mode,gi,counts,gv,prior,q);self.assertAlmostEqual(np.mean(loss*w),q@means)
  w=n.risk_weights('erm',gi,counts,gv,prior,q);self.assertAlmostEqual(np.mean(loss*w),sum(prior[j]*loss[vi==j].mean() for j in range(3)))
 def test_views_keep_labels_and_preserve_total_per_record_risk(self):
  x,y=self.setup_data();xx,yy,vi,gi,k,counts,gv,prior=n.expanded(x,y,True)
  np.testing.assert_array_equal(yy,np.tile(y,3));self.assertTrue(xx.iloc[5:10].src_role.eq(v.MISSING).all());self.assertTrue(xx.iloc[10:].dst_role.eq(v.MISSING).all())
  w=n.risk_weights('erm',gi,counts,gv,prior,None);np.testing.assert_allclose(w.reshape(3,5).sum(0)/len(w),np.ones(5)/5)
 def test_adversary_prefers_harder_group_and_keeps_view_mass(self):
  means=np.array([0.,2.,4.,1.]);gv=np.array([0,0,1,1]);prior=np.array([.5,.5]);q=np.ones(4)/4
  out=n.distribution('dro',means,q,gv,prior);self.assertGreater(out[1],out[0]);self.assertGreater(out[2],out[3]);np.testing.assert_allclose([out[:2].sum(),out[2:].sum()],prior)
 def test_behavior_never_uses_targets_or_metadata_or_exact_port(self):
  x,y=self.setup_data();x['src_port_fixed']='80';a=n.behavior(x,True);x['src_port_fixed']='443';x['timestamp']='2099';x['label_binary']='malicious';np.testing.assert_array_equal(a,n.behavior(x,True))
 def test_library_weighted_gradient_and_regularization(self):
  # Test the actual library gradient on one full batch, with unequal weights.
  a=np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.]]);y=np.array([0,1,2,1]);w=np.array([1.,2.,4.,1.]);lr=.001;alpha=.3
  m=MLPClassifier(hidden_layer_sizes=(3,),solver='sgd',momentum=0,nesterovs_momentum=False,learning_rate_init=lr,alpha=alpha,batch_size=4,shuffle=False,random_state=7)
  m.partial_fit(a,y,classes=[0,1,2]);W=[t.copy() for t in m.coefs_];b=[t.copy() for t in m.intercepts_]
  h=np.maximum(a@W[0]+b[0],0);p=softmax(h@W[1]+b[1],axis=1);d=(p-np.eye(3)[y])*w[:,None]/w.sum();dh=(d@W[1].T)*(h>0)
  grads=[a.T@dh+alpha*W[0]/w.sum(),h.T@d+alpha*W[1]/w.sum()];bias=[dh.sum(0),d.sum(0)]
  m.partial_fit(a,y,sample_weight=w)
  for i in range(2):np.testing.assert_allclose(m.coefs_[i],W[i]-lr*grads[i],rtol=0,atol=1e-12);np.testing.assert_allclose(m.intercepts_[i],b[i]-lr*bias[i],rtol=0,atol=1e-12)
 def test_threshold_ties_use_double_comparison_and_keep_normal(self):
  frames=[]
  for view in n.VIEWS:
   p=np.array([[0,.25,.75],[0,.9,.1],[0,.25,.75],[0,.75,.25],[.8,.1,.1]],dtype=np.float32)
   d=pd.DataFrame(p,columns=['p_0','p_1','p_2']);d['label_index']=[1,1,2,2,0];d['pred']=p.argmax(1);d['route']=['asa']*4+['asa_acl'];d['scenario']=view;d['eligible_stress']=True;frames.append(d)
  d=pd.concat(frames,ignore_index=True);z=choose_threshold(d);self.assertGreater(z['threshold'],.75);self.assertLess(z['threshold'],.75001)
  p=d[['p_0','p_1','p_2']].to_numpy();s=(p[:,1]/(p[:,1]+p[:,2])).astype(np.float64);yp=np.where(d.pred==0,0,np.where(s>=z['threshold'],1,2))
  np.testing.assert_array_equal(yp,np.tile([2,1,2,2,0],3));np.testing.assert_array_equal(z['calibration_errors_full_src_dst_M_S'],[1,0]*3)
if __name__=='__main__':unittest.main()
