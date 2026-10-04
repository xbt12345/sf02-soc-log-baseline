import unittest
import numpy as np
import pandas as pd
import run_v53_asa as v

class InputTests(unittest.TestCase):
 def frame(self):
  return pd.DataFrame([v.observed({'action':'deny','outcome':'blocked','transport_protocol':'tcp','src_role':'outside','dst_role':'dmz','src_port_fixed':1234,'dst_port_fixed':443}),
                       v.observed({'action':'deny','outcome':'blocked','transport_protocol':'tcp','src_role':'outside','dst_role':'dmz','src_port_fixed':65000,'dst_port_fixed':443})],columns=v.FIELDS,dtype=object)
 def test_coarsening_does_not_keep_hidden_port_in_interactions(self):
  x=self.frame();x[v.VIEWS['no_src']]=np.nan;z=v.expanded(x,True)
  pd.testing.assert_series_equal(z.iloc[0],z.iloc[1],check_names=False)
  for i,p in enumerate(v.PAIRS):
   if 5 in p:self.assertTrue(z['joint__'+'__'.join(v.FIELDS[j] for j in p)].isna().all())
 def test_hidden_role_does_not_survive(self):
  x=self.frame();x['src_role']=np.nan;z=v.expanded(x,True)
  for p in v.PAIRS:
   if 3 in p:self.assertTrue(z['joint__'+'__'.join(v.FIELDS[j] for j in p)].isna().all())
 def test_real_ports_preserved_and_untrusted_numbers_rejected(self):
  for n in [0,1,22,443,65535]:self.assertEqual(v.observed({'src_port_fixed':n})['src_port_fixed'],str(n))
  for n in [65536,-1,'CRED-1234','443',True,443.5]:self.assertTrue(pd.isna(v.observed({'src_port_fixed':n})['src_port_fixed']))
 def test_original_label_mass_and_rows_preserved(self):
  x=self.frame();y=np.array([1,2]);xx,yy,w=v.view_data(x,y,True)
  np.testing.assert_array_equal(yy.reshape(6,2),np.tile(y,(6,1)))
  np.testing.assert_allclose(w.reshape(6,2).sum(axis=0),1,atol=1e-15)
  for k in [1,2]:self.assertAlmostEqual(w[yy==k].sum(),1)
  pd.testing.assert_frame_equal(xx.iloc[:2].reset_index(drop=True),x.reset_index(drop=True))
 def test_unknown_and_literal_pair_do_not_collapse(self):
  z=v.expanded(self.frame(),True)
  self.assertNotEqual(z.loc[0,'joint__src_port_fixed__dst_port_fixed'],z.loc[1,'joint__src_port_fixed__dst_port_fixed'])
  self.assertEqual(z.shape[1],len(v.FIELDS)+len(v.PAIRS))
if __name__=='__main__':unittest.main()
