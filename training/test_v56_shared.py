import unittest
import numpy as np
import pandas as pd
import run_v54 as v
import v56_shared_control as s

class Tests(unittest.TestCase):
 def test_novel_combination_still_receives_shared_facts(self):
  x=pd.DataFrame(v.MISSING,index=range(2),columns=v.FIELDS,dtype=object);x['transport_protocol']=['tcp','udp'];x['src_role']=['outside','inside'];x['dst_role']=['dmz','inside'];b=s.fit_design(x,['a','b']);z=x.iloc[:1].copy();z['dst_role']='inside';a=s.design(b,z);indices=set(a.indices);counts={level:sum(i in indices for i in mapping.values()) for level,mapping in b['maps'].items()}
  self.assertEqual(sum(counts[k] for k in ['coarse','fine','exact']),0);self.assertGreater(counts['field:transport_protocol'],0);self.assertGreater(counts['field:dst_role'],0)
 def test_new_shared_branch_has_no_raw_fixed_ports_or_metadata(self):
  self.assertNotIn('src_port_fixed',s.SHARED);self.assertNotIn('dst_port_fixed',s.SHARED);self.assertTrue(set(s.SHARED)<=set(v.FIELDS));self.assertTrue({'timestamp','src_ip','product_name','label_binary','body_group'}.isdisjoint(s.SHARED))
if __name__=='__main__':unittest.main()
