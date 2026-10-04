import unittest
import numpy as np
from v156_neighborhood_geometry import nearest,condition,normalize_dense

class GeometryTests(unittest.TestCase):
    def test_own_only_best_is_excluded_but_other_root_same_local_survives(self):
        r=nearest(np.array([[1.,.9],[1.,.9]]),np.array([4,5]),np.array([[0,2],[3,2]]),{4:{10,11},5:{12}},[10,10])
        np.testing.assert_array_equal(r['local'],[5,4]);np.testing.assert_array_equal(r['tied_original_rows'],[2,3])
        np.testing.assert_array_equal(r['tied_roots'],[1,1])

    def test_ties_keep_original_mass_and_distinct_roots_not_duplicate_nodes(self):
        r=nearest(np.array([[.8,.8,.3]]),np.array([1,2,3]),np.array([[7,5,8]]),{1:{20,21},2:{21,22},3:{23}},[20])
        self.assertEqual(r['local'][0],1);self.assertEqual(r['tied_original_rows'][0],12)
        self.assertEqual(r['tied_roots'][0],2);self.assertEqual(r['tied_locals'][0],2)
        empty=nearest(np.array([[.8,.8,.3]]),np.array([1,2,3]),np.zeros((1,3),int),{1:{20},2:{20},3:{20}},[20])
        self.assertTrue(np.isnan(empty['cosine'][0]));self.assertEqual(empty['local'][0],-1)

    def test_unknown_absent_and_not_applicable_are_distinct_zero_has_no_direction(self):
        known={'transport_protocol':'tcp','src_port_fixed':443}
        unknown={'transport_protocol':'tcp','src_port_fixed':65536}
        absent={'transport_protocol':'tcp'}
        icmp={'transport_protocol':'icmp','src_port_fixed':65536}
        self.assertEqual(len({condition(f) for f in [known,unknown,absent,icmp]}),4)
        a,zero=normalize_dense(np.array([[0,0],[3,4]]))
        np.testing.assert_array_equal(zero,[True,False]);np.testing.assert_allclose(a,[[0,0],[.6,.8]])

if __name__=='__main__':unittest.main()
