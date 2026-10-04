import unittest
import numpy as np
from run_v36_train import representations,evaluate


class ExecutionBoundaries(unittest.TestCase):
    def test_outer_evaluation_values_cannot_fit_any_encoder(self):
        fit=np.array([True,True,True,False])
        texts=['credential failure','credential success','network blocked','never_seen_payload']
        facts=['{"outcome":"failure","src_port":443}','{"outcome":"success","src_port":80}','{}','{"outcome":"novel","src_port":60000}']
        x,inv,t,f=representations(texts,np.arange(4),facts,np.arange(4),'B2',fit)
        texts[-1]='different_future';facts[-1]='{"outcome":"other_future","src_port":65535}'
        z,inv2,t2,f2=representations(texts,np.arange(4),facts,np.arange(4),'B2',fit)
        self.assertEqual(t.names().tolist(),t2.names().tolist())
        self.assertEqual(f.names().tolist(),f2.names().tolist())
        np.testing.assert_array_equal(t.idf,t2.idf)
        np.testing.assert_array_equal(f.scale,f2.scale)
        np.testing.assert_allclose(x[inv[:3]].toarray(),z[inv2[:3]].toarray())

    def test_empty_event_ids_do_not_inflate_group_equal_evidence(self):
        y=np.array([0,0,0,1,2]);p=np.eye(3)[y]
        result=evaluate(y,p,np.arange(5),np.array(['x']*5),np.array([True,True,True,False,False]),
                        np.array([True,True,True,False,False]),[(.001,.5)],np.array([0,0,0,1,2]))
        self.assertEqual(result['all_rows']['rows'],5)
        self.assertEqual(result['group_equal_nonempty']['rows'],2)
        self.assertEqual(sum(result['view_equal_including_collapsed_empty']['class_support']),3)


if __name__=='__main__':unittest.main()
