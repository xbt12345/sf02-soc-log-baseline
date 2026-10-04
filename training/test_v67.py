import unittest
import numpy as np
import pandas as pd
from run_v67_specialist import threshold_from_inner
from run_v67_targeted import target_mask
from verify_v67 import independent_threshold


class TargetedTests(unittest.TestCase):
    def calibration(self):
        rows=[];scores=[];fold=[]
        for k in range(3):
            for protocol in ['tcp','udp']:
                for g in range(200):
                    rows.append({'label':1,'group':k*1000+(0 if protocol=='tcp' else 200)+g,'transport_protocol':protocol})
                    scores.append(.9 if g<3 else .1);fold.append(k)
                rows.append({'label':2,'group':k*1000+900+(0 if protocol=='tcp' else 1),'transport_protocol':protocol})
                scores.append(.95);fold.append(k)
        return pd.DataFrame(rows),np.array(scores),np.array(fold)

    def test_tied_scores_are_not_partially_admitted(self):
        f,s,k=self.calibration();t,_=threshold_from_inner(f,s,k)
        self.assertGreater(t,.9);self.assertLess(t,.95)
        self.assertEqual(t,independent_threshold(f,s,k))

    def test_duplicate_rows_do_not_multiply_source_budget(self):
        f,s,k=self.calibration();t,_=threshold_from_inner(f,s,k)
        ids=np.r_[np.arange(len(f)),np.repeat(0,500)]
        self.assertEqual(t,threshold_from_inner(f.iloc[ids].reset_index(drop=True),s[ids],k[ids])[0])

    def test_no_s_labels_used_to_select_threshold(self):
        f,s,k=self.calibration();t,_=threshold_from_inner(f,s,k)
        s[f.label.eq(2)]=0
        self.assertEqual(t,threshold_from_inner(f,s,k)[0])

    def test_missing_M_support_does_not_pass(self):
        f,s,k=self.calibration();mask=~((k==0)&f.transport_protocol.eq('udp').to_numpy())
        with self.assertRaises(AssertionError):threshold_from_inner(f[mask].reset_index(drop=True),s[mask],k[mask])

    def test_target_definition_preserves_only_original_failed_nonconflict_S(self):
        d=pd.DataFrame({'label':[2,2,2,1], 'H1_pred':[1,1,2,1],
            'validation_input_conflict':[False,True,False,False],
            'behavior':['deny|blocked|tcp|outside|dmz']*4})
        np.testing.assert_array_equal(target_mask(d),[True,False,False,False])


if __name__=='__main__':unittest.main()
