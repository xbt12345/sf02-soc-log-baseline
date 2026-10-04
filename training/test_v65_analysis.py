import copy
import unittest
from analyze_v65_rank_heads import continuation_gate


def metric(s=.3, hard=.1):
    return {'ASA':{'recall_B_M_S':[None,.999,.9],'macro_f1_M_S':.98},
            'M_source_recall':.999,'S_source_recall':s,
            'hard':{'tcp':{'1':.999,'2':hard},'udp':{'1':.999,'2':.1}}}


def fixtures():
    return {str(i):{'S_validation_sources':100,
       'selection':{'selected':{'metrics':metric(.36,.12)}},
       'references':{'h0':metric(),'coarse':metric(.29)}} for i in range(3)}


class GateTests(unittest.TestCase):
    def test_no_feasible_checkpoint_is_not_replaced(self):
        f=fixtures();f['1']['selection']['selected']=None
        self.assertFalse(continuation_gate(f)['passed'])
        self.assertIsNone(continuation_gate(f)['pooled_S_gain'])

    def test_large_gain_cannot_hide_hard_slice_regression(self):
        f=fixtures();f['1']['selection']['selected']['metrics']['hard']['udp']['2']=.09
        g=continuation_gate(f)
        self.assertFalse(g['passed']);self.assertFalse(g['no_hard_S_decline'])

    def test_pooled_gain_uses_source_counts_not_row_counts_or_fold_means(self):
        f=fixtures();f['0']['S_validation_sources']=1000
        f['0']['selection']['selected']['metrics']['S_source_recall']=.31
        g=continuation_gate(f)
        self.assertAlmostEqual(g['pooled_S_gain'],(1000*.01+200*.06)/1200)
        self.assertFalse(g['passed'])

    def test_stronger_reference_is_used_and_valid_gain_can_pass(self):
        f=fixtures();self.assertTrue(continuation_gate(f)['passed'])
        for v in f.values():v['references']['coarse']=metric(.32)
        self.assertFalse(continuation_gate(f)['passed'])


if __name__=='__main__':unittest.main()
