import copy
import unittest
import numpy as np
import pandas as pd
from v64_selection import select
from audit_v64_evidence import pure_rule, leave_source_out, context_shapes


def metrics(row=.985,M=.999,S=.28):
    return {'ASA':{'recall_B_M_S':[None,M,.96],'macro_f1_M_S':row},
            'M_source_recall':M,'S_source_recall':S,'source_balanced':(M+S)/2,
            'hard':{p:{'1':M,'2':.01} for p in ['tcp','udp']}}


class PolicyTests(unittest.TestCase):
    def test_higher_subject_score_cannot_spend_row_budget(self):
        good={'step':1,'metrics':metrics()}
        bad={'step':2,'metrics':metrics(row=.916,S=.4)}
        result=select([good,bad],{'frozen':metrics()})
        self.assertEqual(result['selected']['step'],1)
        self.assertFalse(result['screen_passed'])

    def test_no_feasible_model_means_none_not_best_failure(self):
        result=select([{'step':1,'metrics':metrics(row=.8,S=.9)}],{'frozen':metrics()})
        self.assertIsNone(result['selected']);self.assertFalse(result['screen_passed'])

    def test_every_reference_protects_its_stronger_cells(self):
        result=select([{'step':1,'metrics':metrics(row=.916)}],
                      {'degraded':metrics(row=.916),'strong':metrics()})
        self.assertIsNone(result['selected'])

    def test_missing_or_nan_metrics_fail_closed(self):
        with self.assertRaises(ValueError):select([],{})
        bad=metrics();bad['S_source_recall']=float('nan')
        with self.assertRaises(ValueError):select([{'step':1,'metrics':bad}],{'base':metrics()})

    def test_gain_requires_hard_bucket_improvement_and_preserves_ties(self):
        x=metrics(S=.4)
        self.assertFalse(select([{'step':2,'metrics':x}],{'base':metrics()})['screen_passed'])
        x['hard']['tcp']['2']=.1
        result=select([{'step':2,'metrics':x},{'step':1,'metrics':copy.deepcopy(x)}],{'base':metrics()})
        self.assertTrue(result['screen_passed']);self.assertEqual(result['selected']['step'],1)


class EvidenceTests(unittest.TestCase):
    def test_duplicate_rows_do_not_create_independent_support(self):
        f=pd.DataFrame({'key':['a']*100,'group':[1]*100,'label':[2]*100})
        self.assertEqual(pure_rule(f,'key'),{})
        self.assertTrue((leave_source_out(f,'key')==-1).all())

    def test_target_source_all_labels_are_excluded(self):
        f=pd.DataFrame({'key':['a']*5,'group':[1,2,3,4,4],'label':[2,2,2,2,1]})
        p=leave_source_out(f,'key')
        self.assertEqual(p.tolist(),[-1,-1,-1,2,2])
        self.assertEqual(pure_rule(f,'key'),{})

    def test_three_sources_do_not_support_each_other_at_min_three(self):
        f=pd.DataFrame({'key':['a']*3,'group':[1,2,3],'label':[2]*3})
        self.assertEqual(pure_rule(f,'key'),{'a':2})
        self.assertTrue((leave_source_out(f,'key')==-1).all())

    def test_context_shape_does_not_use_count_magnitude(self):
        x=np.zeros((2,16));x[:,0:4]=[[1,1,1,1],[30,12,2,9]]
        x[:,4]=[.1,.9]
        y=context_shapes(x)
        self.assertTrue(y.iloc[0].equals(y.iloc[1]))
        self.assertEqual(int(y.iloc[0].sum()),2)


if __name__=='__main__':unittest.main()
