import copy
import unittest
import pandas as pd
from v66_selection import guarded_select,select_both
from prepare_v66_supported_folds import supported_folds
from audit_v66_endpoint_mapping import bijection_mask


def metric(errors=0,support=10,s=.3):
    return {'ASA':{'recall_B_M_S':[None,1.,.9],'macro_f1_M_S':.99},
      'M_source_recall':1.,'S_source_recall':s,'source_balanced':(1+s)/2,
      'hard':{p:{'1':1.,'2':.1} for p in ['tcp','udp']},
      'normal_control_errors':errors,'normal_support':support}


class SelectionTests(unittest.TestCase):
    def test_normal_failure_rejected_even_when_S_is_better(self):
        curve=[{'step':1,'metrics':metric()},{'step':2,'metrics':metric(errors=1,s=.9)}]
        result=guarded_select(curve,{'ref':metric()})
        self.assertEqual(result['selected']['step'],1)
        self.assertEqual(result['normal_rejected_steps'],[2])

    def test_missing_normal_support_fails_closed(self):
        result=guarded_select([{'step':1,'metrics':metric(support=0)}],{'ref':metric(support=0)})
        self.assertEqual(result['status'],'normal_validation_unsupported')
        self.assertIsNone(result['selected'])

    def test_all_failed_does_not_return_best_failure(self):
        self.assertIsNone(guarded_select([{'step':1,'metrics':metric(errors=1)}],{'ref':metric()})['selected'])

    def test_arms_use_identical_selector_and_budget(self):
        curve=[{'step':1,'metrics':metric()},{'step':2,'metrics':metric(s=.4)}]
        both=select_both({'H0':curve,'H1':copy.deepcopy(curve)},{'ref':metric()})
        self.assertEqual(both['H0'],both['H1'])
        with self.assertRaises(ValueError):select_both({'H0':curve,'H1':curve[:1]},{'ref':metric()})

    def test_invalid_normal_counts_do_not_pass(self):
        for errors in [-1,11,float('nan')]:
            with self.assertRaises(ValueError):guarded_select([{'step':1,'metrics':metric(errors=errors)}],{'ref':metric()})

    def test_support_repair_moves_whole_mixed_label_sources(self):
        frame=pd.DataFrame({'group':[10,10,20,20,30,30,40,40],
               'label':[0,2,0,1,0,1,1,2],'fold':[0,0,0,0,2,2,1,1]})
        result=supported_folds(frame)
        self.assertEqual(frame.assign(new=result).groupby('group').new.nunique().max(),1)
        self.assertEqual(frame[frame.label.eq(0)].assign(new=result).new.nunique(),3)
        self.assertTrue(result[frame.group.eq(40)].eq(1).all())

    def test_too_few_normal_sources_does_not_invent_validation_support(self):
        frame=pd.DataFrame({'group':[1,1,2,2],'label':[0,1,0,2],'fold':[0,0,1,1]})
        with self.assertRaises(ValueError):supported_folds(frame)

    def test_ambiguous_endpoint_maps_are_quarantined(self):
        frame=pd.DataFrame({'collector':['a']*5,'structured':['x','x','y','z','z'],
                            'body':['u','v','v','w','w']})
        safe,_,_=bijection_mask(frame,'structured','body')
        self.assertEqual(safe.tolist(),[False,False,False,True,True])


if __name__=='__main__':unittest.main()
