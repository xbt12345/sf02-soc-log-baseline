"""Replay actual V131 counterexamples and independent official-row counts. No fits."""
import unittest

import numpy as np
import pandas as pd

from v131_common import ROOT,OUT,OFFICIAL,read,learning_gates,require_run_seal


class ObservedV131Cases(unittest.TestCase):
    def test_actual_perfect_hard_S_with_M_regression_must_fail(self):
        history=read(OUT/'fold1_CO/progress.json')
        partial=next(v for v in history if v['epoch']==60)['stats']
        self.assertEqual(partial['hard_S_errors'],0)
        self.assertEqual(partial['M_errors'],52)
        self.assertFalse(all(learning_gates(partial).values()))
        complete=read(OUT/'fold1_CO/fit.json')['endpoint_training']
        self.assertTrue(all(learning_gates(complete).values()))

    def test_registration_and_fixed_selection_are_intact(self):
        require_run_seal(ROOT/'training/v131_train.py')
        selection=read(OUT/'learning_selection.json')
        self.assertEqual(selection['selected_arm'],'R')
        self.assertFalse(selection['new_held_results_read_before_selection'])
        self.assertEqual(selection['registered_preference'],['R','C','O','CO'])
        for arm in ['R','O','C','CO']:
            self.assertTrue(all(learning_gates(selection['arms'][arm]['stats']).values()))

    def test_actual_full_task_recount_includes_every_original_label(self):
        delivery=read(OUT/'delivery.json')
        ledger=pd.read_parquet(OUT/'full_prediction_ledger.parquet')
        official=pd.read_parquet(OFFICIAL,columns=['label_binary'])
        y=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(np.int8)
        self.assertEqual(len(ledger),2056871)
        np.testing.assert_array_equal(ledger.row_position.to_numpy(),np.arange(len(y)))
        self.assertFalse(ledger.row_position.duplicated().any())
        for cl in range(3):
            p=ledger.pred_R.to_numpy(); m=delivery['metrics']['R']['full_task'][str(cl)]
            self.assertEqual(m['support'],int((y==cl).sum()))
            self.assertEqual(m['correct'],int(((y==cl)&(p==cl)).sum()))
            self.assertEqual(m['missed'],int(((y==cl)&(p!=cl)).sum()))
            self.assertEqual(m['false_called'],int(((y!=cl)&(p==cl)).sum()))
        self.assertFalse(delivery['model_promoted'])
        self.assertEqual(delivery['quality_acceptance'],all(delivery['quality']['R']['gates'].values()))

    def test_all_actual_fits_and_update_budget_are_accounted(self):
        delivery=read(OUT/'delivery.json')
        fits=[read(p) for p in OUT.glob('*/fit.json')]
        self.assertEqual(len(fits),8)
        self.assertEqual(sum(v['optimizer_steps'] for v in fits),31400)
        self.assertEqual(delivery['classifier_fits'],8)
        self.assertEqual(delivery['optimizer_steps'],31400)
        for fit in fits:
            self.assertEqual(fit['endpoint'],2000 if fit['kind']=='probe' else 100)


if __name__=='__main__':unittest.main()
