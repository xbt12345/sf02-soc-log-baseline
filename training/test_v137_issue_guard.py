"""Real original-row populations exercise repair preservation independently of totals."""
import unittest
import pandas as pd
from v137_issue_guard import ROOT,OUT,DEST,protect_original_rows

class RepairRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=pd.read_parquet(OUT/'ASA_prediction_ledger.parquet')
        cls.ref=cls.rows[['row_position','truth']]
        cls.before=cls.rows[['row_position','pred_A0']].rename(columns={'pred_A0':'pred'})
        cls.protected=cls.rows.loc[cls.rows.pred_A0.eq(cls.rows.truth),['row_position']]

    def test_actual_new_errors_rejected_on_hypothetical_reference_guard(self):
        after=self.rows[['row_position','pred_R_decay']].rename(columns={'pred_R_decay':'pred'})
        r=protect_original_rows(self.ref,self.before,after,self.protected)
        self.assertFalse(r['repair_protection_passed'])
        self.assertEqual(r['new_errors_on_protected_original_rows'],1454)

    def test_identity_baseline_passes_without_claiming_quality(self):
        r=protect_original_rows(self.ref,self.before,self.before,self.protected)
        self.assertTrue(r['repair_protection_passed'])
        self.assertEqual(r['new_errors_on_protected_original_rows'],0)
        self.assertTrue(r['not_task_quality_or_runtime_permission'])

    def test_net_improvement_still_rejects_one_repair_regression(self):
        # Constructed candidate on actual official row population, not a new fitted model.
        after=self.ref.rename(columns={'truth':'pred'}).copy()
        row=self.rows[self.rows.pred_A0.eq(self.rows.truth)&self.rows.truth.eq(1)].row_position.iloc[0]
        after.loc[after.row_position.eq(row),'pred']=2
        r=protect_original_rows(self.ref,self.before,after,self.protected)
        self.assertGreater(r['all_reference_errors_repaired'],r['all_reference_correct_regressions'])
        self.assertEqual(r['new_errors_on_protected_original_rows'],1)
        self.assertFalse(r['repair_protection_passed'])

    def test_missing_original_row_cannot_hide_regression(self):
        with self.assertRaises(ValueError):
            protect_original_rows(self.ref,self.before,self.before.iloc[1:],self.protected)

    def test_prediction_supplied_truth_cannot_rewrite_reference(self):
        after=self.rows[['row_position','pred_R_decay']].rename(columns={'pred_R_decay':'pred'}).copy()
        after['truth']=after.pred
        r=protect_original_rows(self.ref,self.before,after,self.protected)
        self.assertEqual(r['new_errors_on_protected_original_rows'],1454)

    def test_protected_failure_cannot_be_forged_as_prior_success(self):
        bad=self.rows.loc[self.rows.pred_A0.ne(self.rows.truth),['row_position']].iloc[:1]
        with self.assertRaises(ValueError):protect_original_rows(self.ref,self.before,self.before,bad)

if __name__=='__main__':unittest.main()
