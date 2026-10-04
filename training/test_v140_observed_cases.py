"""Replay the actual lower-loss-but-regressing candidate and scoped guards."""
import unittest
import numpy as np
import pandas as pd
from v140_runtime import ROOT,OUT,OLD,read,sha
from v135_runtime import load_data,fit_context
from v140_retention_check import check


class Tests(unittest.TestCase):
    def test_actual_candidate_recounts_original_truth_and_cannot_close_issue(self):
        _,d=load_data();pure_total=new_total=repaired_total=0
        for fold in range(3):
            frame,_,pure,_,_=fit_context(d,fold)
            rows=pd.read_parquet(OUT/f'fold{fold}_E/endpoint_original_rows.parquet')
            receipt=read(OUT/f'fold{fold}_E/fit.json')
            self.assertEqual(sha(OUT/f'fold{fold}_E/endpoint_original_rows.parquet'),receipt['rows_sha256'])
            self.assertTrue(np.array_equal(rows.row_position,frame.row_position));self.assertTrue(np.array_equal(rows.truth,frame.truth))
            old=pd.read_parquet(OLD/f'fold{fold}_H_L/endpoint_original_rows.parquet')
            mask=pure[frame.local].astype(bool);y=frame.truth.to_numpy();after=rows.pred.to_numpy();before=old.pred.to_numpy()
            pure_total+=int((mask&(after!=y)).sum());new_total+=int((mask&(before==y)&(after!=y)).sum());repaired_total+=int((mask&(before!=y)&(after==y)).sum())
            initial=read(OUT/'preflight.json')['folds'][fold]['objectives']['E']['risk']
            self.assertLess(receipt['endpoint_losses']['original_ensemble_CE'],initial)
        self.assertEqual((pure_total,new_total,repaired_total),(14,6,8))
        self.assertFalse(pure_total==0 and new_total==0)
        self.assertFalse(read(OUT/'issue_decision.json')['issue_solved'])

    def test_new_scope_accepts_actual_control_rejects_actual_regression(self):
        allrows=pd.read_parquet(OUT/'all_training_role_ledger.parquet')
        good=check(allrows[allrows.arm.eq('C')]);bad=check(allrows[allrows.arm.eq('E')])
        self.assertTrue(good['passed']);self.assertFalse(bad['passed']);self.assertTrue(bad['V138']['passed'])
        self.assertEqual(bad['additional_V140'][0]['new_errors_on_protected_original_rows'],2)

    def test_actual_same_input_saturation_is_diagnostic_not_issue_success(self):
        folder=ROOT/'artifacts/v139_training_decision_review_20261001/v140_saturation_audit'
        audit=read(folder/'audit.json')
        for rel,value in audit['evidence_sha256'].items(): self.assertEqual(sha(ROOT/rel),value)
        self.assertEqual(sha(folder/'same_input_three_states.parquet'),audit['rows_sha256'])
        rows=pd.read_parquet(folder/'same_input_three_states.parquet')
        summaries={a['arm']:a for a in audit['summary']}
        for arm in ['V138','C','E']:
            r=rows[rows.arm.eq(arm)&rows.E_pred.ne(rows.truth)]
            self.assertEqual(int(r.original_rows.sum()),14)
            mean=float(np.average(r.gradient_norm_ratio,weights=r.original_rows))
            self.assertAlmostEqual(mean,summaries[arm]['weighted_logit_gradient_ratio'],places=12)
        self.assertLess(summaries['E']['weighted_logit_gradient_ratio'],.01)
        self.assertGreater(summaries['C']['weighted_logit_gradient_ratio'],.4)
        self.assertFalse(read(OUT/'issue_decision.json')['issue_solved'])


if __name__=='__main__':unittest.main()
