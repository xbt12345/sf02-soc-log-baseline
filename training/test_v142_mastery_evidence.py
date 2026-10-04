"""Actual mastery and regression protections reject previously failed models."""
import unittest
import numpy as np
import pandas as pd
from v142_runtime import ROOT,OUT,read,sha
from v135_runtime import load_data,fit_context
from v142_retention_check import check
from v142_input_identity_audit import audit


class Tests(unittest.TestCase):
    def test_all_original_training_truth_and_conflict_floor(self):
        _,d=load_data();identity,_=audit();self.assertTrue(identity['all_actual_input_identities_recomputed'])
        delivery=read(OUT/'final_delivery.json');self.assertTrue(delivery['issue_solved']);self.assertFalse(delivery['quality_acceptance'])
        pure_errors=0;updates=0;evals=0
        for f in range(3):
            frame,c,pure,_,_=fit_context(d,f);r=read(OUT/f'fold{f}_S2/fit.json');rows=pd.read_parquet(OUT/f'fold{f}_S2/endpoint_original_rows.parquet')
            self.assertEqual(sha(OUT/f'fold{f}_S2/endpoint_original_rows.parquet'),r['rows_sha256']);self.assertTrue(np.array_equal(rows.row_position,frame.row_position));self.assertTrue(np.array_equal(rows.truth,frame.truth))
            error=rows.pred.ne(rows.truth).to_numpy();pure_errors+=int((error&pure[frame.local].astype(bool)).sum())
            self.assertEqual(int((error&rows.truth.eq(1).to_numpy()).sum()),0);self.assertEqual(int((error&rows.truth.eq(2).to_numpy()).sum()),[22,6,28][f])
            mix=frame.groupby(['canonical_key','truth']).size().unstack(fill_value=0);floor=int((mix.sum(1)-mix.max(1)).sum());self.assertEqual(floor,[22,6,28][f])
            updates+=r['accepted_updates'];evals+=r['full_gradient_evaluations']
        self.assertEqual((pure_errors,evals,updates),(0,600,295))

    def test_mastered_scope_accepts_current_rejects_old_control(self):
        current=pd.read_parquet(OUT/'all_training_role_ledger.parquet');self.assertTrue(check(current)['passed'])
        old=pd.read_parquet(ROOT/'artifacts/v140_ensemble_training_round2_20261001/all_training_role_ledger.parquet');r=check(old[old.arm.eq('C')]);self.assertFalse(r['passed'])
        self.assertTrue(r['previous']['passed']);self.assertEqual(sum(v['new_errors_on_protected_original_rows'] for v in r['mastered_TRAIN']),12)

    def test_complete_population_and_cache_free_function_replayed(self):
        v=read(OUT/'verification.json');self.assertEqual(v['official_original_rows'],2056871)
        for a in v['audits']:
            self.assertEqual(a['cache_free_numeric_input_replay_count'],22546);self.assertLess(a['cache_free_probability_max_gap'],2e-6)
            self.assertEqual(a['actual_state_replays'],5);self.assertFalse(a['raw_new_message_preprocessor_external_validation'])
        reg=read(OUT/'verified_TRAIN_mastery_registry.json');self.assertEqual(reg['protected_correct_TRAIN_role_rows'],225558);self.assertEqual(reg['independent_protected_official_rows'],112779)
        self.assertFalse(reg['deployable'])


if __name__=='__main__':unittest.main()
