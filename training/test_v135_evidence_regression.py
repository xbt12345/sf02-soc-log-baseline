"""Historical execution counterexample and qualified real-input kernel replay."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,sha,check_bindings

class ActualEvidenceRegression(unittest.TestCase):
    def test_same_initial_state_is_not_a_complete_matched_control(self):
        root=ROOT/'artifacts/v134_stable_learning_trial_20260930'
        a=root/'fold0_R_const';b=root/'fold0_R_decay'
        meta=[json.loads((p/'started.json').read_text()) for p in [a,b]]
        self.assertEqual(meta[0]['initial_state_sha256'],meta[1]['initial_state_sha256'])
        self.assertEqual(meta[0]['seed'],meta[1]['seed'])
        rows=[pd.read_parquet(p/'epochs/epoch010_rows.parquet',columns=['row_position','truth','pred','p0','p1','p2']) for p in [a,b]]
        self.assertTrue(np.array_equal(rows[0].row_position,rows[1].row_position))
        self.assertTrue(np.array_equal(rows[0].truth,rows[1].truth))
        self.assertGreater(int(rows[0].pred.ne(rows[1].pred).sum()),0)
        self.assertGreater(float(np.abs(rows[0][['p0','p1','p2']].to_numpy()-rows[1][['p0','p1','p2']].to_numpy()).max()),.001)

    def test_kernel_qualification_is_bound_real_data_and_zero_updates(self):
        q=json.loads((ROOT/'artifacts/v135_kernel_qualification_20260930.json').read_text())
        check_bindings(q['source_sha256']);self.assertTrue(q['all_checks_passed'])
        self.assertEqual(q['optimizer_updates'],0)
        self.assertEqual({z['fold'] for z in q['records']},{0,1,2})
        self.assertEqual(len(q['records']),12)
        self.assertTrue(all(z['new_forward_bitwise_identical'] and z['new_backward_bitwise_identical'] and z['parameter_unchanged'] for z in q['records']))

if __name__=='__main__':unittest.main()
