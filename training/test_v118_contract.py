"""Real historical counterexamples, not a test that rewards the old tie policy."""
import json
import unittest
from pathlib import Path
import pandas as pd
from v118_training_contract import ContractViolation, require_terminal_checkpoint, require_complete_population, replication_allowed

ROOT = Path(__file__).resolve().parents[1]
V116 = ROOT / 'artifacts/v116_nested_selection_20260929'


class ContractTests(unittest.TestCase):
    def test_real_v116_early_selected_candidate_rejected_despite_completed_fit(self):
        fit = json.loads((V116 / 'outer_fold1/fit.json').read_text(encoding='utf-8'))
        self.assertEqual(fit['epochs'], 25)
        self.assertEqual(fit['selected_epoch'], 15)
        with self.assertRaises(ContractViolation):
            require_terminal_checkpoint(fit['epochs'], fit['selected_epoch'], fit['status'])
        require_terminal_checkpoint(fit['epochs'], 25, fit['status'])

    def test_real_K_U_scoring_cannot_omit_validation_collateral(self):
        m = pd.read_parquet(V116 / 'inner_split_manifest.parquet')
        d = m[m.outer_fold.eq(1)]
        expected = d.loc[d.role.isin(['K', 'U', 'validation_collateral']), 'row_position']
        old = d.loc[d.role.isin(['K', 'U']), 'row_position']
        self.assertEqual(len(expected) - len(old), 6936)
        with self.assertRaises(ContractViolation):
            require_complete_population(expected, old)
        require_complete_population(expected, expected.iloc[::-1])

    def test_truncated_or_paused_fit_cannot_be_reported_as_complete(self):
        with self.assertRaises(ContractViolation):
            require_terminal_checkpoint(15, 15, 'fit_executed')
        with self.assertRaises(ContractViolation):
            require_terminal_checkpoint(25, 25, 'failed')

    def test_total_improvement_cannot_override_M_or_source_failure(self):
        gates = dict.fromkeys(['complete_six_paired_fits', 'population_and_identity_verified',
            'ASA_M_errors_at_most_318', 'ASA_S_errors_at_most_2094', 'ASA_total_errors_at_most_2170',
            'M_S_not_worse_than_fresh_A', 'two_folds_improve', 'S_outside_top3_not_worse',
            'full_each_class_recall_and_F1_not_worse'], True)
        self.assertTrue(replication_allowed(gates))
        for key in ('M_S_not_worse_than_fresh_A', 'two_folds_improve', 'S_outside_top3_not_worse'):
            changed = dict(gates, **{key: False}, lower_training_CE=True, lower_gradient_variance=True)
            self.assertFalse(replication_allowed(changed))

    def test_unmeasured_results_and_duplicate_scoring_are_rejected(self):
        with self.assertRaises(ContractViolation):
            replication_allowed({'lower_training_CE': True})
        with self.assertRaises(ContractViolation):
            require_complete_population([1, 2], [1, 1, 2])


if __name__ == '__main__':
    unittest.main()
