import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from experiment_review import (ROOT, ReviewError, read, review_plan, seal_run, require_run_seal,
                               require_checkpoint, align_predictions, evaluate_primary)

PLAN = ROOT / 'training/review_policy/next_batch_plan.json'


def small_case():
    y = np.array([0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2])
    ref = pd.DataFrame({'row_position': np.arange(12), 'truth': y,
                        'fold': [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                        'root': np.arange(100, 112), 'route': ['other'] * 4 + ['asa'] * 8})
    a = y.copy(); a[[4, 8]] = 2
    pred = pd.DataFrame({'row_position': np.arange(12), 'pred_A': a, 'pred_B': y})
    profile = {'expected_full_rows': 12, 'asa_error_limits': {'M': 1, 'S': 1, 'total': 1},
               'minimum_improved_folds': 2, 'protected_S_roots': [], 'required_full_classes': [0, 1, 2]}
    return ref, pred, profile


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read(PLAN)
        cls.reference = pd.read_parquet(ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',
                                        columns=['row_position', 'fold', 'root', 'truth'])
        cls.reference['route'] = 'asa'

    def test_current_plan_only_qualifies_trial(self):
        r = review_plan(self.plan)
        self.assertTrue(r['plan_review_passed'])
        self.assertFalse(r['quality_acceptance'])
        self.assertFalse(r['model_promoted'])

    def test_unsupported_early_tie_selector_is_rejected(self):
        p = copy.deepcopy(self.plan)
        p['selector'] = {'kind': 'K_U_F1', 'epoch': 15, 'tie_policy': 'earliest'}
        r = review_plan(p)
        self.assertIn('budget_and_candidate_consistent', r['violations'])

    def test_unhandled_risk_or_incomplete_monitor_blocks_plan(self):
        for change in ('risk', 'monitor'):
            p = copy.deepcopy(self.plan)
            if change == 'risk': p['risk_actions'].pop('LOCAL_MECHANISM')
            else: p['monitor'].remove('per_class_checkpoint_CE')
            self.assertFalse(review_plan(p)['plan_review_passed'])

    def test_posthoc_selection_or_fold_specific_method_rejected(self):
        for key, value in [('selection_labels', 'outer_answers'), ('best_seed_selection', True),
                           ('fold_specific_method_override', True), ('confirmation_condition', 'lower_training_CE')]:
            p = copy.deepcopy(self.plan); p[key] = value
            self.assertFalse(review_plan(p)['plan_review_passed'])

    def test_25_is_scoped_not_a_permanent_training_constant(self):
        p = copy.deepcopy(self.plan); p['training_epochs'] = 30; p['selector']['epoch'] = 30
        self.assertTrue(review_plan(p)['plan_review_passed'])
        require_checkpoint(p, {'status': 'fit_executed', 'completed_epochs': 30, 'prediction_epoch': 30})
        with self.assertRaises(ReviewError):
            require_checkpoint(p, {'status': 'fit_executed', 'completed_epochs': 30, 'prediction_epoch': 25})

    def test_real_V116_selected15_rejected_despite_full25_fit(self):
        fit = read(ROOT / 'artifacts/v116_nested_selection_20260929/outer_fold1/fit.json')
        with self.assertRaises(ReviewError):
            require_checkpoint(self.plan, {'status': fit['status'], 'completed_epochs': fit['epochs'], 'prediction_epoch': fit['selected_epoch']})

    def test_real_collateral_rows_cannot_be_omitted(self):
        m = pd.read_parquet(ROOT / 'artifacts/v116_nested_selection_20260929/inner_split_manifest.parquet')
        d = m[m.outer_fold.eq(1) & m.role.isin(['K', 'U', 'validation_collateral'])].copy()
        self.assertEqual(int(d.role.eq('validation_collateral').sum()), 6936)
        d['route'] = 'asa'; scored = d[d.role.isin(['K', 'U'])][['row_position']].copy()
        scored['pred_A'] = 1; scored['pred_B'] = 1
        with self.assertRaises(ReviewError): align_predictions(d, scored)

    def test_real_V113_total_gain_cannot_hide_M_regression(self):
        d = pd.read_parquet(ROOT / 'artifacts/v113_case_training_20260929/OOF_ASA_decisions.parquet')
        p = d.rename(columns={'A_N1_prediction': 'pred_A', 'B_case_prediction': 'pred_B'})
        r = evaluate_primary(self.reference, p, self.plan['quality_profile'])
        self.assertEqual((r['ASA']['A']['1']['missed'], r['ASA']['B']['1']['missed']), (318, 370))
        self.assertEqual(r['paired_changes']['1']['regressed'], 52)
        self.assertFalse(r['gates']['paired_M_S_protected'])
        self.assertFalse(r['gates']['multiple_folds_improve'])
        self.assertFalse(r['primary_quality_passed'])

    def test_real_V116_M_gain_cannot_hide_S_regression(self):
        d = pd.read_parquet(ROOT / 'artifacts/v116_nested_selection_20260929/OOF_ASA_decisions.parquet')
        p = d.rename(columns={'A_epoch25': 'pred_A', 'B_selected': 'pred_B'})
        r = evaluate_primary(self.reference, p, self.plan['quality_profile'])
        self.assertEqual(r['paired_changes']['2']['regressed'], 2542)
        self.assertEqual(r['paired_changes']['1']['repaired'], 24)
        self.assertFalse(r['gates']['paired_M_S_protected'])
        self.assertFalse(r['primary_quality_passed'])

    def test_99_percent_all_normal_still_fails(self):
        ref, pred, profile = small_case()
        extra = pd.DataFrame({'row_position': np.arange(12, 1012), 'truth': 0, 'fold': 0,
                              'root': np.arange(1000, 2000), 'route': 'other'})
        ref = pd.concat([ref, extra], ignore_index=True)
        p = ref[['row_position']].copy(); p['pred_A'] = ref.truth; p['pred_B'] = 0
        profile['expected_full_rows'] = len(ref)
        self.assertGreater((ref.truth == 0).mean(), .99)
        r = evaluate_primary(ref, p, profile)
        self.assertFalse(r['gates']['paired_M_S_protected'])
        self.assertFalse(r['primary_quality_passed'])

    def test_missing_class_is_unknown_not_perfect(self):
        ref, pred, profile = small_case()
        ref = ref[ref.truth.ne(2)]; pred = pred[pred.row_position.isin(ref.row_position)]
        profile['expected_full_rows'] = len(ref)
        r = evaluate_primary(ref, pred, profile)
        self.assertIsNone(r['full_task']['B']['2']['recall'])
        self.assertFalse(r['gates']['all_required_classes_present'])
        self.assertFalse(r['primary_quality_passed'])

    def test_forged_truth_duplicate_or_missing_predictions_rejected(self):
        ref, pred, _ = small_case()
        fake = pred.copy(); fake['truth'] = ref.truth; fake.loc[4, 'truth'] = 0
        for bad in (fake, pd.concat([pred, pred.iloc[:1]]), pred.iloc[:-1], pred.assign(pred_B=np.nan)):
            with self.assertRaises(ReviewError): align_predictions(ref, bad)

    def test_positive_control_can_pass_primary_but_not_promote(self):
        ref, pred, profile = small_case()
        r = evaluate_primary(ref, pred.iloc[::-1], profile)
        self.assertTrue(r['primary_quality_passed'], r['gates'])
        self.assertFalse(r['model_promoted'])

    def test_actual_seal_rejects_changed_trainer_and_changed_plan(self):
        parent = (ROOT / 'artifacts/v119_review_constraints_20260929').resolve()
        with tempfile.TemporaryDirectory(prefix='seal_test_', dir=parent) as tmp:
            p = Path(tmp).resolve(); self.assertTrue(p.is_relative_to(parent))
            trainer = p / 'trainer.py'; trainer.write_text('# inert test fixture, never executed\n', encoding='utf-8')
            plan = p / 'plan.json'; plan.write_text(json.dumps(self.plan), encoding='utf-8')
            seal = p / 'seal.json'; seal_run(plan, trainer, [], seal)
            require_run_seal(seal, trainer)
            trainer.write_text('# changed inert fixture\n', encoding='utf-8')
            with self.assertRaises(ReviewError): require_run_seal(seal, trainer)
            trainer.write_text('# inert test fixture, never executed\n', encoding='utf-8')
            require_run_seal(seal, trainer)
            edited = copy.deepcopy(self.plan); edited['training_epochs'] = 15
            plan.write_text(json.dumps(edited), encoding='utf-8')
            with self.assertRaises(ReviewError): require_run_seal(seal, trainer)
            with self.assertRaises(ReviewError): seal_run(plan, trainer, [], seal)


if __name__ == '__main__':
    unittest.main()
