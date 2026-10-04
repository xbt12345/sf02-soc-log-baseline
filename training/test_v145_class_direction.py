"""Projection math and the real failed finite-step qualification remain explicit."""
import unittest
import numpy as np
import torch
from v145_class_direction import project_vector,DESCENT_SLACK
from v142_runtime import ROOT,read,sha


class Tests(unittest.TestCase):
    def test_projection_matches_independent_one_active_constraint_solution(self):
        v=torch.tensor([1.,-1.,1.],dtype=torch.float64)
        constraints=[torch.tensor([1.,0.,0.],dtype=torch.float64),torch.tensor([0.,1.,0.],dtype=torch.float64)]
        actual,receipt=project_vector(v,constraints)
        reference=torch.tensor([1/np.sqrt(3),DESCENT_SLACK,1/np.sqrt(3)],dtype=torch.float64);reference/=reference.norm()
        self.assertLess(float((actual-reference).abs().max()),1e-12)
        self.assertEqual(receipt['active_constraints'],[1]);self.assertGreater(receipt['class_M_cosine'],0);self.assertGreater(receipt['class_S_cosine'],0)

    def test_incompatible_strict_descent_is_rejected_without_relaxation(self):
        with self.assertRaisesRegex(ValueError,'No simultaneous'):
            project_vector(torch.tensor([-1.,0.],dtype=torch.float64),[torch.tensor([1.,0.],dtype=torch.float64),torch.tensor([0.,1.],dtype=torch.float64)])

    def test_real_functional_trials_preserve_classification_but_fail_registered_risk_gate(self):
        a=read(ROOT/'artifacts/v145_class_direction_qualification_20261001/qualification.json')
        self.assertFalse(a['qualification_passed']);self.assertEqual(len(a['failures']),6)
        self.assertEqual((a['new_classifier_fits'],a['new_optimizer_steps'],a['new_mutable_parameter_updates'],a['fixed_functional_parameter_evaluations']),(0,0,0,6))
        self.assertTrue(a['baseline_guard']['passed']);self.assertTrue(all(v['passed'] for v in a['fixed_trial_guards'].values()))
        for p,h in a['evidence_bindings'].items():self.assertEqual(sha(ROOT/p),h)
        for r in a['folds']:
            self.assertLess(r['original_CE_gradient_recomposition_max_gap'],1e-12);self.assertTrue(r['parameter_identity_unchanged'])
            self.assertEqual(np.array(r['class_gradient_mass_seen']).sum(0).tolist(),r['original_class_mass_seen'])
            for v in r['directions']:
                self.assertTrue(v['projected_nonzero']);self.assertTrue(v['actual_TRAIN_classification']['mastered'])
                self.assertEqual(v['actual_TRAIN_classification']['new_errors_vs_start'],0)
                self.assertFalse(v['strict_finite_class_risk_descent']);self.assertLess(v['objective_descent_derivative'],0)
                self.assertTrue(all(z<0 for z in v['class_descent_derivatives']))


if __name__=='__main__':unittest.main()
