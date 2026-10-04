"""Actual loss decrease cannot bypass classifier guard; rollback and plan risks."""
import unittest,torch
from v146_optimizer import acceptable,assign,restore
from v146_runtime import review,adversaries


class Tests(unittest.TestCase):
    def test_backtracking_rejects_lower_loss_wrong_class_and_preserves_base(self):
        p=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64));base=(p.detach().clone(),);direction=(torch.ones_like(p),);accepted=None
        for step in [5.,2.5,1.25]:
            assign((p,),base,direction,step);loss=float(p.detach().square().sum());guard=float(p.detach())>0
            ok=acceptable(4.,loss,step,4.,guard,1e-4)
            if ok:accepted=step;break
            restore((p,),base);self.assertTrue(torch.equal(p.detach(),base[0]))
        self.assertEqual(accepted,1.25);self.assertEqual(float(p.detach()),.75)
        self.assertFalse(acceptable(4.,.25,2.5,4.,False,1e-4))

    def test_finite_class_probability_changes_do_not_impose_class_CE_license(self):
        self.assertTrue(acceptable(1.,.99,.01,1.,True,1e-4))
        self.assertFalse(acceptable(1.,1.01,.01,1.,True,1e-4))
        self.assertFalse(acceptable(1.,1.,1e-10,1e-20,True,1e-4))
        self.assertFalse(acceptable(1.,.99,.01,1.,False,1e-4))

    def test_prospective_policy_rejects_observed_risk_regressions(self):
        p=review();r=adversaries(p);self.assertEqual(len(r),11);self.assertTrue(all(v['rejected'] for v in r))


if __name__=='__main__':unittest.main()
