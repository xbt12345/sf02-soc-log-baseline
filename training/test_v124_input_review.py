"""Actual V124 counterexamples: new input is not an old batch experiment."""
import copy
import unittest

from v124_experiment_review import ROOT, read, review_plan
from v124_header import HEADER, transform


class HeaderTrialReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registered=read(ROOT/'training/review_policy/v124_header_trial.json')

    def test_real_plan_passes_but_is_not_quality(self):
        out=review_plan(self.registered)
        self.assertTrue(out['plan_review_passed'])
        self.assertFalse(out['quality_acceptance'])

    def test_old_batch_plan_cannot_impersonate_input_trial(self):
        old=read(ROOT/'training/review_policy/v120_next_batch_plan.json')
        with self.assertRaises((KeyError, ValueError)):
            review_plan(old)

    def test_no_early_checkpoint_or_label_mass_change(self):
        for field,value in [('selection_labels','outer_answers'),('preserve_input_class_mass',False)]:
            p=copy.deepcopy(self.registered);p[field]=value
            self.assertFalse(review_plan(p)['plan_review_passed'])
        p=copy.deepcopy(self.registered);p['selector']['epoch']=15
        self.assertFalse(review_plan(p)['plan_review_passed'])

    def test_no_b_dedup_or_source_fold_override(self):
        for field,value in [('B_re_deduplication',True),('old_local_sampling_units_preserved',False)]:
            p=copy.deepcopy(self.registered);p['input_contract'][field]=value
            self.assertFalse(review_plan(p)['plan_review_passed'])
        p=copy.deepcopy(self.registered);p['fold_specific_method_override']=True
        self.assertFalse(review_plan(p)['plan_review_passed'])

    def test_nested_clock_and_body_boundary(self):
        body='USER-0010-0324 Deny udp src outside:1.2.3.4/1234 dst dmz:2.3.4.5/514'
        a='<164>Sep 13 2022 USER-1856: '+body
        b='<164>Jul 26 USER-9546 ORG-CRED-25677: '+body
        assert HEADER.match(a) and HEADER.match(b)
        self.assertEqual(transform(a)[0],transform(b)[0])
        with self.assertRaises(ValueError):
            transform('<164>Jul 26 05:59:56: unrelated '+body)


if __name__=='__main__':unittest.main()
