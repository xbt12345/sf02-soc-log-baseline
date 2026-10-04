"""Adversarial cases for source support and exclusion of contradictory auxiliary labels."""
import unittest
import pandas as pd
from audit_v63_support import auxiliary_pool, conflict_summary


def rows(items):
    return pd.DataFrame([{'role': 'fit', 'behavior': 'b', 'group': g, 'single': v, 'label': y}
                         for g, v, y in items])


class SupportTests(unittest.TestCase):
    def test_repeated_rows_cannot_supply_independent_sources(self):
        d = rows([(1, 11, 1)] * 100 + [(2, 21, 2)] * 100)
        out, enabled = auxiliary_pool(d)
        self.assertEqual(len(out), 200)
        self.assertFalse(enabled)
        self.assertFalse(out.aux_eligible.any())

    def test_conflict_is_removed_from_auxiliary_only(self):
        d = rows([(i, i, 1) for i in range(1, 5)] + [(i, i, 2) for i in range(5, 9)] +
                 [(20, 100, 1), (21, 100, 2)])
        out, enabled = auxiliary_pool(d)
        self.assertEqual(len(out), len(d))
        self.assertEqual(enabled, {'b'})
        self.assertTrue(out[out.single.ne(100)].aux_eligible.all())
        self.assertFalse(out[out.single.eq(100)].aux_eligible.any())
        self.assertEqual(conflict_summary(out, 'single')['retrospective_minimum_errors'], 1)

    def test_same_input_across_three_sources_is_not_diverse_support(self):
        d = rows([(i, 1, 1) for i in range(1, 4)] + [(i, 2, 2) for i in range(4, 7)])
        out, enabled = auxiliary_pool(d)
        self.assertEqual(enabled, {'b'})
        self.assertFalse(out.aux_eligible.any())

    def test_source_with_another_view_can_still_supply_a_positive(self):
        d = rows([(1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 3, 1)] +
                 [(i, i, 2) for i in range(4, 7)])
        out, _ = auxiliary_pool(d)
        self.assertEqual(int(out[out.group.eq(1)].diverse_positive_sources.iloc[0]), 2)

    def test_selection_rows_fail_closed(self):
        d = rows([(1, 1, 1)])
        d.loc[0, 'role'] = 'selection'
        with self.assertRaises(ValueError):
            auxiliary_pool(d)


if __name__ == '__main__':
    unittest.main()
