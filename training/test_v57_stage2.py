import unittest
import numpy as np
import pandas as pd
import run_v54 as v
import run_v57_stage2 as s
import v57_applicability as m

class IsolatedConditionTests(unittest.TestCase):
    def test_activation_uses_only_observation(self):
        x = pd.DataFrame({'src_role': ['outside', v.MISSING, 'outside', v.MISSING], 'dst_role': ['dmz', 'dmz', v.MISSING, v.MISSING]})
        np.testing.assert_array_equal(s.active(x, 'no_src_role'), [False, True, False, False])
        np.testing.assert_array_equal(s.active(x, 'no_dst_role'), [False, False, True, False])
        x['route'] = ['wrong'] * 4
        x['label_index'] = [2] * 4
        np.testing.assert_array_equal(s.active(x, 'no_src_role'), [False, True, False, False])

    def test_complete_prediction_invariant_despite_large_adapter(self):
        x = pd.DataFrame([{**{f: v.MISSING for f in v.FIELDS}, 'transport_protocol': 'udp', 'src_role': 'dmz', 'dst_role': 'outside', 'action': 'deny', 'outcome': 'blocked'}])
        xx = x.copy(); xx['src_role'] = v.MISSING
        design = m.fit_design(xx, [1])
        weights = np.ones((len(design['feature_keys']), 3)) * [0, 50, -50]
        bundle = {'design': design, 'weights': weights}
        parts = [pd.DataFrame({'scenario': [view], 'pred': [2], 'p_0': [.01], 'p_1': [.09], 'p_2': [.90]}) for view in v.VIEWS]
        baseline = pd.concat(parts, ignore_index=True)
        pred = s.apply_heads(baseline, {'no_src_role': bundle}, x)
        np.testing.assert_array_equal(pred.loc[pred.scenario == 'full', ['pred', 'p_0', 'p_1', 'p_2']], baseline.loc[baseline.scenario == 'full', ['pred', 'p_0', 'p_1', 'p_2']])
        self.assertEqual(pred.loc[pred.scenario == 'no_src_role', 'pred'].iloc[0], 1)
        self.assertEqual(pred.loc[pred.scenario == 'no_dst_role', 'pred'].iloc[0], 2)

if __name__ == '__main__':
    unittest.main()
