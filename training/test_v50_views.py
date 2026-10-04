import unittest
import numpy as np
from v50_views import partial_auth, training_views

class ContextCorruptionTests(unittest.TestCase):
    def test_preserves_behavior_and_does_not_rewrite_reason(self):
        f = dict(auth_result='failure', outcome='failure', response='missing',
                 authentication_interaction='no_response', attempt_count=7, duration_seconds=3,
                 src_port_fixed=80, product='x', timestamp=99)
        v = partial_auth(f)
        self.assertEqual(v['response'], 'missing')
        self.assertNotIn('credential_check', v)
        self.assertEqual((v['attempt_count'], v['duration_seconds']), (7, 3))
        self.assertNotIn('product', v); self.assertNotIn('timestamp', v)

    def test_unknown_or_conflicting_observation_not_augmented(self):
        self.assertIsNone(partial_auth({'outcome': 'failure'}))
        self.assertIsNone(partial_auth({'auth_result': 'success', 'outcome': 'failure'}))
        self.assertIsNone(partial_auth({'auth_result': 'success', 'outcome': 'success', 'credential_check': 'invalid'}))
        self.assertIsNone(partial_auth({'policy_decision': 'allowed'}))

    def test_conflicting_labels_retained_and_each_row_weight_conserved(self):
        f = [{'auth_result':'failure','outcome':'failure'}, {'action':'deny'}]
        v, x, y, w, origin, eligible = training_views(f, [0,0,1], [0,2,1])
        self.assertEqual(len(v),1)
        np.testing.assert_array_equal(y, np.array([0,2,1])[origin])
        np.testing.assert_array_equal(np.bincount(origin,weights=w), [1,1,1])
        self.assertEqual(set(y[x==2]), {0,2})

    def test_evaluation_only_view_not_added(self):
        f = [{'auth_result':'success','outcome':'success'},
             {'auth_result':'failure','outcome':'failure','credential_check':'invalid'}]
        v, *_ = training_views(f, [0], [0])
        self.assertEqual(v, [f[0]])

if __name__ == '__main__': unittest.main()
