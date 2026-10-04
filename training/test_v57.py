import unittest
import numpy as np
import pandas as pd
import v57_applicability as m
import run_v54 as v

class ApplicableEvidenceTests(unittest.TestCase):
    def row(self, **values):
        return {**{field: v.MISSING for field in v.FIELDS}, 'action': 'deny', 'outcome': 'blocked', **values}

    def test_protocol_and_missing_contract(self):
        x = pd.DataFrame([self.row(transport_protocol='udp', icmp_type='3', icmp_code='13', src_role='dmz', dst_port_range='system'), self.row(transport_protocol='icmp', icmp_type='3', icmp_code='13', src_port_range='system')])
        k = m.keys(x)
        self.assertIsNone(k['field:icmp_type'][0])
        self.assertEqual(k['field:icmp_code'][1], '13')
        self.assertIsNone(k['field:src_port_range'][1])
        self.assertIsNone(k['field:src_role'][1])
        self.assertEqual(k['field:dst_port_range'][0], 'system')

    def test_known_facts_survive_novel_combination(self):
        fit = pd.DataFrame([self.row(transport_protocol='tcp', src_role='outside', dst_role='dmz'), self.row(transport_protocol='udp', src_role='dmz', dst_role='outside')])
        design = m.fit_design(fit, [1, 2])
        new = pd.DataFrame([self.row(transport_protocol='udp', src_role='outside', dst_role='dmz')])
        row = m.matrix(design, new).toarray()[0]
        active = [name for (name, value), z in zip(design['feature_keys'], row) if z]
        self.assertIn('field:transport_protocol', active)
        self.assertIn('field:src_role', active)
        self.assertNotIn('exact', active)
        self.assertFalse(any(name.startswith('field:') and value == v.MISSING for name, value in design['feature_keys']))

    def test_all_view_class_guard_catches_hidden_regression(self):
        parts = []
        for view in v.VIEWS:
            parts.append(pd.DataFrame({'scenario': view, 'row_position': [1, 2, 3], 'label_index': [0, 1, 2], 'route': ['asa_acl', 'asa', 'asa'], 'eligible_stress': True, 'pred': [0, 2, 2]}))
        base = pd.concat(parts, ignore_index=True)
        candidate = base.copy()
        candidate.loc[(candidate.scenario == 'full') & (candidate.label_index == 1), 'pred'] = 1
        candidate.loc[(candidate.scenario == 'no_src_role') & (candidate.label_index == 2), 'pred'] = 1
        gate = m.compare(base, candidate)
        self.assertFalse(gate['quality_passed'])
        self.assertTrue(gate['any_M_S_improvement'])
        self.assertEqual(len(gate['failures']), 1)

if __name__ == '__main__':
    unittest.main()
