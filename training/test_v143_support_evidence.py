"""Actual support evidence retains correct controls, masks and legal root isolation."""
import unittest
import json
import numpy as np
import pandas as pd
from v143_fine_support_audit import OUT
from v142_runtime import ROOT,read,sha


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=pd.read_parquet(OUT/'fine_support_population.parquet');cls.a=read(OUT/'audit.json')

    def test_whole_official_population_and_bound_sources(self):
        for path,digest in self.a['source_sha256'].items():self.assertEqual(sha(ROOT/path),digest)
        y=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
        self.assertEqual(len(y),2056871);self.assertEqual(len(self.d),112807)
        self.assertFalse(self.d.row_position.duplicated().any());self.assertTrue(np.array_equal(self.d.truth,y[self.d.row_position]))
        self.assertEqual(int(self.d.wrong.sum()),3543);self.assertEqual(int((self.d.truth.eq(2)&self.d.wrong).sum()),1633)
        self.assertEqual((self.a['new_classifier_fits'],self.a['new_parameter_updates']),(0,0))

    def test_destination_support_comes_only_from_other_source_roots(self):
        # Independent source-root exclusion, rather than trusting the recorded fold selector.
        d=self.d
        for f in range(3):
            q=d[d.fold.eq(f)];roots=set(q.root)
            train=d[~d.root.isin(roots)&d.parameter_observed]
            self.assertTrue(train.fold.ne(f).all())
            for c in [1,2]:
                count=train[train.truth.eq(c)].groupby('destination_key').agg(rows=('row_position','size'),roots=('root','nunique'))
                for field in ['rows','roots']:
                    actual=q.destination_key.map(count[field]).fillna(0).astype(int).to_numpy()
                    self.assertTrue(np.array_equal(actual,q[f'destination_{c}_{field}']))
        s=d[d.truth.eq(2)&d.wrong]
        self.assertTrue(s.behavior_bucket.eq('multiple_same_sources').all())
        self.assertTrue(d.loc[~d.parameter_observed,'destination_bucket'].eq('unobserved_parameter_not_fine_support').all())

    def test_strict_pair_absence_does_not_erase_partial_support_or_icmp(self):
        g=pd.read_parquet(OUT/'legal_full_fact_positive_groups.parquet')
        self.assertTrue(g[g.scope.eq('known_both_transport_ports')].truth.eq(1).all())
        s=g[g.truth.eq(2)];self.assertEqual(s.groupby('training_role').size().tolist(),[13,4,13])
        self.assertTrue(s.key.map(lambda k:json.loads(k).get('src_port_fixed')==65536).all())
        detail=read(OUT/'support_detail.json');a=pd.DataFrame(detail['protocol_availability'])
        q=a[a.truth.eq(2)&a.parameter_observed&~a.record_src_port_observed]
        self.assertEqual(q.protocol.tolist(),['icmp']);self.assertEqual(int(q.population.sum()),719)
        for r in self.a['cross_format_support']:
            self.assertGreater(r['ASA_matching_M_rows'],r['ASA_matching_S_rows']);self.assertEqual(r['exact_fine_semantic_support_rows'],0)
        self.assertFalse(self.a['second_issue_solved']);self.assertFalse(self.a['quality_acceptance'])


if __name__=='__main__':unittest.main()
