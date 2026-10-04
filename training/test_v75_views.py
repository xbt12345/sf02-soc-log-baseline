import json
import unittest
from pathlib import Path
import numpy as np
from v75_views import view,byte_matrix,matrix_hashes


class PreservationTests(unittest.TestCase):
    def test_real_official_cases_roundtrip_and_unresolved_tail(self):
        p=Path(__file__).resolve().parents[1]/'artifacts/v73_full_task_20260921_r2/official_case_messages.json'
        cases=json.loads(p.read_text(encoding='utf-8'))
        for raw in cases.values():
            text,a=view(raw)
            self.assertEqual(''.join(raw[s:e] for s,e,_ in a['spans']),raw)
            self.assertTrue(text)
        self.assertIn('AND 1',view(cases['81090'])[0])

    def test_null_empty_and_whitespace_remain_distinguishable_in_ledger(self):
        self.assertTrue(view(None)[1]['is_null'])
        self.assertFalse(view('')[1]['is_null'])
        self.assertEqual(view('  \n')[0],'  \n')

    def test_assertion_and_known_identity_counterfactual(self):
        a='{"classification":"phish", "msg":"deny tcp", "sender":"user-1@example.net"}'
        b='{"classification":"benign", "msg":"deny tcp", "sender":"user-2@example.net"}'
        self.assertEqual(view(a)[0],view(b)[0])
        self.assertNotEqual(view(a)[1]['sha256'],view(b)[1]['sha256'])

    def test_bytes_have_no_oov_and_tail_is_not_truncated(self):
        rows=['汉字👋', 'x'*10000+'DENY', 'x'*10000+'ALLOW', '\\"A; AND 1=1', '']
        x=byte_matrix(rows)
        self.assertTrue((x.getnnz(axis=1)[:4]>0).all())
        self.assertNotEqual(matrix_hashes(x)[1],matrix_hashes(x)[2])
        self.assertEqual(x.getnnz(axis=1)[-1],0)

    def test_long_order_is_explicitly_a_model_limitation(self):
        # Different byte sequences, identical unigram/bigram counts. Prevents
        # falsely describing the model representation as sequence-lossless.
        a,b='abaca','acaba'
        self.assertNotEqual(a,b)
        np.testing.assert_array_equal(byte_matrix([a]).toarray(),byte_matrix([b]).toarray())

    def test_no_numeric_port_or_negation_deletion(self):
        text,_=view('deny udp dst dmz:10.1.2.3/1433; NOT an attack; duration=00:03:20')
        self.assertIn('/1433',text);self.assertIn('NOT an attack',text);self.assertIn('00:03:20',text)

    def test_vpc_and_native_clocks_cannot_be_residual_shortcuts(self):
        a='2 123456789012 eni-abc 10.1.1.1 10.2.2.2 1234 443 6 2 400 1600000000 1600000010 REJECT OK'
        b=a.replace('123456789012','999999999999').replace('1600000000','1800000000').replace('1600000010','1800000010')
        self.assertEqual(view(a)[0],view(b)[0])

    def test_all_metadata_ports_are_distinct_and_conflicts_do_not_overwrite(self):
        from v75_metadata import encode
        x,key,a=encode([str(i) for i in range(65536)],np.full(65536,65536))
        self.assertEqual(len(np.unique(key)),65536)
        self.assertEqual(a['observed'],65536)
        np.testing.assert_array_equal((x.toarray()[:,:16]*np.sqrt(18)).round().dot(2**np.arange(16)),np.arange(65536))
        _,_,a=encode(['443'],[80]);self.assertEqual(a['conflicting_observations'],1)
        a='<134>Original Address=10.1.1.1 1 2022-01-01T01:00:00Z HOST-0001 flows src=1.1.1.1 dst=2.2.2.2 protocol=tcp sport=1234 dport=443'
        b=a.replace('2022-01-01T01:00:00Z','2026-09-21T02:00:00Z').replace('HOST-0001','HOST-9999')
        self.assertEqual(view(a)[0],view(b)[0])


if __name__=='__main__':unittest.main()
