"""Representation isolation and adversarial native-field boundaries."""
import json
import unittest
import numpy as np
import v41_core as c
import v41_native_auth as n


def native(body, data=None, root='message'):
    return json.dumps({'event': {'provider': 'Microsoft-Windows-Security-Auditing', 'code': '4625'},
                       root: body, 'winlog': {'event_data': data or {}}})


BODY = 'An account failed to log on.\nSubject:\n Security ID: X\nFailure Reason: Unknown user name or bad password.\n Status: 0xC000006D\n Sub Status: 0xC0000064\nCaller Process Name: -\n'


class Isolation(unittest.TestCase):
    def test_f_keeps_aliases_and_changes_only_finite_bits(self):
        f = [{'auth_result': 'failure', 'outcome': 'failure', 'src_port_fixed': x} for x in (0, 1, 65535, 65536)]
        f += [{'authentication_interaction': 'no_response', 'response': 'missing'}, {'status': '0xc000006d'}]
        b = c.prior.SemanticFacts().fit(f); t = c.FiniteOnlyFacts().fit(f)
        check = c.isolation_check('F', b.transform(f), t.transform(f), b.names(), 0)
        self.assertEqual(check['unauthorized_changed_columns'], 0)
        j = list(b.names()).index('outcome=failure')
        np.testing.assert_array_equal(b.transform(f)[:, j].toarray(), t.transform(f)[:, j].toarray())

    def test_d_keeps_all_finite_codes_and_conflicting_aliases(self):
        f = [{'auth_result': 'failure', 'outcome': 'failure', 'src_port_fixed': 65536},
             {'authentication_interaction': 'no_response', 'response': 'missing', 'status': '0xc000006d'},
             {'auth_result': 'failure', 'outcome': 'success', 'src_port_fixed': 0}]
        b = c.prior.SemanticFacts().fit(f); t = c.DeduplicatedFacts().fit(f)
        c.isolation_check('D', b.transform(f), t.transform(f), b.names(), 0)
        self.assertEqual(c.deduplicate(f[2]), f[2])

    def test_all_ports_unique_and_missing_distinct(self):
        bits = c.previous.observed_bits(list(range(65537)), 'src_port_fixed')
        code = bits.dot(2.0**np.arange(17)).astype(np.int64)
        np.testing.assert_array_equal(code[:-1], np.arange(1, 65537))
        self.assertEqual(code[-1], 0)
        self.assertEqual(len(np.unique(code)), 65537)


class NativeBoundary(unittest.TestCase):
    def test_escaped_case_and_redacted_root(self):
        for body in [BODY, BODY.replace('Status:', 'STATUS:').replace('0xC', '0Xc'), BODY.replace(' Status:', '\tStatus:\t')]:
            p = n.extract(native(body, root='CRED-12345'))
            self.assertEqual(p['facts'], {'status': '0xc000006d', 'substatus': '0xc0000064'})

    def test_consistent_carriers_count_once(self):
        p = n.extract(native(BODY, {'Status': '0xc000006d', 'SubStatus': '0xc0000064'}))
        self.assertEqual(len(p['facts']), 2); self.assertEqual(len(p['carriers']), 4)

    def test_conflict_is_explicit_fallback(self):
        p = n.extract(native(BODY, {'Status': '0xc000006a'}))
        self.assertEqual(p['state'], 'conflict'); self.assertFalse(p['facts'])

    def test_quoted_or_process_example_is_not_native_observation(self):
        for body in [BODY.replace('Status: 0xC000006D', 'Status: "0xC000006D"').replace('Sub Status: 0xC0000064', 'Sub Status: "0xC0000064"'),
                     'An account failed to log on.\nCaller Process Name: example\n'+BODY.split('Subject:\n')[1],
                     BODY.replace('Failure Reason:', 'Quoted example:')]:
            self.assertFalse(n.extract(native(body))['facts'])

    def test_request_and_non_auth_are_excluded(self):
        raw = json.dumps({'event': {'provider': 'Microsoft-Windows-Security-Auditing', 'code': '4625'}, 'winlog': {}, 'request': {'message': BODY}})
        self.assertFalse(n.extract(raw)['facts'])
        self.assertFalse(n.extract(native(BODY).replace('4625', '4688'))['facts'])
        self.assertFalse(n.extract(native(BODY).replace('An account failed to log on.', 'An application completed an operation.'))['facts'])

    def test_damaged_parent_only_body_trusted(self):
        raw = native(BODY, {'Status': '0xc000006d'})[:-1].replace('"winlog": {', '"winlog": {"broken": [')
        p = n.extract(raw)
        self.assertEqual(p['facts']['status'], '0xc000006d')
        self.assertTrue(all(v['carrier'] == 'native_rendered_line' for v in p['carriers']))

    def test_native_code_not_double_encoded_in_text(self):
        b = {'text': 'status: 0xc000006d sub status: 0xc0000064 command 0xc0000064', 'facts': {'outcome': 'failure'}}
        p = n.repair(b, n.extract(native(BODY)))
        self.assertEqual(p['text'], 'status: native_code sub status: native_code command 0xc0000064')
        self.assertEqual(p['facts']['outcome'], 'failure')


if __name__ == '__main__': unittest.main()
