"""Safety/meaning tests for new body boundaries and semantic rules, not quality."""
import json
import unittest
import numpy as np
import v39_core as core


def auth(result='success', reason='valid_passcode'):
    return json.dumps({'application': {}, 'user': {}, 'result': result, 'reason': reason, 'txid': 'event-x', 'timestamp': 1})


class MeaningTests(unittest.TestCase):
    def test_ICMP_semantics_retain_code_distinction(self):
        a = {'transport_protocol': 'icmp', 'icmp_type': 3, 'icmp_code': 13}
        b = dict(a, icmp_code=3)
        x = core.semantic('', a, 'asa')['facts']; y = core.semantic('', b, 'asa')['facts']
        self.assertEqual(x['icmp_message'], y['icmp_message'])
        self.assertNotEqual(x['icmp_unreachable'], y['icmp_unreachable'])
        enc = core.SemanticFacts().fit([x, y])
        self.assertGreater((enc.transform([x])-enc.transform([y])).nnz, 0)

    def test_ICMPv6_does_not_receive_IPv4_meaning(self):
        self.assertNotIn('icmp_message', core.semantic('', {'transport_protocol': 'icmp6', 'icmp_type': 3}, 'asa')['facts'])

    def test_policy_allow_not_authentication_success(self):
        p = core.prepare_message(auth('allow'))['facts']
        self.assertEqual(p['policy_decision'], 'allowed')
        self.assertNotIn('auth_result', p)
        self.assertNotIn('outcome', p)

    def test_unanswered_is_not_invalid_password_or_guessed_failure(self):
        p = core.prepare_message(auth('USER-9999', 'no_response'))['facts']
        self.assertEqual(p['authentication_interaction'], 'no_response')
        self.assertNotIn('credential_check', p)
        self.assertNotIn('auth_result', p)

    def test_explicit_failure_and_invalid_credential_survive(self):
        p = core.prepare_message(auth('failure', 'invalid_passcode'))['facts']
        self.assertEqual((p['auth_result'], p['credential_check']), ('failure', 'invalid'))

    def test_outer_fields_not_input(self):
        raw = auth()
        self.assertEqual(core.prepare_record({'message_sanitized': raw}), core.prepare_record({'message_sanitized': raw, 'label_binary': 'malicious', 'product_name': 'X', 'timestamp': '2099', 'src_ip': '1.2.3.4'}))

    def test_absolute_date_removed_duration_retained(self):
        p = core.semantic('at 2124-07-02t11:11:17z duration 123:02:03', {'duration_seconds': 2}, 'windows_message')
        self.assertNotIn('2124', p['text'])
        self.assertIn('123:02:03', p['text'])
        self.assertEqual(p['facts']['duration_seconds'], 2)

    def test_redacted_scheduled_boundary_invariant(self):
        a = '<startboundary>entity:04:44z</startboundary><interval>pt30m</interval>'
        b = '<startboundary>2099-01-01t00:00:00z</startboundary><interval>pt30m</interval>'
        self.assertEqual(core.semantic(a, {}, 'windows_message')['text'], core.semantic(b, {}, 'windows_message')['text'])
        self.assertIn('pt30m', core.semantic(a, {}, 'windows_message')['text'])

    def test_group_keeps_interface_but_ignores_wrapper(self):
        body = 'Deny icmp src outside:192.0.2.1 dst dmz-1:192.0.2.2 (type 3, code 13) by acl-group "outside_acl_in" [0x0, 0x0]'
        a = '<164>Jul 23 2022 09:02:26: host ' + body
        b = '<164>Jul 24 2023 10:02:26: newhost ' + body
        c = b.replace('dmz-1', 'dmz-2')
        self.assertEqual(core.body_identity(a, 'asa', []), core.body_identity(b, 'asa', []))
        self.assertNotEqual(core.body_identity(a, 'asa', []), core.body_identity(c, 'asa', []))
        self.assertEqual(core.prepare_message(a), core.prepare_message(c))

    def test_auth_clock_does_not_split_body(self):
        a = auth(); b = a.replace('"timestamp": 1', '"timestamp": 999')
        self.assertEqual(core.body_identity(a, 'authentication', []), core.body_identity(b, 'authentication', []))
        self.assertNotEqual(core.body_identity(a, 'authentication', []), core.body_identity(a.replace('event-x', 'event-y'), 'authentication', []))

    def test_empty_has_no_shared_event_identity(self):
        self.assertNotEqual(core.body_identity('', 'empty', [], 1), core.body_identity('', 'empty', [], 2))

    def test_unrecognized_bodies_not_coarsened(self):
        self.assertNotEqual(core.body_identity('first 1', 'unsupported', []), core.body_identity('first 2', 'unsupported', []))

    def test_vocabulary_is_fit_only(self):
        e = core.learning.FrequencyTfidf().fit(['known event'], np.array([3]))
        self.assertEqual(e.transform(['totallyunseenholdout']).nnz, 0)
        self.assertEqual(e.fit_rows, 3)


if __name__ == '__main__':
    unittest.main()
