import ast
from pathlib import Path
import unittest

import numpy as np

import soc_behavior_v2 as v2


class ParserTests(unittest.TestCase):
    def flags(self, message, **metadata):
        vector, detail = v2.parse(dict(metadata, message_sanitized=message))
        return dict(zip(v2.FEATURES, vector.tolist())), detail

    def test_python38_syntax(self):
        ast.parse(Path(v2.__file__).read_text(encoding="utf-8"), feature_version=(3, 8))

    def test_windows_structured_event_wins_over_explanatory_text(self):
        flags, _ = self.flags('USER-001 ::: {"agent":{"type":"winlogbeat"},"event":{"code":"4624"},"message":"An account failed to log on is another event."}')
        self.assertEqual(flags["auth_success"], 1)
        self.assertEqual(flags["auth_failure"], 0)

    def test_damaged_json_preserves_verified_readable_fields(self):
        flags, evidence = self.flags('USER-001 ::: {"agent":"winlogbeat","process":{"pid":CRED-999},"event":{"code":"4625"},"message":"An account failed to log on. Unknown user name or bad password."}')
        self.assertEqual(flags["auth_failure"], 1)
        self.assertEqual(flags["bad_credentials"], 1)
        self.assertIn("auth_failure", evidence["evidence"])

    def test_xml_json_authentication_equivalence(self):
        a, _ = self.flags('<Event><System><EventID>4625</EventID></System><EventData><Data Name="LogonType">3</Data></EventData></Event>')
        b, _ = self.flags('{"agent":"winlogbeat","event_id":4625,"LogonType":"3"}')
        self.assertEqual(a, b)

    def test_json_conflicting_codes_are_not_guessed(self):
        flags, _ = self.flags('{"agent":"winlogbeat","code":"4625","event_id":4624}')
        self.assertEqual(flags["auth_success"] + flags["auth_failure"], 0)

    def test_vpc_logging_status_is_not_traffic_outcome(self):
        flags, _ = self.flags('2 123456789012 eni-demo 192.0.2.1 198.51.100.2 54321 53 17 1 44 1720000000 1720000010 REJECT OK')
        self.assertEqual(flags["network_denied"], 1)
        self.assertEqual(flags["network_allowed"], 0)
        self.assertEqual(flags["service_dns"], 1)
        self.assertEqual(flags["protocol_udp"], 1)
        flags, _ = self.flags('2 123456789012 eni-demo - - - - - - - 1720000000 1720000010 - NODATA')
        self.assertEqual(flags["network_denied"] + flags["network_allowed"], 0)

    def test_partial_redacted_port_is_unknown_not_an_invented_number(self):
        flags, _ = self.flags('<164>Jul 26 12:13:14 host: Deny udp src inside:192.0.2.1/4321 dst outside:198.51.100.2/5CRED-003 by access-group rule')
        self.assertEqual(flags["network_denied"], 1)
        self.assertEqual(flags["protocol_udp"], 1)
        self.assertFalse(any(flags[name] for name in v2.CONTEXT if name.startswith("service_")))

    def test_no_response_and_invalid_passcode_remain_distinct(self):
        a, _ = self.flags('{"reason":"no_response","result":"denied"}')
        b, _ = self.flags('{"reason":"invalid_passcode","result":"denied"}')
        self.assertEqual(a["auth_failure"], b["auth_failure"])
        self.assertEqual(a["no_response"], 1)
        self.assertEqual(b["invalid_passcode"], 1)
        self.assertNotEqual(a, b)

    def test_false_or_string_true_does_not_indicate_action(self):
        a, _ = self.flags('{"quarantine_file":false,"operation_blocked":"true"}')
        self.assertEqual(a["file_quarantined"] + a["operation_blocked"], 0)
        b, _ = self.flags('{"quarantine_file":true,"operation_blocked":true}')
        self.assertEqual(b["file_quarantined"] + b["operation_blocked"], 2)

    def test_http_status_not_used_as_authentication_outcome(self):
        flags, _ = self.flags('CEF:0|demo|http|1|id|request|1|requestMethod=GET outcome=403')
        self.assertEqual(flags["http_get"], 1)
        self.assertEqual(flags["http_status_4xx"], 1)
        self.assertEqual(flags["auth_failure"], 0)

    def test_source_and_missing_metadata_do_not_change_features(self):
        message = 'sshd: pam_unix(sshd:session): session opened for user USER-001'
        a, _ = self.flags(message, product_name="Linux PAM", src_host=None)
        b, _ = self.flags(message, product_name="Unknown product", src_host="", timestamp="2035-01-01")
        self.assertEqual(a, b)

    def test_calendar_and_entity_variation_same_group(self):
        a = {"message_sanitized": "Jul 23 2022 01:02:03 alice: authentication failure 192.0.2.1", "username": "alice"}
        b = {"message_sanitized": "Aug 01 2024 10:20:30 bob: authentication failure 198.51.100.2", "username": "bob"}
        self.assertEqual(v2.canonical_group(a), v2.canonical_group(b))
        np.testing.assert_array_equal(v2.parse(a)[0], v2.parse(b)[0])

    def test_transitive_old_new_group_bridges_are_preserved(self):
        # old group A bridges new groups X and Y; Y also belongs to old B.
        groups = v2.connected_groups(["A", "A", "B", "C"], ["X", "Y", "Y", "Z"])
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[0], groups[3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
