"""Semantic and leakage regressions; none measure SOC model quality."""
import json
import unittest

import numpy as np

import v331_prepare as old
import v351_safeguards as p

BODY = 'Deny tcp src outside:100.64.1.2/12345 dst dmz-1:10.2.3.4/443 by access-group "outside_acl" [0x0, 0x0]'
AUTH = {"application": {}, "user": {}, "txid": "ORG-123", "reason": "invalid_passcode",
        "result": "denied", "timestamp": 1720000000}


class Safeguards(unittest.TestCase):
    def test_missing_product_bucket_is_audit_only(self):
        for v in (None, "", " \t", float("nan")):
            self.assertEqual(p.product_bucket(v), "<missing>")
        self.assertEqual(p.product_bucket("new_product"), "new_product")

    def test_outer_values_cannot_change_text(self):
        for raw in (BODY, json.dumps(AUTH), "", 'command: echo fail > output'):
            expected = p.model_text({"message_sanitized": raw})
            for product in (None, "", " \t", "Duo", "UNSEEN", "malicious"):
                row = dict(message_sanitized=raw, product_name=product, vendor_name=product,
                           timestamp=0, event_id="other", label_binary="suspicious", pipeline="other")
                self.assertEqual(p.model_text(row), expected)
                row["timestamp"] = 9999999999
                self.assertEqual(p.model_text(row), expected)

    def test_only_message_is_read(self):
        class Projection(dict):
            def get(self, key, *args):
                if key != "message_sanitized":
                    raise AssertionError("Metadata access")
                return BODY
        self.assertEqual(p.model_text(Projection()), p.model_text({"message_sanitized": BODY}))

    def test_asa_date_and_header_missingness_invariance(self):
        prefixes = ["", "<164>Jul 23 2022 09:02:26: USER-0010-0324 ",
                    "<180>Jul 23 USER-9564 09:02:26: USER-0010-0324 ",
                    "<180>2026-09-12T12:00:00Z collector.example "]
        self.assertEqual(len({p.model_text({"message_sanitized": h+BODY}) for h in prefixes}), 1)

    def test_auth_date_change_and_absence_invariance(self):
        texts = []
        for v in (None, 0, 1720000000, 9999999999):
            obj = dict(AUTH)
            if v is None:
                del obj["timestamp"]
            else:
                obj["timestamp"] = v
            texts.append(p.model_text({"message_sanitized": json.dumps(obj)}))
        self.assertEqual(len(set(texts)), 1)

    def test_rfc5424_redacted_fraction_full_and_missing_stamp_invariance(self):
        times = ['USER-9546-07-26T06:26:11.CRED-2411823-05:00',
                 '2026-09-12T12:00:00.123456+08:00', '2024-07-26T12:00:00Z', '-']
        texts = [p.model_text({'message_sanitized': '<134>1 '+t+' host app - - - body Duration: 00:20:03.657974'}) for t in times]
        self.assertEqual(len(set(texts)), 1)
        self.assertIn('00:20:03.657974', texts[0])

    def test_rfc5424_payload_date_not_processed_as_header_slot(self):
        raw = 'command: echo "<134>1 2024-07-26T12:00:00Z host app - - - body"'
        self.assertEqual(p.neutralize_collection_stamp(raw), raw)
        raw = '<134>1 2024-07-26T12:00:00Z host app - - - payload 2030-01-02T03:04:05Z'
        self.assertIn('payload 2030-01-02T03:04:05Z', p.neutralize_collection_stamp(raw))

    def test_bad_rfc5424_header_is_not_guessed(self):
        for raw in ('<999>1 2024-07-26T12:00:00Z host app - - - body',
                    '<134>1 payload_looks_like_time host app - - - body',
                    '<134>1 2024-07-26T12:00:00Z host app - - incomplete'):
            self.assertEqual(p.neutralize_collection_stamp(raw), raw)

    def test_duration_previously_lost_is_preserved(self):
        raw = 'Duration: 12:34:56 Count: 20'
        self.assertNotIn('12:34:56', old.prepare_message(raw)["text"])
        self.assertIn('12:34:56', p.model_text({"message_sanitized": raw}))

    def test_changing_duration_changes_evidence(self):
        a = p.model_text({"message_sanitized": "Duration: 12:34:56"})
        b = p.model_text({"message_sanitized": "Duration: 12:34:57"})
        self.assertNotEqual(a, b)

    def test_named_duration_not_plain_clock(self):
        text = p.model_text({"message_sanitized": 'start 12:34:56 Duration: 12:34:56'})
        self.assertEqual(text.count("12:34:56"), 1)

    def test_json_duration_and_multiple_values(self):
        raw = json.dumps({"duration": "12:34:56", "elapsed": "0:00:02.125", "timestamp": "2024-01-01"})
        text = p.model_text({"message_sanitized": raw})
        self.assertIn("12:34:56", text)
        self.assertIn("0:00:02.125", text)
        self.assertNotIn("2024-01-01", text)

    def test_invalid_or_redacted_duration_is_not_repaired(self):
        for value in ("CRED-1:34:56", "1CRED-2:34:56", "12:61:00", "12:34:56CRED-1", "12:34:56:78"):
            self.assertIsNone(p.DURATION.search("Duration: " + value))

    def test_duration_placeholder_cannot_collide_with_payload(self):
        raw = 'preserveddurationplaceholderzqz Duration: 12:34:56'
        text = p.model_text({"message_sanitized": raw})
        self.assertIn('preserveddurationplaceholderzqz', text)
        self.assertIn('12:34:56', text)

    def test_payload_operators_are_not_globally_removed(self):
        text = p.model_text({"message_sanitized": 'cmd: echo x >> file && cat file | tool'})
        for v in ('>>', '&&', '|'):
            self.assertIn(v, text)

    def test_complete_ports_are_preserved_and_direction_matters(self):
        f = p.asa_visible_facts(BODY)
        self.assertEqual(f['src']['port'], 12345)
        self.assertEqual(f['dst']['port'], 443)
        self.assertNotEqual(f, p.asa_visible_facts(BODY.replace('/443', '/445')))
        self.assertNotEqual(f, p.asa_visible_facts(BODY.replace('src outside:', 'src inside:')))

    def test_port_redaction_digits_do_not_become_values(self):
        for value in ('CRED-12345', '5CRED-2CRED-24947', '65536', '-1'):
            self.assertIsNone(p.endpoint_facts('outside:10.1.2.3/'+value)['port'])

    def test_address_changes_do_not_split_visible_facts(self):
        self.assertEqual(p.asa_visible_facts(BODY), p.asa_visible_facts(BODY.replace('100.64.1.2', '10.9.8.7')))

    def test_quoted_attack_lookalike_is_not_parsed_as_outer_asa(self):
        self.assertIsNone(p.asa_visible_facts('request body: '+BODY))

    def test_fit_counts_do_not_read_evaluation_labels(self):
        groups, roles = [1, 1, 2, 2], [0, 0, 3, 3]
        a = p.fit_group_distributions(groups, [1, 2, 1, 2], roles)
        b = p.fit_group_distributions(groups, [1, 2, 0, 0], roles)
        self.assertEqual(a, b)
        self.assertEqual(a[0]['distribution'], [0, .5, .5])

    def test_soft_counts_are_exact_loss_aggregation_not_new_labels(self):
        labels = np.array([1, 1, 2, 2, 2])
        counts = np.bincount(labels, minlength=3)
        probability = np.array([.1, .3, .6])
        self.assertAlmostEqual(-np.log(probability[labels]).sum(), -len(labels)*np.dot(counts/len(labels), np.log(probability)))

    def test_repeats_cannot_cross_active_roles(self):
        with self.assertRaises(ValueError):
            p.assert_group_isolation([1, 1, 2], [0, 3, 0])
        p.assert_group_isolation([1, 1, 2], [-1, 3, 0])

    def test_all_excluded_and_empty_isolation(self):
        p.assert_group_isolation([1], [-1])
        p.assert_group_isolation([], [])

    def test_repeat_weights_bounded_mean_one_label_blind(self):
        keys = np.array(['a']*1000 + ['b']*10 + ['c'])
        weights = p.bounded_repeat_weights(keys)
        self.assertAlmostEqual(weights.mean(), 1, places=12)
        self.assertTrue(np.all((weights >= .2) & (weights <= 5)))
        self.assertLess(weights[0], weights[-1])
        self.assertEqual(len(set(weights[:1000])), 1)

    def test_no_observation_repeat_is_not_unique_row_weight(self):
        w = p.bounded_repeat_weights(['']*50 + ['visible_1', 'visible_2'])
        self.assertLess(w[0], w[-1])

    def test_equal_repeat_counts_do_not_change_effective_regularization(self):
        np.testing.assert_allclose(p.bounded_repeat_weights(['a', 'a', 'b', 'b']), np.ones(4))


if __name__ == '__main__':
    unittest.main(verbosity=2)
