"""Synthetic representation-contract counterexamples, not SOC quality validation.

These tests assert information retention and removal boundaries. They deliberately
do not assign benign/malicious/suspicious labels or require any model prediction.
Run from the project root with:
    python -m unittest discover -s training -p test_soc_v3_prepare.py -v
"""

import copy
import json
import unittest

from soc_v3_prepare import prepare_record
import soc_v3_prepare as v3


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class PrepareRecordContractTests(unittest.TestCase):
    def prepare(self, message, **metadata):
        row = dict(metadata, message_sanitized=message)
        before = copy.deepcopy(row)
        result = prepare_record(row)
        self.assertEqual(row, before, "Preparation must not alter its raw evidence")
        self.assertIsInstance(result, dict)
        for key in ("text", "diagnostic_text", "format"):
            self.assertIsInstance(result[key], str, key)
        for key in ("original_empty", "filtered_empty", "unknown_format"):
            self.assertIs(type(result[key]), bool, key)
        self.assertIsInstance(result["removed_spans"], list)
        raw = "" if message is None else message
        for span in result["removed_spans"]:
            self.assertIsInstance(span, dict)
            self.assertIs(type(span["start"]), int)
            self.assertIs(type(span["end"]), int)
            self.assertGreaterEqual(span["start"], 0)
            self.assertLess(span["start"], span["end"])
            self.assertLessEqual(span["end"], len(raw))
            self.assertIsInstance(span["reason"], str)
            self.assertTrue(span["reason"].strip())
        return result

    def test_reporting_metadata_cannot_change_the_model_text(self):
        message = encoded({"event": {"code": "4625"},
                           "message": "An account failed to log on."})
        first = self.prepare(message, event_id="row-a", timestamp="2022-01-01",
                             product_name="Windows", vendor_name="Vendor A",
                             pipeline="collector-a", src_host=None)
        second = self.prepare(message, event_id="row-b", timestamp="2035-12-31",
                              product_name=None, vendor_name="Vendor B",
                              pipeline="collector-b", src_host="")
        self.assertEqual(first["text"], second["text"])

    def test_explicit_identities_and_absolute_dates_are_invariant(self):
        def sample(user, address, stamp):
            return encoded({"@timestamp": stamp, "user": {"name": user},
                            "source": {"ip": address},
                            "event": {"code": "4625"},
                            "message": "An account failed to log on."})

        first = self.prepare(sample("alice_example", "192.0.2.17", "2022-04-05T06:07:08Z"),
                             username="alice_example", src_ip="192.0.2.17")
        second = self.prepare(sample("bob_example", "198.51.100.29", "2035-11-12T13:14:15Z"),
                              username="bob_example", src_ip="198.51.100.29")
        self.assertEqual(first["text"], second["text"])
        for identity in ("alice_example", "192.0.2.17", "2022-04-05"):
            self.assertNotIn(identity, first["text"])
        self.assertIn("4625", first["text"])

    def test_redaction_ids_change_without_erasing_adjacent_action(self):
        a = self.prepare('USER-001 ::: {"actor":"ORG-002",'
                         '"credential":"CRED-003","action":"denied","dst_port":443}')
        b = self.prepare('USER-919 ::: {"actor":"ORG-828",'
                         '"credential":"CRED-737","action":"denied","dst_port":443}')
        self.assertEqual(a["text"], b["text"])
        self.assertIn("denied", a["text"].casefold())
        self.assertIn("443", a["text"])
        for identity in ("001", "002", "003"):
            self.assertNotIn(identity, a["text"])

    def test_windows_success_and_failure_codes_do_not_collapse(self):
        a = self.prepare(encoded({"agent": {"type": "winlogbeat"},
                                  "event": {"code": "4624"}}))
        b = self.prepare(encoded({"agent": {"type": "winlogbeat"},
                                  "event": {"code": "4625"}}))
        self.assertNotEqual(a["text"], b["text"])
        self.assertIn("4624", a["text"])
        self.assertIn("4625", b["text"])

    def test_http_response_codes_do_not_collapse(self):
        def message(status):
            return encoded({"http": {"request": {"method": "GET"},
                                      "response": {"status_code": status}}})
        a, b = self.prepare(message(200)), self.prepare(message(403))
        self.assertNotEqual(a["text"], b["text"])
        self.assertIn("200", a["text"])
        self.assertIn("403", b["text"])

    def test_explicit_service_port_difference_is_retained(self):
        def message(port):
            return encoded({"network": {"transport": "tcp"},
                            "destination": {"port": port}, "action": "allowed"})
        a, b = self.prepare(message(53)), self.prepare(message(443))
        self.assertNotEqual(a["text"], b["text"])
        self.assertIn("53", a["text"])
        self.assertIn("443", b["text"])

    def test_boolean_false_cannot_become_an_affirmative_action(self):
        a = self.prepare(encoded({"operation_blocked": False, "quarantine_file": False}))
        b = self.prepare(encoded({"operation_blocked": True, "quarantine_file": True}))
        self.assertNotEqual(a["text"], b["text"])
        self.assertIn("false", a["text"].casefold())
        self.assertIn("true", b["text"].casefold())

    def test_readable_negation_is_not_removed_as_wrapper_noise(self):
        a = self.prepare(encoded({"message": "authentication failure observed"}))
        b = self.prepare(encoded({"message": "no authentication failure observed"}))
        self.assertNotEqual(a["text"], b["text"])
        self.assertRegex(b["text"].casefold(), r"\bno\b")

    def test_upstream_judgment_changes_only_the_diagnostic_view(self):
        def message(severity, confidence):
            return encoded({"severity": severity, "confidence": confidence,
                            "process": {"command_line": "whoami /groups"},
                            "action": "process_created"})
        a = self.prepare(message("critical", 0.99))
        b = self.prepare(message("informational", 0.01))
        self.assertEqual(a["text"], b["text"])
        self.assertNotEqual(a["diagnostic_text"], b["diagnostic_text"])
        self.assertIn("whoami", a["text"].casefold())
        self.assertNotIn("critical", a["text"].casefold())
        self.assertTrue(a["removed_spans"], "Removed conclusions need raw-span evidence")

    def test_judgment_words_inside_command_arguments_are_not_global_filters(self):
        message = encoded({"severity": "critical", "confidence": 0.99,
                           "process": {"command_line": "echo severity confidence suspicious_activity"}})
        result = self.prepare(message)
        for argument in ("echo", "severity", "confidence", "suspicious_activity"):
            self.assertIn(argument, result["text"].casefold())
        self.assertNotIn("critical", result["text"].casefold())

    def test_post_event_workflow_outcomes_do_not_enter_the_fact_view(self):
        def message(verdict, duration, resolved):
            return encoded({"verdict": verdict, "resolution": "analyst_reviewed",
                            "triaged": True, "resolved": resolved,
                            "seconds_to_triaged": duration,
                            "seconds_to_resolved": duration + 1,
                            "action": "file_written",
                            "process": {"command_line": "echo verdict resolved"}})
        a = self.prepare(message("malicious", 731, True))
        b = self.prepare(message("benign", 952, False))
        self.assertEqual(a["text"], b["text"])
        self.assertNotEqual(a["diagnostic_text"], b["diagnostic_text"])
        self.assertIn("file_written", a["text"].casefold())
        self.assertIn("verdict", a["text"].casefold())
        self.assertIn("resolved", a["text"].casefold())
        self.assertNotIn("malicious", a["text"].casefold())
        self.assertNotIn("731", a["text"])

    def test_mixed_detection_description_loss_is_visible_in_diagnostic_and_spans(self):
        description = "process downloaded a file; analyst calls this suspicious"
        message = encoded({"description": description, "action": "file_written"})
        result = self.prepare(message)
        self.assertNotIn("downloaded", result["text"].casefold())
        self.assertIn("downloaded", result["diagnostic_text"].casefold())
        self.assertIn("file_written", result["text"].casefold())
        fact_position = message.index("downloaded")
        self.assertTrue(any(span["start"] <= fact_position < span["end"]
                            for span in result["removed_spans"]),
                        "Removing a mixed paragraph must expose the lost fact span")

    def test_original_empty_is_separate_from_filtering_away_the_evidence(self):
        for message in (None, "", " \n\t"):
            with self.subTest(raw=message):
                result = self.prepare(message)
                self.assertTrue(result["original_empty"])
                self.assertFalse(result["filtered_empty"])
                self.assertFalse(result["text"].strip())
        filtered = self.prepare(encoded({"severity": "critical", "confidence": 0.99}))
        self.assertFalse(filtered["original_empty"])
        self.assertTrue(filtered["filtered_empty"])
        self.assertFalse(filtered["text"].strip())
        self.assertTrue(filtered["diagnostic_text"].strip())

    def test_readable_fields_in_damaged_json_are_not_discarded(self):
        message = ('{"agent":"winlogbeat","process":{"pid":CRED-003},'
                   '"event":{"code":"4625"},"message":"access denied"}')
        result = self.prepare(message)
        self.assertIn("4625", result["text"])
        self.assertIn("denied", result["text"].casefold())
        self.assertFalse(result["original_empty"])
        self.assertFalse(result["filtered_empty"])

    def test_unrecognized_text_stays_visible_and_is_not_declared_safe(self):
        message = "custom_unrecognized_probe emitted an amber result"
        result = self.prepare(message)
        self.assertTrue(result["unknown_format"])
        self.assertIn("custom_unrecognized_probe", result["text"].casefold())
        self.assertNotIn("pred_label", result)

    def test_removal_offsets_refer_to_original_unicode_evidence(self):
        message = encoded({"message": "审计记录：process started", "severity": "critical",
                           "process": {"command_line": "echo severity"}})
        result = self.prepare(message)
        self.assertTrue(result["removed_spans"])
        judgment_position = message.index("critical")
        self.assertTrue(any(span["start"] <= judgment_position < span["end"]
                            for span in result["removed_spans"]))
        command_position = message.index("echo severity")
        self.assertFalse(any(span["start"] <= command_position < span["end"]
                             for span in result["removed_spans"]),
                         "Removing a conclusion must not swallow the command evidence")
        self.assertIn("echo", result["text"].casefold())

    def test_unknown_curl_json_payload_is_not_reinterpreted_as_log_metadata(self):
        message = ('curl --data \'{"description":"payload_written_to_service",'
                   '"severity":"critical","action":"delete"}\' /api')
        result = self.prepare(message)
        for fact in ("curl", "--data", "description", "payload_written_to_service",
                     "severity", "critical", "delete"):
            self.assertIn(fact, result["text"].casefold())
        self.assertFalse(any(span["reason"] == "upstream_or_post_event"
                             for span in result["removed_spans"]),
                         "An unknown command payload is not an upstream log object")

    def test_root_json_boundary_does_not_consume_a_trailing_description(self):
        message = (encoded({"description": "upstream_root_conclusion", "action": "file_written"})
                   + ' trailing "description":"trailing_observed_payload"')
        result = self.prepare(message)
        self.assertNotIn("upstream_root_conclusion", result["text"].casefold())
        self.assertIn("trailing_observed_payload", result["text"].casefold())
        self.assertIn("file_written", result["text"].casefold())
        trailing_position = message.index("trailing_observed_payload")
        self.assertFalse(any(span["start"] <= trailing_position < span["end"]
                             for span in result["removed_spans"]),
                         "The closing root brace terminates its property-removal scope")

    def test_redacted_vpc_interface_keeps_flow_semantics_and_masks_collection_values(self):
        def flow(account, start, finish, marker):
            return ("2 {} ORG-1504 192.0.2.1 198.51.100.2 54321 {} "
                    "6 1 44 {} {} REJECT OK").format(account, marker, start, finish)
        raw_a = flow("123456789012", "1720000000", "1720000010", "4CRED-003")
        raw_b = flow("987654321098", "1820000000", "1820000010", "4CRED-999")
        a, b = self.prepare(raw_a), self.prepare(raw_b)
        self.assertEqual(a["format"], "vpc14")
        self.assertEqual(a["text"], b["text"])
        values = a["text"].split()
        self.assertEqual(len(values), 14)
        self.assertTrue(values[6].startswith("unknown"),
                        "A partially redacted port is wholly unknown, not a leftover digit")
        self.assertNotIn("4entity", a["text"])
        self.assertEqual(values[7:10], ["6", "1", "44"])
        self.assertEqual(values[-2:], ["reject", "ok"])
        for collection_value in ("123456789012", "1720000000", "1720000010"):
            self.assertNotIn(collection_value, a["text"])

    def test_fourteen_ordinary_words_are_not_enough_to_trigger_vpc_masking(self):
        message = "2 alpha beta gamma delta epsilon zeta eta theta iota kappa lambda REJECT OK"
        self.assertEqual(len(message.split()), 14)
        result = self.prepare(message)
        self.assertNotEqual(result["format"], "vpc14")
        for fact in ("alpha", "beta", "gamma", "kappa", "lambda"):
            self.assertIn(fact, result["text"].casefold())

    def test_collector_name_inside_request_body_does_not_remove_the_payload(self):
        message = encoded({"agent": {"type": "winlogbeat", "id": "collection_only_id"},
                           "request": {"body": {"type": "winlogbeat", "severity": "critical",
                                                  "description": "observed_request_argument",
                                                  "action": "delete"}},
                           "event": {"code": 4625}})
        result = self.prepare(message)
        for fact in ("winlogbeat", "critical", "observed_request_argument", "delete", "4625"):
            self.assertIn(fact, result["text"].casefold())
        self.assertNotIn("collection_only_id", result["text"].casefold())
        payload_position = message.index("observed_request_argument")
        self.assertFalse(any(span["start"] <= payload_position < span["end"]
                             for span in result["removed_spans"]))

    def test_cef_collection_id_is_removed_without_erasing_an_identical_url_parameter(self):
        message = ('CEF:0|Acme|WAF|1.0|42|request|5|externalId=123 rt=1720000000000 '
                   'request="/api?externalId=123&action=delete" requestMethod=GET')
        result = self.prepare(message)
        self.assertIn("/api?externalid=123&action=delete", result["text"].casefold())
        self.assertEqual(result["text"].casefold().count("externalid=123"), 1)
        self.assertIn("requestmethod=get", result["text"].casefold())
        self.assertNotIn("1720000000000", result["text"])
        collection_position = message.index("externalId=123")
        request_position = message.index("/api?externalId=123&action=delete")
        self.assertTrue(any(span["start"] <= collection_position < span["end"]
                            for span in result["removed_spans"]))
        self.assertFalse(any(span["start"] <= request_position < span["end"]
                             for span in result["removed_spans"]))

    def test_asa_partial_port_redaction_is_unknown_without_collapsing_real_ports(self):
        def message(port):
            return ("Deny tcp src inside:192.0.2.1/54321 dst outside:198.51.100.2/{} "
                    "by access-group example_rule").format(port)
        a, b = self.prepare(message("4CRED-003")), self.prepare(message("4CRED-999"))
        self.assertEqual(a["text"], b["text"])
        self.assertIn("/unknown_port", a["text"])
        self.assertNotIn("4entity", a["text"])
        self.assertIn("deny tcp", a["text"].casefold())
        clear = self.prepare(message("443"))
        self.assertIn("/443", clear["text"])
        self.assertNotEqual(a["text"], clear["text"])

    def test_nested_raw_objects_are_not_mutated(self):
        row = {"message_sanitized": encoded({"event": {"code": 4625}}),
               "collector_note": {"tags": ["original", "unchanged"]}}
        original = copy.deepcopy(row)
        prepare_record(row)
        self.assertEqual(row, original)


class GroupingContractTests(unittest.TestCase):
    def test_filtered_nonempty_raw_duplicates_keep_their_original_link(self):
        raw = encoded({"severity": "critical", "confidence": 0.99})
        prepared = prepare_record({"message_sanitized": raw})
        self.assertFalse(prepared["original_empty"])
        self.assertTrue(prepared["filtered_empty"])
        first = v3.group_key(raw, prepared, 7)
        repeated = v3.group_key(raw, prepared, 901)
        self.assertIsInstance(first, bytes)
        self.assertEqual(first, repeated,
                         "Filtering away evidence cannot release raw duplicates across folds")
        other_raw = encoded({"severity": "informational", "confidence": 0.01})
        other = prepare_record({"message_sanitized": other_raw})
        self.assertTrue(other["filtered_empty"])
        self.assertNotEqual(first, v3.group_key(other_raw, other, 902),
                            "Different filtered-away inputs are not one confirmed incident")

    def test_only_original_empty_uses_row_units_and_real_text_uses_its_content(self):
        empty = prepare_record({"message_sanitized": ""})
        self.assertNotEqual(v3.group_key("", empty, 7), v3.group_key("", empty, 901))

        def event(stamp, code):
            raw = encoded({"@timestamp": stamp, "event": {"code": code}})
            prepared = prepare_record({"message_sanitized": raw})
            return raw, prepared

        raw_a, a = event("2022-01-01T00:00:00Z", 4625)
        raw_b, b = event("2035-12-31T23:59:59Z", 4625)
        self.assertNotEqual(raw_a, raw_b)
        self.assertEqual(a["text"], b["text"])
        self.assertEqual(v3.group_key(raw_a, a, 8), v3.group_key(raw_b, b, 902))
        raw_c, c = event("2035-12-31T23:59:59Z", 4624)
        self.assertNotEqual(v3.group_key(raw_b, b, 902), v3.group_key(raw_c, c, 903))

    def test_empty_units_and_repeated_rows_cannot_manufacture_group_support(self):
        import numpy as np

        # Normal: two original-empty row units and two copies of filtered text.
        # Malicious: two informative rows in one group. No attack truth is asserted.
        y = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.uint8)
        groups = np.array([0, 1, 2, 2, 3, 3, 4], dtype=np.int32)
        informative = np.array([False, False, False, False, True, True, True])
        selected = np.array([True, True, True, True, True, True, False])
        support = v3.group_support(y, groups, selected, informative)
        self.assertEqual(support["benign"]["rows"], 4)
        self.assertEqual(support["benign"]["informative_rows"], 0)
        self.assertEqual(support["benign"]["informative_groups"], 0)
        self.assertFalse(support["benign"]["engineering_support_30_groups"])
        self.assertIsNone(support["benign"]["largest_group_fraction_in_informative_rows"])
        self.assertEqual(support["malicious"]["informative_rows"], 2)
        self.assertEqual(support["malicious"]["informative_groups"], 1)
        self.assertEqual(support["malicious"]["remaining_rows_after_largest_group"], 0)
        self.assertEqual(support["suspicious"]["rows"], 0,
                         "Out-of-role records must not contribute support")


if __name__ == "__main__":
    unittest.main(verbosity=2)
