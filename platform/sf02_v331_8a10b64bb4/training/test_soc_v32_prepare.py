"""Old retention contracts plus observed v0.3 failures. No attack labels invented."""
import json
from pathlib import Path
import re
import unittest

import soc_v32_prepare as v32
import test_soc_v3_prepare as original_contracts


class RepairContractTests(unittest.TestCase):
    def prep(self, raw):
        return v32.prepare_record({"message_sanitized": raw})

    def test_json_windows_path_is_preserved(self):
        path = r"C:\tools\runner.exe"
        result = self.prep(json.dumps({"command_line": path}))
        self.assertIn(json.dumps(path.casefold())[1:-1], result["text"])

    def test_unknown_windows_path_not_globally_decoded(self):
        path = r"C:\tools\runner.exe"
        self.assertIn(path.casefold(), self.prep("shell command " + path)["text"])

    def test_literal_backslash_t_is_not_a_tab(self):
        self.assertIn(r"\\t", self.prep(json.dumps({"command_line": r"echo \t"}))["text"])

    def test_invalid_json_value_retains_path_and_masks_address(self):
        raw = r'{"message":"\USER-7625 address:\t192.168.118.146\n cmd C:\tools\runner.exe","event":{"code":4625}}'
        result = self.prep(raw)
        self.assertNotIn("192.168.118.146", result["text"])
        self.assertIn(r"c:\tools\runner.exe", result["text"])
        self.assertIn("4625", result["text"])
        self.assertGreater(result["invalid_string_values_preserved"], 0)

    def test_valid_json_newline_is_decoded_before_ip_masking(self):
        result = self.prep(json.dumps({"message": "address:\t192.168.118.146\nfailed"}))
        self.assertNotIn("192.168.118.146", result["text"])
        self.assertIn("failed", result["text"])

    def test_ip_lookalike_version_not_blindly_removed(self):
        for value in ["version 1.2.3.4", "C:\\app\\23.3.3.264\\run.exe"]:
            self.assertIn(value.casefold(), self.prep(value)["text"])

    def test_falcon_alias_and_entire_identity_values_are_isolated(self):
        def raw(identity, verdict):
            return json.dumps({"detection_id": "ldt:"+identity, "cid": identity,
              "behaviors": [{"display_name": verdict, "behavior_id": "10381", "tactic_id": "TA0005",
               "technique_id": "T0001", "md5": identity, "parent_details": {"parent_process_graph_id": identity},
               "command_line": "whoami /groups", "operation_blocked": False}]})
        a, b = self.prep(raw("abcCRED-123fff", "critical")), self.prep(raw("xyzCRED-999aaa", "benign"))
        self.assertEqual(a["text"], b["text"])
        self.assertNotEqual(a["diagnostic_text"], b["diagnostic_text"])
        for fragment in ["ta0005", "10381", "abc", "fff", "critical", "t0001"]:
            self.assertNotIn(fragment, a["text"])
        self.assertIn("whoami", a["text"])
        self.assertIn("false", a["text"])

    def test_same_named_falcon_words_in_request_are_protected(self):
        raw = json.dumps({"detection_id": "opaque", "behaviors": [], "request": {
            "body": {"tactic_id": "actual_request_argument", "md5": "request_payload_hash"}}})
        result = self.prep(raw)
        self.assertIn("actual_request_argument", result["text"])
        self.assertIn("request_payload_hash", result["text"])

    def test_collector_source_after_json_and_guid_path_before_json(self):
        body=json.dumps({"behaviors":[{"behavior_id":"10381","tactic_id":"TA0005","cmdline":"whoami /groups"}],"detection_id":"ldt:opaque"})
        for prefix in ["ORG-1780 ::: CRED-23501=",r"ORG-1780 ::: filePath=C:\cache\{guid}\app.exe ::: CRED-23501="]:
            raw=prefix+body+" ::: streamName=Crowdstrike Detection"
            p=self.prep(raw)
            self.assertTrue(p["falcon_scoped_policy"])
            self.assertNotIn("tactic_id",p["text"])
            self.assertNotIn("behavior_id",p["text"])
            self.assertIn("whoami",p["text"])

    def test_real_official_failure_fixtures(self):
        path = Path(__file__).resolve().parents[1] / "evidence/2026-09-12/v32_actual_fixtures.json"
        self.assertTrue(path.exists(), "Real fixtures must be extracted before this suite")
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        known = {112255: ["192.168.118.146"], 112259: ["10.243.130.79"],
                 131538: ["192.168.39.160"], 148619: ["192.168.115.129", "100.64.52.192"],
                 149243: ["192.168.112.207"]}
        for row in rows:
            result = self.prep(row["message_sanitized"])
            with self.subTest(row=row["row_position"]):
                for address in known.get(row["row_position"], []):
                    self.assertNotIn(address, result["text"])
                if row["row_position"] in {129493, 442429}:
                    self.assertTrue(result["falcon_scoped_policy"])
                    self.assertNotRegex(result["text"], r"tactic_id|technique_id|display_name|behavior_id|detection_id|\"cid\"|process_graph_id|\"md5\"|\"sha256\"")
                    self.assertIn("cmdline", result["text"])
                    self.assertIn("alleged_filetype", result["text"])
                    self.assertIn("exe", result["text"])
                    self.assertNotIn("parent_md5", result["text"])
                    self.assertNotIn("behaviors_processed", result["text"])
                    self.assertIn("pattern_disposition_details", result["diagnostic_text"])
                    self.assertIn("false", result["diagnostic_text"])


def load_tests(loader, tests, pattern):
    # Retarget the historical contracts in this isolated test process only.
    original_contracts.prepare_record = v32.prepare_record
    original_contracts.v3 = v32
    tests.addTests(loader.loadTestsFromModule(original_contracts))
    return tests


if __name__ == "__main__":
    unittest.main(verbosity=2)
