"""Focused tests for risks that can invalidate this experiment's conclusions."""
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import soc_signal_pilot as pilot


class ContractTests(unittest.TestCase):
    def test_python38_syntax(self):
        ast.parse(Path(pilot.__file__).read_text(encoding="utf-8"), feature_version=(3, 8))

    def test_false_boolean_does_not_mean_blocked(self):
        names = [name for name, _ in pilot.RULES]
        vector, _ = pilot.extract({"message_sanitized": '{"quarantine_file":false,"operation_blocked":false}'})
        self.assertEqual(vector[names.index("quarantine_true")], 0)
        self.assertEqual(vector[names.index("operation_blocked_true")], 0)
        vector, _ = pilot.extract({"message_sanitized": '{"quarantine_file":true,"operation_blocked":true}'})
        self.assertEqual(vector[names.index("quarantine_true")], 1)
        self.assertEqual(vector[names.index("operation_blocked_true")], 1)

    def test_ids_dates_and_metadata_do_not_change_signals_or_group(self):
        first = {"message_sanitized": "2024-01-02T01:02:03Z authentication failure. user alice src=192.0.2.1 id=157", "username": "alice", "src_ip": "192.0.2.1", "product_name": "one"}
        second = {"message_sanitized": "2035-06-07T03:04:05Z authentication failure. user bob src=198.51.100.2 id=900", "username": "bob", "src_ip": "198.51.100.2", "product_name": "two"}
        a, ga = pilot.extract(first)
        b, gb = pilot.extract(second)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ga, gb)
        self.assertEqual(pilot.role_for(ga, 20260911), pilot.role_for(gb, 20260911))

    def test_absent_and_empty_message_have_same_representation(self):
        vectors = [pilot.extract({"message_sanitized": value, "dst_host": value}) for value in (None, "", "  ", float("nan"))]
        for vector, group in vectors:
            np.testing.assert_array_equal(vector, vectors[0][0])
            self.assertEqual(group, vectors[0][1])

    def test_entity_perturbation_preserves_distinct_identities(self):
        row = {"src_host": "alice", "dst_host": "bob", "username": "alice", "message_sanitized": "alice to bob, alice authentication failure; USER-0010 and USER-0011"}
        renamed = pilot.rename_known_entities(row)
        self.assertNotEqual(renamed["src_host"], renamed["dst_host"])
        self.assertEqual(renamed["src_host"], renamed["username"])
        self.assertEqual(renamed["message_sanitized"].count(renamed["src_host"]), 2)
        self.assertEqual(len(set(pilot.PLACEHOLDER.findall(renamed["message_sanitized"]))), 4)
        np.testing.assert_array_equal(pilot.extract(row)[0], pilot.extract(renamed)[0])

    def test_threshold_respects_budget_with_ties(self):
        scores = np.asarray([0.9] * 20 + [0.1] * 80)
        for budget in (0.0001, 0.001, 0.01, 0.10, 0.20):
            threshold = pilot.threshold_from_normal(scores, budget)
            self.assertLessEqual(float((scores >= threshold).mean()), budget)

    def test_threshold_changes_with_calibration_not_audit(self):
        y = np.asarray([0, 0, 1, 2])
        probabilities = np.asarray([[.9, .05, .05], [.8, .1, .1], [.1, .8, .1], [.2, .1, .7]])
        groups = np.asarray(["a", "b", "c", "d"])
        one = pilot.evaluate_thresholds(y, probabilities, groups, y, probabilities)
        two = pilot.evaluate_thresholds(y, probabilities, groups, y[::-1], probabilities[::-1])
        self.assertEqual([row["threshold"] for row in one], [row["threshold"] for row in two])

    def test_absent_classes_marked_and_all_benign_accuracy_is_insufficient(self):
        y = np.asarray([0] * 980 + [1] * 10 + [2] * 10)
        probabilities = np.zeros((len(y), 3))
        probabilities[:, 0] = 1
        report = pilot.metric_summary(y, probabilities)
        self.assertEqual(report["accuracy"], .98)
        self.assertEqual(report["classes"]["malicious"]["recall"], 0)
        self.assertLess(report["macro_f1_all_three_zero_for_absent"], .34)
        missing = pilot.metric_summary(y[:3], probabilities[:3])
        self.assertIsNone(missing["classes"]["malicious"]["recall"])
        self.assertIsNone(missing["classes"]["malicious"]["average_precision"])

    def test_streaming_sample_reads_selected_positions(self):
        rows = []
        for index in range(20000):
            row = {name: "" for name in ["event_id", "timestamp", "pipeline", "product_name", "vendor_name", "message_sanitized", "label_binary"] + pilot.ENTITIES}
            row["event_id"] = str(index)
            rows.append(row)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1200)
            selected, positions = pilot.sample_records(path, 150, 23)
            self.assertEqual([int(row["event_id"]) for row in selected], positions.tolist())
            self.assertGreater(int(positions[-1]), 16000)
            self.assertEqual(len(set(positions)), 150)

    def test_identity_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.parquet"
            pq.write_table(pa.table({"value": [1]}), path)
            with patch.object(pilot, "EXPECTED", {"train.parquet": (1, "0" * 64)}):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    pilot.verify_inputs(Path(temp))

    def test_zero_error_bootstrap_does_not_claim_zero_uncertainty(self):
        y = np.asarray([0, 0, 1, 2])
        probabilities = np.eye(3)[y]
        report = pilot.group_bootstrap(y, probabilities, np.asarray(["a", "b", "c", "d"]), 1, repeats=20)
        self.assertIsNone(report["normal_fpr_95_percent_interval"])
        self.assertIn("cannot establish", report["zero_error_interval_note"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
