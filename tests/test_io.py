import csv
import tempfile
import unittest
from pathlib import Path

from sf02_model.io import iter_input_batches


class RawInputTests(unittest.TestCase):
    def test_prediction_does_not_read_labels_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["event_id", "message_sanitized", "label_binary"])
                writer.writerow(["001", "message, containing a comma\nand newline", "not-a-label"])
                writer.writerow(["002", "", "not-a-label"])
            batches = list(iter_input_batches(path, batch_size=1))
            self.assertEqual([b.event_ids for b in batches], [["001"], ["002"]])
            self.assertEqual(batches[0].messages, ["message, containing a comma\nand newline"])
            self.assertIsNone(batches[0].labels)
            with self.assertRaisesRegex(ValueError, "label_binary"):
                list(iter_input_batches(path, require_labels=True))

    def test_null_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.csv"
            path.write_text("event_id,message_sanitized\n,hello\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "event_id"):
                list(iter_input_batches(path))

    def test_parquet_numeric_ids_are_rejected_before_prediction(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.parquet"
            pq.write_table(pa.table({"event_id": [123], "message_sanitized": [""]}), path)
            with self.assertRaisesRegex(ValueError, "event_id"):
                list(iter_input_batches(path))
