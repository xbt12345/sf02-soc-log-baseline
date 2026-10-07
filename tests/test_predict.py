import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import predict
from sf02_model.io import RawBatch


class ConstantModel:
    def __init__(self, *args, **kwargs):
        pass

    def predict_routed_proba(self, x, *args):
        return np.tile([0.2, 0.7, 0.1], (x.shape[0], 1))


class CompetitionPredictionTests(unittest.TestCase):
    def test_parallel_preprocessing_preserves_batch_order(self):
        source = [RawBatch(["003"], [""], [None]),
                  RawBatch(["001", "002"], ["", ""], [None, "443"])]
        results = list(predict.iter_prepared_batches(iter(source), workers=2))
        self.assertEqual([raw.event_ids for raw, _ in results], [["003"], ["001", "002"]])
        self.assertEqual([p.full_features.shape for _, p in results], [(1, 66287), (2, 66287)])

    def make_input(self, folder, ids):
        path = Path(folder) / "input.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "message_sanitized", "src_port"])
            writer.writerows((i, "", "") for i in ids)
        return path

    @patch("predict.FoldModel", ConstantModel)
    def test_preserves_every_id_and_writes_exact_official_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.make_input(temp, ["001", "002", "003"])
            output = Path(temp) / "res.csv"
            report = predict.predict_file(source, output, weights=temp, device="cpu", batch_size=2)
            with output.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows, [["event_id", "pred_label"], ["001", "malicious"],
                                    ["002", "malicious"], ["003", "malicious"]])
            self.assertEqual(report["output_rows"], 3)
            self.assertFalse(report["labels_read"])
            self.assertEqual(report["folds"], [0, 1, 2])

    @patch("predict.FoldModel", ConstantModel)
    def test_duplicate_input_leaves_no_submission_or_locked_temporary_database(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.make_input(temp, ["001", "001"])
            output = Path(temp) / "res.csv"
            with self.assertRaisesRegex(ValueError, "duplicates"):
                predict.predict_file(source, output, weights=temp, device="cpu", batch_size=1)
            self.assertFalse(output.exists())
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()), ["input.csv"])

    def test_existing_result_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.make_input(temp, ["001"])
            output = Path(temp) / "res.csv"
            output.write_text("existing result", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                predict.predict_file(source, output, weights=temp)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing result")
