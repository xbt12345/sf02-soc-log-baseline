import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import sparse
import torch

from sf02_model.model import FoldModel, WIDTH
from sf02_model.training import (TrainingConfig, TrainingDataset, classification_metrics,
                                 initialize_base, initialize_residual, train_bundle)


def example_dataset(prefix="train"):
    labels = np.array([1, 2, 1, 2, 1, 2, 0, 0, 0], dtype=np.int64)
    mask = np.array([True] * 6 + [False] * 3)
    rows = np.arange(len(labels))
    columns = 100 + labels * 3
    x = sparse.csr_matrix((np.ones(len(labels), dtype=np.float32), (rows, columns)),
                          shape=(len(labels), WIDTH))
    bodies = np.array([[65, 20 + int(y)] for y in labels[mask]], dtype=np.uint8)
    p = SimpleNamespace(full_features=x, asa_mask=mask, header_features=x[mask],
                        body_bytes=bodies, lengths=np.full(6, 2, dtype=np.int16),
                        unsupported_rows=())
    return TrainingDataset(p, labels, [f"{prefix}-{i}" for i in rows])


class TrainingTests(unittest.TestCase):
    def test_initializer_is_finite_reproducible_and_zero_residual_preserves_base(self):
        a = initialize_base(10201)
        b = initialize_base(10201)
        for k, v in a.state_dict().items():
            self.assertTrue(bool(torch.isfinite(v).all()))
            torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
        branch = initialize_residual(12701)
        with torch.no_grad():
            z = branch(torch.tensor([[65, 66]], dtype=torch.uint8), torch.tensor([2]))
        torch.testing.assert_close(z, torch.zeros_like(z), rtol=0, atol=0)

    def test_real_training_save_reload_prediction_and_corruption_guard(self):
        data = example_dataset()
        config = TrainingConfig(base_epochs=1, residual_epochs=2, batch_size=9,
                                router_max_iterations=30, cpu_threads=1)
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "trained"
            validation = example_dataset("valid")
            result = train_bundle(data, output, config, validation=validation)
            self.assertTrue(result["base"]["parameter_identity_changed"])
            self.assertTrue(result["residual"]["parameter_identity_changed"])
            self.assertTrue(result["router"]["parameter_identity_changed"])
            self.assertEqual(result["base"]["optimizer_steps"], 1)
            self.assertEqual(result["residual"]["optimizer_steps"], 2)
            self.assertEqual(result["training_metrics"]["rows"], 9)
            self.assertTrue(result["frozen_base_retained"])
            self.assertFalse(result["published_historical_metrics_applicable"])
            self.assertTrue(result["validation_evaluated"])
            self.assertEqual(result["validation"]["shared_numerical_input_rows"], 9)
            self.assertEqual(result["validation_metrics"]["rows"], 9)
            plan = json.loads((output / "training_plan.json").read_text(encoding="utf-8"))
            self.assertFalse(plan["validation_used_for_gradients_or_checkpoint_selection"])
            model = FoldModel(output, 0)
            p = data.prepared
            probs = model.predict_routed_proba(p.full_features, p.asa_mask, p.header_features,
                                              p.body_bytes, p.lengths)
            self.assertEqual(probs.shape, (9, 3))
            self.assertTrue(np.isfinite(probs).all())
            np.testing.assert_allclose(probs.sum(1), 1, atol=2e-6)
            without_validation = train_bundle(data, Path(parent) / "train_only", config)
            for component in ("base", "residual"):
                self.assertEqual(result[component]["final_parameter_sha256"],
                                 without_validation[component]["final_parameter_sha256"])
            with self.assertRaises(FileExistsError):
                train_bundle(data, output, config)
            with (output / "fold0/residual.pt").open("ab") as stream:
                stream.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "SHA-256|hash"):
                FoldModel(output, 0)

    def test_duplicate_ids_validation_leakage_and_missing_class_are_rejected(self):
        data = example_dataset()
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "blocked"
            with self.assertRaisesRegex(ValueError, "same complete dataset"):
                train_bundle(data, target, validation=data)
            self.assertFalse(target.exists())
            duplicates = example_dataset()
            duplicates.event_ids[1] = duplicates.event_ids[0]
            with self.assertRaisesRegex(ValueError, "Duplicate event_id"):
                train_bundle(duplicates, target)
            self.assertFalse(target.exists())
            data.labels[:] = 1
            with self.assertRaisesRegex(ValueError, "all three classes"):
                train_bundle(data, target)
            self.assertFalse(target.exists())

    def test_missing_validation_class_does_not_receive_a_full_macro_score(self):
        metrics = classification_metrics(np.array([0, 1]), np.array([0, 1]))
        self.assertIsNone(metrics["classes"]["suspicious"]["f1"])
        self.assertIsNone(metrics["macro_F1"])
        self.assertFalse(metrics["all_classes_present"])


if __name__ == "__main__":
    unittest.main()
