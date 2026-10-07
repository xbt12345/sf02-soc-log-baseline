import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse
import torch

from sf02_model.model import (BodyResidual, FoldModel, SparseFirstBatchEnsemble,
                             WIDTH, MEMBERS, HIDDEN, csr_tensor, _features, _verify_file)


class ModelInputTests(unittest.TestCase):
    def test_compact_first_layer_matches_full_equation_with_repeated_unsorted_columns(self):
        torch.set_num_threads(2)
        torch.manual_seed(1701)
        model = SparseFirstBatchEnsemble()
        for parameter in model.parameters():
            torch.nn.init.uniform_(parameter, -0.1, 0.1)
        x = sparse.csr_matrix((np.array([1.0, 2.0, 3.0, 4.0], np.float32),
                               np.array([17, 0, 17, WIDTH - 1]),
                               np.array([0, 3, 4])), shape=(2, WIDTH))
        tx = csr_tensor(x, "cpu")
        with torch.no_grad():
            actual = model(tx)
            expected = torch.stack([
                torch.sparse.mm(tx, model.weight.T * model.r[k, :, None])
                * model.s[k] + model.bias[k] for k in range(MEMBERS)
            ], dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_empty_sparse_input_produces_first_layer_bias(self):
        model = SparseFirstBatchEnsemble()
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        torch.nn.init.ones_(model.bias)
        with torch.no_grad():
            result = model(csr_tensor(sparse.csr_matrix((2, WIDTH), dtype=np.float32), "cpu"))
        self.assertEqual(tuple(result.shape), (2, MEMBERS, HIDDEN))
        torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)

    def test_corrupted_checkpoint_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            path.write_bytes(b"unexpected checkpoint bytes")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _verify_file(path, hashlib.sha256(b"expected bytes").hexdigest())

    def test_incompatible_feature_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            _features(sparse.csr_matrix((2, WIDTH - 1), dtype=np.float32))
        x = sparse.csr_matrix(([np.nan], ([0], [0])), shape=(1, WIDTH))
        with self.assertRaises(ValueError):
            _features(x)

    def test_body_is_neither_silently_truncated_nor_misaligned(self):
        model = FoldModel.__new__(FoldModel)
        x = sparse.csr_matrix((1, WIDTH), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "1..176"):
            model.predict_asa(x, np.zeros((1, 177), dtype=np.uint8), np.array([177]))
        with self.assertRaisesRegex(ValueError, "aligned uint8"):
            model.predict_asa(x, np.zeros((2, 1), dtype=np.uint8), np.array([1]))

    def test_padding_is_ignored_and_real_zero_byte_remains_data(self):
        torch.manual_seed(1701)
        model = BodyResidual().eval()
        raw = torch.tensor([[0, 12, 25, 3]], dtype=torch.uint8)
        changed_padding = torch.tensor([[0, 12, 200, 255]], dtype=torch.uint8)
        length = torch.tensor([2])
        with torch.no_grad():
            original = model(raw, length)
            modified = model(changed_padding, length)
        torch.testing.assert_close(original, modified, rtol=0, atol=0)
        self.assertEqual(model.embed.padding_idx, 0)
        self.assertEqual(int((raw[0, 0].long() + 1).item()), 1)


if __name__ == "__main__":
    unittest.main()
