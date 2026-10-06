import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse
import torch

from sf02_model.model import BodyResidual, FoldModel, WIDTH, _features, _verify_file


class ModelInputTests(unittest.TestCase):
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
