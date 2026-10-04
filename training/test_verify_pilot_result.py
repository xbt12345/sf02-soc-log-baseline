import ast
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import verify_pilot_result as verifier


class ContentIdentityTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            ["official_train", 2, "2", "g2", "audit", "suspicious", "Duo"],
            ["official_train", 1, "1", "g1", "train", "benign", ""],
        ], columns=verifier.MANIFEST_COLUMNS)

    def test_python38_syntax(self):
        ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"), feature_version=(3, 8))

    def test_row_order_and_parquet_encoding_do_not_change_content_identity(self):
        frame = self.frame()
        expected = verifier.content_sha(frame, verifier.MANIFEST_COLUMNS)
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a.parquet", Path(temp) / "b.parquet"
            frame.to_parquet(a, compression="snappy", index=False)
            frame.iloc[::-1].to_parquet(b, compression=None, index=False)
            self.assertNotEqual(verifier.file_sha(a), verifier.file_sha(b))
            self.assertEqual(verifier.content_sha(pd.read_parquet(a), verifier.MANIFEST_COLUMNS), expected)
            self.assertEqual(verifier.content_sha(pd.read_parquet(b), verifier.MANIFEST_COLUMNS), expected)

    def test_changed_role_or_group_changes_identity(self):
        frame = self.frame()
        expected = verifier.content_sha(frame, verifier.MANIFEST_COLUMNS)
        for column, value in [("role", "development"), ("group_id", "different"), ("label", "malicious")]:
            changed = frame.copy()
            changed.loc[0, column] = value
            self.assertNotEqual(verifier.content_sha(changed, verifier.MANIFEST_COLUMNS), expected)

    def test_duplicate_row_positions_rejected(self):
        frame = self.frame()
        frame.loc[0, "row_position"] = 1
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verifier.content_sha(frame, verifier.MANIFEST_COLUMNS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
