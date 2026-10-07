import unittest

import numpy as np

from sf02_model.preprocess import BYTE_FEATURES, PreparedBatch, prepare_batch


ASA = (
    '<164>Jul 26 USER-9546 06:15:20: USER-0010-0324 Deny tcp '
    'src outside:203.0.113.5/9936 dst dmz-1:10.0.0.5/16572 '
    'by ORG-1738-group "outside_ORG-1738_in" [0x0, 0x0]'
)


def same_sparse(a, b):
    delta = a - b
    return not delta.nnz or np.all(delta.data == 0)


class PreprocessTests(unittest.TestCase):
    def test_complete_asa_contract(self):
        batch = prepare_batch([ASA], [9936])
        self.assertIsInstance(batch, PreparedBatch)
        self.assertEqual(batch.full_features.shape, (1, 66287))
        self.assertEqual(batch.header_features.shape, (1, 66287))
        self.assertEqual(batch.asa_indices.tolist(), [0])
        self.assertEqual(batch.asa_mask.tolist(), [True])
        self.assertEqual(batch.route_names.tolist(), ["asa"])
        self.assertEqual(batch.unsupported_count, 0)
        self.assertTrue(0 < batch.lengths[0] <= 176)
        self.assertEqual(len(batch.body_bytes[0, :batch.lengths[0]]), batch.lengths[0])
        self.assertTrue(same_sparse(batch.full_features[:, BYTE_FEATURES:],
                                    batch.header_features[:, BYTE_FEATURES:]))

    def test_independent_record_port_does_not_overwrite_message_facts(self):
        batch = prepare_batch([ASA, ASA, ASA], [None, 9936, 65535])
        self.assertTrue(same_sparse(batch.full_features[0, :-18], batch.full_features[1, :-18]))
        self.assertTrue(same_sparse(batch.full_features[1, :-18], batch.full_features[2, :-18]))
        meta = batch.full_features[:, -18:].toarray()
        self.assertTrue(np.array_equal(meta[0], np.zeros(18)))
        self.assertAlmostEqual(float(meta[1, 16]), 1 / np.sqrt(18), places=7)
        self.assertEqual(float(meta[1, 17]), 0)
        self.assertAlmostEqual(float(meta[2, 17]), 1 / np.sqrt(18), places=7)
        self.assertTrue(np.array_equal(batch.body_bytes[0], batch.body_bytes[1]))

    def test_duplicate_inputs_keep_original_rows_and_order(self):
        batch = prepare_batch([ASA, "", ASA], [9936, None, 9936])
        self.assertEqual(batch.full_features.shape[0], 3)
        self.assertEqual(batch.asa_indices.tolist(), [0, 2])
        self.assertEqual(batch.route_names.tolist(), ["asa", "empty", "asa"])
        self.assertTrue(same_sparse(batch.full_features[0], batch.full_features[2]))

    def test_unknown_registered_header_is_reported_for_router_fallback(self):
        raw = ASA.replace("USER-0010-0324 Deny", "%ASA-4-106023: Deny")
        batch = prepare_batch([raw], [9936])
        self.assertEqual(batch.route_names.tolist(), ["asa"])
        self.assertEqual(batch.full_features.shape, (1, 66287))
        self.assertEqual(batch.source_asa_mask.tolist(), [True])
        self.assertEqual(batch.asa_mask.tolist(), [False])
        self.assertEqual(batch.unsupported_asa_mask.tolist(), [True])
        self.assertEqual(batch.header_features.shape[0], 0)
        self.assertTrue(batch.unsupported_rows[0]["reason"].startswith("asa_grammar_unsupported"))

    def test_long_body_is_not_truncated_or_discarded(self):
        raw = ASA.replace("outside_ORG-1738_in", "outside_ORG-1738_in " + "x" * 240)
        batch = prepare_batch([ASA, raw], [9936, 9936])
        self.assertEqual(batch.full_features.shape[0], 2)
        self.assertEqual(batch.asa_indices.tolist(), [0])
        self.assertEqual(batch.unsupported_rows[0]["index"], 1)
        self.assertTrue(batch.unsupported_rows[0]["reason"].startswith("asa_body_too_long:"))
        self.assertNotEqual(batch.full_features[1, :BYTE_FEATURES].nnz, 0)

    def test_empty_batch_and_null_message_keep_defined_shapes(self):
        empty = prepare_batch([], [])
        self.assertEqual(empty.full_features.shape, (0, 66287))
        self.assertEqual(empty.header_features.shape, (0, 66287))
        self.assertEqual(empty.body_bytes.shape, (0, 0))
        batch = prepare_batch([None, np.nan, ""], [None, None, None])
        self.assertEqual(batch.full_features.shape, (3, 66287))
        self.assertFalse(batch.asa_mask.any())
        self.assertTrue(same_sparse(batch.full_features[0], batch.full_features[1]))
        self.assertTrue(same_sparse(batch.full_features[1], batch.full_features[2]))

    def test_identity_changes_reproduce_identical_registered_asa_inputs(self):
        changed = ASA.replace("203.0.113.5", "203.0.113.17").replace("10.0.0.5", "10.1.1.6")
        batch = prepare_batch([ASA, changed], [9936, 9936])
        self.assertTrue(same_sparse(batch.full_features[0], batch.full_features[1]))
        self.assertTrue(same_sparse(batch.header_features[0], batch.header_features[1]))
        self.assertTrue(np.array_equal(batch.body_bytes[0], batch.body_bytes[1]))

    def test_input_columns_require_matching_row_counts(self):
        with self.assertRaises(ValueError):
            prepare_batch([ASA], [])


if __name__ == "__main__":
    unittest.main()
