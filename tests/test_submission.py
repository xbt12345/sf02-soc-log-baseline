import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from validate_submission import main, validate_submission


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / 'input.parquet'
        pq.write_table(pa.table({'event_id': ['001', '002', '003'],
                                 'label_binary': ['DO_NOT_READ'] * 3}), self.raw)
        self.output = self.root / 'res.csv'

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, text):
        self.output.write_text(text, encoding='utf-8')
        report = validate_submission(self.raw, self.output, batch_size=1, temp_dir=self.root)
        self.assertEqual(list(self.root.glob('sf02-validate-*')), [])
        return report

    def test_reordered_ids_and_leading_zeros_are_valid_without_reading_labels(self):
        report = self._run('event_id,pred_label\n003,suspicious\n001,benign\n002,malicious\n')
        self.assertTrue(report['passed'])
        self.assertEqual(report['counts']['unique_output_ids'], 3)
        self.assertFalse(report['labels_or_private_answers_read'])

    def test_missing_and_extra_ids_fail_even_when_row_count_matches(self):
        report = self._run('event_id,pred_label\n001,benign\n002,malicious\n004,suspicious\n')
        self.assertFalse(report['passed'])
        self.assertEqual(report['counts']['missing_ids'], 1)
        self.assertEqual(report['counts']['extra_ids'], 1)

    def test_duplicate_id_across_batches_fails(self):
        report = self._run('event_id,pred_label\n001,benign\n001,malicious\n003,suspicious\n')
        self.assertFalse(report['passed'])
        self.assertEqual(report['counts']['output_duplicate_ids'], 1)

    def test_schema_requires_exact_two_ordered_columns(self):
        for header in ('event_id,label', 'event_id,pred_label,index', 'pred_label,event_id', 'event_id,event_id'):
            with self.subTest(header=header):
                self.assertFalse(self._run(header + '\n')['passed'])

    def test_invalid_labels_and_empty_ids_are_rejected(self):
        for row in ('001,0', '001,normal', '001,', '001, benign', ',benign', '   ,benign'):
            with self.subTest(row=row):
                report = self._run('event_id,pred_label\n' + row + '\n002,malicious\n003,suspicious\n')
                self.assertFalse(report['passed'])

    def test_malformed_output_rows_are_rejected(self):
        for row in ('001,benign,extra', '001'):
            with self.subTest(row=row):
                self.assertFalse(self._run('event_id,pred_label\n' + row + '\n002,malicious\n003,suspicious\n')['passed'])

    def test_duplicate_and_nonstring_input_ids_are_rejected(self):
        for ids in (['001', '001', '003'], ['001', None, '003'], [1, 2, 3]):
            with self.subTest(ids=ids):
                pq.write_table(pa.table({'event_id': ids}), self.raw)
                report = self._run('event_id,pred_label\n001,benign\n002,malicious\n003,suspicious\n')
                self.assertFalse(report['passed'])

    def test_csv_input_handles_multiline_messages_and_ignores_class_labels(self):
        self.raw = self.root / 'input.csv'
        with self.raw.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['event_id', 'message_sanitized', 'label_binary'])
            writer.writerows([['001', 'line 1\nline 2, quoted', 'DO_NOT_READ'],
                              ['002', '', 'DO_NOT_READ'], ['003', 'value', 'DO_NOT_READ']])
        self.assertTrue(self._run('event_id,pred_label\n003,suspicious\n001,benign\n002,malicious\n')['passed'])

    def test_empty_input_is_rejected(self):
        pq.write_table(pa.table({'event_id': pa.array([], type=pa.string())}), self.raw)
        self.assertFalse(self._run('event_id,pred_label\n')['passed'])

    def test_parse_exception_closes_and_removes_temporary_database(self):
        self.raw = self.root / 'bad.csv'
        self.raw.write_text('event_id,message_sanitized\n001,raw,extra\n', encoding='utf-8')
        self.output.write_text('event_id,pred_label\n001,benign\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            validate_submission(self.raw, self.output, temp_dir=self.root)
        self.assertEqual(list(self.root.glob('sf02-validate-*')), [])

    def test_cli_failure_is_json_and_nonzero(self):
        self.output.write_text('event_id,pred_label\n001,benign\n', encoding='utf-8')
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status = main(['--input', str(self.raw), '--submission', str(self.output), '--temp-dir', str(self.root)])
        self.assertEqual(status, 1)
        self.assertFalse(json.loads(buffer.getvalue())['passed'])


if __name__ == '__main__':
    unittest.main()
