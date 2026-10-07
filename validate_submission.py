"""Independently verify official res.csv schema and exact event_id coverage.

Only event_id is read from the original input. IDs are indexed in a temporary
on-disk SQLite database so validation does not retain millions of IDs in RAM.
This checks submission validity, not model accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

LABELS = frozenset(('benign', 'malicious', 'suspicious'))
COLUMNS = ['event_id', 'pred_label']


def _input_ids(path: Path, batch_size: int):
    if path.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        source = pq.ParquetFile(path)
        if source.schema_arrow.names.count('event_id') != 1:
            raise ValueError('Input must have exactly one event_id column')
        for batch in source.iter_batches(batch_size=batch_size, columns=['event_id'], use_threads=False):
            yield batch.column(0).to_pylist()
    elif path.suffix.lower() == '.csv':
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or fields.count('event_id') != 1:
                raise ValueError('Input CSV must have unique columns and one event_id column')
            batch = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('Input CSV row width differs from its header')
                batch.append(row['event_id'])
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
    else:
        raise ValueError('Input must be an official .parquet or UTF-8 .csv file')


def _valid_id(value):
    # Preserve the original string, including leading zeros. Do not coerce
    # numeric/null values into strings or silently strip IDs.
    return isinstance(value, str) and bool(value.strip())


def _insert_ids(connection, table, values):
    before = connection.total_changes
    connection.executemany(f'INSERT OR IGNORE INTO {table} VALUES (?)', ((value,) for value in values))
    return connection.total_changes - before


def _validate(expected_batches, stream, connection, batch_size):
    counts, labels = Counter(), Counter()
    failures = []
    connection.execute('CREATE TABLE expected (id TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID')
    connection.execute('CREATE TABLE emitted (id TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID')
    for group in expected_batches:
        counts['input_rows'] += len(group)
        valid = [value for value in group if _valid_id(value)]
        counts['input_invalid_ids'] += len(group) - len(valid)
        inserted = _insert_ids(connection, 'expected', valid)
        counts['input_duplicate_ids'] += len(valid) - inserted
    connection.commit()
    reader = csv.DictReader(stream)
    if reader.fieldnames != COLUMNS:
        failures.append('CSV header must be exactly event_id,pred_label')
    else:
        pending = []
        for row in reader:
            counts['output_rows'] += 1
            if None in row or row['pred_label'] is None:
                counts['malformed_csv_rows'] += 1
                continue
            event_id, label = row['event_id'], row['pred_label']
            if label in LABELS:
                labels[label] += 1
            else:
                counts['invalid_labels'] += 1
            if not _valid_id(event_id):
                counts['output_invalid_ids'] += 1
                continue
            pending.append(event_id)
            if len(pending) >= batch_size:
                counts['output_duplicate_ids'] += len(pending) - _insert_ids(connection, 'emitted', pending)
                pending.clear()
        if pending:
            counts['output_duplicate_ids'] += len(pending) - _insert_ids(connection, 'emitted', pending)
    connection.commit()
    counts['unique_input_ids'] = connection.execute('SELECT COUNT(*) FROM expected').fetchone()[0]
    counts['unique_output_ids'] = connection.execute('SELECT COUNT(*) FROM emitted').fetchone()[0]
    counts['missing_ids'] = connection.execute(
        'SELECT COUNT(*) FROM expected e LEFT JOIN emitted o ON e.id=o.id WHERE o.id IS NULL').fetchone()[0]
    counts['extra_ids'] = connection.execute(
        'SELECT COUNT(*) FROM emitted o LEFT JOIN expected e ON e.id=o.id WHERE e.id IS NULL').fetchone()[0]
    for key in ('input_invalid_ids', 'input_duplicate_ids', 'output_invalid_ids',
                'output_duplicate_ids', 'malformed_csv_rows', 'invalid_labels', 'missing_ids', 'extra_ids'):
        if counts[key]:
            failures.append(key)
    if counts['input_rows'] == 0:
        failures.append('Input must contain at least one event')
    if counts['input_rows'] != counts['output_rows']:
        failures.append('input/output row count mismatch')
    result = {
        'passed': not failures,
        'counts': dict(counts),
        'predicted_label_counts': {label: labels[label] for label in sorted(LABELS)},
        'failures': failures,
        'coverage': 'all and only input event_ids; one prediction per ID; output order may differ',
        'labels_or_private_answers_read': False,
        'quality_acceptance': False,
        'id_index_storage': 'temporary disk SQLite with 8 MiB page cache',
    }
    return result


def validate_submission(input_path, submission_path, *, batch_size=8192, temp_dir=None):
    """Return a bounded-memory validity report; never read input class labels."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    input_path, submission_path = Path(input_path), Path(submission_path)
    if not input_path.is_file() or not submission_path.is_file():
        raise FileNotFoundError('Input and submission must both exist')
    with tempfile.TemporaryDirectory(prefix='sf02-validate-', dir=temp_dir) as temporary:
        connection = sqlite3.connect(str(Path(temporary) / 'event_ids.sqlite'))
        try:
            connection.execute('PRAGMA journal_mode=OFF')
            connection.execute('PRAGMA synchronous=OFF')
            connection.execute('PRAGMA cache_size=-8192')
            connection.execute('PRAGMA temp_store=FILE')
            with submission_path.open('r', encoding='utf-8-sig', newline='') as stream:
                result = _validate(_input_ids(input_path, batch_size), stream, connection, batch_size)
        finally:
            # Close before TemporaryDirectory cleanup, including failure paths
            # on Windows, where an open SQLite file cannot be removed.
            connection.close()
    digest = hashlib.sha256()
    with submission_path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    result.update(input_file=input_path.name, submission_file=submission_path.name,
                  submission_bytes=submission_path.stat().st_size,
                  submission_sha256=digest.hexdigest())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--temp-dir', type=Path, help='Directory with free space for the temporary ID index')
    parser.add_argument('--report', type=Path, help='Optional JSON report; existing files are preserved')
    args = parser.parse_args(argv)
    try:
        if args.report is not None and args.report.exists():
            raise FileExistsError('Report already exists; choose a new path')
        result = validate_submission(args.input, args.submission, batch_size=args.batch_size, temp_dir=args.temp_dir)
        if args.report is not None:
            with args.report.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
    except (ValueError, OSError, csv.Error, sqlite3.Error) as error:
        print(json.dumps({'passed': False, 'error_type': type(error).__name__, 'error': str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
