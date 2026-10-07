"""Read official raw records in bounded batches without using prediction labels."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LABEL_INDEX = {"benign": 0, "malicious": 1, "suspicious": 2}


@dataclass
class RawBatch:
    event_ids: list[str]
    messages: list[str | None]
    src_ports: list[object | None]
    labels: np.ndarray | None = None


def _batch(columns, require_labels):
    ids = columns["event_id"]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("event_id must be a nonempty string")
    messages = columns["message_sanitized"]
    if any(value is not None and not isinstance(value, str) for value in messages):
        raise ValueError("message_sanitized must contain strings or nulls")
    labels = None
    if require_labels:
        truth = columns["label_binary"]
        if any(value not in LABEL_INDEX for value in truth):
            raise ValueError("label_binary must be benign, malicious or suspicious")
        labels = np.asarray([LABEL_INDEX[value] for value in truth], dtype=np.int64)
    return RawBatch([str(value) for value in ids], messages,
                    columns.get("src_port", [None] * len(ids)), labels)


def iter_input_batches(path, batch_size=2048, require_labels=False):
    """Yield original row order; prediction never even reads label_binary."""
    path = Path(path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    required = {"event_id", "message_sanitized"}
    if require_labels:
        required.add("label_binary")
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        source = pq.ParquetFile(path)
        names = set(source.schema_arrow.names)
        if required - names:
            raise ValueError(f"Missing required input columns: {sorted(required - names)}")
        columns = ["event_id", "message_sanitized"]
        if "src_port" in names:
            columns.append("src_port")
        if require_labels:
            columns.append("label_binary")
        for block in source.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
            yield _batch(block.to_pydict(), require_labels)
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            names = reader.fieldnames or []
            if len(names) != len(set(names)) or required - set(names):
                raise ValueError("CSV needs unique columns event_id, message_sanitized"
                                 + (", label_binary" if require_labels else ""))
            columns = {name: [] for name in required | ({"src_port"} & set(names))}
            for row in reader:
                if None in row or any(row[name] is None for name in columns):
                    raise ValueError("Malformed CSV row: column count differs from header")
                for name in columns:
                    columns[name].append(row[name])
                if len(columns["event_id"]) == batch_size:
                    yield _batch(columns, require_labels)
                    columns = {name: [] for name in columns}
            if columns["event_id"]:
                yield _batch(columns, require_labels)
    else:
        raise ValueError("Input must be an official .parquet file or a UTF-8 .csv file")
