"""Predict official raw Parquet/CSV records and write competition res.csv."""
from __future__ import annotations

import argparse
from collections import Counter
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from sf02_model.io import iter_input_batches


def FoldModel(*args, **kwargs):
    from sf02_model.model import FoldModel as model_class
    return model_class(*args, **kwargs)


def _prepare_worker(messages, src_ports):
    from sf02_model.preprocess import prepare_batch
    return prepare_batch(messages, src_ports)


def iter_prepared_batches(raw_batches, workers=1):
    """Preserve original order with at most two pending batches per worker."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        for raw in raw_batches:
            yield raw, _prepare_worker(raw.messages, raw.src_ports)
        return
    pending = deque()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for raw in raw_batches:
            future = executor.submit(_prepare_worker, raw.messages, raw.src_ports)
            pending.append((raw, future))
            if len(pending) >= workers * 2:
                record, result = pending.popleft()
                yield record, result.result()
        while pending:
            record, result = pending.popleft()
            yield record, result.result()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_file(input_path, output_path, weights="weights", fold="ensemble",
                 device="auto", batch_size=2048, threads=4, strict_asa=False,
                 report_path=None, workers=1):
    import numpy as np
    import torch
    from sf02_model.model import CLASS_NAMES, available_folds

    input_path, output_path = Path(input_path), Path(output_path)
    report_path = Path(report_path) if report_path else output_path.with_suffix(".report.json")
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Competition output must be a .csv file")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_stat = input_path.stat()
    if output_path.resolve() == report_path.resolve():
        raise ValueError("CSV and report paths must be different")
    for target in (output_path, report_path):
        if target.exists():
            raise FileExistsError(f"Choose a new output path; refusing to overwrite {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    if threads < 1 or batch_size < 1 or workers < 1:
        raise ValueError("threads, workers and batch_size must be positive")
    torch.set_num_threads(threads)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = available_folds(weights) if str(fold) == "ensemble" else [int(fold)]
    models = [FoldModel(weights, role, device=device) for role in folds]
    counts, routes, fallbacks = Counter(), Counter(), Counter()
    rows = 0
    started = time.monotonic()
    temporary_csv = None
    try:
        with (tempfile.TemporaryDirectory(prefix="sf02-ids-", dir=output_path.parent) as temp,
              closing(sqlite3.connect(str(Path(temp) / "ids.sqlite"))) as db):
            db.execute("CREATE TABLE ids (event_id TEXT PRIMARY KEY) WITHOUT ROWID")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                             prefix=".sf02-", suffix=".csv.partial",
                                             dir=output_path.parent, delete=False) as output:
                temporary_csv = Path(output.name)
                writer = csv.writer(output, lineterminator="\n")
                writer.writerow(["event_id", "pred_label"])
                for raw, prepared in iter_prepared_batches(
                        iter_input_batches(input_path, batch_size=batch_size), workers):
                    try:
                        db.executemany("INSERT INTO ids VALUES (?)", ((i,) for i in raw.event_ids))
                        db.commit()
                    except sqlite3.IntegrityError as error:
                        raise ValueError("Input event_id contains duplicates; submission would be ambiguous") from error
                    if strict_asa and prepared.unsupported_rows:
                        raise ValueError(f"Unsupported ASA input: {prepared.unsupported_rows[:3]}")
                    probabilities = np.zeros((len(raw.event_ids), 3), dtype=np.float64)
                    for model in models:
                        probabilities += model.predict_routed_proba(
                            prepared.full_features, prepared.asa_mask, prepared.header_features,
                            prepared.body_bytes, prepared.lengths)
                    probabilities /= len(models)
                    if not np.isfinite(probabilities).all():
                        raise ValueError("Model produced nonfinite probabilities")
                    labels = probabilities.argmax(axis=1)
                    writer.writerows((event_id, CLASS_NAMES[int(label)])
                                     for event_id, label in zip(raw.event_ids, labels))
                    rows += len(raw.event_ids)
                    counts.update(CLASS_NAMES[int(label)] for label in labels)
                    routes.update(prepared.route_names)
                    for unsupported in prepared.unsupported_rows:
                        reason = (unsupported.get("reason", "unsupported_ASA")
                                  if isinstance(unsupported, dict) else str(unsupported[1]))
                        fallbacks.update([reason])
                    if rows == len(raw.event_ids) or rows % (batch_size * 25) == 0:
                        print(f"Predicted {rows:,} rows ({time.monotonic() - started:.1f}s)", flush=True)
                output.flush()
                os.fsync(output.fileno())
            unique_rows = db.execute("SELECT COUNT(*) FROM ids").fetchone()[0]
            if rows == 0 or rows != unique_rows or sum(counts.values()) != rows:
                raise ValueError("Prediction input must be nonempty with exactly one output per ID")
        final_stat = input_path.stat()
        if (input_stat.st_size, input_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError("Input changed during prediction; refusing to publish mixed records")
        report = {
            "input_file": input_path.name, "input_sha256": file_sha256(input_path),
            "output_file": output_path.name, "output_sha256": file_sha256(temporary_csv),
            "input_rows": rows, "output_rows": rows, "unique_event_ids": unique_rows,
            "columns": ["event_id", "pred_label"], "label_counts": dict(counts),
            "route_counts": dict(routes), "ASA_router_fallbacks": dict(fallbacks),
            "folds": folds, "aggregation": "mean_probability" if len(folds) > 1 else "single_fold",
            "device": str(device), "batch_size": batch_size,
            "preprocessing_workers": workers,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "labels_read": False, "training_updates": 0,
            "published_development_score_applies_to_this_run": False,
        }
        # Same-filesystem hard link is an atomic, non-overwriting publication.
        # The partial name is removed in finally; no incomplete res.csv is exposed.
        os.link(temporary_csv, output_path)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"Saved {rows:,} predictions to {output_path}; report: {report_path}", flush=True)
        return report
    finally:
        if temporary_csv is not None:
            temporary_csv.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Official test .parquet or .csv")
    parser.add_argument("--output", type=Path, default=Path("res.csv"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--fold", default="ensemble", choices=("ensemble", "0", "1", "2"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1, help="Bounded raw preprocessing processes; each needs its own parser memory")
    parser.add_argument("--strict-asa", action="store_true", help="Reject unsupported ASA rather than use the linear classifier")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    predict_file(args.input, args.output, args.weights, args.fold, args.device,
                 args.batch_size, args.threads, args.strict_asa, args.report, args.workers)


if __name__ == "__main__":
    main()
