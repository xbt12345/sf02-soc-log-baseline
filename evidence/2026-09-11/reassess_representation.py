"""Read-only full training-data audit; no fitting, no official validation files."""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LABELS = ["benign", "malicious", "suspicious"]
EXPECTED = "6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def counts(values):
    return {label: int(sum(values == label)) for label in LABELS}


def run(root, out):
    start = time.perf_counter()
    train = root / "data/official/train.parquet"
    actual = digest(train)
    if actual != EXPECTED:
        raise ValueError("Official training file hash mismatch")
    by_message = {}
    source_table = collections.Counter()
    label_counts = collections.Counter()
    empty_counts = collections.Counter()
    offset = 0
    for batch in pq.ParquetFile(train).iter_batches(batch_size=8192, columns=["message_sanitized", "label_binary", "product_name"]):
        messages, labels, products = [batch.column(i).to_pylist() for i in range(3)]
        for i, (message, label, product) in enumerate(zip(messages, labels, products)):
            # No normalization: null and empty have distinct identities.
            payload = b"null" if message is None else b"str:" + message.encode("utf-8")
            key = hashlib.sha256(payload).digest()
            if key not in by_message:
                by_message[key] = [0, 0, 0, offset + i]
            by_message[key][LABELS.index(label)] += 1
            label_counts[label] += 1
            is_empty = message is None or not message.strip()
            if is_empty:
                empty_counts[label] += 1
            source_table[(product or "<missing>", label, is_empty)] += 1
        offset += len(messages)
        if offset % (8192 * 32) == 0:
            print("Read {} training rows".format(offset), flush=True)
    values = np.asarray(list(by_message.values()), dtype=np.int64)
    c = values[:, :3]
    mixed = (c > 0).sum(axis=1) > 1
    ids = np.argsort(-(c.sum(axis=1) - c.max(axis=1)))[0:12]
    conflicts = [{"first_row_position": int(values[i, 3]), "counts": dict(zip(LABELS, map(int, c[i])))} for i in ids if mixed[i]]
    report = {
        "scope": "Full official train only; all outputs are development diagnostics, not unseen validation",
        "input_sha256": actual,
        "rows": int(offset),
        "labels": dict(label_counts),
        "empty_or_whitespace_message_by_class": dict(empty_counts),
        "raw_message_exact_hash": {
            "unique_groups": len(by_message),
            "mixed_label_groups": int(mixed.sum()),
            "rows_in_mixed_label_groups": int(c[mixed].sum()),
            "minimum_empirical_mistakes_for_deterministic_message_only_classifier": int((c.sum(axis=1) - c.max(axis=1)).sum()),
            "per_class_raw_unique_groups_including_empty": {label: int((c[:, i] > 0).sum()) for i, label in enumerate(LABELS)},
            "largest_conflict_groups": conflicts,
            "interpretation": "Same exact visible message may be separate real events. Conflict may mean missing context or differing annotations, not necessarily mislabeled data. This bound applies only to a deterministic message-only classifier on this observed dataset; not to models with additional valid evidence or generalization. Hash collision not separately checked."
        },
        "source_label_message_presence": [{"source": source, "label": label, "message_empty": empty, "rows": count} for (source, label, empty), count in sorted(source_table.items())]
    }
    old = pd.read_parquet(root / "artifacts/signal_pilot_local_v11_20260911/split_manifest.parquet")
    plan = pd.read_parquet(root / "artifacts/behavior_v2_local_20260911/component_split_manifest.parquet")
    merged = old.merge(plan[["row_position", "excluded_cross_role", "model_scope", "component", "v2_usage"]], on="row_position", validate="one_to_one")
    train_dev = merged.role.isin(["train", "development"])
    report["v2_selection_by_class"] = {label: {
        "old_train_dev_rows": int((train_dev & (merged.label == label)).sum()),
        "retained_rows": int((merged.model_scope & (merged.label == label)).sum()),
        "retained_fraction": float((merged.model_scope & (merged.label == label)).sum() / (train_dev & (merged.label == label)).sum())
    } for label in LABELS}
    report["v2_usage_class_component_counts"] = []
    for (usage, label), group in merged[merged.model_scope].groupby(["v2_usage", "label"]):
        sizes = group.groupby("component").size().to_numpy()
        report["v2_usage_class_component_counts"].append({"usage": usage, "label": label, "rows": len(group), "components": len(sizes), "largest_component_rows": int(sizes.max())})
    report["elapsed_seconds"] = time.perf_counter() - start
    report["audit_code_sha256"] = digest(Path(__file__))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "source_label_message_presence"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep prior evidence; output already exists")
    run(args.root, args.output)
