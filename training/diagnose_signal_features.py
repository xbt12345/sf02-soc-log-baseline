"""Read-only diagnosis of v1.1 features on frozen train/development rows only."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

PILOT_SHA = "82dd742a813618f8408f55055181cdf8e14f5dc32978237e5efd85c0f997021b"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def diagnose(args):
    started = time.perf_counter()
    run = Path(args.run_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    pilot_path = run / "soc_signal_pilot.py"
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if sha(pilot_path) != PILOT_SHA or result["code_sha256"] != PILOT_SHA:
        raise ValueError("Only the frozen v1.1 source is supported")
    for name, key in [("split_manifest.parquet", "manifest_sha256"), ("model.cbm", "model_sha256")]:
        if sha(run / name) != result[key]:
            raise ValueError("Artifact hash mismatch: " + name)
    spec = importlib.util.spec_from_file_location("frozen_signal_pilot", str(pilot_path))
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    source = Path(args.train_file)
    if sha(source) != pilot.EXPECTED["train.parquet"][1]:
        raise ValueError("Official training file hash mismatch")
    # Calibration/audit labels and messages are not used for this diagnosis.
    manifest = pd.read_parquet(run / "split_manifest.parquet")
    selected = manifest.loc[manifest.role.isin(["train", "development"])].sort_values("row_position").reset_index(drop=True)
    positions = selected.row_position.to_numpy()
    records = []
    offset = 0
    columns = ["event_id", "message_sanitized"] + pilot.ENTITIES
    for batch in pq.ParquetFile(source).iter_batches(batch_size=8192, columns=columns):
        left, right = np.searchsorted(positions, [offset, offset + batch.num_rows])
        if left < right:
            records.extend(batch.take(pa.array(positions[left:right] - offset)).to_pylist())
        offset += batch.num_rows
    if len(records) != len(selected):
        raise AssertionError("Selected row count mismatch")
    x, groups = pilot.feature_matrix(records)
    if not np.array_equal(groups, selected.group_id.to_numpy()):
        raise AssertionError("Reconstructed message groups differ from the frozen manifest")
    if [pilot.string(row["event_id"]) for row in records] != selected.event_id.tolist():
        raise AssertionError("Event identity mismatch")
    signatures = (x.astype(np.int64) * (1 << np.arange(x.shape[1], dtype=np.int64))).sum(axis=1)
    selected["signature"] = signatures
    names = [name for name, _ in pilot.RULES]
    for index, name in enumerate(names):
        selected[name] = x[:, index]
    selected.to_parquet(output / "train_development_feature_cache.parquet", index=False)
    model = CatBoostClassifier()
    model.load_model(str(run / "model.cbm"))
    model_predictions = np.asarray(pilot.LABELS)[np.argmax(model.predict_proba(x), axis=1)]
    reports = {}
    tables = {}
    for role in ("train", "development"):
        subset = selected.loc[selected.role == role]
        table = pd.crosstab(subset.signature, subset.label).reindex(columns=pilot.LABELS, fill_value=0)
        table["total"] = table[pilot.LABELS].sum(axis=1)
        table["active_signals"] = [", ".join(name for i, name in enumerate(names) if int(signature) & (1 << i)) or "<none>" for signature in table.index]
        table["majority_label"] = table[pilot.LABELS].idxmax(axis=1)
        table["class_count"] = (table[pilot.LABELS] > 0).sum(axis=1)
        table.to_csv(output / (role + "_signature_counts.csv"), encoding="utf-8-sig", index_label="signature")
        tables[role] = table
        mask = selected.role.to_numpy() == role
        row_predictions = model_predictions[mask]
        reports[role] = {
            "rows": len(subset), "distinct_feature_vectors": len(table),
            "features_ever_nonzero": int(x[mask].any(axis=0).sum()),
            "zero_vector_rows": int((subset.signature == 0).sum()),
            "mixed_label_feature_vectors": int((table.class_count > 1).sum()),
            "rows_in_mixed_label_vectors": int(table.loc[table.class_count > 1, "total"].sum()),
            "suspicious_rows": int((subset.label == "suspicious").sum()),
            "suspicious_rows_in_vectors_with_other_labels": int(table.loc[(table.benign + table.malicious) > 0, "suspicious"].sum()),
            "suspicious_majority_vectors": int((table.majority_label == "suspicious").sum()),
            "empirical_best_deterministic_accuracy_for_this_representation": float(table[pilot.LABELS].max(axis=1).sum() / len(subset)),
            "accuracy_ceiling_scope": "Empirical observed rows only; not a population or Macro-F1 bound; no validation claim.",
            "model_predictions": {label: int((row_predictions == label).sum()) for label in pilot.LABELS},
            "suspicious_signature_groups": table.loc[table.suspicious > 0].sort_values("suspicious", ascending=False).reset_index().to_dict(orient="records"),
        }
    # All source contrasts and examples come from training/development, not audit.
    source_counts = selected.groupby(["role", "product", "signature", "label"], dropna=False).size().reset_index(name="rows")
    source_counts.to_csv(output / "source_signature_counts.csv", encoding="utf-8-sig", index=False)
    training_signatures = set(tables["train"].loc[tables["train"].suspicious > 0].index)
    examples = []
    counts = {}
    for index, row in selected.iterrows():
        if row.role != "train" or row.signature not in training_signatures:
            continue
        key = (int(row.signature), row["label"], row["product"])
        if counts.get(key, 0) >= 2:
            continue
        counts[key] = counts.get(key, 0) + 1
        normalized = pilot.normalize(records[index])
        examples.append({"row_position": int(row.row_position), "label": row["label"], "product": row["product"],
                         "signature": int(row.signature), "active_signals": [name for i, name in enumerate(names) if x[index, i]],
                         "message_preview": normalized[:2400], "full_characters": len(normalized),
                         "preview_truncated": len(normalized) > 2400})
    save(output / "training_contrast_examples.json", examples)
    report = {"scope": "Frozen v1.1 training/development diagnostics only; no fitting, no threshold tuning, no audit evaluation",
              "source_code_sha256": PILOT_SHA, "manifest_byte_sha256": result["manifest_sha256"],
              "diagnostic_code_sha256": sha(Path(__file__)), "elapsed_seconds": time.perf_counter() - started,
              "roles": reports, "group_and_event_identity_reconstruction_passed": True}
    save(output / "feature_diagnosis.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    diagnose(parser.parse_args())
