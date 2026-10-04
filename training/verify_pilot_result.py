"""Read-only cross-runtime verification for the frozen signal-pilot-v1.1 run."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

KNOWN_CODE_SHA = "82dd742a813618f8408f55055181cdf8e14f5dc32978237e5efd85c0f997021b"
REFERENCE_MANIFEST = "110999c366b4d05dd6d6da9e0e3e39c91eb064a27d6ae0084c6ab3ca12e2a77e"
REFERENCE_DECISIONS = "34fa545526683904d2cef54a2567239d72ceeffdc28fa40a30e433e386a433dc"
INPUT_HASHES = {"train.parquet": "6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742",
                "valid_input.parquet": "731af597711aae0b870912c36fc1a0f592fd6e53d911c7fec6fe1405ec1a97b0"}
MANIFEST_COLUMNS = ["dataset_id", "row_position", "event_id", "group_id", "role", "label", "product"]
DECISION_COLUMNS = ["dataset_id", "row_position", "event_id", "group_id", "role", "label", "prediction", "needs_review_no_action_signal"]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha(frame, columns):
    # Canonical UTF-8 JSON arrays, column order explicit, rows sorted by official
    # row position. No Parquet compression/metadata or floating-point scores.
    if not frame.row_position.is_unique:
        raise ValueError("Duplicate row positions")
    frame = frame.sort_values("row_position").loc[:, columns]
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False, name=None):
        values = []
        for name, value in zip(columns, row):
            if pd.isna(value):
                value = None
            elif name == "row_position":
                value = int(value)
            elif name == "needs_review_no_action_signal":
                value = bool(value)
            else:
                value = str(value)
            values.append(value)
        encoded = json.dumps(values, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n"
        digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()


def verify(run_dir):
    run = Path(run_dir)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    checks = {}
    for filename, key in [("soc_signal_pilot.py", "code_sha256"), ("configuration.json", "configuration_sha256"),
                          ("split_manifest.parquet", "manifest_sha256"), ("model.cbm", "model_sha256")]:
        checks[filename] = file_sha(run / filename) == result[key]
    checks["known_v11_source"] = result["code_sha256"] == KNOWN_CODE_SHA
    identity = json.loads((run / "input_identity.json").read_text(encoding="utf-8"))
    checks["pilot_input_hash_receipts"] = all(identity[name]["sha256"] == digest for name, digest in INPUT_HASHES.items())
    manifest = pd.read_parquet(run / "split_manifest.parquet")
    predictions = pd.read_parquet(run / "audit_predictions.parquet")
    checks["groups_do_not_cross_roles"] = bool((manifest.groupby("group_id").role.nunique() == 1).all())
    checks["audit_rows_match_manifest"] = set(predictions.row_position) == set(manifest.loc[manifest.role == "audit", "row_position"])
    manifest_sha = content_sha(manifest, MANIFEST_COLUMNS)
    decision_sha = content_sha(predictions, DECISION_COLUMNS)
    checks["manifest_content_matches_local"] = manifest_sha == REFERENCE_MANIFEST
    checks["audit_decisions_match_local"] = decision_sha == REFERENCE_DECISIONS
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    # Recompute class recall from the actual decision file, not just metrics.json.
    recalls = {}
    for label in ("benign", "malicious", "suspicious"):
        mask = predictions.label == label
        recall = float((predictions.loc[mask, "prediction"] == label).mean()) if mask.any() else None
        recalls[label] = {"support": int(mask.sum()), "recall": recall}
        reported = metrics["audit"]["classes"][label]["recall"]
        checks["recall_matches_report_" + label] = (recall is None and reported is None) or (recall is not None and reported is not None and abs(recall - reported) < 1e-12)
    return {"all_checks_passed": all(checks.values()), "checks": checks,
            "manifest_content_sha256": manifest_sha, "audit_decisions_content_sha256": decision_sha,
            "recomputed_audit_class_recall": recalls,
            "calibration_derived_thresholds": [{key: row[key] for key in ["requested_calibration_fpr_budget", "audit_benign_alert_rate", "audit_malicious_alert_rate", "audit_suspicious_alert_rate"]} for row in metrics["calibration_derived_thresholds"]],
            "scope": "No training and no writes. Local-reference split/decision contents compared; probability arrays are not compared. Input hashes checked against pilot receipts, original data files not rehashed."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    report = verify(args.run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if report["all_checks_passed"] else 1)
