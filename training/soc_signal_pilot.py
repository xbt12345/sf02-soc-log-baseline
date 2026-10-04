#!/usr/bin/env python3
"""Python 3.8+ CPU pilot. Lexical signals are weak evidence, not attack labels."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             log_loss, precision_recall_fscore_support,
                             roc_auc_score)

VERSION = "signal-pilot-v1.1"
LABELS = ["benign", "malicious", "suspicious"]
ROLES = ["train", "development", "calibration", "audit"]
ENTITIES = ["src_ip", "dst_ip", "src_host", "dst_host", "username"]
EXPECTED = {
    "train.parquet": (2056871, "6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742"),
    "valid_input.parquet": (2014052, "731af597711aae0b870912c36fc1a0f592fd6e53d911c7fec6fe1405ec1a97b0"),
}
# Fixed before seeing pilot scores. Only occurrence features; no product, length,
# missingness, parsing-success, severity, entity, timestamp or identifier features.
RULE_TEXT = [
    ("authentication_failure", r"\bauthentication (?:failure|failed)\b|\bfailed password\b|\blog(?:in|on) (?:failed|failure)\b"),
    ("session_opened", r"\bsession opened\b|\blogged on\b|\blogin success(?:ful)?\b"),
    ("session_closed", r"\bsession closed\b|\blogged off\b|\bhas exited tty\b"),
    ("access_denied", r"\b(?:access|permission) denied\b|\bdeny (?:tcp|udp|icmp)\b"),
    ("packet_dropped", r"\b(?:dropped|dropping) (?:packet|connection)s?\b|\b(?:packet|connection)s? (?:dropped|rejected)\b"),
    ("connection_built", r"\bbuilt (?:inbound|outbound)\b|\bconnection (?:opened|established)\b"),
    ("connection_closed", r"\bteardown\b|\bconnection (?:closed|terminated)\b"),
    ("file_written", r"\ba file written\b|\bfile (?:created|written)\b|\bcreated (?:a )?file\b"),
    ("process_created", r"\b(?:new )?process (?:created|started)\b|\bprocess creation\b"),
    ("process_terminated", r"\bprocess (?:terminated|killed)\b"),
    ("powershell_mentioned", r"\bpowershell(?:\.exe)?\b"),
    ("encoded_command", r"-(?:encodedcommand|enc)\s+\S+"),
    ("scheduled_task_action", r"\bscheduled task (?:created|registered|updated|deleted)\b|\bschtasks(?:\.exe)?\s+/(?:create|change|delete)\b"),
    ("account_created", r"\b(?:user )?account (?:was )?created\b|\bnew user (?:was )?created\b"),
    ("group_member_added", r"\bmember (?:was )?added\b|\badded to (?:a |the )?(?:security|local|global|universal|group)\b"),
    ("group_member_removed", r"\bmember (?:was )?removed\b|\bremoved from (?:a |the )?(?:security|local|global|universal|group)\b"),
    ("password_change", r"\bpassword (?:was )?(?:changed|reset)\b|\bchange (?:an account.s |the )?password\b"),
    ("audit_log_cleared", r"\b(?:audit|security) log (?:was )?cleared\b"),
    ("service_installed", r"\bservice (?:was )?installed\b"),
    ("content_downloaded", r"\bdownload(?:ed|ing) (?:content|file|update)\b"),
    # Explicit true is necessary: the mere presence of blocked/quarantine keys
    # in endpoint JSON must not count when their values are false.
    ("quarantine_true", r'"quarantine_file"\s*:\s*true\b|\bfile (?:was )?quarantined\b'),
    ("process_kill_true", r'"(?:kill_process|process_killed)"\s*:\s*true\b'),
    ("operation_blocked_true", r'"(?:operation_blocked|process_blocked|network_connection_blocked)"\s*:\s*true\b'),
]
RULES = [(name, re.compile(pattern, re.I)) for name, pattern in RULE_TEXT]
PLACEHOLDER = re.compile(r"\b(?:USER|ORG|CRED)(?:-(?:USER|ORG|CRED))*-\d+(?:-\d+)*", re.I)
IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
MAC = re.compile(r"\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", re.I)
UUID = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)
HEX = re.compile(r"\b[0-9a-f]{16,}\b", re.I)
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[t ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?)?", re.I)
CLOCK = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b")
NUMBER = re.compile(r"\d+")
SPACE = re.compile(r"\s+")


def string(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize(row):
    text = string(row.get("message_sanitized")).casefold()
    # Known entity values are removed from the message, not turned into features.
    # Exact bounded replacement cannot identify every implicit/unknown entity.
    values = sorted({string(row.get(k)).casefold() for k in ENTITIES}, key=len, reverse=True)
    for value in values:
        if len(value) >= 3 and value not in ("unknown", "null", "none", "nan"):
            text = re.sub(r"(?<!\w)" + re.escape(value) + r"(?!\w)", "<entity>", text)
    for pattern in (DATE, CLOCK, UUID, MAC, IP, HEX, PLACEHOLDER):
        text = pattern.sub("<entity>", text)
    return SPACE.sub(" ", text).strip()


def extract(row):
    normalized = normalize(row)
    vector = np.array([bool(pattern.search(normalized)) for _, pattern in RULES], dtype=np.uint8)
    # Deliberately conservative, approximate message grouping. Numeric variation
    # does not make a new template. Similar-but-not-equal templates may remain.
    template = NUMBER.sub("<number>", normalized)
    group = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return vector, group


def role_for(group, seed):
    digest = hashlib.sha256((str(seed) + ":" + group).encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 70 else "development" if bucket < 80 else "calibration" if bucket < 90 else "audit"


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def log(message):
    print(time.strftime("%H:%M:%S") + " " + message, flush=True)


def verify_inputs(data_dir):
    result = {}
    for name, (rows, digest) in EXPECTED.items():
        path = data_dir / name
        actual = file_hash(path)
        parquet = pq.ParquetFile(path)
        result[name] = {"path": str(path.resolve()), "sha256": actual, "rows": parquet.metadata.num_rows}
        if actual != digest or parquet.metadata.num_rows != rows:
            raise ValueError("Official file identity mismatch: " + name)
    return result


def sample_records(path, size, seed):
    parquet = pq.ParquetFile(path)
    count = parquet.metadata.num_rows
    if not 1 <= size <= count:
        raise ValueError("sample-size must be within 1..train row count")
    # Uniform over all row positions, without using labels or taking the head.
    positions = np.sort(np.random.default_rng(seed).choice(count, size=size, replace=False))
    columns = ["event_id", "timestamp", "pipeline", "product_name", "vendor_name",
               "message_sanitized", "label_binary"] + ENTITIES
    records = []
    offset = 0
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        left, right = np.searchsorted(positions, [offset, offset + batch.num_rows])
        if left != right:
            selected = batch.take(pa.array(positions[left:right] - offset))
            records.extend(selected.to_pylist())
        offset += batch.num_rows
    if len(records) != size:
        raise AssertionError("Sample row count mismatch")
    return records, positions


def feature_matrix(records):
    x = np.empty((len(records), len(RULES)), dtype=np.uint8)
    groups = []
    for index, row in enumerate(records):
        x[index], group = extract(row)
        groups.append(group)
        if (index + 1) % 25000 == 0:
            log("Extracted {}/{} rows".format(index + 1, len(records)))
    return x, np.asarray(groups)


def rename_known_entities(original):
    """One-to-one, consistent renaming; do not merge distinct entity identities."""
    changed = dict(original)
    message = string(original.get("message_sanitized"))
    mapping = {}
    generated = set()
    def replacement(value):
        key = value.casefold()
        if key in generated:
            return value
        if key not in mapping:
            number = 900000 + len(mapping)
            candidate = "USER-" + str(number)
            while candidate.casefold() in message.casefold() or candidate.casefold() in generated:
                number += 1
                candidate = "USER-" + str(number)
            mapping[key] = candidate
            generated.add(candidate.casefold())
        return mapping[key]
    values = sorted({string(original.get(k)) for k in ENTITIES if len(string(original.get(k))) >= 3}, key=lambda s: (-len(s), s))
    for value in values:
        target = replacement(value)
        message = re.sub(r"(?<!\w)" + re.escape(value) + r"(?!\w)", target, message, flags=re.I)
    message = PLACEHOLDER.sub(lambda match: replacement(match.group(0)), message)
    changed["message_sanitized"] = message
    for key in ENTITIES:
        value = string(original.get(key))
        if len(value) >= 3:
            changed[key] = mapping[value.casefold()]
    return changed


def representation_checks(records, x, limit=256):
    # Actual sample transformations plus a semantic fixture in the test suite.
    count = min(limit, len(records))
    flips = {"metadata_and_missing_representation": 0, "known_entities_in_message": 0,
             "iso_dates_and_clock_in_message": 0, "reverse_batch_order": 0}
    for index in range(count):
        original = records[index]
        changed = dict(original)
        for key in ("timestamp", "event_id", "pipeline", "product_name", "vendor_name"):
            changed[key] = "changed metadata"
        for key in ENTITIES:
            if not string(changed.get(key)):
                changed[key] = ""
        flips["metadata_and_missing_representation"] += int(not np.array_equal(extract(changed)[0], x[index]))
        changed = rename_known_entities(original)
        flips["known_entities_in_message"] += int(not np.array_equal(extract(changed)[0], x[index]))
        changed = dict(original)
        changed["message_sanitized"] = CLOCK.sub("13:14:15", DATE.sub("2035-02-03T13:14:15Z", string(original.get("message_sanitized"))))
        flips["iso_dates_and_clock_in_message"] += int(not np.array_equal(extract(changed)[0], x[index]))
    for index in reversed(range(count)):
        flips["reverse_batch_order"] += int(not np.array_equal(extract(records[index])[0], x[index]))
    return {"sample_rows": count, "feature_flip_counts": flips, "passed": not any(flips.values()),
            "scope": "Fixed rules; named perturbations only. Not all date/entity formats or semantic equivalence."}


def metric_summary(y, probabilities):
    predicted = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y, predicted, labels=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=[0, 1, 2], zero_division=0)
    per_class = {}
    for label_index, label in enumerate(LABELS):
        binary = y == label_index
        both = bool(binary.any() and (~binary).any())
        per_class[label] = {"support": int(support[label_index]),
                            "precision": float(precision[label_index]),
                            "recall": float(recall[label_index]) if binary.any() else None,
                            "f1": float(f1[label_index]) if binary.any() else None,
                            "average_precision": float(average_precision_score(binary, probabilities[:, label_index])) if both else None,
                            "roc_auc": float(roc_auc_score(binary, probabilities[:, label_index])) if both else None}
    normal = int((y == 0).sum())
    false_alerts = int(((y == 0) & (predicted != 0)).sum())
    return {"rows": len(y), "classes": per_class, "confusion_matrix": matrix.tolist(),
            "matrix_order": LABELS, "accuracy": float((y == predicted).mean()),
            "macro_f1_all_three_zero_for_absent": float(f1.mean()),
            "all_classes_present": bool((support > 0).all()),
            "normal_false_alerts": false_alerts,
            "false_alerts_per_10000_normal": 10000.0 * false_alerts / normal if normal else None,
            "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
            "multiclass_brier_sum_over_classes": float(np.mean(np.sum((probabilities - np.eye(3)[y]) ** 2, axis=1))),
            "probabilities_calibrated": False}


def threshold_from_normal(scores, budget):
    if not len(scores):
        raise ValueError("Calibration has no normal observations")
    allowed = int(math.floor(budget * len(scores)))
    ordered = np.sort(scores)[::-1]
    return float(np.nextafter(ordered[allowed], np.inf))


def evaluate_thresholds(cal_y, cal_p, cal_groups, audit_y, audit_p):
    output = []
    normal = cal_y == 0
    normal_groups = len(np.unique(cal_groups[normal]))
    cal_risk = 1.0 - cal_p[:, 0]
    audit_risk = 1.0 - audit_p[:, 0]
    for budget in (0.0001, 0.001, 0.01):
        threshold = threshold_from_normal(cal_risk[normal], budget)
        entry = {"requested_calibration_fpr_budget": budget, "threshold": threshold,
                 "comparison": "1 - P(benign) >= threshold", "calibration_normal_rows": int(normal.sum()),
                 "calibration_normal_groups": normal_groups,
                 "empirical_calibration_fpr": float((cal_risk[normal] >= threshold).mean()),
                 "independent_low_fpr_guarantee": False,
                 "support_warning": "No independence proof; template group count does not establish a reliable low FPR guarantee."}
        for index, label in enumerate(LABELS):
            mask = audit_y == index
            entry["audit_" + label + "_alert_rate"] = float((audit_risk[mask] >= threshold).mean()) if mask.any() else None
            entry["audit_" + label + "_support"] = int(mask.sum())
        output.append(entry)
    return output


def group_bootstrap(y, probabilities, groups, seed, repeats=200):
    names, inverse = np.unique(groups, return_inverse=True)
    tables = np.zeros((len(names), 3, 3), dtype=np.int64)
    predicted = np.argmax(probabilities, axis=1)
    np.add.at(tables, (inverse, y, predicted), 1)
    rng = np.random.default_rng(seed)
    macro, fpr, mal_recall = [], [], []
    missing_class_repeats = 0
    for _ in range(repeats):
        matrix = tables[rng.integers(len(names), size=len(names))].sum(axis=0)
        row = matrix.sum(axis=1)
        denominator = row + matrix.sum(axis=0)
        values = np.divide(2.0 * np.diag(matrix), denominator, out=np.zeros(3), where=denominator > 0)
        if (row > 0).all():
            macro.append(float(values.mean()))
        else:
            missing_class_repeats += 1
        if row[0]:
            fpr.append(float((row[0] - matrix[0, 0]) / row[0]))
        if row[1]:
            mal_recall.append(float(matrix[1, 1] / row[1]))
    def interval(values):
        return np.quantile(values, [0.025, 0.975]).tolist() if values else None
    no_normal_errors = not bool(((y == 0) & (predicted != 0)).any())
    return {"method": "Percentile bootstrap of observed message groups, preserving all sampled rows per group",
            "repeats": repeats, "groups": len(names), "repeats_missing_a_class": missing_class_repeats,
            "macro_f1_95_percent_interval_conditional_on_class_presence": interval(macro),
            "normal_fpr_95_percent_interval": None if no_normal_errors else interval(fpr),
            "zero_error_interval_note": "With zero observed errors, ordinary bootstrap only repeats zero and cannot establish an upper risk bound; interval suppressed." if no_normal_errors else None,
            "malicious_recall_95_percent_interval": interval(mal_recall),
            "limitation": "Descriptive within sampled source mixture; no external-domain guarantee."}


def coverage_report(records, x, groups, roles, y):
    output = {}
    sources = np.asarray([string(row.get("product_name")) or "<missing>" for row in records])
    for source in sorted(set(sources)):
        mask = sources == source
        output[source] = {"rows": int(mask.sum()), "message_groups": len(np.unique(groups[mask])),
                          "class_counts": {label: int(((y == i) & mask).sum()) for i, label in enumerate(LABELS)},
                          "no_action_signal_rows": int((~x[mask].any(axis=1)).sum())}
    json_like = invalid = empty = 0
    for row in records:
        text = string(row.get("message_sanitized"))
        empty += int(not text)
        if text.startswith(("{", "[")):
            json_like += 1
            try:
                json.loads(text)
            except (ValueError, RecursionError):
                invalid += 1
    sizes = pd.Series(groups).value_counts()
    frame = pd.DataFrame({"group": groups, "y": y})
    mixed = int((frame.groupby("group")["y"].nunique() > 1).sum())
    return {"per_product": output, "empty_messages": empty, "json_like_messages": json_like,
            "invalid_json_like_messages": invalid, "json_diagnostic_used_as_feature": False,
            "no_action_signal_rows": int((~x.any(axis=1)).sum()),
            "action_coverage_per_class": {label: {"rows": int((y == i).sum()), "with_signal": int(x[y == i].any(axis=1).sum())} for i, label in enumerate(LABELS)},
            "group_count": len(sizes), "largest_group_rows": int(sizes.iloc[0]),
            "top_10_group_sizes": [int(v) for v in sizes.head(10)], "groups_with_mixed_labels": mixed,
            "split_class_counts": {role: {label: int(((roles == role) & (y == i)).sum()) for i, label in enumerate(LABELS)} for role in ROLES},
            "split_group_counts": {role: len(np.unique(groups[roles == role])) for role in ROLES},
            "grouping_limit": "Exact equality after fixed normalization, not complete near-duplicate, incident, or entity isolation."}


def run(args, out):
    started = time.perf_counter()
    shutil.copyfile(Path(__file__), out / "soc_signal_pilot.py")
    configuration = vars(args).copy()
    configuration.update({"version": VERSION, "labels": LABELS, "feature_rules": RULE_TEXT,
                          "role_percentages": [70, 10, 10, 10], "class_weights": None,
                          "source_holdout_executed": False, "external_validation_executed": False})
    save_json(out / "configuration.json", configuration)
    log("Verifying both official input hashes (validation labels are not read)")
    identities = verify_inputs(Path(args.data_dir))
    save_json(out / "input_identity.json", identities)
    log("Uniformly sampling {} training rows".format(args.sample_size))
    records, positions = sample_records(Path(args.data_dir) / "train.parquet", args.sample_size, args.seed)
    y_text = [string(row["label_binary"]) for row in records]
    unknown = set(y_text) - set(LABELS)
    if unknown:
        raise ValueError("Unknown target labels: " + repr(unknown))
    y = np.asarray([LABELS.index(label) for label in y_text])
    x, groups = feature_matrix(records)
    roles = np.asarray([role_for(group, args.seed) for group in groups])
    # Manifest is persisted before any fitting or evaluation.
    manifest = pd.DataFrame({"dataset_id": "official_train", "row_position": positions,
                             "event_id": [string(row["event_id"]) for row in records],
                             "group_id": groups, "role": roles, "label": y_text,
                             "product": [string(row.get("product_name")) for row in records]})
    manifest.to_parquet(out / "split_manifest.parquet", index=False)
    if manifest.groupby("group_id")["role"].nunique().max() != 1:
        raise AssertionError("Group crosses roles")
    coverage = coverage_report(records, x, groups, roles, y)
    save_json(out / "coverage.json", coverage)
    checks = representation_checks(records, x)
    save_json(out / "representation_checks.json", checks)
    if not checks["passed"]:
        raise ValueError("Representation invariance failed; see representation_checks.json")
    lacking = [role for role in ROLES if set(y[roles == role]) != {0, 1, 2}]
    if lacking:
        save_json(out / "result.json", {"status": "blocked_by_class_coverage", "roles": lacking,
                                       "action": "Do not change seed to chase coverage or replace grouping with row-random split."})
        log("STOP: insufficient class coverage in " + ", ".join(lacking))
        return 2
    log("Fitting unweighted CPU CatBoost; early stopping uses development only")
    model = CatBoostClassifier(iterations=args.iterations, depth=5, learning_rate=0.08,
                               loss_function="MultiClass", eval_metric="MultiClass",
                               task_type="CPU", thread_count=args.threads, random_seed=args.seed,
                               allow_writing_files=False, verbose=False)
    train = roles == "train"
    dev = roles == "development"
    feature_names = [name for name, _ in RULES]
    model.fit(Pool(x[train], y[train], feature_names=feature_names),
              eval_set=Pool(x[dev], y[dev], feature_names=feature_names),
              early_stopping_rounds=30, use_best_model=True)
    if list(model.classes_) != [0, 1, 2]:
        raise AssertionError("Unexpected model class order")
    model.save_model(str(out / "model.cbm"))
    predictions = model.predict_proba(x)
    batch_error = float(np.max(np.abs(predictions[:128] - model.predict_proba(x[:128][::-1])[::-1])))
    single_error = max(float(np.max(np.abs(predictions[i] - model.predict_proba(x[i:i + 1])[0]))) for i in range(min(8, len(x))))
    checks["prediction_batch_max_abs_error"] = batch_error
    checks["prediction_single_row_max_abs_error"] = single_error
    save_json(out / "representation_checks.json", checks)
    if max(batch_error, single_error) > 1e-12:
        raise ValueError("Prediction depends on batching")
    audit = roles == "audit"
    calibration = roles == "calibration"
    metrics = {role: metric_summary(y[roles == role], predictions[roles == role]) for role in ROLES[1:]}
    all_normal = np.zeros((int(audit.sum()), 3))
    all_normal[:, 0] = 1
    metrics["audit_all_benign_control"] = metric_summary(y[audit], all_normal)
    metrics["calibration_derived_thresholds"] = evaluate_thresholds(y[calibration], predictions[calibration], groups[calibration], y[audit], predictions[audit])
    metrics["audit_group_bootstrap"] = group_bootstrap(y[audit], predictions[audit], groups[audit], args.seed)
    per_source = {}
    sources = manifest["product"].to_numpy()
    for source in sorted(set(sources[audit])):
        mask = audit & (sources == source)
        per_source[source or "<missing>"] = metric_summary(y[mask], predictions[mask])
    metrics["audit_per_product"] = per_source
    save_json(out / "metrics.json", metrics)
    audit_output = manifest.loc[audit].copy()
    for index, label in enumerate(LABELS):
        audit_output["p_" + label] = predictions[audit, index]
    audit_output["prediction"] = np.asarray(LABELS)[np.argmax(predictions[audit], axis=1)]
    audit_output["needs_review_no_action_signal"] = ~x[audit].any(axis=1)
    audit_output.to_parquet(out / "audit_predictions.parquet", index=False)
    wrong = np.flatnonzero(audit & (np.argmax(predictions, axis=1) != y))
    errors = []
    # At most 10 per true/predicted pair; diagnostics only, never fed back here.
    counts = {}
    for index in wrong:
        prediction = int(np.argmax(predictions[index]))
        pair = (int(y[index]), prediction)
        if counts.get(pair, 0) >= 10:
            continue
        counts[pair] = counts.get(pair, 0) + 1
        normalized = normalize(records[index])
        errors.append({"row_position": int(positions[index]), "event_id": string(records[index]["event_id"]),
                       "truth": LABELS[y[index]], "prediction": LABELS[prediction], "product": sources[index],
                       "matched_evidence": {name: pattern.search(normalized).group(0) for name, pattern in RULES if pattern.search(normalized)},
                       "message_preview": normalized[:1200], "preview_truncated": len(normalized) > 1200})
    save_json(out / "audit_error_examples.json", errors)
    save_json(out / "feature_importance.json", dict(zip([name for name, _ in RULES], model.get_feature_importance().tolist())))
    peak_bytes = None
    try:
        import resource
        peak_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    except ImportError:
        pass
    result = {"status": "pilot_executed_not_transfer_validated", "version": VERSION,
              "elapsed_seconds": time.perf_counter() - started, "peak_rss_bytes": peak_bytes,
              "peak_rss_note": "Unavailable on Windows without an OS monitor" if peak_bytes is None else "Process high-water resident set, not container-wide memory",
              "sample_rows": len(records), "audit_rows": int(audit.sum()), "fitted_tree_count": int(model.tree_count_),
              "audit_macro_f1": metrics["audit"]["macro_f1_all_three_zero_for_absent"],
              "audit_all_benign_macro_f1": metrics["audit_all_benign_control"]["macro_f1_all_three_zero_for_absent"],
              "audit_malicious_recall": metrics["audit"]["classes"]["malicious"]["recall"],
              "audit_false_alerts_per_10000_normal": metrics["audit"]["false_alerts_per_10000_normal"],
              "no_action_signal_fraction": coverage["no_action_signal_rows"] / len(records),
              "code_sha256": file_hash(out / "soc_signal_pilot.py"), "configuration_sha256": file_hash(out / "configuration.json"),
              "manifest_sha256": file_hash(out / "split_manifest.parquet"), "model_sha256": file_hash(out / "model.cbm"),
              "python": sys.version, "platform": sys.platform,
              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "catboost")},
              "limitations": ["Lexical evidence can be negated, quoted, redacted, or source-specific; these signals are not attack truth.",
                              "Conservative message groups do not guarantee complete incident or near-template isolation.",
                              "Hash-group splitting preserves rows within groups but may shift source and class proportions.",
                              "No source holdout, external environment test, temporal chain, weighted comparison, probability calibration, or official submission executed.",
                              "Audit has now been inspected; do not reuse it as an unseen test after adapting to its errors."]}
    save_json(out / "result.json", result)
    log("Completed pilot; no transfer claim. Results: " + str(out.resolve()))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory only; never overwrite an experiment")
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.threads < 1 or args.iterations < 1:
        parser.error("threads and iterations must be positive")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        return run(args, out)
    except Exception as exc:
        save_json(out / "failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                                         "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
