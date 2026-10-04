#!/usr/bin/env python3
"""Structured behavior experiment; Python 3.8+, CPU, no external service."""
import argparse
import hashlib
import importlib.util
import json
import math
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

VERSION = "behavior-v2.0"
V11_SHA = "82dd742a813618f8408f55055181cdf8e14f5dc32978237e5efd85c0f997021b"
LABELS = ["benign", "malicious", "suspicious"]
CORE = ["auth_failure", "auth_success", "session_closed", "object_handle_closed",
        "bad_credentials", "unknown_user", "invalid_passcode", "no_response", "policy_denied",
        "network_denied", "network_allowed", "network_closed", "account_created",
        "group_member_added", "group_member_removed", "password_changed", "audit_log_cleared",
        "service_installed", "process_created", "process_terminated", "file_written",
        "file_quarantined", "process_killed", "operation_blocked"]
CONTEXT = ["protocol_tcp", "protocol_udp", "protocol_icmp", "service_dns", "service_http",
           "service_ssh", "service_rdp", "service_smb", "service_ldap", "service_ntp",
           "service_other", "auth_kerberos", "auth_ntlm", "logon_network", "logon_remote",
           "http_get", "http_post", "http_write_method", "http_status_2xx", "http_status_4xx", "http_status_5xx"]
FEATURES = CORE + CONTEXT
ENTITY_COLUMNS = ["src_ip", "dst_ip", "src_host", "dst_host", "username"]
REASONS = {"invalid_passcode": "invalid_passcode", "no_response": "no_response",
           "denied_by_policy": "policy_denied", "denied_network": "policy_denied"}
EVENT_ACTIONS = {"4624": "auth_success", "4625": "auth_failure", "4634": "session_closed", "4658": "object_handle_closed"}
LEXICAL = {
    "auth_failure": r"\bauthentication (?:failure|failed)\b|\bfailed password\b|\ban account failed to log on\b|\bauth_failure\b",
    "auth_success": r"\bsession opened\b|\blogin successful\b|\ban account was successfully logged on\b",
    "session_closed": r"\bsession closed\b|\bhas exited tty\b",
    "account_created": r"\b(?:user )?account (?:was )?created\b",
    "group_member_added": r"\bmember (?:was )?added\b|\badded to (?:a |the )?(?:security|local|global|universal|group)\b",
    "group_member_removed": r"\bmember (?:was )?removed\b",
    "password_changed": r"\bpassword (?:was )?(?:changed|reset)\b",
    "audit_log_cleared": r"\b(?:audit|security) log (?:was )?cleared\b",
    "service_installed": r"\bservice (?:was )?installed\b",
    "process_created": r"\b(?:new )?process (?:was )?(?:created|started)\b",
    "process_terminated": r"\bprocess (?:was )?(?:terminated|killed)\b",
    "file_written": r"\ba file written\b|\bfile (?:was )?(?:created|written)\b",
}
LEXICAL = {key: re.compile(value, re.I) for key, value in LEXICAL.items()}
SCALARS = re.compile(r'"(?P<key>code|event_id|outcome|result|reason|status|substatus|logontype|logon_type|authenticationpackagename|requestmethod|request_method|quarantine_file|kill_process|process_killed|operation_blocked|process_blocked|network_connection_blocked)"\s*:\s*(?P<value>"(?:\\.|[^"\\])*"|true|false|null|-?\d+)\s*(?=[,}\]])', re.I)
XML_EVENT = re.compile(r"<EventID(?:\s[^>]*)?>(\d+)</EventID>", re.I)
XML_FIELD = re.compile(r'<Data\s+Name=[\"\x27](LogonType|AuthenticationPackageName|Status|SubStatus)[\"\x27]\s*>([^<]*)</Data>', re.I)
ASA = re.compile(r"\b(deny)\s+(tcp|udp|icmp6?)\s+src\s+(\S+)\s+dst\s+(\S+)", re.I)
BUILT = re.compile(r"\b(built|teardown)\s+(?:(?:inbound|outbound)\s+)?(tcp|udp|icmp)\b", re.I)
KV_METHOD = re.compile(r"\brequestMethod=(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b", re.I)
KV_STATUS = re.compile(r"\b(?:outcome|status|statusCode)=(\d{3})(?=\s|$)", re.I)
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
REDACTED_TOKEN = re.compile(r"[\w.-]*(?:user|cred|org)-[\w.-]+", re.I)
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[t ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?)?", re.I)
SYSLOG_DATE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:\s+\d{4})?(?:\s+\d{1,2}:\d{2}:\d{2})?", re.I)
IDENTIFIER = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b|\b[0-9a-f]{16,}\b|\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", re.I)


def text(value):
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value).strip()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def log(message):
    print(time.strftime("%H:%M:%S") + " " + message, flush=True)


def canonical_group(row):
    raw = text(row.get("message_sanitized")).casefold()
    # Only grouping/reporting uses this text. No vocabulary is fitted to it.
    raw = re.sub(r"\\[nrt]", " ", raw)
    for value in sorted({text(row.get(key)).casefold() for key in ENTITY_COLUMNS}, key=lambda s: (-len(s), s)):
        if len(value) >= 3 and value not in ("unknown", "none", "null", "nan"):
            raw = re.sub(r"(?<!\w)" + re.escape(value) + r"(?!\w)", "<id>", raw)
    for pattern in (ISO_DATE, SYSLOG_DATE):
        raw = pattern.sub("<date>", raw)
    raw = re.sub(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b", "<date>", raw)
    for pattern in (IPV4, IDENTIFIER, REDACTED_TOKEN):
        raw = pattern.sub("<id>", raw)
    raw = re.sub(r"\d+", "<number>", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse(row):
    raw = text(row.get("message_sanitized"))
    lower = raw.casefold()
    values = {key: 0 for key in FEATURES}
    evidence = {}
    families = set()
    def mark(name, snippet):
        values[name] = 1
        evidence.setdefault(name, str(snippet)[:220])
    scalars = {}
    for match in SCALARS.finditer(raw):
        try:
            value = json.loads(match.group("value"))
        except ValueError:
            continue
        key = match.group("key").casefold()
        scalars.setdefault(key, []).append((value, match.group(0)))
    for match in XML_FIELD.finditer(raw):
        scalars.setdefault(match.group(1).casefold(), []).append((match.group(2), match.group(0)))
    windows = "winlogbeat" in lower or '"winlog"' in lower or ("<eventid" in lower and "<system" in lower)
    event_codes = []
    if windows:
        families.add("windows_event")
        event_codes += [(m.group(1), m.group(0)) for m in XML_EVENT.finditer(raw)]
        event_codes += [(str(v), snippet) for key in ("code", "event_id") for v, snippet in scalars.get(key, [])]
    unique_codes = {value for value, _ in event_codes}
    # Conflicting event codes are left unresolved, not chosen by a label heuristic.
    if len(unique_codes) == 1:
        code, snippet = event_codes[0]
        if code in EVENT_ACTIONS:
            mark(EVENT_ACTIONS[code], snippet)
    for name, pattern in LEXICAL.items():
        if windows and name in ("auth_failure", "auth_success", "session_closed") and unique_codes:
            continue  # Avoid explanatory boilerplate overriding a structured event.
        match = pattern.search(raw)
        if match and not re.search(r"\b(?:not|no)\s*$", raw[max(0, match.start() - 12):match.start()], re.I):
            mark(name, match.group(0))
    reason_values = [(str(value).casefold(), snippet) for value, snippet in scalars.get("reason", [])]
    if any(value in REASONS for value, _ in reason_values):
        families.add("authentication_result")
        for value, snippet in reason_values:
            if value in REASONS:
                mark(REASONS[value], snippet)
        for value, snippet in scalars.get("result", []):
            if str(value).casefold() in ("denied", "failure", "error"):
                mark("auth_failure", snippet)
            elif str(value).casefold() == "success":
                mark("auth_success", snippet)
    if values["auth_failure"]:
        for name, pattern in [("bad_credentials", r"unknown user name or bad password|bad password|invalid password"),
                              ("unknown_user", r"unknown user name")]:
            match = re.search(pattern, raw, re.I)
            if match:
                mark(name, match.group(0))
    for key in ("logontype", "logon_type"):
        for value, snippet in scalars.get(key, []):
            if windows and str(value) in ("3", "10"):
                mark("logon_network" if str(value) == "3" else "logon_remote", snippet)
    if windows or values["auth_failure"] or values["auth_success"]:
        for value, snippet in scalars.get("authenticationpackagename", []):
            if str(value).casefold() in ("kerberos", "ntlm"):
                mark("auth_" + str(value).casefold(), snippet)
    def network_context(protocol, port, snippet):
        protocol = {"6": "tcp", "17": "udp", "1": "icmp", "58": "icmp"}.get(protocol.casefold(), protocol.casefold())
        if protocol in ("tcp", "udp", "icmp", "icmp6"):
            mark("protocol_" + ("icmp" if protocol == "icmp6" else protocol), snippet)
        if port.isdigit() and 0 <= int(port) <= 65535:
            service = {53: "dns", 80: "http", 443: "http", 22: "ssh", 3389: "rdp", 445: "smb", 139: "smb", 389: "ldap", 636: "ldap", 123: "ntp"}.get(int(port), "other")
            mark("service_" + service, snippet)
    asa = ASA.search(raw)
    if asa:
        families.add("firewall_flow")
        mark("network_denied", asa.group(0))
        port = asa.group(4).rsplit("/", 1)[-1] if "/" in asa.group(4) else ""
        network_context(asa.group(2), port, asa.group(0))
    built = BUILT.search(raw)
    if built:
        families.add("firewall_flow")
        mark("network_allowed" if built.group(1).casefold() == "built" else "network_closed", built.group(0))
        network_context(built.group(2), "", built.group(0))
    tokens = raw.split()
    # Default v2 VPC layout only. Damaged numeric fields do not destroy the
    # positions of readable action/protocol fields. Other layouts are not guessed.
    if len(tokens) == 14 and tokens[0] == "2" and tokens[-1].upper() in ("OK", "NODATA", "SKIPDATA"):
        families.add("vpc_flow")
        action = tokens[-2].upper()
        if action in ("ACCEPT", "REJECT"):
            mark("network_allowed" if action == "ACCEPT" else "network_denied", " ".join(tokens[-2:]))
            network_context(tokens[7], tokens[6], "protocol={} dstport={}".format(tokens[7], tokens[6]))
    for key, flag in [("quarantine_file", "file_quarantined"), ("kill_process", "process_killed"),
                      ("process_killed", "process_killed"), ("operation_blocked", "operation_blocked"),
                      ("process_blocked", "operation_blocked"), ("network_connection_blocked", "operation_blocked")]:
        for value, snippet in scalars.get(key, []):
            if value is True:
                mark(flag, snippet)
    method_matches = [(m.group(1), m.group(0)) for m in KV_METHOD.finditer(raw)]
    for key in ("requestmethod", "request_method"):
        method_matches += [(str(value), snippet) for value, snippet in scalars.get(key, [])]
    for method, snippet in method_matches:
        method = method.upper()
        if method in ("GET", "POST", "PUT", "DELETE", "PATCH"):
            mark("http_get" if method == "GET" else "http_post" if method == "POST" else "http_write_method", snippet)
            families.add("http_request")
    if method_matches:
        for match in KV_STATUS.finditer(raw):
            if match.group(1)[0] in ("2", "4", "5"):
                mark("http_status_" + match.group(1)[0] + "xx", match.group(0))
    vector = np.asarray([values[key] for key in FEATURES], dtype=np.uint8)
    return vector, {"evidence": evidence, "families": sorted(families), "no_core_evidence": not bool(vector[:len(CORE)].any())}


def connected_groups(old_groups, new_groups):
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    for old, new in zip(old_groups, new_groups):
        a, b = find("old:" + old), find("new:" + new)
        if a != b:
            low, high = sorted((a, b))
            parent[high] = low
    return np.asarray([find("old:" + old) for old in old_groups])


def load_v11(run):
    source = run / "soc_signal_pilot.py"
    if sha(source) != V11_SHA:
        raise ValueError("Previous run must contain the verified v1.1 source snapshot")
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if sha(run / "split_manifest.parquet") != result["manifest_sha256"]:
        raise ValueError("Previous manifest hash mismatch")
    spec = importlib.util.spec_from_file_location("v11_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_counts(y):
    return {name: int((y == index).sum()) for index, name in enumerate(LABELS)}


def fit_evaluate(reference, x, y, components, fit, calibration, development, evaluation, args, names, weighted=False):
    info = {"fit_class_counts": class_counts(y[fit]), "calibration_class_counts": class_counts(y[calibration]),
            "early_stopping_class_counts": class_counts(y[development]), "evaluation_class_counts": class_counts(y[evaluation]),
            "evaluation_groups": len(set(components[evaluation]))}
    if set(y[fit]) != {0, 1, 2} or not development.any() or not evaluation.any() or not (y[calibration] == 0).any():
        return None, dict(info, status="insufficient_support", reason="Need all fit classes, development/evaluation rows and calibration normal rows; split is not repaired")
    fit_groups = set(components[fit])
    if fit_groups & set(components[calibration]) or fit_groups & set(components[evaluation]) or set(components[calibration]) & set(components[evaluation]):
        raise AssertionError("Fit/calibration/evaluation component overlap")
    counts = np.bincount(y[fit], minlength=3)
    weights = np.minimum(np.sqrt(counts.max() / counts), 5.0).tolist() if weighted else None
    model = CatBoostClassifier(iterations=args.iterations, depth=5, learning_rate=0.08, loss_function="MultiClass",
                               eval_metric="MultiClass", class_weights=weights, random_seed=20260911,
                               task_type="CPU", thread_count=args.threads, allow_writing_files=False, verbose=False)
    model.fit(Pool(x[fit], y[fit], feature_names=names), eval_set=Pool(x[development], y[development], feature_names=names),
              early_stopping_rounds=30, use_best_model=True)
    p = model.predict_proba(x[evaluation])
    cal_p = model.predict_proba(x[calibration])
    info.update(status="executed_diagnostic", class_weights=weights, trees=int(model.tree_count_),
                metrics=reference.metric_summary(y[evaluation], p),
                thresholds=reference.evaluate_thresholds(y[calibration], cal_p, components[calibration], y[evaluation], p))
    return model, info


def main_run(args, out):
    started = time.perf_counter()
    previous = Path(args.previous_run)
    reference = load_v11(previous)
    shutil.copyfile(Path(__file__), out / "soc_behavior_v2.py")
    shutil.copyfile(previous / "soc_signal_pilot.py", out / "v11_reference.py")
    settings = dict(vars(args), version=VERSION, core_features=CORE, contextual_features=CONTEXT,
                    comparison_variants=["v11_refit", "core", "core_sqrt_weights", "core_context"],
                    source_holdout_variants=["core", "core_context"],
                    policy="Only old train/development used for models; quarantine components spanning old roles; group-hash 15% of old train for inner calibration; no automatic promotion")
    save(out / "configuration.json", settings)
    log("Verify official input hashes and load the frozen 100000-row manifest")
    identity = reference.verify_inputs(Path(args.data_dir))
    save(out / "input_identity.json", identity)
    manifest = pd.read_parquet(previous / "split_manifest.parquet").sort_values("row_position").reset_index(drop=True)
    if len(manifest) != 100000 or not manifest.row_position.is_unique:
        raise ValueError("This controlled follow-up requires the original 100000-row pilot manifest")
    positions = manifest.row_position.to_numpy()
    records, offset = [], 0
    for batch in pq.ParquetFile(Path(args.data_dir) / "train.parquet").iter_batches(batch_size=8192, columns=["event_id", "message_sanitized"] + ENTITY_COLUMNS):
        a, b = np.searchsorted(positions, [offset, offset + batch.num_rows])
        if a < b:
            records.extend(batch.take(pa.array(positions[a:b] - offset)).to_pylist())
        offset += batch.num_rows
    if [text(row["event_id"]) for row in records] != manifest.event_id.tolist():
        raise AssertionError("Source event order does not match manifest")
    new_groups = np.asarray([canonical_group(row) for row in records])
    component = connected_groups(manifest.group_id.tolist(), new_groups)
    plan = manifest.drop(columns=["label"]).copy()
    plan["v2_group"] = new_groups
    plan["component"] = component
    crossing = plan.groupby("component").role.nunique()
    quarantined = plan.component.isin(crossing[crossing > 1].index).to_numpy()
    usable = (~quarantined) & manifest.role.isin(["train", "development"]).to_numpy()
    plan["excluded_cross_role"] = quarantined
    plan["model_scope"] = usable
    # Calibration/audit messages were used only to find duplicate components;
    # their labels do not enter feature diagnostics, fitting or score selection.
    indices = np.flatnonzero(usable)
    x = np.zeros((len(indices), len(FEATURES)), dtype=np.uint8)
    old_x = np.zeros((len(indices), len(reference.RULES)), dtype=np.uint8)
    parsed = []
    log("Parse {} eligible training/development rows".format(len(indices)))
    for j, index in enumerate(indices):
        x[j], detail = parse(records[index])
        parsed.append(detail)
        old_x[j], _ = reference.extract(records[index])
        if (j + 1) % 20000 == 0:
            log("Parsed {}/{}".format(j + 1, len(indices)))
    selected = manifest.iloc[indices].copy().reset_index(drop=True)
    selected["component"] = component[indices]
    y = np.asarray([LABELS.index(value) for value in selected.label])
    components = selected.component.to_numpy()
    buckets = np.asarray([int(hashlib.sha256(("inner-cal-v2:" + value).encode()).hexdigest()[:12], 16) % 100 for value in components])
    train_role = selected.role.to_numpy() == "train"
    calibration = train_role & (buckets < 15)
    fit = train_role & ~calibration
    dev = selected.role.to_numpy() == "development"
    selected["v2_usage"] = np.where(dev, "development", np.where(calibration, "inner_calibration", "fit"))
    plan["v2_usage"] = "excluded"
    plan.loc[indices, "v2_usage"] = selected.v2_usage.to_numpy()
    plan.to_parquet(out / "component_split_manifest.parquet", index=False)
    sources = selected["product"].fillna("").replace("", "<missing>").to_numpy()
    candidates = {"v11_refit": (old_x, [name for name, _ in reference.RULES], False),
                  "core": (x[:, :len(CORE)], CORE, False),
                  "core_sqrt_weights": (x[:, :len(CORE)], CORE, True),
                  "core_context": (x, FEATURES, False)}
    # Source cohort selection uses counts, before any fitting or metric lookup.
    held_sources = sorted(source for source in set(sources) if int((sources == source).sum()) >= 100)
    save(out / "source_holdout_manifest.json", {"sources": held_sources, "minimum_rows": 100,
          "scope": "Product metadata holdout diagnostic, not an unseen external environment; <missing> mixes unknown sources"})
    coverage = {"original_rows": len(manifest), "cross_role_components": int((crossing > 1).sum()),
                "quarantined_rows_all_roles": int(quarantined.sum()),
                "quarantined_rows_by_role": {role: int((quarantined & (manifest.role.to_numpy() == role)).sum()) for role in sorted(set(manifest.role))},
                "model_scope_rows": len(selected), "class_counts": class_counts(y),
                "per_usage_class_counts": {name: class_counts(y[mask]) for name, mask in [("fit", fit), ("inner_calibration", calibration), ("development", dev)]},
                "core_unique_vectors": len(np.unique(x[:, :len(CORE)], axis=0)), "context_unique_vectors": len(np.unique(x, axis=0)),
                "v11_unique_vectors": len(np.unique(old_x, axis=0)), "nonzero_core_features": int(x[:, :len(CORE)].any(axis=0).sum()),
                "no_core_evidence_rows": int((~x[:, :len(CORE)].any(axis=1)).sum()),
                "per_class_no_core_evidence": {name: int((~x[y == i, :len(CORE)].any(axis=1)).sum()) for i, name in enumerate(LABELS)},
                "per_source": {source: {"rows": int((sources == source).sum()), "class_counts": class_counts(y[sources == source]),
                                          "no_core_evidence": int((~x[sources == source, :len(CORE)].any(axis=1)).sum())} for source in sorted(set(sources))}}
    save(out / "coverage.json", coverage)
    cache = selected.copy()
    for i, name in enumerate(FEATURES):
        cache[name] = x[:, i]
    cache.to_parquet(out / "feature_cache.parquet", index=False)
    examples, counts_by_kind = [], {}
    for j, index in enumerate(indices):
        key = (sources[j], selected.label.iloc[j])
        if counts_by_kind.get(key, 0) >= 2:
            continue
        counts_by_kind[key] = counts_by_kind.get(key, 0) + 1
        examples.append({"row_position": int(positions[index]), "role": selected.v2_usage.iloc[j], "product": sources[j],
                         "label": selected.label.iloc[j], **parsed[j]})
    save(out / "extraction_evidence_examples.json", examples)
    # Actual-sample metadata/date/identifier perturbations. No inverse mapping
    # from anonymization tokens to guessed original semantics.
    counter = {"metadata_changed": 0, "dates_changed": 0, "redaction_numbers_changed": 0}
    for j, index in enumerate(indices[:256]):
        row = records[index]
        changed = dict(row, product_name="different", pipeline=None, timestamp="2035-01-01", event_id="different")
        counter["metadata_changed"] += int(not np.array_equal(parse(changed)[0], x[j]))
        changed = dict(row, message_sanitized=SYSLOG_DATE.sub("Jan 01 2035", ISO_DATE.sub("2035-01-01T01:02:03Z", text(row.get("message_sanitized")))))
        counter["dates_changed"] += int(not np.array_equal(parse(changed)[0], x[j]))
        changed = dict(row, message_sanitized=REDACTED_TOKEN.sub(lambda m: re.sub(r"\d", "9", m.group(0)), text(row.get("message_sanitized"))))
        counter["redaction_numbers_changed"] += int(not np.array_equal(parse(changed)[0], x[j]))
    save(out / "representation_checks.json", {"actual_rows": min(256, len(indices)), "feature_flip_counts": counter,
                                               "scope": "Named formats only; identifier perturbation here covers numeric redaction spelling, not arbitrary semantic renaming"})
    if any(counter.values()):
        raise ValueError("Feature invariance check failed")
    main_reports, model_hashes = {}, {}
    for name, (matrix, feature_names, weighted) in candidates.items():
        log("Main comparison: " + name)
        model, report = fit_evaluate(reference, matrix, y, components, fit, calibration, dev, dev, args, feature_names, weighted)
        main_reports[name] = report
        if model is not None:
            model.save_model(str(out / (name + ".cbm")))
            model_hashes[name] = sha(out / (name + ".cbm"))
            prediction = model.predict_proba(matrix[dev])
            if not np.allclose(prediction[:128], model.predict_proba(matrix[dev][:128][::-1])[::-1], atol=1e-12, rtol=0):
                raise ValueError("Batch-order prediction invariance failed")
            save(out / (name + "_importance.json"), dict(zip(feature_names, model.get_feature_importance().tolist())))
    save(out / "development_comparison.json", main_reports)
    holdouts = []
    for source in held_sources:
        evaluation = sources == source
        held_groups = set(components[evaluation])
        eligible = np.asarray([group not in held_groups for group in components])
        log("Source holdout: " + source)
        for name in ("core", "core_context"):
            matrix, feature_names, weighted = candidates[name]
            _, report = fit_evaluate(reference, matrix, y, components, fit & eligible, calibration & eligible,
                                     dev & eligible, evaluation, args, feature_names, weighted)
            holdouts.append({"source": source, "variant": name, "excluded_overlap_rows_other_sources": int((~eligible & ~evaluation).sum()), **report})
        save(out / "source_holdout_results.json", holdouts)
    peak = None
    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    except ImportError:
        pass
    executed = any(report["status"] == "executed_diagnostic" for report in main_reports.values())
    result = {"status": "behavior_v2_experiments_executed_not_transfer_accepted" if executed else "blocked_by_group_or_class_support", "version": VERSION,
              "elapsed_seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak,
              "model_scope_rows": len(selected), "quarantined_rows_all_roles": int(quarantined.sum()),
              "core_unique_vectors": coverage["core_unique_vectors"], "context_unique_vectors": coverage["context_unique_vectors"],
              "main_comparison": {name: {"status": report["status"], "metrics": report.get("metrics")} for name, report in main_reports.items()},
              "source_holdout_runs_executed": sum(row["status"] == "executed_diagnostic" for row in holdouts),
              "source_holdout_runs_insufficient_support": sum(row["status"] == "insufficient_support" for row in holdouts),
              "code_sha256": sha(out / "soc_behavior_v2.py"), "v11_source_sha256": V11_SHA,
              "configuration_sha256": sha(out / "configuration.json"), "split_manifest_sha256": sha(out / "component_split_manifest.parquet"),
              "model_sha256": model_hashes, "python": sys.version,
              "limitations": ["Development used for early stopping; its metrics are optimistic diagnostics, not independent test scores.",
                              "Context features can encode source or collection differences despite semantic interpretation.",
                              "Product holdout is not a substitute for a new environment; absent classes are explicitly reported.",
                              "Old calibration/audit labels were not used for v2 training, thresholds, selection or evaluation.",
                              "Fixed parsers cover only named formats, may miss or misinterpret damaged/quoted text; unknowns remain unresolved.",
                              "No automatic model promotion, external data, probability calibration or final submission."]}
    save(out / "result.json", result)
    log("Finished; inspect source holdouts before judging development scores")
    print(json.dumps({key: value for key, value in result.items() if key != "main_comparison"}, ensure_ascii=False, indent=2), flush=True)
    for name, report in main_reports.items():
        m = report.get("metrics", {})
        log("{}: macro_f1={} suspicious_recall={}".format(name, m.get("macro_f1_all_three_zero_for_absent"), m.get("classes", {}).get("suspicious", {}).get("recall")))
    return 0 if executed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.iterations < 1 or args.threads < 1:
        parser.error("iterations and threads must be positive")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        sys.exit(main_run(args, out))
    except Exception as exc:
        save(out / "failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
