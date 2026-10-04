"""Official-train V0/V1: bounded-memory text preparation and group support audit.

No classifier, vocabulary or IDF is fitted. Python 3.8 compatible.
"""
import argparse
import collections
import functools
import hashlib
import json
import math
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

VERSION = "v3-prepare-0.3"
LABELS = ["benign", "malicious", "suspicious"]
EXPECTED_SHA = "6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742"
DECODER = json.JSONDecoder()
MARKER = re.compile(r"(?:USER|ORG|CRED)(?:-(?:USER|ORG|CRED))*-\d+(?:-\d+)*", re.I)
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
UUID = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)
MAC = re.compile(r"\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", re.I)
ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[Tt ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:?\d{2})?)?\b")
PARTIAL_DATE = re.compile(r"(?:USER|ORG|CRED)-\d+(?:-\d+)*-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:?\d{2})?", re.I)
SYSLOG_DATE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:\s+\d{4})?(?:\s+\d{2}:\d{2}:\d{2})?\b", re.I)
CLOCK = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b")
LONG_HEX = re.compile(r"\b[0-9a-f]{16,}\b", re.I)
FQDN = re.compile(r"(?<![\w.])(?:[a-z0-9][a-z0-9-]*\.)+(?:com|net|org|local|internal|example|invalid)(?![\w.])", re.I)
KV_ID = re.compile(r"\b(?:src_ip|dst_ip|src_host|dst_host|username|user_name|hostname)\s*=\s*(?:\"[^\"]*\"|[^\s,;]+)", re.I)
XML_ID = re.compile(r"<(?:Computer|TimeCreated|EventRecordID|Correlation|Execution|Security)\b[^>]*(?:/>|>[^<]*</(?:Computer|EventRecordID)>)", re.I)
XML_USER = re.compile(r"<Data\s+Name=[\"'](?:SubjectUserName|TargetUserName|SubjectDomainName|TargetDomainName|SubjectUserSid|TargetUserSid|SubjectLogonId|TargetLogonId|ProcessId|HandleId|IpAddress|WorkstationName)[\"']\s*>[^<]*</Data>", re.I)
UPSTREAM = {"severity", "confidence", "scenario", "verdict", "resolution", "triaged", "resolved",
            "seconds_to_triaged", "seconds_to_resolved", "tactic", "technique", "description",
            "detection_name", "threat_name", "risk_score", "triage_status", "resolution_status"}
UPSTREAM.update({"max_severity", "max_severity_displayname", "max_confidence", "max_confidence_displayname"})
IDENTITY = {"src_ip", "dst_ip", "src_host", "dst_host", "username", "user_name", "hostname",
            "record_id", "ephemeral_id", "device_id", "agent_id", "event_uuid", "organization",
            "sender", "sender_ip", "sender_host", "product_name", "vendor_name", "pipeline"}
CONTAINERS = {"agent", "@metadata", "ecs", "observer"}
IDENTITY_PARENTS = {"host", "user", "source", "destination", "device", "sensor", "computer"}
PROTECTED_PARENTS = {"request", "body", "args", "arguments", "parameters", "query", "command_line"}


def string(value):
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8388608), b""):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def value_end(raw, start):
    """Find a complete quoted/container boundary; tolerate damaged bare values."""
    if start >= len(raw):
        return None
    if raw[start] == '"':
        try:
            _, end = DECODER.raw_decode(raw, start)
            return end
        except ValueError:
            return None
    if raw[start] in "[{":
        stack = [raw[start]]
        i = start + 1
        while i < len(raw):
            if raw[i] == '"':
                try:
                    _, i = DECODER.raw_decode(raw, i)
                except ValueError:
                    return None
                continue
            if raw[i] in "[{":
                stack.append(raw[i])
            elif raw[i] in "]}":
                if not stack or (stack[-1], raw[i]) not in [("[", "]"), ("{", "}")]:
                    return None
                stack.pop()
                if not stack:
                    return i + 1
            i += 1
        return None
    i = start
    while i < len(raw) and raw[i] not in ",}]\r\n":
        i += 1
    return i if i > start else None


def object_root(raw):
    start = len(raw) - len(raw.lstrip())
    if raw[start:start + 1] == "{":
        return start
    first = raw.find("{")
    if first < 0:
        return None
    prefix = raw[:first]
    if re.fullmatch(r"\s*(?:USER|ORG|CRED)-[\w.-]+\s*:::\s*", prefix, re.I):
        return first
    if (re.match(r"\s*(?:USER|ORG|CRED)-", prefix, re.I) and "crowdstrike" in prefix.casefold()
            and re.search(r":::\s*(?:USER|ORG|CRED)-[\w.-]+=\s*$", prefix, re.I)):
        return first
    return None


def members(raw):
    """Walk only actual JSON-style property boundaries, never quoted commands."""
    result = []
    def walk(start, stop, path):
        i = start + 1
        while i < stop:
            if raw[i] != '"':
                i += 1
                continue
            try:
                key, after = DECODER.raw_decode(raw, i)
            except ValueError:
                break
            j = after
            while j < stop and raw[j].isspace():
                j += 1
            if j >= stop or raw[j] != ":":
                i = after
                continue
            begin = j + 1
            while begin < stop and raw[begin].isspace():
                begin += 1
            end = value_end(raw, begin)
            if end is None or end > stop:
                # Do not interpret the interior of an incomplete quoted value.
                if begin < stop and raw[begin] == '"':
                    break
                i = after
                continue
            full_path = path + (string(key).casefold(),)
            result.append((i, end, full_path, begin))
            if raw[begin:begin + 1] == "{":
                walk(begin, end, full_path)
            elif raw[begin:begin + 1] == "[":
                k = begin + 1
                while k < end:
                    if raw[k] in '[{"':
                        last = value_end(raw, k)
                        if last is None:
                            break
                        if raw[k] == "{":
                            walk(k, last, full_path + ("[]",))
                        k = last
                    else:
                        k += 1
            i = end
    first = object_root(raw)
    if first is not None:
        end = value_end(raw, first)
        # An unmatched root container may have readable complete members; never
        # scan beyond a known matching root, or inside unknown command payloads.
        walk(first, end if end is not None else len(raw), ())
    return result


def removal_reason(path):
    key = path[-1]
    parent = path[-2] if len(path) > 1 else ""
    if any(p in PROTECTED_PARENTS for p in path[:-1]):
        return None
    if key in CONTAINERS or key.startswith("@meta") or key in IDENTITY or key in {"@timestamp", "timestamp", "created", "modified"}:
        return "identity_or_collection"
    if parent in IDENTITY_PARENTS and key in {"name", "id", "ip", "domain", "fqdn", "sid", "mac", "version"}:
        return "identity_or_collection"
    if parent in {"process", "thread"} and key in {"pid", "tid", "id"}:
        return "identity_or_collection"
    if key in UPSTREAM and (len(path) == 1 or not any(p in {"process", "event_data", "eventdata"} for p in path[:-1])):
        return "upstream_or_post_event"
    return None


def choose_spans(items):
    chosen = []
    for item in sorted(items, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        if chosen and item["start"] < chosen[-1]["end"]:
            continue
        chosen.append(item)
    return chosen


def apply_spans(raw, spans, include_upstream=False):
    result = []
    previous = 0
    for span in spans:
        if include_upstream and span["reason"] == "upstream_or_post_event":
            continue
        result.extend((raw[previous:span["start"]], " "))
        previous = span["end"]
    result.append(raw[previous:])
    return "".join(result)


def format_of(raw):
    low = raw.casefold()
    tokens = raw.split()
    if not raw.strip():
        return "empty"
    if (len(tokens) == 14 and tokens[0] == "2" and tokens[-1].upper() in {"OK", "NODATA", "SKIPDATA"}
            and tokens[-2].upper() in {"ACCEPT", "REJECT", "-"}
            and (tokens[2].startswith("eni-") or MARKER.fullmatch(tokens[2]))
            and all(re.fullmatch(r"[0-9]+|-", t) or MARKER.search(t) for t in [tokens[1]] + tokens[5:12])
            and all(IPV4.fullmatch(t) or MARKER.fullmatch(t) or t == "-" for t in tokens[3:5])):
        return "vpc14"
    if re.search(r"\b(?:deny|built|teardown)\s+(?:(?:inbound|outbound)\s+)?(?:tcp|udp|icmp)", low):
        return "asa_like"
    if "flows " in low and "protocol=" in low:
        return "network_kv"
    if "<event" in low:
        return "windows_xml_like"
    if "winlog" in low:
        return "windows_json_like"
    if "cef:" in low:
        return "cef_like"
    if "{" in raw:
        return "json_fragment"
    if raw.startswith("<") or SYSLOG_DATE.match(raw):
        return "syslog_like"
    return "unknown"


def normalize_body(raw, fmt):
    # Source ports with partial redaction are unknown whole values, not numbers.
    raw = re.sub(r"\b(?:sport|dport|src_port|dst_port)=([^\s,;]+)",
                 lambda m: m.group(0).split("=", 1)[0] + "=unknown_port" if MARKER.search(m.group(1)) else m.group(0), raw, flags=re.I)
    if fmt == "asa_like":
        raw = re.sub(r"\b(src|dst)\s+(\S+)", lambda m: m.group(1) + " " +
                     (m.group(2).rsplit("/", 1)[0] + "/unknown_port" if "/" in m.group(2) and MARKER.search(m.group(2).rsplit("/", 1)[1]) else m.group(2)), raw, flags=re.I)
    if fmt == "vpc14":
        toks = raw.split()
        if len(toks) == 14:
            for i in (1, 2, 3, 4):
                toks[i] = "entity"
            for i in (5, 6, 7, 8, 9):
                if MARKER.search(toks[i]):
                    toks[i] = "unknown_value"
            toks[10] = toks[11] = "time"
            raw = " ".join(toks)
    raw = PARTIAL_DATE.sub("time", raw)
    raw = ISO.sub("time", raw)
    raw = SYSLOG_DATE.sub("time", raw)
    raw = CLOCK.sub("time", raw)
    for pattern in (UUID, MAC, IPV4, FQDN, LONG_HEX):
        raw = pattern.sub("entity", raw)
    raw = MARKER.sub("entity", raw)
    raw = KV_ID.sub(lambda m: m.group(0).split("=", 1)[0].strip() + "=entity", raw)
    # PRI, RFC5424 version and wrapper timestamps are collection attributes.
    raw = re.sub(r"^\s*<\d{1,3}>\s*(?:1\s+)?", "", raw)
    raw = re.sub(r"\\[nrt]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip().casefold()
    # Delimiters left after removing an entire object are not evidence.
    return raw if re.search(r"[a-z0-9]", raw) else ""


@functools.lru_cache(maxsize=512)
def _prepare_message(raw):
    fmt = format_of(raw)
    spans = []
    props = members(raw)
    collector_paths = set()
    for _, _, path, begin in props:
        if path[-1] in {"type", "beat"}:
            try:
                value, _ = DECODER.raw_decode(raw, begin)
                if value in ("winlogbeat", "filebeat"):
                    collector_paths.add(path[:-1])
            except (ValueError, TypeError):
                pass
    for start, end, path, _ in props:
        reason = removal_reason(path)
        if path in collector_paths and not any(p in PROTECTED_PARENTS for p in path):
            reason = "identity_or_collection"
        if path == ("status",) and "crowdstrike" in raw[:max(0, raw.find("{"))].casefold():
            reason = "upstream_or_post_event"
        if reason:
            spans.append({"start": start, "end": end, "reason": reason, "path": ".".join(path)})
    if fmt == "network_kv":
        wrapper = re.match(r"\s*<\d+>Original Address=\S+\s+1\s+\S+\s+\S+\s+(?=flows\b)", raw, re.I)
        if wrapper:
            spans.append({"start": wrapper.start(), "end": wrapper.end(), "reason": "identity_or_collection", "path": "network_wrapper"})
    if fmt == "cef_like":
        # Only extension-level fields; skip whole quoted request values. An
        # externalId substring inside a URL or command is not a log property.
        cef_start = raw.find("CEF:")
        pipes = list(re.finditer(r"(?<!\\)\|", raw[cef_start:])) if cef_start >= 0 else []
        if len(pipes) >= 7:
            extension_start = cef_start + pipes[6].end()
            field = re.compile(r'''(?<!\S)([A-Za-z][\w]*)=(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s]*)''')
            for match in field.finditer(raw[extension_start:]):
                if match.group(1) in {"externalId", "rt"}:
                    spans.append({"start": extension_start + match.start(), "end": extension_start + match.end(),
                                  "reason": "identity_or_collection", "path": "cef_collection_field"})
    for pattern in (XML_ID, XML_USER):
        for match in pattern.finditer(raw):
            spans.append({"start": match.start(), "end": match.end(), "reason": "identity_or_collection", "path": "xml_identity"})
    diagnostic_spans = choose_spans([s for s in spans if s["reason"] != "upstream_or_post_event"])
    spans = choose_spans(spans)
    main = normalize_body(apply_spans(raw, spans), fmt)
    diagnostic = normalize_body(apply_spans(raw, diagnostic_spans), fmt)
    return {"text": main, "diagnostic_text": diagnostic, "format": fmt, "removed_spans": spans,
            "original_empty": not bool(raw.strip()), "filtered_empty": bool(raw.strip()) and not bool(main),
            "unknown_format": fmt in {"unknown", "json_fragment", "syslog_like", "cef_like"},
            "upstream_fields_removed": sum(s["reason"] == "upstream_or_post_event" for s in spans)}


def prepare_record(row):
    # Do not infer identity linkage from possibly inconsistent outer columns.
    value = _prepare_message(string(row.get("message_sanitized")))
    return dict(value, removed_spans=[dict(s) for s in value["removed_spans"]])


def group_key(raw, prepared, row_position):
    if prepared["text"]:
        return b"text:" + hashlib.sha256(prepared["text"].encode("utf-8")).digest()
    if not prepared["original_empty"]:
        return b"filtered_raw:" + hashlib.sha256(string(raw).encode("utf-8")).digest()
    return b"empty_row:" + str(row_position).encode("ascii")


def log(msg):
    print(time.strftime("%H:%M:%S") + " " + msg, flush=True)


def peak_memory_bytes():
    """Process high-water memory; neither total machine nor container usage."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not query(process(), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def group_support(y, groups, mask, informative):
    import numpy as np
    result = {}
    for index, label in enumerate(LABELS):
        selected = mask & (y == index)
        _, sizes = np.unique(groups[selected & informative], return_counts=True)
        result[label] = {"rows": int(selected.sum()), "informative_rows": int((selected & informative).sum()),
                         "informative_groups": len(sizes), "largest_informative_group_rows": int(sizes.max()) if len(sizes) else 0,
                         "largest_group_fraction_in_informative_rows": float(sizes.max() / sizes.sum()) if len(sizes) else None,
                         "remaining_rows_after_largest_group": int(sizes.sum() - sizes.max()) if len(sizes) else 0,
                         "engineering_support_30_groups": len(sizes) >= 30}
    return result


def run(args, out):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold

    started = time.perf_counter()
    train = Path(args.train)
    actual_sha = file_hash(train)
    if actual_sha != EXPECTED_SHA:
        raise ValueError("Official training SHA mismatch")
    shutil.copyfile(Path(__file__), out / "soc_v3_prepare.py")
    source_sha = file_hash(out / "soc_v3_prepare.py")
    config = {"version": VERSION, "scope": "Official train only; no validation files, no external data, no model fit",
              "input_sha256": actual_sha, "code_sha256": source_sha, "seed": 20260912,
              "outer_folds": 3, "inner_folds": 5, "fit_inner": [0, 1, 2], "selection_inner": 3,
              "calibration_inner": 4, "minimum_informative_groups_per_class": 30,
              "group_rule": "Identical nonempty normalized full text together; filtered-empty nonempty raw duplicates stay grouped by raw hash. Only originally empty messages use row units. Zero-text rows never establish independent support.",
              "class_weights": None, "row_weight": 1, "sklearn": sklearn.__version__, "python": sys.version,
              "batch_size": 512, "message_cache_max_entries": 512,
              "workers": args.workers, "parallel_scope": "Text parsing only; parent assigns groups in original input order",
              "normalization_warning": "Limited named field boundaries and formats; retained unknown body is not certified pure fact. No verified incident linkage, no full near-duplicate isolation."}
    save(out / "configuration.json", config)
    parquet = pq.ParquetFile(train)
    n = parquet.metadata.num_rows
    y = np.empty(n, dtype=np.uint8)
    groups = np.empty(n, dtype=np.int32)
    informative = np.empty(n, dtype=bool)
    group_map = {}
    next_group = 0
    categories = collections.Counter()
    examples = []
    example_counts = collections.Counter()
    identity_residual = collections.Counter()
    writer = None
    offset = 0
    log("Prepare full official training data in bounded batches")
    pool = None
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=args.workers)
    try:
        for batch in parquet.iter_batches(batch_size=512, columns=["event_id", "message_sanitized", "product_name", "label_binary"]):
            output_rows = []
            raw_rows = batch.to_pylist()
            prepared_rows = pool.map(prepare_record, raw_rows, chunksize=32) if pool else map(prepare_record, raw_rows)
            for row, prepared in zip(raw_rows, prepared_rows):
                text_value = prepared["text"]
                if not prepared["original_empty"]:
                    h = group_key(row["message_sanitized"], prepared, offset)
                    group = group_map.get(h)
                    if group is None:
                        group = next_group
                        next_group += 1
                        group_map[h] = group
                else:
                    group = next_group
                    next_group += 1
                label = row["label_binary"]
                y[offset] = LABELS.index(label)
                groups[offset] = group
                informative[offset] = bool(text_value)
                source = string(row["product_name"]) or "<missing>"
                categories[(source, prepared["format"], label, prepared["original_empty"], prepared["filtered_empty"], prepared["unknown_format"])] += 1
                for name, pattern in [("known_redaction_marker", MARKER), ("ipv4", IPV4), ("iso_date", ISO), ("syslog_date", SYSLOG_DATE)]:
                    identity_residual[name] += bool(pattern.search(text_value))
                key = (source, label, prepared["format"])
                if example_counts[key] < 2:
                    example_counts[key] += 1
                    examples.append({"row_position": offset, "event_id": row["event_id"], "product": source, "label": label,
                                     "raw_message": row["message_sanitized"], **prepared})
                output_rows.append({"row_position": offset, "event_id": row["event_id"], "label": label,
                                    "product": source, "format": prepared["format"], "group_id": group,
                                    "original_empty": prepared["original_empty"], "filtered_empty": prepared["filtered_empty"],
                                    "unknown_format": prepared["unknown_format"], "text": text_value,
                                    "diagnostic_text": prepared["diagnostic_text"], "removed_field_count": len(prepared["removed_spans"]),
                                    "upstream_field_count": prepared["upstream_fields_removed"]})
                offset += 1
            table = pa.Table.from_pylist(output_rows)
            if writer is None:
                writer = pq.ParquetWriter(out / "prepared_corpus.parquet", table.schema, compression="zstd")
            writer.write_table(table)
            if offset % 65536 == 0:
                log("Prepared {}/{} rows; {} nonempty-raw group keys".format(offset, n, len(group_map)))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if writer is not None:
            writer.close()
    if offset != n:
        raise AssertionError("Training row count mismatch")
    informative_groups = len(np.unique(groups[informative]))
    del group_map
    counts = np.bincount(groups[informative] * 3 + y[informative], minlength=next_group * 3).reshape(-1, 3)
    conflict = (counts > 0).sum(axis=1) > 1
    conflict_ids = np.flatnonzero(conflict)
    ids = conflict_ids[np.argsort(-counts[conflict_ids].sum(axis=1))[:30]]
    collision = {"informative_unique_texts": informative_groups, "mixed_label_text_groups": int(conflict.sum()),
                 "rows_in_mixed_label_text_groups": int(counts[conflict].sum()),
                 "minimum_observed_mistakes_deterministic_text_only": int((counts.sum(axis=1) - counts.max(axis=1)).sum()),
                 "largest_conflict_groups": [{"group_id": int(g), "counts": dict(zip(LABELS, map(int, counts[g])))} for g in ids],
                 "note": "Conflicts may reflect normalization loss, missing evidence, or annotation semantics; not necessarily wrong labels or irreversible irreducibility. Empty inputs excluded here and reported separately."}
    del counts
    save(out / "text_collisions.json", collision)
    save(out / "normalization_examples.json", examples)
    save(out / "coverage_by_source_format.json", [{"product": k[0], "format": k[1], "label": k[2], "original_empty": k[3], "filtered_empty": k[4], "unknown_format": k[5], "rows": v} for k, v in sorted(categories.items())])
    save(out / "residual_pattern_checks.json", {"rows": n, "residual_counts": dict(identity_residual), "scope": "Only named regex patterns; not proof that all identities/absolute times are removed"})
    # Persist group identities before any splitting; no label-driven text repair.
    pd.DataFrame({"row_position": np.arange(n, dtype=np.int32), "group_id": groups, "label_index": y,
                  "informative": informative}).to_parquet(out / "group_manifest.parquet", index=False)
    log("Build fixed 3 outer folds and 5 inner parts; no model fitting")
    outer = np.full(n, -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20260912)
    for fold, (_, idx) in enumerate(splitter.split(np.zeros(n, dtype=np.uint8), y, groups)):
        outer[idx] = fold
    if not np.isin(outer, [0, 1, 2]).all():
        raise AssertionError("Outer fold coverage incomplete")
    fold_table = {"row_position": np.arange(n, dtype=np.int32), "outer_fold": outer}
    supports = []
    for fold in range(3):
        available = np.flatnonzero(outer != fold)
        inner = np.full(n, -1, dtype=np.int8)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260912)
        for part, (_, idx) in enumerate(splitter.split(np.zeros(len(available), dtype=np.uint8), y[available], groups[available])):
            inner[available[idx]] = part
        if not np.isin(inner[available], [0, 1, 2, 3, 4]).all():
            raise AssertionError("Inner fold coverage incomplete")
        fold_table["inner_for_outer_" + str(fold)] = inner
        masks = {"fit": (inner >= 0) & (inner <= 2), "selection": inner == 3, "calibration": inner == 4, "outer_development": outer == fold}
        if not (sum(mask.astype(np.uint8) for mask in masks.values()) == 1).all():
            raise AssertionError("Each row must have exactly one role per outer fold")
        support = {role: group_support(y, groups, mask, informative) for role, mask in masks.items()}
        support["fold"] = fold
        support["support_gate_passed"] = all(c["engineering_support_30_groups"] for role in masks for c in support[role].values())
        supports.append(support)
        usage = np.where(outer == fold, 3, np.where(inner <= 2, 0, np.where(inner == 3, 1, 2)))
        if pd.DataFrame({"g": groups, "u": usage}).groupby("g").u.nunique().max() != 1:
            raise AssertionError("Group crosses an inner/outer role")
        log("Fold {} support gate={}".format(fold, support["support_gate_passed"]))
    pd.DataFrame(fold_table).to_parquet(out / "split_manifest.parquet", index=False)
    save(out / "split_support.json", supports)
    result = {"status": "prepared_pending_semantic_audit" if all(s["support_gate_passed"] for s in supports) else "prepared_but_class_group_support_insufficient",
              "version": VERSION, "rows": n, "informative_rows": int(informative.sum()), "informative_groups": informative_groups,
              "mixed_label_text_groups": collision["mixed_label_text_groups"], "all_fold_support_gates_passed": all(s["support_gate_passed"] for s in supports),
              "model_trained": False, "semantic_normalization_accepted": False, "transfer_validated": False,
              "elapsed_seconds": time.perf_counter() - started, "peak_process_memory_bytes": peak_memory_bytes(),
              "peak_memory_scope": "Parent-process high-water working set / resident memory only; excludes worker processes, machine-wide and container-wide usage", "code_sha256": source_sha,
              "corpus_sha256": file_hash(out / "prepared_corpus.parquet"), "split_sha256": file_hash(out / "split_manifest.parquet"),
              "limitations": ["All official training data are development material; no new blind test created.",
                              "Unknown body and unrecognized identifiers may retain source or verdict evidence.",
                              "Text grouping is not verified real incident independence; near duplicates may remain.",
                              "Class support gate alone is not a model training authorization gate; semantic review is still required."]}
    save(out / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error("--workers must be between 1 and 8")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(args, output)
    except Exception as exc:
        save(output / "failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
