"""v3.2 representation repair. Raw evidence is never modified. Python 3.8+."""
import functools
import ipaddress
import json
import re

import soc_v3_prepare as base

VERSION = "v32-prepare-1.1"
DECODER = json.JSONDecoder()
PROTECTED = base.PROTECTED_PARENTS
# These paths are scoped to confirmed Falcon objects, not command payloads.
FALCON_UPSTREAM = {"tactic_id", "technique_id", "display_name", "behavior_id",
                   "pattern_disposition", "pattern_disposition_details",
                   "pattern_disposition_description", "behaviors_processed", "show_in_ui",
                   "ioc_description", "ioc_type", "ioc_value"}
FALCON_IDENTITIES = {"detection_id", "cid", "aid", "user_id", "device_id",
                     "process_id", "process_graph_id", "parent_process_id",
                     "parent_process_graph_id", "triggering_process_graph_id",
                     "md5", "sha1", "sha256", "sha512", "local_ip", "external_ip",
                     "mac_address", "bios_serial_number", "serial_number", "parent_md5", "parent_sha256"}
FALCON_COLLECTION = {"first_behavior", "last_behavior", "first_seen", "last_seen",
                     "created_timestamp", "modified_timestamp", "timestamp"}
ADDRESS_FIELDS = {"ipaddress", "sourceaddress", "destaddress", "destinationaddress",
                  "sourceip", "destinationip", "srcip", "dstip", "clientip"}
WIN_IDS = {"subjectusersid", "targetusersid", "subjectusername", "targetusername",
           "subjectdomainname", "targetdomainname", "subjectlogonid", "targetlogonid",
           "workstationname", "processid", "handleid", "eventrecordid", "activityid"}
IP_CANDIDATE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
# Escape boundary is used for address masking only, never global escape decoding.
ESCAPED_IP = re.compile(r"(\\[nrt])((?:\d{1,3}\.){3}\d{1,3})(?![\w.])")
QUOTED = re.compile(r'"(?:\\.|[^"\\])*"', re.S)


def quoted_end(raw, start):
    if raw[start:start+1] != '"':
        return None
    i = start + 1
    while i < len(raw):
        if raw[i] == "\\":
            i += 2
        elif raw[i] == '"':
            return i + 1
        else:
            i += 1
    return None


def value_end(raw, start):
    if start >= len(raw):
        return None
    if raw[start] == '"':
        return quoted_end(raw, start)
    if raw[start] in "[{":
        stack = [raw[start]]
        i = start + 1
        while i < len(raw):
            if raw[i] == '"':
                i = quoted_end(raw, i)
                if i is None:
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
    previous=base.object_root(raw)
    if previous is not None:return previous
    # A ::: collector record can put its source name after the JSON. Find the
    # explicit redacted field = { property boundary, not a GUID in a file path.
    if re.match(r"\s*(?:USER|ORG|CRED)-[\w.-]+\s*:::",raw,re.I):
        for m in re.finditer(r":::\s*([A-Za-z0-9_-]+)\s*=\s*(\{)",raw):
            if not base.MARKER.fullmatch(m.group(1)):continue
            root=m.start(2); k=root+1
            while k<len(raw) and raw[k].isspace():k+=1
            end=quoted_end(raw,k)
            if end is not None and re.match(r"\s*:",raw[end:]) and raw[k:end]=='"behaviors"':
                last=value_end(raw,root)
                if last is not None and "crowdstrike" in (raw[:root]+raw[last:]).casefold():return root
    return None


def members(raw):
    """Lexical field spans survive invalid value escapes; no guessed decoding."""
    result = []
    def walk(start, stop, path):
        i = start + 1
        while i < stop:
            if raw[i] != '"':
                i += 1
                continue
            after = quoted_end(raw, i)
            if after is None or after > stop:
                break
            try:
                key = json.loads(raw[i:after])
            except ValueError:
                i = after
                continue
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
                if begin < stop and raw[begin] == '"':
                    break
                i = after
                continue
            full_path = path + (str(key).casefold(),)
            result.append((i, end, full_path, begin))
            if raw[begin:begin+1] == "{":
                walk(begin, end, full_path)
            elif raw[begin:begin+1] == "[":
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
    start = object_root(raw)
    if start is not None:
        walk(start, value_end(raw, start) or len(raw), ())
    return result


def valid_ip(value):
    try:
        ipaddress.IPv4Address(value)
        return True
    except ipaddress.AddressValueError:
        return False


def normalize_body(raw, fmt):
    raw = re.sub(r"\b(?:sport|dport|src_port|dst_port)=([^\s,;]+)",
        lambda m: m.group(0).split("=", 1)[0] + "=unknown_port" if base.MARKER.search(m.group(1)) else m.group(0), raw, flags=re.I)
    if fmt == "asa_like":
        raw = re.sub(r"\b(src|dst)\s+(\S+)", lambda m: m.group(1) + " " +
            (m.group(2).rsplit("/", 1)[0] + "/unknown_port" if "/" in m.group(2) and base.MARKER.search(m.group(2).rsplit("/", 1)[1]) else m.group(2)), raw, flags=re.I)
    if fmt == "vpc14":
        toks = raw.split()
        if len(toks) == 14:
            for i in (1, 2, 3, 4):
                toks[i] = "entity"
            for i in (5, 6, 7, 8, 9):
                if base.MARKER.search(toks[i]):
                    toks[i] = "unknown_value"
            toks[10] = toks[11] = "time"
            raw = " ".join(toks)
    for pattern in (base.PARTIAL_DATE, base.ISO, base.SYSLOG_DATE, base.CLOCK):
        raw = pattern.sub("time", raw)
    # Preserve explicit software versions and out-of-range dotted numbers.
    protected = []
    def keep_version(m):
        protected.append(m.group(0))
        return "versionvalueplaceholderz" + str(len(protected)-1) + "z"
    raw = re.sub(r"\b(?:version|file_version|product_version|build)\s*[:= ]\s*(?:\d+\.){2,}\d+", keep_version, raw, flags=re.I)
    raw = ESCAPED_IP.sub(lambda m: m.group(1) + ("entity" if valid_ip(m.group(2)) else m.group(2)), raw)
    raw = IP_CANDIDATE.sub(lambda m: "entity" if valid_ip(m.group()) else m.group(), raw)
    for pattern in (base.UUID, base.MAC, base.FQDN, base.LONG_HEX):
        raw = pattern.sub("entity", raw)
    raw = base.MARKER.sub("entity", raw)
    raw = base.KV_ID.sub(lambda m: m.group(0).split("=", 1)[0].strip() + "=entity", raw)
    raw = re.sub(r"^\s*<\d{1,3}>\s*(?:1\s+)?", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip().casefold()
    for i, value in enumerate(protected):
        raw = raw.replace("versionvalueplaceholderz" + str(i) + "z", value.casefold())
    return raw


def is_falcon(raw, props):
    prefix = raw[:max(0, raw.find("{"))].casefold()
    roots = {p[-1] for _, _, p, _ in props if len(p) == 1}
    return "crowdstrike" in prefix or {"behaviors", "detection_id"}.issubset(roots)


def removal_reason(path, falcon=False):
    if any(p in PROTECTED for p in path[:-1]):
        return None
    key = path[-1]
    if falcon:
        if key in FALCON_IDENTITIES or key in FALCON_COLLECTION:
            return "identity_or_collection"
        if key.endswith("_graph_id"):
            return "identity_or_collection"
        if base.MARKER.search(key):
            # The field name itself has been redacted; do not guess its semantics.
            return "unverified_field"
        if key in FALCON_UPSTREAM and ("behaviors" in path or len(path) == 1):
            return "upstream_or_post_event"
        if path == ("status",):
            return "upstream_or_post_event"
    if ("winlog" in path or "eventdata" in path or "event_data" in path) and key in WIN_IDS | ADDRESS_FIELDS:
        return "identity_or_collection"
    return base.removal_reason(path)


def render(raw, spans, props, fmt):
    edits = [(s["start"], s["end"], " ") for s in spans]
    decoded, invalid = 0, 0
    for _, end, path, begin in props:
        if raw[begin:begin+1] != '"' or any(a <= begin < b for a, b, _ in edits):
            continue
        try:
            value = json.loads(raw[begin:end])
            decoded += 1
            # Collapse actual decoded whitespace, then serialize literal backslashes
            # without guessing their meaning. Address masking runs once afterward.
            if path[-1] in {"version", "file_version", "product_version"}:
                value = "version " + value
            value = re.sub(r"\s+", " ", value)
            edits.append((begin, end, json.dumps(value, ensure_ascii=False)))
        except ValueError:
            invalid += 1
    previous, parts = 0, []
    for start, end, replacement in sorted(edits):
        if start < previous:
            continue
        parts.extend((raw[previous:start], replacement))
        previous = end
    parts.append(raw[previous:])
    text = normalize_body("".join(parts), fmt)
    if not re.search(r"[a-z0-9]", text):
        text = ""
    return text, decoded, invalid


@functools.lru_cache(maxsize=256)
def _prepare(raw):
    fmt = base.format_of(raw)
    props = members(raw)
    falcon = is_falcon(raw, props)
    spans = []
    collector_paths = set()
    for _, end, path, begin in props:
        if path[-1] in {"type", "beat"}:
            try:
                if json.loads(raw[begin:end]) in ("winlogbeat", "filebeat"):
                    collector_paths.add(path[:-1])
            except (ValueError, TypeError):
                pass
    for start, end, path, begin in props:
        reason = removal_reason(path, falcon)
        if path in collector_paths and not any(p in PROTECTED for p in path):
            reason = "identity_or_collection"
        if reason:
            spans.append({"start": start, "end": end, "reason": reason, "path": ".".join(path)})
    if falcon:
        # Confirmed ::: collector wrapper. Keep visible file/path facts; remove
        # whole identity/hash/source fields, never fragments inside the JSON body.
        first = object_root(raw)
        last = value_end(raw, first) if first is not None else None
        ranges = [(0, first)] if first is not None else []
        if last is not None:
            ranges.append((last, len(raw)))
        for a, b in ranges:
            for match in re.finditer(r"(?:^|:::)(.*?)(?=:::|$)", raw[a:b], re.S):
                segment = match.group(1)
                if not re.match(r"\s*(?:filename|filepath)\s*=", segment, re.I):
                    spans.append({"start": a+match.start(), "end": a+match.end(), "reason": "identity_or_collection", "path": "falcon_collector_wrapper"})
    if fmt == "network_kv":
        wrapper = re.match(r"\s*<\d+>Original Address=\S+\s+1\s+\S+\s+\S+\s+(?=flows\b)", raw, re.I)
        if wrapper:
            spans.append({"start": wrapper.start(), "end": wrapper.end(), "reason": "identity_or_collection", "path": "network_wrapper"})
    if fmt == "cef_like":
        cef_start = raw.find("CEF:")
        pipes = list(re.finditer(r"(?<!\\)\|", raw[cef_start:])) if cef_start >= 0 else []
        if len(pipes) >= 7:
            extension_start = cef_start + pipes[6].end()
            field = re.compile(r'''(?<!\S)([A-Za-z][\w]*)=(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s]*)''')
            for match in field.finditer(raw[extension_start:]):
                if match.group(1) in {"externalId", "rt"}:
                    spans.append({"start": extension_start+match.start(), "end": extension_start+match.end(), "reason": "identity_or_collection", "path": "cef_collection_field"})
    for pattern in (base.XML_ID, base.XML_USER):
        for match in pattern.finditer(raw):
            spans.append({"start": match.start(), "end": match.end(), "reason": "identity_or_collection", "path": "xml_identity"})
    diagnostic_spans = base.choose_spans([s for s in spans if s["reason"] == "identity_or_collection"])
    main_spans = base.choose_spans(spans)
    main, decoded, invalid = render(raw, main_spans, props, fmt)
    diagnostic, _, _ = render(raw, diagnostic_spans, props, fmt)
    return {"text": main, "diagnostic_text": diagnostic, "format": fmt,
            "removed_spans": main_spans, "original_empty": not bool(raw.strip()),
            "filtered_empty": bool(raw.strip()) and not bool(main),
            "unknown_format": fmt in {"unknown", "json_fragment", "syslog_like", "cef_like"},
            "upstream_fields_removed": sum(s["reason"] == "upstream_or_post_event" for s in main_spans),
            "unverified_fields_isolated": sum(s["reason"] == "unverified_field" for s in main_spans),
            "decoded_string_values": decoded, "invalid_string_values_preserved": invalid,
            "falcon_scoped_policy": falcon}


def prepare_record(row):
    value = _prepare(base.string(row.get("message_sanitized")))
    return dict(value, removed_spans=[dict(s) for s in value["removed_spans"]])


group_key = base.group_key
group_support = base.group_support
