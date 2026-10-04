"""v3.3.1 scoped repairs; source labels and outer metadata never route parsing."""
import functools
import json
import re

import soc_v3_prepare as base
import soc_v32_prepare as old
import v32_finalize as final

VERSION = "v331-input-1.0"
ASA_BODY = re.compile(
    r'(?P<action>Deny)\s+(?P<protocol>tcp|udp|icmp6?|[0-9]+)\s+src\s+'
    r'(?P<src>\S+)\s+dst\s+(?P<dst>\S+)'
    r'(?P<icmp>\s+\(type\s+\S+,\s*code\s+\S+\))?'
    r'\s+by\s+(?P<aclword>[A-Za-z0-9_-]+[-_]group)\s+"(?P<acl>[^"\r\n]*)"'
    r'\s+\[(?P<hex>0x[0-9a-f]+,\s*0x[0-9a-f]+)\]\s*$', re.I)
ASA_CODE = re.compile(r'%(?:ASA|PIX)-([0-7])-([0-9]{6}):\s*$', re.I)
WRAPPED_OBJECT = re.compile(r'(?:^|:::)\s*[\w-]+\s*=\s*(\{)')
AUTH_SIGNATURE = {"application", "user", "result", "reason", "txid"}
AUTH_FACTS = {"result", "reason", "factor", "event_type", "remembered_factor"}
AUTH_UPSTREAM = {"adaptive_trust_assessments", "rbfs_triggered_attacks",
                 "risk_score", "trust_level", "trusted_endpoint_status"}
AUTH_IDENTITIES = {"application", "user", "alias", "txid", "timestamp",
                   "isotimestamp", "access_device", "auth_device"}
SOURCES = [
    "https://duo.com/docs/adminapi",
    "https://www.cisco.com/c/en/us/support/docs/security/adaptive-security-appliance-asa-software/217679-asa-access-control-list-configuration-ex.html",
]


def asa_parts(raw):
    """Only a complete observed ACL grammar after a checked collection prefix."""
    if "deny" not in raw.casefold():
        return None
    matches = list(ASA_BODY.finditer(raw))
    if len(matches) != 1:
        return None
    m = matches[0]
    prefix = raw[:m.start()]
    event = ASA_CODE.search(prefix)
    event_text = ""
    if event:
        event_text = "event_code {} native_severity {} ".format(event.group(2), event.group(1))
        prefix = prefix[:event.start()]
    if prefix.strip():
        check = re.sub(r'^\s*<\d{1,3}>\s*(?:1\s+)?', '', prefix)
        found_time = False
        for pat in (base.PARTIAL_DATE, base.ISO, base.SYSLOG_DATE, base.CLOCK):
            check, n = pat.subn(" TIME ", check)
            found_time |= n > 0
        check = base.MARKER.sub(" ENTITY ", check)
        # Collector host slots are bounded; quoted/application payload prefixes fail closed.
        tokens = check.replace(":", " ").split()
        others = [t for t in tokens if t not in {"TIME", "ENTITY", "-"}]
        if not found_time or len(others) > 2 or any(not re.fullmatch(r"[A-Za-z][\w.-]*", t) for t in others):
            return None
    return {"body_start": m.start(), "body_end": len(raw), "body": raw[m.start():],
            "event_text": event_text, "facts": m.groupdict()}


def role(endpoint):
    value = endpoint.split(":", 1)[0].casefold()
    if value in {"inside", "outside", "identity", "internet"}:
        return value
    if re.fullmatch(r"dmz(?:[-_]\d+)?", value):
        return "dmz"
    return "other_interface"


def template_of(parts):
    if parts is None:
        return None
    f = parts["facts"]
    # Label/score blind family: do not retain ports, addresses, ACL names or hex hashes.
    return "|".join([f["action"].casefold(), f["protocol"].casefold(),
                     role(f["src"]), role(f["dst"]), "icmp_fields" if f["icmp"] else "no_icmp_fields"])


def auth_object(raw):
    if not all('"' + k + '"' in raw for k in AUTH_SIGNATURE):
        return None
    candidates = [m.start(1) for m in WRAPPED_OBJECT.finditer(raw)]
    if raw.lstrip().startswith("{"):
        candidates.append(len(raw) - len(raw.lstrip()))
    accepted = []
    for start in candidates:
        end = old.value_end(raw, start)
        fragment = raw[start:end] if end is not None else raw[start:]
        props = old.members(fragment)
        roots = [p for p in props if len(p[2]) == 1]
        keys = [p[2][0] for p in roots]
        partial_tail = None
        if end is None and roots and (AUTH_SIGNATURE - {"user"}).issubset(keys):
            last = max(q[1] for q in roots)
            # A real observed failure is an unbalanced final identity object.
            # Use only complete preceding scalar fields; quarantine the remainder.
            tail = re.match(r'\s*,\s*"user"\s*:\s*\{', fragment[last:])
            if tail and "user" not in keys:
                partial_tail = last
                end = len(raw)
        enough = AUTH_SIGNATURE.issubset(keys) or partial_tail is not None
        if enough and end is not None and len(keys) == len(set(keys)):
            accepted.append((start, end, fragment, roots, partial_tail))
    return accepted[0] if len(accepted) == 1 else None


def route(raw):
    if asa_parts(raw) is not None:
        return "asa"
    if auth_object(raw) is not None:
        return "authentication"
    return "unchanged"


def auth_view(found):
    start, end, fragment, roots, partial_tail = found
    facts = {}
    removed = []
    unknown = []
    for a, b, path, begin in roots:
        key = path[0]
        if key in AUTH_FACTS:
            try:
                value = json.loads(fragment[begin:b])
            except (ValueError, TypeError):
                value = None
            if not isinstance(value, str) or base.MARKER.search(value):
                unknown.append(key)
                value = "unknown_value"
            facts[key] = value
        else:
            reason = ("upstream_or_post_event" if key in AUTH_UPSTREAM
                      else "identity_or_collection" if key in AUTH_IDENTITIES
                      else "unverified_field")
            removed.append({"start": start+a, "end": start+b, "path": key, "reason": reason})
    if partial_tail is not None:
        removed.append({"start":start+partial_tail,"end":end,"path":"unparsed_identity_tail",
                        "reason":"unverified_field"})
    # Complete scalar values only. No repair of malformed numeric or string content.
    text = " ".join("{} {}".format(k, facts[k]) for k in sorted(facts))
    text = final.finalize_text(old.normalize_body(text, "json_fragment"))[0]
    return text, facts, removed, unknown


@functools.lru_cache(maxsize=1024)
def prepare_message(raw):
    parts = asa_parts(raw)
    auth = auth_object(raw) if parts is None else None
    previous = final.prepare_record({"message_sanitized": raw})
    p = dict(previous)
    p.update(route="unchanged", asa_template=None, boundary_verified=False,
             authentication_result_unknown=False, no_observable_auth_facts=False,
             auth_facts={}, repair_spans=[], retained_body_sha256=None)
    if parts:
        body = parts["body"]
        text, spans = final.finalize_text(old.normalize_body(body, "asa_like"))
        p.update(text=parts["event_text"]+text, route="asa", boundary_verified=True,
                 asa_template=template_of(parts),
                 repair_spans=[{"start": 0, "end": parts["body_start"], "reason": "collection_prefix",
                                "preserved_native_event": parts["event_text"]}],
                 final_quarantined_fragments=list(spans))
        import hashlib
        p["retained_body_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    elif auth:
        text, facts, spans, unknown = auth_view(auth)
        p.update(text=text, route="authentication", boundary_verified=True,
                 repair_spans=spans, auth_facts=facts,
                 authentication_result_unknown=facts.get("result") == "unknown_value",
                 no_observable_auth_facts=all(v == "unknown_value" for v in facts.values()),
                 removed_spans=spans,
                 upstream_fields_removed=sum(s["reason"] == "upstream_or_post_event" for s in spans),
                 unverified_fields_isolated=sum(s["reason"] == "unverified_field" for s in spans),
                 final_quarantined_fragments=[])
    p["filtered_empty"] = bool(raw.strip()) and not bool(p["text"])
    return p


def prepare_record(row):
    # Never pass label, product, source, pipeline, timestamp or event ID to the parser.
    return dict(prepare_message(base.string(row.get("message_sanitized"))))
