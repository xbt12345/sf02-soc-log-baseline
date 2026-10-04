"""Scoped input and fitting safeguards, not a trained v3.5 model. Python 3.8+."""
import collections
import json
import re

import numpy as np

import soc_v3_prepare as base
import v331_prepare as legacy

VERSION = "v351-scoped-input-1.1"
RFC5424_HEADER = re.compile(
    r'^\s*<(?P<pri>[0-9]{1,3})>1\s+(?P<stamp>\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+(?=[-\[])')
STAMP = re.compile(
    r'(?:[0-9]{4}|' + base.MARKER.pattern + r')-[0-9]{2}-[0-9]{2}[Tt]'
    r'[0-9]{2}:[0-9]{2}:[0-9]{2}'
    r'(?:\.(?:[0-9]+|[0-9-]*(?:USER|ORG|CRED)[A-Za-z0-9-]*))?'
    r'(?:[Zz]|[+-][0-9]{2}:?[0-9]{2})?', re.I)
# Require a whole explicitly named duration; never complete a redacted value.
DURATION = re.compile(
    r'\b(?:duration|elapsed)\b["\x27]?\s*[:=]\s*["\x27]?'
    r'(?P<value>[0-9]+:[0-5][0-9]:[0-5][0-9](?:\.[0-9]+)?)'
    r'(?![\w.:-])', re.I)


def product_bucket(value):
    """Audit only. This bucket must never be passed to a classifier or router."""
    value = base.string(value).strip()
    return value or "<missing>"


def neutralize_collection_stamp(raw):
    """Only the bounded RFC 5424 TIMESTAMP slot, including observed redaction.

    A canonical raw timestamp is an intermediate token for the legacy parser;
    it becomes 'time' and is not a recovered or invented original event time.
    """
    header = RFC5424_HEADER.match(raw)
    if header is None or int(header['pri']) > 191:
        return raw
    stamp = header['stamp']
    if stamp != '-' and STAMP.fullmatch(stamp) is None:
        return raw
    a, b = header.span('stamp')
    return raw[:a] + '2000-01-01T00:00:00Z' + raw[b:]


def model_text(row):
    """Explicit projection: no metadata, label, date, or missing-product feature.

    Retains legacy text for a matched diagnostic candidate, with bounded header
    time and duration fixes. It is NOT the complete B1/B2 representation and retains legacy
    implicit source/format risks. Old fitted models cannot be used unchanged.
    """
    raw = neutralize_collection_stamp(base.string(row.get("message_sanitized")))
    matches = list(DURATION.finditer(raw))
    if not matches:
        return legacy.prepare_message(raw)["text"]
    # Collision-free alphabetic placeholders survive the legacy normalizer.
    # They are always restored/removed before returning predictor text.
    prefix = "preserveddurationplaceholderz"
    while prefix in raw.casefold():
        prefix += "z"
    values = []
    def protect(match):
        value = match.group("value")
        i = len(values)
        token = prefix + "q" * (i + 1) + "z"
        values.append((token, value))
        a, b = match.span("value")
        return match.group()[:a-match.start()] + token + match.group()[b-match.start():]
    protected = DURATION.sub(protect, raw)
    text = legacy.prepare_message(protected)["text"]
    for token, value in values:
        text = text.replace(token, value)
    if prefix in text:
        raise AssertionError("Unrestored duration placeholder")
    return text


def endpoint_facts(endpoint):
    """Conservative visible fields for collision diagnosis; identities excluded."""
    interface, sep, remainder = endpoint.partition(":")
    if not sep:
        return {"interface": None, "port": None, "port_state": "unparsed"}
    address, slash, port = remainder.rpartition("/")
    if not slash:
        return {"interface": interface.casefold(), "port": None, "port_state": "absent"}
    if base.MARKER.search(port):
        number, state = None, "redacted"
    elif re.fullmatch(r"[0-9]+", port) and 0 <= int(port) <= 65535:
        number, state = int(port), "observed"
    else:
        number, state = None, "invalid"
    return {"interface": interface.casefold(), "port": number, "port_state": state}


def asa_visible_facts(raw):
    """Not an attack label or complete behavioral context.

    Exact interface is retained in this audit fingerprint to avoid hiding a
    genuine role difference. Deployment role encoding remains a B2 decision.
    """
    p = legacy.asa_parts(raw)
    if p is None:
        return None
    f = p["facts"]
    return {"action": f["action"].casefold(), "protocol": f["protocol"].casefold(),
            "src": endpoint_facts(f["src"]), "dst": endpoint_facts(f["dst"]),
            "icmp": f["icmp"], "native_event": p["event_text"],
            "aclword": f["aclword"], "acl": f["acl"], "acl_hashes": f["hex"]}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def assert_group_isolation(groups, roles):
    """All active occurrences of a repeat/union group have exactly one role."""
    groups, roles = np.asarray(groups), np.asarray(roles)
    if groups.ndim != 1 or groups.shape != roles.shape:
        raise ValueError("Aligned one-dimensional groups and roles required")
    active = roles >= 0
    _, inv = np.unique(groups[active], return_inverse=True)
    low = np.full(len(np.unique(inv)), 127, dtype=np.int16)
    high = np.full(len(low), -127, dtype=np.int16)
    np.minimum.at(low, inv, roles[active])
    np.maximum.at(high, inv, roles[active])
    if np.any(low != high):
        raise ValueError("A repeated representation crosses active roles")


def fit_group_distributions(groups, labels, roles, fit_roles=(0,)):
    """Fit-side label counts for review or exact loss aggregation, never inference.

    A group distribution is descriptive. Replacing per-row cross entropy with
    the count-weighted distribution has the SAME loss, not new supervision.
    """
    groups, labels, roles = map(np.asarray, (groups, labels, roles))
    if groups.ndim != 1 or not (groups.shape == labels.shape == roles.shape):
        raise ValueError("Aligned one-dimensional arrays required")
    take = np.isin(roles, fit_roles)
    if np.any(~np.isin(labels[take], [0, 1, 2])):
        raise ValueError("Unknown fit label")
    counts = collections.defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for group, label in zip(groups[take], labels[take]):
        counts[int(group)][int(label)] += 1
    return [{"group_id": g, "counts": c.tolist(),
             "distribution": (c / c.sum()).tolist(), "mixed": bool((c > 0).sum() > 1)}
            for g, c in sorted(counts.items())]


def bounded_repeat_weights(fit_feature_keys, lower=0.2, upper=5.0):
    """One preregistered candidate: mean-one clipped inverse-sqrt repeat weights.

    Input MUST be only currently allowed fit rows. Keys are label-blind final
    feature identities (including empty text), not event IDs or audit groups
    made unique by row. This function does not decide whether to use weighting.
    """
    if not 0 < lower <= 1 <= upper:
        raise ValueError("Bounds must contain one and be positive")
    keys = np.asarray(fit_feature_keys)
    if keys.ndim != 1 or not len(keys):
        raise ValueError("Nonempty one-dimensional fit keys required")
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    raw = 1 / np.sqrt(counts[inv].astype(np.float64))
    lo, hi = 0.0, upper / raw.min()
    for _ in range(80):
        scale = (lo + hi) / 2
        if np.clip(scale * raw, lower, upper).mean() < 1:
            lo = scale
        else:
            hi = scale
    result = np.clip((lo + hi) / 2 * raw, lower, upper)
    if not np.isclose(result.mean(), 1, atol=1e-12):
        raise AssertionError("Weight normalization failed")
    return result
