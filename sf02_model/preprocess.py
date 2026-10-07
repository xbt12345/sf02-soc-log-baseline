"""Reconstruct the fitted input contract directly from official raw-log columns.

Only message_sanitized and the independent record src_port are consumed.
Labels, event IDs, products and source-group assignments never enter features.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize as l2_normalize

from ._parser import asa_header, firewall, native_adapter, record_port, semantics
from ._parser.placeholder import normalize as normalize_placeholders
from ._parser.text_mask import stable
from ._parser.text_view import BYTE_FEATURES, byte_matrix, view

WIDTH = 66287
PARSED_FACTS = 477
RECORD_PORT_FEATURES = 18
MAX_BODY_BYTES = 176
_ENCODER = semantics.SemanticFacts()
_ASA_EVENT = re.compile(r"\bUSER-0010-0324\s+Deny\b|%(?:ASA|PIX)-[0-7]-[0-9]{6}:", re.I)


@dataclass(frozen=True)
class PreparedBatch:
    full_features: sparse.csr_matrix
    asa_mask: np.ndarray
    asa_indices: np.ndarray
    header_features: sparse.csr_matrix
    body_bytes: np.ndarray
    lengths: np.ndarray
    route_names: np.ndarray
    source_asa_mask: np.ndarray
    unsupported_asa_mask: np.ndarray
    unsupported_rows: tuple[dict, ...]

    @property
    def unsupported_count(self) -> int:
        return len(self.unsupported_rows)


def _message(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return ""
    return value if isinstance(value, str) else str(value)


def _n1_text(raw: str, route: str) -> str:
    text = stable(view(raw)[0])
    # The fitted full-format router used the old R0 input plus an ASA-only N1
    # delta. Applying the ASA placeholder repair to other routes would change
    # their already fitted numerical representation.
    return normalize_placeholders(text, "placeholder_cluster") if route == "asa" else text


def _ordered_body(raw: str) -> bytes:
    match = asa_header.HEADER.match(raw)
    if match is None:
        raise ValueError("ASA header did not match registered grammar")
    body = raw[match.end():]
    if not body.startswith("USER-0010-0324 Deny "):
        raise ValueError("ASA header/body boundary changed")
    text = normalize_placeholders(stable(view(body)[0]), "placeholder_cluster")
    return text.encode("utf-8")


@lru_cache(maxsize=8192)
def _prepare_message(raw: str):
    error = None
    try:
        parsed = native_adapter.prepare_record({"message_sanitized": raw})
        route = parsed["route"]
        facts = dict(parsed["facts"])
        # This frozen literal-firewall adapter is part of the original full
        # router input. It changes facts only on the unsupported parser route.
        patch = firewall.parse(raw) if route == "unsupported" else None
        if patch is not None:
            facts = dict(patch["facts"])
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        route, facts = "unsupported", {}
        error = f"parser_error:{type(exc).__name__}:{str(exc)[:160]}"
    text = _n1_text(raw, route)
    # Other verified Cisco/ASA routes (ACL and protocol messages) retain the
    # linear router. Their native ASA event codes do not make them P_IS inputs.
    candidate = route == "asa" or (route == "unsupported" and bool(_ASA_EVENT.search(raw)))
    header_text, body = None, None
    if candidate:
        if route != "asa":
            error = error or "asa_parser_unsupported"
        else:
            try:
                header_text = asa_header.transform(raw)[0]
                body = _ordered_body(raw)
                if not body:
                    error = "asa_body_empty"
                elif len(body) > MAX_BODY_BYTES:
                    error = f"asa_body_too_long:{len(body)}>{MAX_BODY_BYTES}"
            except (ValueError, TypeError) as exc:
                error = f"asa_grammar_unsupported:{str(exc)[:160]}"
        if error is not None:
            header_text, body = None, None
    return route, facts, text, candidate, header_text, body, error


def prepare_batch(messages: Iterable, src_ports: Iterable) -> PreparedBatch:
    """Preserve every row; expose unsupported ASA rows for explicit router fallback.

    full_features covers all rows in input order. ASA-specific arrays cover
    asa_indices only, in the same order. No row is dropped or body truncated.
    The caller should report unsupported_rows when deploying router fallback.
    Message caches are bounded and contain no event IDs, labels or dataset IDs.
    """
    messages = [_message(value) for value in messages]
    ports = list(src_ports)
    if len(messages) != len(ports):
        raise ValueError("messages and src_ports must have the same number of rows")
    n = len(messages)
    if not n:
        empty = sparse.csr_matrix((0, WIDTH), dtype=np.float32)
        mask = np.empty(0, dtype=bool)
        return PreparedBatch(empty, mask, np.empty(0, dtype=np.int64), empty.copy(),
                             np.empty((0, 0), dtype=np.uint8), np.empty(0, dtype=np.int16),
                             np.empty(0, dtype=object), mask.copy(), mask.copy(), ())

    # Factorization is a computation optimization only. The final feature rows
    # are expanded back to every original record, including its own src_port.
    unique = list(dict.fromkeys(messages))
    lookup = {raw: i for i, raw in enumerate(unique)}
    inverse = np.asarray([lookup[raw] for raw in messages], dtype=np.int64)
    records = [_prepare_message(raw) for raw in unique]
    facts = [record[1] for record in records]
    errors = [record[6] for record in records]
    try:
        factual = _ENCODER.transform(facts).astype(np.float32)
    except (ValueError, TypeError, OverflowError):
        # A malformed new observation must not abort unrelated records. This
        # explicitly reported deployment fallback has no training-equivalence
        # claim; its facts use the fitted empty-facts representation.
        pieces = []
        for i, item in enumerate(facts):
            try:
                piece = _ENCODER.transform([item]).astype(np.float32)
            except (ValueError, TypeError, OverflowError) as exc:
                facts[i] = {}
                errors[i] = f"fact_encoding_error:{type(exc).__name__}:{str(exc)[:160]}"
                piece = _ENCODER.transform([{}]).astype(np.float32)
            pieces.append(piece)
        factual = sparse.vstack(pieces, format="csr")
    if factual.shape != (len(unique), PARSED_FACTS):
        raise RuntimeError("The frozen parsed-fact schema changed")
    factual = l2_normalize(factual, norm="l2", copy=False)
    text = byte_matrix([record[2] for record in records])
    message_ports = np.asarray([item.get("src_port_fixed", 65536) for item in facts])[inverse]
    metadata, _, _ = record_port.encode(ports, message_ports)
    full = sparse.hstack([text[inverse], factual[inverse], metadata], format="csr", dtype=np.float32)
    full.sort_indices()
    if full.shape != (n, WIDTH) or not np.isfinite(full.data).all():
        raise RuntimeError("Invalid reconstructed full-format feature matrix")

    routes = np.asarray([record[0] for record in records], dtype=object)[inverse]
    source_mask = np.asarray([record[3] for record in records], dtype=bool)[inverse]
    supported = np.asarray([record[3] and record[4] is not None and errors[i] is None
                            for i, record in enumerate(records)], dtype=bool)[inverse]
    indices = np.flatnonzero(supported).astype(np.int64)
    row_errors = tuple({"index": i, "route": str(routes[i]), "reason": errors[int(inverse[i])]}
                       for i in range(n) if errors[int(inverse[i])] is not None)
    if len(indices):
        selected = [records[int(inverse[i])] for i in indices]
        header_text = byte_matrix([record[4] for record in selected])
        header = sparse.hstack([header_text, full[indices, BYTE_FEATURES:]], format="csr", dtype=np.float32)
        header.sort_indices()
        lengths = np.asarray([len(record[5]) for record in selected], dtype=np.int16)
        body = np.zeros((len(indices), int(lengths.max())), dtype=np.uint8)
        for i, record in enumerate(selected):
            body[i, :lengths[i]] = np.frombuffer(record[5], dtype=np.uint8)
    else:
        header = sparse.csr_matrix((0, WIDTH), dtype=np.float32)
        body = np.empty((0, 0), dtype=np.uint8)
        lengths = np.empty(0, dtype=np.int16)
    return PreparedBatch(full, supported, indices, header, body, lengths, routes,
                         source_mask, source_mask & ~supported, row_errors)


def clear_preprocess_cache() -> None:
    """Release the public message cache; lower parser caches remain bounded."""
    _prepare_message.cache_clear()
