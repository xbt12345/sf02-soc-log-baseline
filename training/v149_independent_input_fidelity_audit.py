"""Decode actual input coordinates and independently parse observed ASA body facts.

No model imports, model forwards, gradients, fitting or parameter updates.
This measures input fidelity, not whether a classifier learned these fields.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v149_independent_input_fidelity_20261001'
BODY = re.compile(r'\bDeny\s+(tcp|udp|icmp)\s+src\s+([^:\s]+):(\S+)\s+dst\s+([^:\s]+):(\S+)', re.I)
ICMP = re.compile(r'\(type\s+([^,\s)]+),\s*code\s+([^)\s]+)\)', re.I)
ENUM_FIELDS = ('action', 'outcome', 'transport_protocol', 'src_role', 'dst_role',
               'src_port_range', 'dst_port_range', 'icmp_message', 'icmp_unreachable')
BIT_FIELDS = {'src_port_fixed': (17, 65536), 'dst_port_fixed': (17, 65536),
              'icmp_type': (9, 256), 'icmp_code': (9, 256)}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def numeric(token, maximum, missing):
    if not re.fullmatch(r'[0-9]+', token):
        return missing
    n = int(token)
    return n if 0 <= n <= maximum else missing


def parse(raw):
    m = BODY.search(raw)
    if m is None:
        return None
    protocol, sz, se, dz, de = m.groups()
    protocol = protocol.lower()
    fields = dict(action='deny', outcome='blocked', transport_protocol=protocol)
    for side, zone, endpoint in (('src', sz, se), ('dst', dz, de)):
        role = re.fullmatch(r'(inside|outside|dmz)(?:-[0-9]+)?', zone, re.I)
        if role:
            fields[side + '_role'] = role[1].lower()
        if protocol in ('tcp', 'udp'):
            token = endpoint.rsplit('/', 1)[1] if '/' in endpoint else ''
            port = numeric(token, 65535, 65536)
            fields[side + '_port_fixed'] = port
            if port < 65536:
                fields[side + '_port_range'] = 'system' if port < 1024 else 'user' if port < 49152 else 'dynamic'
        else:
            fields[side + '_port_fixed'] = 65536
    if protocol == 'icmp':
        code = ICMP.search(raw[m.end():])
        if code:
            fields['icmp_type'] = numeric(code[1], 255, 256)
            fields['icmp_code'] = numeric(code[2], 255, 256)
    return fields


def main():
    assert not OUT.exists(), 'Never overwrite executed audit evidence.'
    trace_path = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    identity_path = ROOT / 'artifacts/v147_independent_observability_20261001/observable_identity_ledger.parquet'
    known_path = ROOT / 'artifacts/v147_independent_conditional_support_20261001/all_known_port_conditional_support.parquet'
    contract_path = ROOT / 'artifacts/v92_evidence_training_20260928/representation_contract.json'
    input_path = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    cache_path = ROOT / 'artifacts/v138_single_issue_round1_20260930/facts.npy'
    official_path = ROOT / 'data/official/train.parquet'
    source = pd.read_parquet(trace_path, columns=['row_position', 'local', 'root', 'fold', 'truth', 'facts_json', 'raw_message'])
    identity = pd.read_parquet(identity_path)
    for k in ['row_position', 'local', 'root', 'fold', 'truth']:
        assert np.array_equal(source[k], identity[k])
    assert len(source) == 112807 and not source.row_position.duplicated().any()
    # Verify original strings/labels against the actual official file, not only a derived trace.
    source_by_row = source.set_index('row_position'); seen = offset = 0
    for batch in pq.ParquetFile(official_path).iter_batches(batch_size=32768, columns=['message_sanitized', 'label_binary']):
        raw = batch.to_pandas()
        positions = source.row_position[(source.row_position >= offset) & (source.row_position < offset + len(raw))].to_numpy()
        if len(positions):
            actual = raw.iloc[positions - offset]
            ref = source_by_row.loc[positions]
            assert np.array_equal(actual.message_sanitized, ref.raw_message)
            labels = actual.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
            assert np.array_equal(labels, ref.truth)
            seen += len(positions)
        offset += len(raw)
    assert offset == 2056871 and seen == len(source)
    x = sparse.load_npz(input_path).tocsr()
    assert x.shape == (22546, 66287)
    coords = x[:, 65792:].toarray()
    assert coords.shape == (22546, 495)
    assert np.array_equal(coords, np.load(cache_path))
    names = json.loads(contract_path.read_text(encoding='utf-8'))['old_fact_coordinate_names']
    assert len(names) == 477 and len(set(names)) == len(names)
    loc = source.local.to_numpy(); encoded = coords[loc, :len(names)]
    facts = source.facts_json.map(json.loads).to_list()
    decoded, reports, bad = {}, [], np.zeros(len(source), dtype=bool)
    for field in ENUM_FIELDS:
        indices = [i for i, n in enumerate(names) if n.startswith(field + '=')]
        values = [names[i].split('=', 1)[1] for i in indices]
        present = encoded[:, indices] > 0
        assert (present.sum(1) <= 1).all()
        result = np.array([values[int(np.argmax(row))] if row.any() else None for row in present], dtype=object)
        decoded[field] = result
        expected = np.array([f.get(field) if f.get(field) in values else None for f in facts], dtype=object)
        mismatch = result != expected; bad |= mismatch
        reports.append(dict(field=field, original_rows=len(source), observed_rows=int(present.any(1).sum()),
            mismatches=int(mismatch.sum()), unsupported_original_values=dict(Counter(str(f[field]) for f in facts if field in f and f[field] not in values))))
    for field, (width, missing) in BIT_FIELDS.items():
        indices = [names.index(field + ':bit' + str(b)) for b in range(width)]
        present = encoded[:, indices] > 0
        result = (present.astype(np.int64) * (1 << np.arange(width))).sum(1)
        decoded[field] = result
        expected = np.array([f.get(field, missing) for f in facts], dtype=np.int64)
        mismatch = result != expected; bad |= mismatch
        reports.append(dict(field=field, original_rows=len(source), observed_rows=int((result != missing).sum()),
            mismatches=int(mismatch.sum()), unknown_sentinel=missing))
    parsed = [parse(s) for s in source.raw_message]
    raw_comparisons, raw_bad = [], np.zeros(len(source), dtype=bool)
    for field in ['action', 'outcome', 'transport_protocol', 'src_role', 'dst_role',
                  'src_port_fixed', 'dst_port_fixed', 'src_port_range', 'dst_port_range', 'icmp_type', 'icmp_code']:
        check = np.array([p is not None and field in p for p in parsed])
        expected = np.array([p.get(field) if p else None for p in parsed], dtype=object)
        values = decoded[field]
        mismatch = check & (values != expected); raw_bad |= mismatch
        raw_comparisons.append(dict(field=field, independently_observed_comparisons=int(check.sum()),
            raw_to_coordinate_mismatches=int(mismatch.sum())))
    known = pd.read_parquet(known_path, columns=['row_position', 'known_578_cohort'])
    hard_ids = set(known.loc[known.known_578_cohort, 'row_position'])
    hard = source.row_position.isin(hard_ids).to_numpy()
    assert int(hard.sum()) == 578
    out_rows = source[['row_position', 'local', 'root', 'fold', 'truth']].copy()
    out_rows['body_grammar_matched'] = [p is not None for p in parsed]
    out_rows['parsed_facts_to_coordinates_mismatch'] = bad
    out_rows['original_body_to_coordinates_mismatch'] = raw_bad
    out_rows['known_578_cohort'] = hard
    for field, values in decoded.items():
        out_rows['decoded_' + field] = values
    OUT.mkdir()
    out_rows.to_parquet(OUT / 'all_ASA_input_decoding_ledger.parquet', index=False)
    report = dict(status='actual_input_fidelity_measured_not_model_learning', latest_actual_training='V146',
        new_model_forwards=0, new_gradients=0, new_fits=0, new_updates=0, official_rows=offset,
        original_ASA_rows=len(source), official_original_strings_and_labels_exact=True,
        actual_classifier_fact_cache_exact_to_CSR=True, base_semantic_coordinates=477, trailing_other_coordinates=18,
        direct_fields=len(decoded), field_decoding=reports, original_body_comparisons=raw_comparisons,
        grammar_unmatched_original_rows=int(sum(p is None for p in parsed)),
        all_original_rows_with_fact_coordinate_mismatch=int(bad.sum()),
        all_original_rows_with_raw_body_coordinate_mismatch=int(raw_bad.sum()),
        known_578=dict(original_rows=int(hard.sum()), parsed_fact_coordinate_mismatches=int(bad[hard].sum()),
            raw_body_coordinate_mismatches=int(raw_bad[hard].sum()),
            raw_grammar_unmatched=int(sum(parsed[i] is None for i in np.flatnonzero(hard)))),
        unsupported_grammar_row_ids=source.loc[[p is None for p in parsed], 'row_position'].astype(int).tolist(),
        scope=['Actual typed field coordinates and known finite-domain values are decoded without learned weights.',
            'This does not prove arbitrary raw text or ordering is preserved, or hidden representations learned the fields.',
            'The last 18 metadata coordinates are retained but not assigned semantic meanings by this audit.',
            'No label-conditioned field selection, loss weights, pairs, classifier targets or decisions are constructed.',
            'Masked original tokens cannot be reconstructed; protocol applicability and unknown sentinels remain distinct.',
            'Original numeric values matching coordinates does not make ports a threat classifier or prove source transfer.'],
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), trace_path, identity_path,
            known_path, contract_path, input_path, cache_path, official_path]})
    (OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('field_decoding', 'original_body_comparisons',
        'source_sha256', 'scope', 'unsupported_grammar_row_ids')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
