"""Audit context shortcuts and ordered-view header fidelity, without models.

Uses independent pure-text reconstruction of current V124 normalization.
No classifier, decoder, optimizer or model runtime is imported or executed.
Projection counts are descriptive; no threat decision rule is constructed.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from v75_views import SYSLOG, view

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v151_independent_context_qualification_20261001'
IDENTITY_CLOCK = r'(?:(?:USER|HOST|CRED|ORG)-)+[0-9]+(?:-[0-9]+)*'
HEADER = re.compile(r'^<\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+'
                    r'(?:(?:\d{4}|' + IDENTITY_CLOCK + r')\s+)?'
                    r'(?:\d{2}:\d{2}:\d{2}|' + IDENTITY_CLOCK + r'):?\s*'
                    r'(?=USER-0010-0324\s+Deny\b)')
EMBEDDED = re.compile(r'(?:USER|HOST|CRED|ORG)-[0-9]+(?:-[0-9]+)*')
CLUSTER = re.compile(r'[ \t]*(?:(?:CRED|HOST|USER|ORG)-[ \t]*)*<IDENTITY>[ \t]*')
MONTHDAY = re.compile(r'^(<\d{1,3}>)(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}')
PRI = re.compile(r'^<([0-9]{1,3})>')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def current_text(raw):
    header = HEADER.match(raw)
    assert header is not None, 'Unknown grammar cannot silently lose a prefix'
    if SYSLOG.match(raw):
        text = view(raw)[0]
    else:
        head, body = raw[:header.end()], raw[header.end():]
        assert re.search(IDENTITY_CLOCK + r':?\s*$', head)
        assert body.startswith('USER-0010-0324 Deny ')
        text = ' <ABSOLUTE_CLOCK> ' + view(body)[0]
    return CLUSTER.sub(' <IDENTITY> ', EMBEDDED.sub(' <IDENTITY> ', text)), header.end(), SYSLOG.match(raw) is None


def state(v):
    if v is None or pd.isna(v):
        return 'null'
    return 'empty' if isinstance(v, str) and not v.strip() else 'observed'


def collisions(frame, key):
    # Group actual strings/tuples rather than assuming digest equality.
    c = frame.groupby([key, 'truth'], sort=False).size().unstack(fill_value=0).reindex(columns=[1, 2], fill_value=0)
    mixed = c[(c > 0).sum(1) > 1]
    return dict(original_rows=len(frame), unique_inputs=len(c), mixed_groups=len(mixed),
                mixed_original_rows=int(mixed.sum().sum()), empirical_projection_minimum_errors=int((c.sum(1) - c.max(1)).sum()))


def main():
    assert not OUT.exists(), 'Preserve already executed evidence'
    trace_path = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    official_path = ROOT / 'data/official/train.parquet'
    patched_path = ROOT / 'artifacts/v124_header_trial_20260929/header_span_ledger.parquet'
    input_audit_path = ROOT / 'artifacts/v149_independent_input_fidelity_20261001/audit.json'
    v151 = ROOT / 'artifacts/v151_context_coherence_20261001/audit.json'
    trace = pd.read_parquet(trace_path).sort_values('row_position').reset_index(drop=True)
    positions = trace.row_position.to_numpy(); offset = 0; parts = []
    for batch in pq.ParquetFile(official_path).iter_batches(batch_size=32768, columns=['message_sanitized', 'label_binary', 'product_name', 'vendor_name']):
        rows = batch.to_pandas(); take = positions[(positions >= offset) & (positions < offset + len(rows))]
        if len(take):
            actual = rows.iloc[take - offset].copy(); actual['row_position'] = take; parts.append(actual)
        offset += len(rows)
    actual = pd.concat(parts).sort_values('row_position').reset_index(drop=True)
    assert offset == 2056871 and len(actual) == len(trace) == 112807
    assert np.array_equal(actual.row_position, trace.row_position)
    assert np.array_equal(actual.message_sanitized, trace.raw_message)
    assert np.array_equal(actual.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}), trace.truth)
    expected_hash = json.loads(input_audit_path.read_text())['source_sha256']['data/official/train.parquet']
    assert sha(official_path) == expected_hash
    cached = {}; values = []
    for raw in actual.message_sanitized:
        if raw not in cached:
            old, spans = view(raw)
            assert ''.join(raw[a:b] for a, b, _ in spans['spans']) == raw
            current, boundary, gap = current_text(raw)
            changed_date = MONTHDAY.sub(lambda m: m[1] + 'Jan 01', raw, count=1)
            assert changed_date[HEADER.match(changed_date).end():] == raw[boundary:]
            level = int(PRI.match(raw)[1]); facility, severity = divmod(level, 8)
            cached[raw] = (old, current, boundary, gap, old != view(changed_date)[0], current != current_text(changed_date)[0], facility, severity)
        values.append(cached[raw])
    frame = trace[['row_position', 'local', 'root', 'fold', 'truth']].copy()
    frame['old_v75_partial_clock_view'] = [v[0] for v in values]
    frame['current_v124_normalized_ordered_view'] = [v[1] for v in values]
    frame['header_gap'] = [v[3] for v in values]
    frame['old_view_changes_under_date_only'] = [v[4] for v in values]
    frame['current_view_changes_under_date_only'] = [v[5] for v in values]
    frame['syslog_facility'] = [v[6] for v in values]
    frame['syslog_severity'] = [v[7] for v in values]
    frame['product_state'] = actual.product_name.map(state)
    frame['vendor_state'] = actual.vendor_name.map(state)
    frame['complete_facts'] = trace.facts_json.map(lambda s: json.dumps(json.loads(s), sort_keys=True, separators=(',', ':')))
    for name in ['syslog_facility', 'syslog_severity', 'product_state']:
        frame['facts_plus_' + name] = list(zip(frame.complete_facts, frame[name]))
    frame['facts_plus_current_ordered'] = list(zip(frame.complete_facts, frame.current_v124_normalized_ordered_view))
    patched = pd.read_parquet(patched_path).set_index('row_position')
    assert set(frame.loc[frame.header_gap, 'row_position']) == set(patched.index) and int(frame.header_gap.sum()) == 682
    for i in frame.index[frame.header_gap]:
        ref = patched.loc[int(frame.row_position.iloc[i])]
        assert values[i][2] == ref.header_end and digest(values[i][1]) == ref.new_text_sha256
        assert digest(actual.message_sanitized.iloc[i]) == ref.raw_sha256
        assert digest(actual.message_sanitized.iloc[i][values[i][2]:]) == ref.body_sha256
    assert int(frame.old_view_changes_under_date_only.sum()) == 682
    assert not frame.current_view_changes_under_date_only.any()
    assert frame.syslog_severity.eq(4).all()
    assert frame.loc[frame.truth.eq(1), 'product_state'].eq('empty').all()
    assert frame.loc[frame.truth.eq(2), 'product_state'].eq('observed').all()
    keys = ['complete_facts', 'old_v75_partial_clock_view', 'current_v124_normalized_ordered_view',
            'facts_plus_current_ordered', 'facts_plus_syslog_facility', 'facts_plus_syslog_severity', 'facts_plus_product_state']
    profiles = []
    for name in keys:
        profiles.append(dict(role='whole_observed_development', projection=name, **collisions(frame, name)))
        for fold in range(3):
            legal = frame[frame.fold.ne(fold)]
            assert not set(legal.root) & set(frame.loc[frame.fold.eq(fold), 'root'])
            profiles.append(dict(role='legal_TRAIN', fold=fold, projection=name, **collisions(legal, name)))
    original_worker = json.loads(v151.read_text())
    worker_scope = [r for r in original_worker['projections'] if r['role'] == 'whole_observed_population']
    assert profiles[4 * keys.index('old_v75_partial_clock_view')]['empirical_projection_minimum_errors'] == next(r['deterministic_observed_population_error_floor'] for r in worker_scope if r['projection'] == 'ordered_identity_clock_removed_key')
    level_and_product = frame.groupby(['syslog_facility', 'syslog_severity', 'product_state', 'truth']).size().rename('original_rows').reset_index()
    exported = frame.drop(columns=keys)
    exported['old_v75_ordered_sha256'] = frame.old_v75_partial_clock_view.map(digest)
    exported['current_ordered_sha256'] = frame.current_v124_normalized_ordered_view.map(digest)
    OUT.mkdir()
    exported.to_parquet(OUT / 'all_original_context_qualification_ledger.parquet', index=False)
    level_and_product.to_parquet(OUT / 'facility_product_class_mass.parquet', index=False)
    source_files = [Path(__file__), trace_path, official_path, patched_path, input_audit_path, v151,
                    ROOT / 'training/v75_views.py', ROOT / 'training/v124_header.py', ROOT / 'training/v75_corrective.py',
                    ROOT / 'training/v99_normalization_feasibility.py']
    report = dict(status='context_shortcut_and_ordered_header_qualification_executed', latest_actual_classifier='V146',
                  new_model_forwards=0, new_gradients=0, new_fits=0, new_updates=0,
                  official_rows=offset, original_ASA_rows=len(frame), official_body_and_gold_exact=True,
                  old_v75_date_sensitive_original_rows=682, current_date_sensitive_original_rows=0,
                  current_patch_exact_to_executed_V124_682=True,
                  product_state_threat_class_proxy=dict(malicious_empty=78748, suspicious_observed=34059,
                                                       warning='Perfect observed separation is a dataset shortcut, not independent threat evidence.'),
                  all_syslog_severity_is_4=True, facility_product_class_mass=level_and_product.to_dict('records'),
                  projections=profiles, classifier_gain=False, quality_acceptance=False, issue_solved=False,
                  limits=['Current text is independently reconstructed from sealed V124 code and checked against its executed 682 patch receipts.',
                          'Actual CSR bytes are not regenerated here; ordered projections are not the classifier input or evidence of classification gain.',
                          'Raw strings retain unresolved interface/ACL suffixes and formatting. Collision reduction does not establish transferable threat semantics.',
                          'All role counts preserve original frequency; known conflicts, unknown states and correct controls remain.',
                          'Logging facility is a configurable logging source field; correlations are not official M/S rationale.',
                          'No conditional classifier, lookup threat rule, training weight, pseudo-label or selection from HELD is generated.'],
                  source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source_files})
    (OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'old_v75_date_sensitive_original_rows', 'current_date_sensitive_original_rows',
                      'product_state_threat_class_proxy', 'facility_product_class_mass']}, ensure_ascii=False), flush=True)
    print(json.dumps([r for r in profiles if r['role'] == 'whole_observed_development'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
