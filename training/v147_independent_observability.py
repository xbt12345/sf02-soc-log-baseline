"""Original observable-input collision audit; no models, fitting or label rules."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from v144_independent_decision_review import ROOT, read, sha

RUN = ROOT/'artifacts/v146_guarded_pair_training_20261001'
OUT = ROOT/'artifacts/v147_independent_observability_20261001'
OFFICIAL = ROOT/'data/official/train.parquet'
INPUT = ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
FACTS = ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
FIELDS = ['message_sanitized', 'src_port', 'pipeline', 'src_ip', 'dst_ip',
          'src_host', 'dst_host', 'username', 'product_name', 'vendor_name']


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def read_original_rows(positions):
    chunks = []
    start = 0
    reader = pq.ParquetFile(OFFICIAL)
    for batch in reader.iter_batches(batch_size=32768, columns=FIELDS+['timestamp', 'label_binary']):
        end = start+batch.num_rows
        ids = positions[np.searchsorted(positions, start):np.searchsorted(positions, end)]
        if len(ids):
            frame = batch.to_pandas().iloc[ids-start].copy()
            frame['row_position'] = ids
            chunks.append(frame)
        start = end
    assert start == 2056871
    raw = pd.concat(chunks, ignore_index=True)
    assert np.array_equal(raw.row_position, positions)
    return raw


def summary(frame, col):
    c = frame.groupby([col, 'truth']).size().unstack(fill_value=0).reindex(columns=[1, 2], fill_value=0)
    mixed = c[(c[1] > 0) & (c[2] > 0)]
    return dict(original_rows=len(frame), unique_inputs=len(c), mixed_inputs=len(mixed),
                original_rows_in_mixed_inputs=int(mixed.sum().sum()),
                empirical_deterministic_minimum_errors=int((mixed.sum(1)-mixed.max(1)).sum()))


def main():
    if OUT.exists():
        raise FileExistsError('Preserve the original independent audit')
    delivery = read(RUN/'final_delivery.json')
    assert delivery['latest_actual'] == 'V146' and not delivery['model_promoted']
    d = pd.read_parquet(RUN/'ASA_prediction_ledger.parquet').sort_values('row_position').reset_index(drop=True)
    assert len(d) == 112807 and not d.row_position.duplicated().any()
    raw = read_original_rows(d.row_position.to_numpy())
    truth = raw.label_binary.map({'benign':0, 'malicious':1, 'suspicious':2}).to_numpy()
    assert np.array_equal(d.truth, truth)
    x = sparse.load_npz(INPUT).astype(np.float32).tocsr()
    x.sum_duplicates(); x.sort_indices()
    keys = [hashlib.sha256(x.indices[x.indptr[i]:x.indptr[i+1]].astype('<i8').tobytes()+
                           x.data[x.indptr[i]:x.indptr[i+1]].astype('<f4').tobytes()).hexdigest()
            for i in range(x.shape[0])]
    assert x.shape == (22546, 66287)
    assert np.array_equal(d.canonical_key, d.local.map(dict(enumerate(keys))))
    facts = pd.read_parquet(FACTS, columns=['row_position', 'facts_json']).set_index('row_position')
    fact_objects = facts.loc[d.row_position, 'facts_json'].map(json.loads).tolist()
    d['full_body_facts'] = [digest(f) for f in fact_objects]
    views = {
        'raw_message': ['message_sanitized'],
        'raw_message_and_record_src_port': ['message_sanitized', 'src_port'],
        'all_observables_without_event_id_or_clock': FIELDS,
        'all_observables_without_event_id': FIELDS+['timestamp'],
    }
    for name, fields in views.items():
        d[name] = [digest(row) for row in raw[fields].itertuples(index=False, name=None)]
    roles = []
    for role in ['whole_development_population', 0, 1, 2]:
        a = d if isinstance(role, str) else d[d.fold.ne(role)]
        if not isinstance(role, str):
            assert not set(a.root) & set(d.loc[d.fold.eq(role), 'root'])
        roles.append(dict(training_role=role, views={k:summary(a, k) for k in ['canonical_key', 'full_body_facts']+list(views)}))
    mixed_keys = set(d.groupby('canonical_key').truth.nunique().loc[lambda s:s > 1].index)
    conflict = d[d.canonical_key.isin(mixed_keys)].copy()
    all_fields = FIELDS+['timestamp']
    class_conflicts = []
    for key, a in conflict.groupby('canonical_key'):
        idx = a.index
        details = raw.loc[idx, all_fields]
        varying = [k for k in all_fields if details[k].nunique(dropna=False) > 1]
        class_conflicts.append(dict(canonical_key=key, original_rows=len(a), sources=int(a.root.nunique()),
                                   M=int(a.truth.eq(1).sum()), S=int(a.truth.eq(2).sum()),
                                   varying_original_fields=varying,
                                   raw_message_mixed_inputs=summary(a, 'raw_message')['mixed_inputs'],
                                   no_clock_full_observable_floor=summary(a, 'all_observables_without_event_id_or_clock')['empirical_deterministic_minimum_errors']))
    known_both = np.array([f.get('transport_protocol') in ['tcp', 'udp'] and
                          0 <= f.get('src_port_fixed',65536) <= 65535 and
                          0 <= f.get('dst_port_fixed',65536) <= 65535 for f in fact_objects])
    hard = d.truth.eq(2).to_numpy() & d.pred_V142.ne(2).to_numpy() & known_both
    hard_summary = dict(original_rows=int(hard.sum()), sources=int(d.loc[hard,'root'].nunique()),
                        A_remaining_errors=int(d.loc[hard,'pred_A'].ne(2).sum()),
                        B_remaining_errors=int(d.loc[hard,'pred_B'].ne(2).sum()),
                        numeric_input_conflict_rows=int(d.loc[hard,'canonical_key'].isin(mixed_keys).sum()))
    bindings = {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__), OFFICIAL, INPUT, FACTS,
                                                             RUN/'ASA_prediction_ledger.parquet', RUN/'final_delivery.json']}
    output = dict(status='original_observable_and_actual_numeric_input_collisions_recounted',
                  latest_actual_training='V146', new_fits=0, new_forwards=0, new_gradients=0, new_updates=0,
                  all_actual_numeric_identities_recomputed=True, original_ASA_rows=len(raw),
                  roles=roles, numeric_conflict_groups=class_conflicts, known_both_hard_S=hard_summary,
                  limitations=['Observed collision floors apply only to that exact representation and recorded population.',
                               'Different original values prove observable differences, not causal or transferable label evidence.',
                               'Timestamp, IP, host, user and product may separate rows by identity; none is licensed as a classifier shortcut.',
                               'Raw sanitized logs are not recovered pre-redaction traffic or ground-truth attack context.',
                               'HELD truth is retrospective diagnosis only; no matched pairs, weights or new labels constructed.',
                               'Zero empirical collision floor does not prove sufficient data, learnability or unseen-source generalization.'],
                  source_sha256=bindings)
    OUT.mkdir()
    d.to_parquet(OUT/'observable_identity_ledger.parquet', index=False)
    pd.DataFrame(class_conflicts).to_parquet(OUT/'numeric_conflict_original_field_profiles.parquet', index=False)
    (OUT/'audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:output[k] for k in ['status', 'roles', 'known_both_hard_S']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
