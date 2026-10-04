"""Recount legal same/opposite-class factual support, including correct controls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v144_independent_decision_review import ROOT, read, sha

SRC = ROOT/'artifacts/v147_nearest_support_audit_20261001'
OBS = ROOT/'artifacts/v147_independent_observability_20261001'
OUT = ROOT/'artifacts/v147_independent_conditional_support_20261001'


def observed_key(f):
    if f.get('transport_protocol') not in ['tcp', 'udp'] or not all(
        0 <= f.get(k, 65536) <= 65535 for k in ['src_port_fixed', 'dst_port_fixed']):
        return None
    value = {k:v for k,v in f.items() if k != 'src_port_fixed'}
    value['src_port_observed'] = value['dst_port_observed'] = True
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def main():
    assert not OUT.exists(), 'Preserve executed independent evidence'
    q = pd.read_parquet(SRC/'all_known_port_query_support.parquet')
    d = pd.read_parquet(OBS/'observable_identity_ledger.parquet').set_index('row_position')
    assert len(q) == 63208 and not q.row_position.duplicated().any()
    ref = d.loc[q.row_position]
    for name in ['truth', 'fold', 'root', 'canonical_key', 'pred_A0', 'pred_V142', 'pred_A', 'pred_B']:
        assert np.array_equal(q[name], ref[name])
    fact_path = ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    facts = pd.read_parquet(fact_path, columns=['row_position', 'facts_json']).set_index('row_position')
    keys = facts.loc[q.row_position, 'facts_json'].map(json.loads).map(observed_key).to_numpy()
    assert np.array_equal(q.key, keys)
    assert q.groupby('canonical_key').truth.nunique().max() == 1
    profiles, full = [], []
    for fold in range(3):
        train = q[q.fold.ne(fold)]
        held = q[q.fold.eq(fold)].copy()
        assert not set(train.root) & set(held.root)
        counts = train.groupby(['key', 'truth']).agg(rows=('truth', 'size'), roots=('root', 'nunique'),
                                                    numeric_inputs=('canonical_key', 'nunique')).reset_index()
        for cls in [1, 2]:
            c = counts[counts.truth.eq(cls)].set_index('key')
            for name in ['rows', 'roots', 'numeric_inputs']:
                held[f'class{cls}_{name}'] = held.key.map(c[name]).fillna(0).astype(int)
        held['same_class_rows'] = np.where(held.truth.eq(1), held.class1_rows, held.class2_rows)
        held['same_class_roots'] = np.where(held.truth.eq(1), held.class1_roots, held.class2_roots)
        held['opposite_class_rows'] = np.where(held.truth.eq(1), held.class2_rows, held.class1_rows)
        assert np.array_equal(held.same_class_rows, held.same_key_rows)
        assert np.array_equal(held.same_class_roots, held.same_key_roots)
        held['same_class_bucket'] = np.select([held.same_class_rows.eq(0), held.same_class_roots.eq(1)],
                                               ['none', 'one_source'], default='multiple_sources')
        held['opposite_class_present'] = held.opposite_class_rows.gt(0)
        for (cls, bucket, opposite), g in held.groupby(['truth', 'same_class_bucket', 'opposite_class_present']):
            profiles.append(dict(fold=fold, truth=int(cls), same_class_bucket=bucket,
                                 opposite_class_present=bool(opposite), original_rows=len(g),
                                 source_roots=int(g.root.nunique()),
                                 **{a+'_errors':int(g['pred_'+a].ne(cls).sum()) for a in ['A0', 'V142', 'A', 'B']}))
        full.append(held)
    allq = pd.concat(full, ignore_index=True)
    hard = allq[allq.known_578_cohort].copy()
    assert len(hard) == 578
    hard_groups = []
    for (bucket, opposite), g in hard.groupby(['same_class_bucket', 'opposite_class_present']):
        hard_groups.append(dict(same_class_bucket=bucket, opposite_class_present=bool(opposite),
                                original_rows=len(g), source_roots=int(g.root.nunique()),
                                V142_errors=int(g.pred_V142.ne(2).sum()), A_errors=int(g.pred_A.ne(2).sum()),
                                B_errors=int(g.pred_B.ne(2).sum())))
    assert int(hard.same_class_rows.gt(0).sum()) == 48
    assert int((hard.same_class_rows.gt(0) & hard.opposite_class_present).sum()) == 38
    assert not hard.same_class_roots.ge(2).any()
    report = dict(status='conditional_same_and_opposite_class_support_recounted', latest_actual_training='V146',
                  new_fits=0, new_gradients=0, new_forwards=0, new_updates=0,
                  known_port_original_rows=len(allq), all_correct_controls_preserved=True,
                  key_reconstructed_from_original_parsed_facts=True, role_source_overlap=0,
                  population_profiles=profiles, known_578_hard_S=hard_groups,
                  new_method_selected=False, HELD_labels_used_for_training_or_rules=False,
                  limits=['Factual key omits exact observed source-port value, preserving range and other parsed fields.',
                          'This key is neither raw-log identity nor proof of a label-preserving transformation.',
                          'Single-source support is observed supervision, not independent multi-source confirmation.',
                          'Opposite-class support forbids copying a same-key S label onto all matching inputs.',
                          'No opposite-class TRAIN match does not prove it cannot appear in a new source.',
                          'Historical HELD class counts describe failures, not candidate construction or weights.'],
                  source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),
                                 SRC/'all_known_port_query_support.parquet', OBS/'audit.json',
                                 OBS/'observable_identity_ledger.parquet', fact_path]})
    OUT.mkdir()
    allq.to_parquet(OUT/'all_known_port_conditional_support.parquet', index=False)
    (OUT/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status', 'known_port_original_rows', 'known_578_hard_S']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
