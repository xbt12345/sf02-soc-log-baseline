"""Full ASA population companion to the declared known-port V147 diagnostic."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v146_guarded_pair_training_20261001'
OUT = ROOT / 'artifacts/v147_nearest_support_audit_20261001'


def main():
    target = OUT / 'whole_population_audit.json'
    assert not target.exists(), 'Preserve completed companion'
    report = json.loads((OUT / 'audit.json').read_text(encoding='utf-8'))
    d = pd.read_parquet(RUN / 'ASA_prediction_ledger.parquet')
    facts_path = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    f = pd.read_parquet(facts_path, columns=['row_position', 'facts_json'])
    assert np.array_equal(f.row_position, d.row_position)
    facts = f.facts_json.map(json.loads)
    d['protocol'] = facts.map(lambda v: v.get('transport_protocol', '<absent>'))
    d['body_src_port_observed'] = facts.map(lambda v: 0 <= v.get('src_port_fixed', 65536) <= 65535)
    d['body_dst_port_observed'] = facts.map(lambda v: 0 <= v.get('dst_port_fixed', 65536) <= 65535)
    d['port_pair_diagnostic_eligible'] = d.protocol.isin(['tcp', 'udp']) & d.body_src_port_observed & d.body_dst_port_observed
    d['whole_numeric_label_conflict'] = d.groupby('canonical_key').truth.transform('nunique').gt(1)
    q = pd.read_parquet(OUT / 'all_known_port_query_support.parquet')
    columns = ['row_position', 'same_key_rows', 'same_key_roots', 'old_registered_key', 'nearest_differing_fields',
        'nearest_mismatch_patterns', 'nearest_roots', 'nearest_original_rows']
    d = d.merge(q[columns], on='row_position', how='left', validate='one_to_one')
    assert len(d) == 112807 and d.row_position.nunique() == 112807
    assert d.port_pair_diagnostic_eligible.sum() == report['known_port_query_rows'] == len(q)
    assert d.loc[~d.port_pair_diagnostic_eligible, 'same_key_rows'].isna().all()
    raw_path = ROOT / 'data/official/train.parquet'
    y = pd.read_parquet(raw_path, columns=['label_binary']).label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy(np.int8)
    assert len(y) == 2056871 and np.array_equal(d.truth, y[d.row_position])
    groups = []
    for (cl, proto, src, dst, conflict, eligible), g in d.groupby(['truth', 'protocol', 'body_src_port_observed',
            'body_dst_port_observed', 'whole_numeric_label_conflict', 'port_pair_diagnostic_eligible'], dropna=False):
        groups.append(dict(truth=int(cl), protocol=str(proto), body_src_port_observed=bool(src),
            body_dst_port_observed=bool(dst), whole_numeric_label_conflict=bool(conflict), diagnostic_eligible=bool(eligible),
            rows=len(g), roots=int(g.root.nunique()), V142_errors=int(g.pred_V142.ne(g.truth).sum()),
            A_errors=int(g.pred_A.ne(g.truth).sum()), B_errors=int(g.pred_B.ne(g.truth).sum())))
    d.to_parquet(OUT / 'whole_ASA_support_population.parquet', index=False)
    result = dict(status='all_ASA_unknown_conflict_and_correct_controls_retained', latest_actual_training='V146',
        new_fits=0, new_gradients=0, new_model_forwards=0, new_updates=0, ASA_rows=len(d),
        known_port_diagnostic_rows=len(q), other_parameter_states_rows=len(d)-len(q),
        whole_population_numeric_conflict_rows=int(d.whole_numeric_label_conflict.sum()),
        all_groups=groups, independently_verified_official_truth=True,
        limits=['ICMP has no transport-port requirement; port diagnostic ineligibility does not classify it as malformed.',
                'Whole-population input label conflict is diagnostic; legal TRAIN purity remains defined per role.',
                'Unknown parameter states are retained and never counted as equal concrete port values.',
                'Complete V146 task scoring remains unchanged; nearest-field audit does not remove difficult rows.'],
        source_sha256={str(p.relative_to(ROOT)).replace(chr(92), '/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), OUT/'audit.json', OUT/'all_known_port_query_support.parquet', RUN/'ASA_prediction_ledger.parquet', facts_path, raw_path]})
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status', 'ASA_rows', 'known_port_diagnostic_rows', 'other_parameter_states_rows',
        'whole_population_numeric_conflict_rows']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
