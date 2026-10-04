"""Inspect observed sanitized address ranges without asserting true topology."""
import hashlib
import ipaddress
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v153_independent_training_transfer_gap_20261001'
CAPTURES = ROOT / 'artifacts/v153_raw_residual_delta_20261001/all_original_capture_provenance.parquet'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
GAP = OUT / 'all_original_classifier_gap_and_control_ledger.parquet'
TARGET = OUT / 'literal_scope_qualification.json'
NETWORKS = [(name, ipaddress.ip_network(cidr)) for name, cidr in [
    ('RFC1918_10', '10.0.0.0/8'), ('RFC1918_172', '172.16.0.0/12'),
    ('RFC1918_192', '192.168.0.0/16'), ('shared_CGN', '100.64.0.0/10')]]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not TARGET.exists(), 'Preserve executed evidence.'
    c = pd.read_parquet(CAPTURES).sort_values('row_position').reset_index(drop=True)
    d = pd.read_parquet(GAP).sort_values('row_position').reset_index(drop=True)
    t = pd.read_parquet(TRACE).sort_values('row_position').reset_index(drop=True)
    for col in ['row_position', 'truth', 'root', 'fold']:
        assert c[col].equals(d[col]) and c[col].equals(t[col])
    assert len(c) == 112807
    assert c.raw_sha256.equals(t.raw_message.map(lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()))
    for side in ['src', 'dst']:
        literal = c[side].str.split(':', n=1).str[1].str.split('/').str[0]
        categories = {}
        for value in literal.unique():
            address = ipaddress.ip_address(value)
            categories[value] = next((name for name, network in NETWORKS if address in network), 'other')
        c[side + '_observed_literal_scope'] = literal.map(categories)
    cols = ['src_observed_literal_scope', 'dst_observed_literal_scope']
    data = c[['row_position', 'root', 'fold', 'truth'] + cols].copy()
    data['known_578_cohort'] = d.known_578_cohort
    data['same_family_outer_fold_correct_control_S'] = d.same_family_and_outer_fold_control_S
    data['outer_B_pred'] = d.outer_B_pred
    population = data.groupby(['truth'] + cols).size().rename('original_rows').reset_index()
    selected = data.loc[data.known_578_cohort | data.same_family_outer_fold_correct_control_S]
    groups = selected.groupby(['known_578_cohort', 'same_family_outer_fold_correct_control_S'] + cols).size().rename(
        'original_rows').reset_index()
    assert selected[cols].drop_duplicates().shape[0] == 1
    assert int(selected.known_578_cohort.sum()) == 578
    assert int(selected.same_family_outer_fold_correct_control_S.sum()) == 51
    data.to_parquet(OUT / 'all_original_observed_literal_scope.parquet', index=False)
    result = dict(status='sanitized_literal_range_candidate_examined_not_new_behavior_teacher',
        original_ASA_rows=len(data), new_model_forwards=0, new_fits=0, new_gradients=0, new_updates=0,
        population=population.to_dict('records'), hard_and_strict_controls=groups.to_dict('records'),
        hard_and_strict_correct_controls_have_same_observed_scope=True,
        original_namespace_or_topology_preservation_known=False,
        candidate_added_to_model=False, new_qualified_classification_mechanism=False,
        limits=['These are literal ranges in already sanitized strings, not established real-network scopes.',
                'No prefixes, exact addresses, root identifiers or HELD labels become a classifier input or teacher.',
                'Matching ranges do not prove all IP relationships are irrelevant or all full-input information is complete.',
                'The diagnosis rules out these two literal categories as a separator of these 578/51 records only.'],
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), CAPTURES, TRACE, GAP]})
    TARGET.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'population', 'hard_and_strict_controls']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
