"""Original-row class-condition audit within legal TRAIN, excluding own root.

No models, prediction rules, gradients, label correction, or loss weights.
Root is an existing leakage-isolation group, not an asserted real-world entity.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v152_independent_leave_root_conditions_20261001'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
IDENTITY = ROOT / 'artifacts/v147_independent_observability_20261001/observable_identity_ledger.parquet'
KNOWN = ROOT / 'artifacts/v147_independent_conditional_support_20261001/all_known_port_conditional_support.parquet'
FIDELITY = ROOT / 'artifacts/v149_independent_input_fidelity_20261001/audit.json'
CONTEXT = ROOT / 'artifacts/v151_independent_context_qualification_20261001/audit.json'
OFFICIAL = ROOT / 'data/official/train.parquet'
FIELDS = ('action', 'outcome', 'transport_protocol', 'src_role', 'dst_role',
          'src_port_fixed', 'dst_port_fixed', 'src_port_range', 'dst_port_range',
          'icmp_type', 'icmp_code', 'icmp_message', 'icmp_unreachable')
SENTINELS = dict(src_port_fixed=65536, dst_port_fixed=65536,
                 icmp_type=256, icmp_code=256)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def condition(facts, omit_source_port=False):
    # Preserve observed/unknown states, applicability through protocol, all
    # values except an explicitly labelled retrieval-only source-port omission.
    entries = []
    for field in FIELDS:
        value = facts.get(field)
        observed = value is not None and value != SENTINELS.get(field, object())
        retained = value if observed else None
        if omit_source_port and field == 'src_port_fixed':
            retained = None
        entries.append((field, observed, retained))
    return json.dumps(entries, ensure_ascii=False, separators=(',', ':'))


def main():
    assert not OUT.exists(), 'Preserve executed evidence; never rerun over output.'
    fidelity = load(FIDELITY)
    context = load(CONTEXT)
    official_sha = sha(OFFICIAL)
    assert official_sha == fidelity['source_sha256']['data/official/train.parquet']
    assert official_sha == context['source_sha256']['data/official/train.parquet']
    data = pd.read_parquet(TRACE).sort_values('row_position').reset_index(drop=True)
    obs = pd.read_parquet(IDENTITY).sort_values('row_position').reset_index(drop=True)
    for name in ['row_position', 'local', 'root', 'fold', 'truth']:
        assert np.array_equal(data[name], obs[name])
    assert len(data) == 112807 and data.row_position.is_unique
    assert data.groupby('root').fold.nunique().eq(1).all()
    facts = data.facts_json.map(json.loads)
    assert set().union(*(set(x) for x in facts)) <= set(FIELDS)
    data['complete_facts_mask_key'] = facts.map(condition)
    data['omit_source_port_key'] = facts.map(lambda f: condition(f, True))
    data['protocol'] = facts.map(lambda f: f.get('transport_protocol', 'unknown'))
    data['both_ports_observed'] = facts.map(lambda f: all(
        0 <= f.get(n, 65536) <= 65535 for n in ['src_port_fixed', 'dst_port_fixed']))
    data['canonical_key'] = obs.canonical_key.to_numpy()
    data['actual_V146_B_prediction'] = obs.pred_B.to_numpy()
    data['actual_V146_B_correct'] = data.actual_V146_B_prediction.eq(data.truth)
    known = pd.read_parquet(KNOWN, columns=['row_position', 'known_578_cohort']).set_index('row_position')
    data['known_578_cohort'] = data.row_position.map(known.known_578_cohort).eq(True)
    assert int(data.known_578_cohort.sum()) == 578
    class_masses = data.groupby(['canonical_key', 'truth']).size().unstack(fill_value=0)
    mixed_keys = class_masses.index[(class_masses > 0).sum(axis=1).gt(1)]
    data['canonical_mixed_label'] = data.canonical_key.isin(mixed_keys)
    assert int(data.canonical_mixed_label.sum()) == 206
    outputs, profiles, cases = [], [], []
    for fold in range(3):
        legal = data.loc[data.fold.ne(fold)].copy()
        assert not set(legal.root) & set(data.loc[data.fold.eq(fold), 'root'])
        for key in ['complete_facts_mask_key', 'omit_source_port_key']:
            q = legal[['row_position', 'local', 'root', 'fold', 'truth', 'canonical_key',
                       'protocol', 'both_ports_observed', 'known_578_cohort',
                       'canonical_mixed_label', 'actual_V146_B_correct', key]].copy()
            q['outer_training_role'] = fold
            q['condition_key'] = q.pop(key)
            q['key_scope'] = key
            global_counts = legal.groupby([key, 'truth']).agg(
                rows=('row_position', 'size'), roots=('root', 'nunique'))
            own_counts = legal.groupby([key, 'truth', 'root']).size()
            for cls in [1, 2]:
                counts = global_counts.xs(cls, level='truth')
                q[f'class{cls}_other_root_rows'] = q.condition_key.map(counts.rows).fillna(0).astype(np.int64)
                q[f'class{cls}_other_root_count'] = q.condition_key.map(counts.roots).fillna(0).astype(np.int64)
                lookup = pd.MultiIndex.from_arrays([q.condition_key, np.full(len(q), cls), q.root],
                                                  names=[key, 'truth', 'root'])
                own = own_counts.reindex(lookup, fill_value=0).to_numpy(np.int64)
                q[f'class{cls}_other_root_rows'] -= own
                q[f'class{cls}_other_root_count'] -= (own > 0).astype(np.int64)
            same = np.where(q.truth.eq(1), q.class1_other_root_rows, q.class2_other_root_rows)
            opposite = np.where(q.truth.eq(1), q.class2_other_root_rows, q.class1_other_root_rows)
            same_roots = np.where(q.truth.eq(1), q.class1_other_root_count, q.class2_other_root_count)
            q['same_class_other_root_rows'] = same
            q['opposite_class_other_root_rows'] = opposite
            q['same_class_other_root_count'] = same_roots
            q['support_state'] = np.select([
                (same == 0) & (opposite == 0), (same == 0) & (opposite > 0),
                (same > 0) & (opposite > 0), (same_roots == 1) & (opposite == 0)],
                ['no_other_root_support', 'only_opposite_class', 'both_classes', 'one_root_same_class_only'],
                default='multiple_roots_same_class_only')
            cols = [c for c in q if c.startswith('class') or c.endswith('_rows') or c.endswith('_count')]
            assert (q[cols].to_numpy() >= 0).all()
            # Independent explicit exclusion recount for every distinct key/root
            # query, not only a favorable sample or the historically hard cohort.
            pairs = q[['condition_key', 'root']].drop_duplicates()
            first_checks = q.drop_duplicates(['condition_key', 'root']).set_index(['condition_key', 'root'])
            member = legal.groupby(key).indices
            for key_value, root in pairs.itertuples(index=False, name=None):
                g = legal.iloc[member[key_value]]
                other = g.loc[g.root.ne(root)]
                check = first_checks.loc[(key_value, root)]
                for cls in [1, 2]:
                    actual = other.loc[other.truth.eq(cls)]
                    assert len(actual) == check[f'class{cls}_other_root_rows']
                    assert actual.root.nunique() == check[f'class{cls}_other_root_count']
            profiles.append(dict(outer_training_role=fold, key_scope=key,
                                 original_role_rows=len(q), distinct_key_root_checks=len(pairs)))
            outputs.append(q)
            # Fixed descriptive cases from legal TRAIN only: large complete-fact
            # cross-root mixed groups. Quotes diagnose missing conditions, not
            # a threat-rule teacher, lookup classifier, or HELD error selection.
            if key == 'complete_facts_mask_key':
                groups = legal.groupby(key).agg(rows=('row_position', 'size'),
                                                classes=('truth', 'nunique'), roots=('root', 'nunique'))
                for value in groups.loc[groups.classes.eq(2) & groups.roots.ge(2)].sort_values(
                        'rows', ascending=False, kind='stable').head(3).index:
                    group = legal.loc[legal[key].eq(value)].sort_values('row_position')
                    m = group.loc[group.truth.eq(1)].iloc[0]
                    s_other = group.loc[group.truth.eq(2) & group.root.ne(m.root)]
                    if s_other.empty:
                        s = group.loc[group.truth.eq(2)].iloc[0]
                        m = group.loc[group.truth.eq(1) & group.root.ne(s.root)].iloc[0]
                    else:
                        s = s_other.iloc[0]
                    assert m.root != s.root and m.fold != fold and s.fold != fold
                    cases.append(dict(outer_training_role=fold, condition_key=value,
                        group_original_rows=len(group), group_roots=int(group.root.nunique()),
                        examples=[{name: row[name].item() if isinstance(row[name], np.generic) else row[name]
                                   for name in ['row_position', 'root', 'fold', 'truth', 'facts_json', 'raw_message']}
                                  for row in [m, s]]))
    allq = pd.concat(outputs, ignore_index=True)
    assert len(allq) == 112807 * 2 * 2
    assert allq.groupby(['key_scope', 'row_position']).size().eq(2).all()
    assert set(allq.row_position) == set(data.row_position)
    summary = allq.groupby(['outer_training_role', 'key_scope', 'truth', 'protocol',
                           'both_ports_observed', 'canonical_mixed_label', 'known_578_cohort',
                           'actual_V146_B_correct', 'support_state'], dropna=False).agg(
        original_role_rows=('row_position', 'size'), query_roots=('root', 'nunique')).reset_index()
    OUT.mkdir()
    allq.to_parquet(OUT / 'all_legal_train_leave_root_support.parquet', index=False)
    summary.to_parquet(OUT / 'original_role_support_profiles.parquet', index=False)
    (OUT / 'legal_train_original_counterexamples.json').write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    broad = allq.groupby(['key_scope', 'truth', 'support_state']).size().rename('original_role_rows').reset_index()
    hard = allq.loc[allq.known_578_cohort].groupby(['key_scope', 'support_state']).size().rename(
        'original_role_rows').reset_index()
    result = dict(status='legal_train_leave_own_root_conditions_executed_not_classifier_gain',
        latest_actual_classifier='V146', original_ASA_rows=len(data), legal_TRAIN_role_rows=225614,
        two_key_query_rows=len(allq), new_model_forwards=0, new_gradients=0, new_fits=0, new_updates=0,
        source_file_identity_verified=True, all_distinct_key_root_exclusions_recounted=True,
        legal_TRAIN_no_own_root_support=True, all_unknown_ICMP_mixed_correct_rows_retained=True,
        classifier_predictions_are_historical_outer_diagnostics_only=True,
        held_labels_used_for_training_or_rules=False, cases_selected_only_from_legal_TRAIN=True,
        profiles=profiles, broad_original_role_support=broad.to_dict('records'),
        known_578_legal_training_role_diagnostics=hard.to_dict('records'),
        quality_acceptance=False, issue_solved=False, next_training_registered=False,
        limits=['Each original row appears in two legal outer TRAIN roles; role counts are not new independent data.',
                'No exact-key support does not prove absence of compositional support or universal impossibility.',
                'Mask/value keys retain full parsed semantics but exclude other actual text/metadata inputs.',
                'Source-port omission is a descriptive retrieval comparison, not a label-preserving augmentation.',
                'Cross-root conditional agreement is a sampling diagnostic, not attack truth or a deployment guarantee.',
                'No noisy labels, new labels, confidence scores or threat classification rules are inferred.'],
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in
                       [Path(__file__), TRACE, IDENTITY, KNOWN, FIDELITY, CONTEXT, OFFICIAL]})
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'original_ASA_rows', 'legal_TRAIN_role_rows',
        'broad_original_role_support', 'known_578_legal_training_role_diagnostics']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
