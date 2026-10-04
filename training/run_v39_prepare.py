"""Freeze v39 body groups, development folds and semantic projections."""
import argparse
import collections
import datetime
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import v39_core as core


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def assign_folds(frame, seed=3901):
    """Whole body groups, balancing route/class totals; no model view involved."""
    strata, names = pd.factorize(pd.MultiIndex.from_frame(frame[['route', 'label_index']]), sort=True)
    g, back = np.unique(frame.body_group.to_numpy(), return_inverse=True)
    n = len(names)
    count = np.bincount(back * n + strata, minlength=len(g) * n).reshape(len(g), n)
    total = np.maximum(count.sum(axis=0), 1)
    normalized = count / total
    rng = np.random.default_rng(seed)
    tie = rng.random(len(g))
    order = np.lexsort((tie, -count.sum(axis=1), -normalized.max(axis=1)))
    balance = np.zeros((3, n))
    fold = np.full(len(g), -1, dtype=np.int8)
    for j in order:
        delta = (2 * balance * normalized[j] + normalized[j] ** 2).sum(axis=1)
        choices = np.flatnonzero(np.isclose(delta, delta.min(), atol=1e-16, rtol=0))
        k = int(choices[int(tie[j] * 1000000000) % len(choices)])
        fold[j] = k
        balance[k] += normalized[j]
    return fold[back]


def main(a):
    start = time.perf_counter()
    root, out = Path(a.root), Path(a.output)
    if out.exists():
        raise FileExistsError('New prepared directory required')
    old = root / 'artifacts/v38_local_r1_20260913'
    legacy = root / 'artifacts/v37_prepared_r13_20260913'
    receipt = json.loads((old / 'prepared/complete.json').read_text(encoding='utf-8'))
    assert sha(root / 'data/official/train.parquet') == receipt['official_sha256']
    for name in ('rows.parquet', 'projections.parquet'):
        assert sha(old / 'prepared' / name) == receipt['files'][name]
    assert sha(legacy / 'prepared.parquet') == receipt['previous_prepared_sha256']
    out.mkdir(parents=True)
    shutil.copyfile(root / 'docs/V39_NEXT_PLAN.md', out / 'approved_plan.md')
    configuration = {'version': core.VERSION, 'C': 0.1, 'folds': 3, 'seed': 3901,
        'models': ['BASELINE_C', 'SEMANTIC', 'AVAILABILITY'], 'weights': 'original row multiplicity, no class or duplicate weighting',
        'nominal_allowed_rows': 1378650, 'fresh_blind_test': False,
        'old_outer_evaluation_used_to_fit': False,
        'primary_fits': 9, 'optional_pair_fits': 3,
        'pair_gate': {'min_distinct_bodies_per_class_per_fit_cell': 3, 'excluded': 'all fit-side conflicting actual input keys',
                     'target_fields': ['credential_check', 'response', 'outcome', 'icmp_code'],
                     'max_per_body': 4, 'strength': 0.1,
                     'requirement': 'a documented non-identity fact varies; same family/mask and other facts; distinct full body groups'},
        'probability_tolerance': 1e-10, 'regression_recall_drop': 0.02, 'regression_fpr_increase': 0.001,
        'meaningful_slice_min_groups_per_class': 3,
        'same_protocol_baseline_required': True,
        'old_template_stress_retained': True, 'source_holdout_conditional_on_development': True}
    save(out / 'configuration.json', configuration)
    rows = pq.read_table(old / 'prepared/rows.parquet').to_pandas()
    original_pids = rows.projection_id.to_numpy(copy=True)
    base = pq.read_table(old / 'prepared/projections.parquet', columns=['text', 'C_BOTH']).to_pandas()
    base_text = base.text.tolist(); base_facts = [json.loads(v) for v in base.C_BOTH]
    del base
    routes = rows.route.to_numpy()
    labels = rows.label_index.to_numpy()
    group_map, projection_map, records = {}, {}, []
    body_ids = np.empty(len(rows), dtype=np.int32)
    model_ids = np.empty(len(rows), dtype=np.int32)
    methods = collections.Counter(); rule_rows = collections.Counter()
    old_hashes = []
    provenance, cases = [], []
    sampled = collections.Counter()
    original_batches = pq.ParquetFile(root / 'data/official/train.parquet').iter_batches(batch_size=4096,
        columns=['event_id', 'label_binary', 'message_sanitized'], use_threads=False)
    previous_batches = pq.ParquetFile(legacy / 'prepared.parquet').iter_batches(batch_size=4096,
        columns=['raw_hash', 'evidence'], use_threads=False)
    position = 0
    for batch, oldbatch in zip(original_batches, previous_batches):
        originals = batch.to_pylist(); oldrecords = oldbatch.to_pylist()
        assert len(originals) == len(oldrecords)
        for original, previous in zip(originals, oldrecords):
            raw = original['message_sanitized'] or ''
            assert original['event_id'] == rows.event_id.iloc[position]
            assert original['label_binary'] == ['benign', 'malicious', 'suspicious'][labels[position]]
            raw_hash = core.digest(raw)
            assert raw_hash == previous['raw_hash']
            route = routes[position]
            evidence = json.loads(previous['evidence'])
            body, method = core.body_identity(raw, route, evidence, position)
            if body not in group_map:
                group_map[body] = len(group_map)
            body_ids[position] = group_map[body]
            methods[method] += 1
            auth, spans = core.auth_values(raw) if route == 'authentication' else ({}, [])
            key = (int(original_pids[position]), core.canonical(auth))
            if key not in projection_map:
                old_id = key[0]
                semantic = core.semantic(base_text[old_id], base_facts[old_id], route, auth)
                index = len(records); projection_map[key] = index
                records.append({'projection_id': index, 'previous_projection_id': old_id,
                    'baseline_text': base_text[old_id], 'baseline_facts': core.canonical(base_facts[old_id]),
                    'text': semantic['text'], 'facts': core.canonical(semantic['facts']),
                    'audit': core.canonical(semantic['audit']), 'route_audit_only': route})
            index = projection_map[key]; model_ids[position] = index
            audit = json.loads(records[index]['audit'])
            rule_rows.update(audit['rules'])
            if audit['rules']:
                provenance.append({'row_position': position, 'projection_id': index,
                    'rules': core.canonical(audit['rules']), 'source_spans': core.canonical(spans or evidence)})
            sample_key = (route, int(labels[position]))
            if rows['product'].iloc[position] in ('Duo', 'Barracuda WAF') or sampled[sample_key] < 12:
                cases.append({'row_position': position, 'projection_id': index, 'raw': raw, 'route': route,
                              'body_group': int(body_ids[position])})
                sampled[sample_key] += 1
            old_hashes.append(raw_hash)
            position += 1
        if position % 131072 == 0:
            print(json.dumps({'stage': 'native_body_and_semantics', 'rows': position, 'seconds': round(time.perf_counter() - start)}), flush=True)
    assert position == len(rows) == 2056871
    rows['body_group'] = body_ids
    rows['projection_id'] = model_ids
    rows['previous_projection_id'] = original_pids
    nominal = rows.inner_role.to_numpy() >= 0
    assert int(nominal.sum()) == configuration['nominal_allowed_rows']
    # If a new body crosses the inherited excluded region, quarantine its
    # allowed copies; never quietly fit an inherited evaluation body.
    forbidden_groups = np.unique(body_ids[~nominal])
    overlap = nominal & np.isin(body_ids, forbidden_groups)
    eligible = nominal & ~overlap
    fold = np.full(len(rows), -1, dtype=np.int8)
    print(json.dumps({'stage': 'assigning_folds', 'eligible': int(eligible.sum()), 'quarantined': int(overlap.sum())}), flush=True)
    fold[eligible] = assign_folds(rows.loc[eligible])
    rows['fold'] = fold; rows['outer_body_quarantined'] = overlap
    assert rows.loc[eligible].groupby('body_group').fold.nunique().max() == 1
    # Nonempty exact raw duplicates must not cross folds.
    nonempty = ~rows.original_empty.to_numpy()
    key_table = pd.DataFrame({'raw': old_hashes, 'fold': fold})
    duplicate_cross = key_table[eligible & nonempty].groupby('raw').fold.nunique()
    assert not (duplicate_cross > 1).any()
    del key_table, old_hashes; gc.collect()
    for k in range(3):
        assert set(labels[fold == k]) == {0, 1, 2}
    special = rows[(rows.union_group == 2054164) & (rows.inner_role == 2)]
    assert len(special) == 865 and special.body_group.nunique() == 2
    waf = rows.loc[rows.row_position.isin([1477712, 1477722])]
    assert len(waf) == 2 and waf.body_group.nunique() == 1
    supports = rows[eligible].groupby(['route', 'label_index']).agg(rows=('body_group', 'size'), body_keys=('body_group', 'nunique')).reset_index()
    folds = rows[eligible].groupby(['fold', 'label_index']).size().reset_index(name='rows')
    old_to_new = rows[eligible].groupby('union_group').body_group.nunique()
    new_to_old = rows[eligible].groupby('body_group').union_group.nunique()
    # Near-copy upper bound: model projection collisions are flagged but never
    # silently merged into duplicate groups or declared independent events.
    near = rows[eligible].groupby('projection_id').agg(rows=('body_group', 'size'), body_keys=('body_group', 'nunique'), classes=('label_index', 'nunique')).reset_index()
    near = near[near.body_keys > 1]
    pq.write_table(pa.Table.from_pandas(rows, preserve_index=False), out / 'rows.parquet', compression='zstd')
    pq.write_table(pa.Table.from_pylist(records), out / 'projections.parquet', compression='zstd')
    pq.write_table(pa.Table.from_pylist(provenance), out / 'provenance.parquet', compression='zstd')
    pq.write_table(pa.Table.from_pandas(near, preserve_index=False), out / 'near_copy_candidates.parquet', compression='zstd')
    save(out / 'audit_cases.json', cases)
    summary = {'rows_original_verified': position, 'nominal_allowed': int(nominal.sum()), 'eligible': int(eligible.sum()),
        'quarantined_prior_outer_body_copies': int(overlap.sum()), 'body_keys': int(rows.loc[eligible].body_group.nunique()),
        'body_methods_all_rows': dict(methods), 'semantic_rule_rows_all': dict(rule_rows),
        'route_support': supports.to_dict('records'), 'fold_class_rows': folds.to_dict('records'),
        'old_components_split_into_multiple_bodies': int((old_to_new > 1).sum()),
        'body_keys_spanning_multiple_old_components': int((new_to_old > 1).sum()),
        'special_ASA_865_body_keys': 2, 'WAF_two_rows_body_keys': 1,
        'exact_nonempty_raw_fold_overlap': 0, 'body_fold_overlap': 0,
        'near_copy_candidates': len(near), 'near_copy_isolation_complete': False,
        'limitations': ['Body keys are not verified incidents.', 'Unrecognized clocks and native event identities remain literal; near-copy candidates remain explicit.',
                        'Projections shared across body groups are association diagnostics, not automatic event linkage.',
                        'All folds are development; prior reviewed targets remain reviewed.']}
    save(out / 'audit.json', summary)
    save(out / 'complete.json', {'version': core.VERSION, 'official_sha256': receipt['official_sha256'],
        'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seconds': time.perf_counter() - start,
        'new_model_trained': False})
    print(json.dumps({'stage': 'prepared', 'seconds': round(time.perf_counter() - start), 'summary': {k: summary[k] for k in ['eligible', 'body_keys', 'quarantined_prior_outer_body_copies', 'semantic_rule_rows_all']}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    main(parser.parse_args())
