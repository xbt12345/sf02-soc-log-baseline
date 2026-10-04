"""Official-only metadata/collision/duration audit. No fit, relabeling or test labels."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import v331_prepare as old
import v351_safeguards as guard

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['benign', 'malicious', 'suspicious']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''):
            h.update(b)
    return h.hexdigest()


def save(out, name, value):
    (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run(out):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'audit.json').exists():
        raise FileExistsError('Use a new output directory for a new audit')
    train = ROOT/'data/official/train.parquet'
    ready = ROOT/'artifacts/v331_ready_20260912_r2'
    hashes = {str(train.relative_to(ROOT)): sha(train),
              str((ready/'group_manifest.parquet').relative_to(ROOT)): sha(ready/'group_manifest.parquet'),
              str((ready/'prepared_corpus.parquet').relative_to(ROOT)): sha(ready/'prepared_corpus.parquet')}
    assert hashes[str(train.relative_to(ROOT))] == '6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
    assert hashes[str((ready/'group_manifest.parquet').relative_to(ROOT))] == '9ee5642c33e17fe3e848012f9f18823e0bec3308104fc79f3aa465eab3532c90'
    assert hashes[str((ready/'prepared_corpus.parquet').relative_to(ROOT))] == 'fba196159411a30ea31744b3416c71ae58b47a2e6c5a5ff0f7d3130647ca1bce'
    metadata = {}
    train_buckets = None
    for name in ('train.parquet', 'valid_input.parquet'):
        path = ROOT/'data/official'/name
        # Validation is used only for input availability, never label/threshold selection.
        t = pq.read_table(path, columns=['product_name', 'timestamp'])
        vals = t['product_name'].to_pylist()
        buckets = collections.Counter(guard.product_bucket(v) for v in vals)
        times = t['timestamp'].to_numpy()
        metadata[name] = {'rows': len(t), 'product_null_rows': t['product_name'].null_count,
                         'product_empty_string_rows': sum(v == '' for v in vals),
                         'product_missing_bucket_rows': buckets['<missing>'],
                         'product_missing_fraction': buckets['<missing>']/len(t),
                         'product_buckets': dict(sorted(buckets.items())),
                         'timestamp_nonfinite_or_missing_rows': int((~np.isfinite(times)).sum())}
        if name == 'train.parquet':
            train_buckets = set(buckets)
            official_y = pq.read_table(path, columns=['label_binary'])['label_binary'].to_pylist()
            metadata[name]['missing_bucket_labels'] = dict(collections.Counter(y for v, y in zip(vals, official_y) if guard.product_bucket(v) == '<missing>'))
        else:
            metadata[name]['unseen_named_products'] = sorted(set(buckets)-train_buckets)
            hashes[str(path.relative_to(ROOT))] = sha(path)
    print('Official train/validation input metadata checked; no validation answers read', flush=True)
    gm = pq.read_table(ready/'group_manifest.parquet')
    g, y, informative = (gm[c].to_numpy() for c in ('group_id', 'label_index', 'informative'))
    counts = np.bincount(g[informative].astype(np.int64)*3+y[informative], minlength=(int(g.max())+1)*3).reshape(-1, 3)
    gids = np.flatnonzero((counts > 0).sum(1) > 1)
    positions = np.flatnonzero(np.isin(g, gids))
    assert len(gids) == 75 and len(positions) == 4075
    selected = pq.read_table(ready/'prepared_corpus.parquet',
                            columns=['row_position', 'group_id', 'label', 'text', 'repair_route'],
                            filters=[('group_id', 'in', gids.tolist())]).to_pylist()
    by_position = {r['row_position']: r for r in selected}
    duration_matches, duration_changed, duration_examples = 0, [], []
    header_rows, header_changed, header_examples = 0, [], []
    header_variants, header_variant_changes = 0, 0
    patches = []
    offset = 0
    for batch in pq.ParquetFile(train).iter_batches(batch_size=16384,
            columns=['message_sanitized', 'product_name', 'timestamp', 'label_binary'], use_threads=False):
        table = pa.Table.from_batches([batch])
        local = positions[(positions >= offset) & (positions < offset+len(table))]-offset
        for pos, row in zip(local+offset, table.take(pa.array(local, type=pa.int64())).to_pylist()):
            by_position[int(pos)].update(row)
        # Arrow prefilter avoids turning every full log into a Python object.
        has_duration = pc.fill_null(pc.match_substring_regex(table['message_sanitized'], r'(?i)\b(duration|elapsed)\b'), False).to_numpy()
        has_header = pc.fill_null(pc.match_substring_regex(table['message_sanitized'], r'^\s*<[0-9]{1,3}>1\s'), False).to_numpy()
        di = np.flatnonzero(has_duration | has_header)
        for pos, row in zip(di+offset, table.take(pa.array(di, type=pa.int64())).to_pylist()):
            raw = row['message_sanitized'] or ''
            named_duration = bool(guard.DURATION.search(raw))
            neutral = guard.neutralize_collection_stamp(raw)
            recognized_header = neutral != raw
            if not named_duration and not recognized_header:
                continue
            duration_matches += named_duration
            before = old.prepare_message(raw)['text']
            time_only = old.prepare_message(neutral)['text']
            after = guard.model_text({'message_sanitized': raw})
            if before != after:
                patches.append({'row_position': int(pos), 'text': after, 'old_text_sha256': hashlib.sha256(before.encode()).hexdigest()})
            if recognized_header:
                header_rows += 1
                hm = guard.RFC5424_HEADER.match(raw)
                a, b = hm.span('stamp')
                for stamp in ('-', '2099-12-31T23:59:59Z'):
                    shifted = raw[:a]+stamp+raw[b:]
                    header_variant_changes += guard.neutralize_collection_stamp(shifted) != neutral
                    header_variants += 1
            if before != time_only:
                header_changed.append(int(pos))
                if len(header_examples) < 8:
                    header_examples.append({'row_position': int(pos), 'raw': raw, 'before': before, 'after_time_only': time_only})
            if time_only != after:
                duration_changed.append(int(pos))
                if len(duration_examples) < 12:
                    duration_examples.append({'row_position': int(pos), 'raw': raw, 'before': before, 'after': after})
        offset += len(table)
        if offset % (16384*32) == 0:
            print('Scanned {} original rows'.format(offset), flush=True)
    assert offset == len(g)
    print('All original messages scanned; every conflicting group retrieved', flush=True)
    patch_table = pa.Table.from_pylist(patches)
    pq.write_table(patch_table, out/'candidate_text_patch.parquet', compression='zstd')
    # Rebuild exact new-text equivalence and its union with historical grouping.
    # No class value is used to construct either grouping.
    patch_positions = np.array([r['row_position'] for r in patches], dtype=np.int32)
    prepared_text = pq.read_table(ready/'prepared_corpus.parquet', columns=['text'])
    old_patch_text = prepared_text.take(pa.array(patch_positions))['text'].to_pylist()
    assert all(hashlib.sha256(s.encode()).hexdigest() == r['old_text_sha256'] for s, r in zip(old_patch_text, patches))
    new_values = pa.array(sorted({r['text'] for r in patches}), type=prepared_text['text'].type)
    match = pc.is_in(prepared_text['text'], value_set=new_values).to_numpy().copy()
    match[patch_positions] = False  # A changed row no longer has its old text.
    matching_positions = np.flatnonzero(match)
    matching_texts = prepared_text.take(pa.array(matching_positions))['text'].to_pylist()
    new_text_id = {s: int(g[pos]) for s, pos in zip(matching_texts, matching_positions)}
    next_id = int(g.max())+1
    new_groups = g.copy()
    parent = np.arange(next_id, dtype=np.int32)
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = int(parent[k])
        return k
    anchors = {}
    for row in patches:
        s, pos = row['text'], row['row_position']
        if s not in new_text_id:
            new_text_id[s] = next_id
            next_id += 1
        new_groups[pos] = new_text_id[s]
        anchor = anchors.setdefault(s, int(g[pos]))
        left, right = find(anchor), find(int(g[pos]))
        parent[right] = left
    for s, pos in zip(matching_texts, matching_positions):
        if s in anchors:
            left, right = find(anchors[s]), find(int(g[pos]))
            parent[right] = left
    roots = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    union_groups = roots[g]
    new_counts = np.bincount(new_groups[informative].astype(np.int64)*3+y[informative], minlength=next_id*3).reshape(-1, 3)
    new_mixed = (new_counts > 0).sum(1) > 1
    group_change = {'changed_text_rows': len(patches), 'new_nonempty_mixed_groups': int(new_mixed.sum()),
                    'new_nonempty_mixed_rows': int(new_counts[new_mixed].sum()),
                    'new_text_empirical_min_errors': int((new_counts.sum(1)-new_counts.max(1)).sum()),
                    'historical_group_components_merged': int(len(parent)-len(np.unique(roots))),
                    'labels_and_row_order_unchanged': True,
                    'scope': 'Exact repeat constraints, not guaranteed incident/near-template isolation; no role reassignment yet'}
    pq.write_table(pa.table({'row_position': np.arange(len(g), dtype=np.int32),
                            'candidate_text_group': new_groups, 'old_new_union_group': union_groups,
                            'label_index': y, 'informative': informative}), out/'candidate_group_manifest.parquet', compression='zstd')
    print('Sparse text patch verified and old/new repeat union rebuilt', flush=True)
    members = [by_position[p] for p in sorted(by_position)]
    assert all(LABELS[y[r['row_position']]] == r['label_binary'] == r['label'] for r in members)
    pq.write_table(pa.Table.from_pylist(members), out/'collision_members.parquet', compression='zstd')
    by_group = collections.defaultdict(list)
    for row in members:
        by_group[row['group_id']].append(row)
    summaries, meta_changed, time_changed, variants = [], 0, 0, 0
    for gid, rows in sorted(by_group.items()):
        fingerprints, port_states = set(), collections.Counter()
        per_field = collections.defaultdict(set)
        examples = {}
        for row in rows:
            raw = row['message_sanitized']
            parts = old.asa_parts(raw)
            assert parts is not None and row['repair_route'] == 'asa'
            facts = guard.asa_visible_facts(raw)
            fingerprints.add(guard.canonical(facts))
            for key, val in parts['facts'].items():
                per_field[key].add(val)
            for side in ('src', 'dst'):
                port_states[side+':'+facts[side]['port_state']] += 1
            expected = guard.model_text(row)
            for product, timestamp in [(None, None), ('UNSEEN', 0), ('', 9999999999)]:
                changed = dict(row, product_name=product, timestamp=timestamp, label_binary='not_a_model_input')
                meta_changed += guard.model_text(changed) != expected
                variants += 1
            # Preserve native event information, replace only checked collection prefix.
            original_prefix = raw[:parts['body_start']]
            code = old.ASA_CODE.search(original_prefix)
            native = original_prefix[code.start():] if code else ''
            transformed = '<180>2099-12-31T23:59:59Z collector.example '+native+parts['body']
            time_changed += guard.model_text({'message_sanitized': transformed}) != expected
            examples.setdefault(row['label'], row['row_position'])
        summaries.append({'group_id': gid, 'rows': len(rows), 'class_counts': counts[gid].tolist(),
                          'text': rows[0]['text'], 'visible_fact_variants': len(fingerprints),
                          'raw_field_unique_counts': {k: len(v) for k, v in per_field.items()},
                          'port_state_counts': dict(port_states), 'example_positions_by_class': examples,
                          'empirical_min_errors_for_old_text': int(counts[gid].sum()-counts[gid].max()),
                          'finding': 'same_checked_visible_facts_context_or_label_rule_unresolved' if len(fingerprints)==1 else 'visible_difference_requires_review'})
    protocol = pq.read_table(ready/'protocol_manifest.parquet')
    isolation = {}
    needs_new_roles = []
    for name in protocol.column_names:
        if name == 'row_position':
            continue
        roles = protocol[name].to_numpy()
        guard.assert_group_isolation(g, roles)
        isolation[name] = True
        try:
            guard.assert_group_isolation(union_groups, roles)
        except ValueError:
            needs_new_roles.append(name)
    roles = protocol['domain_0'].to_numpy()
    # Only fit labels enter counts; all-fit feature multiplicities enter weights.
    fit_distributions = guard.fit_group_distributions(g[positions], y[positions], roles[positions])
    fit = roles == 0
    feature_keys = np.where(informative[fit], g[fit].astype(np.int64)+1, 0)
    weights = guard.bounded_repeat_weights(feature_keys)
    no_obs = feature_keys == 0
    weight_audit = {'scope': 'Candidate construction only on domain_0 fit; not used to train or select a model',
                    'rows': int(fit.sum()), 'min': float(weights.min()), 'max': float(weights.max()),
                    'mean': float(weights.mean()), 'no_observation_rows': int(no_obs.sum()),
                    'no_observation_total_weight': float(weights[no_obs].sum()),
                    'all_labels_unchanged': True}
    old_min_errors = int((counts[gids].sum(1)-counts[gids].max(1)).sum())
    result = {'version': guard.VERSION, 'source_hashes': hashes,
              'scope': 'Full official metadata, all training-message duration scan, all 75 collision groups; no classifier fit or validation labels',
              'metadata': metadata,
              'collisions': {'groups': len(gids), 'rows': len(positions), 'fraction_of_training_rows': len(positions)/len(g),
                             'groups_with_same_checked_visible_facts': sum(s['visible_fact_variants']==1 for s in summaries),
                             'old_text_empirical_min_errors': old_min_errors,
                             'old_text_min_error_fraction_all_rows': old_min_errors/len(g),
                             'remaining_groups_not_deleted_or_relabelled': len(gids)},
              'invariance': {'actual_collision_rows': len(members), 'metadata_variants': variants,
                             'metadata_text_changes': int(meta_changed), 'asa_collection_time_variants': len(members),
                             'asa_time_text_changes': int(time_changed),
                             'scope': 'Text equality on these actual rows and variants; not whole-model generalization or all-format time invariance'},
              'duration': {'all_training_rows_scanned': offset, 'rows_with_complete_named_hms_duration': duration_matches,
                           'changed_rows': len(duration_changed), 'changed_positions': duration_changed,
                           'scope': 'Preserve complete named durations; no assertion duration indicates attack or clock/linkage validity'},
              'bounded_syslog_header': {'recognized_rows': header_rows, 'changed_text_rows': len(header_changed),
                                       'counterfactual_variants': header_variants, 'canonical_header_changes': int(header_variant_changes),
                                       'scope': 'Only checked leading RFC5424 timestamp slot, not all embedded time fields'},
              'historical_group_isolation': isolation, 'candidate_weight_audit': weight_audit,
              'candidate_group_change': dict(group_change, historical_tasks_requiring_new_role_allocation=needs_new_roles),
              'model_trained': False, 'transfer_validated': False,
              'unresolved': ['Unknown-format embedded collection values are not exhaustively validated.',
                             'Product/format and class support confounding remains despite explicit metadata exclusion.',
                             'Same visible ASA facts do not establish wrong labels; policy/entity/temporal context is unverified.',
                             'No automatic removal, majority relabeling, pseudo labels, or target-derived threshold.'],
              'elapsed_seconds': time.perf_counter()-start}
    result['source_code_hashes'] = {p.name: sha(p) for p in (Path(__file__), Path(guard.__file__))}
    assert meta_changed == 0 and time_changed == 0 and header_variant_changes == 0
    save(out, 'collision_group_review.json', summaries)
    save(out, 'duration_examples.json', duration_examples)
    save(out, 'header_examples.json', header_examples)
    pq.write_table(pa.table({'row_position': pa.array(header_changed, type=pa.int32())}), out/'header_changed_positions.parquet', compression='zstd')
    save(out, 'fit_only_collision_distributions.json', fit_distributions)
    save(out, 'audit.json', result)
    print(json.dumps({k: result[k] for k in ('collisions', 'invariance', 'duration', 'bounded_syslog_header', 'candidate_weight_audit', 'elapsed_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(ROOT/'evidence/2026-09-12/v351_input_review'))
    run(Path(parser.parse_args().output_dir))
