"""Fit-only auxiliary eligibility; retrospective development coverage, no model fit."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from v61_common import FIELDS, fit_vocab, encode_facts, read, save, sha

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR = FIELDS[:5]


def auxiliary_pool(fit, min_sources=3):
    """Only fit labels may determine eligibility; repeated copies add no support."""
    if not fit.role.eq('fit').all():
        raise ValueError('Auxiliary eligibility requires fit rows only')
    fit = fit.copy()
    mixed = fit.groupby('single').label.nunique().gt(1)
    fit['fit_mixed_single'] = fit.single.map(mixed)
    pure = fit[~fit.fit_mixed_single & fit.label.isin([1, 2])]
    support = pure.groupby(['behavior', 'label']).group.nunique().unstack(fill_value=0)
    support = support.reindex(columns=[1, 2], fill_value=0)
    enabled = set(support.index[support.min(axis=1).ge(min_sources)])
    fit['behavior_enabled'] = fit.behavior.isin(enabled)
    # A same-input copy from another source is not a diverse positive example.
    pools = {}
    for key, part in pure.groupby(['behavior', 'label']):
        by_source = part.groupby('group').single.agg(lambda x: set(x))
        singleton_sources = {}
        for source, views in by_source.items():
            if len(views) == 1:
                singleton_sources.setdefault(next(iter(views)), set()).add(source)
        pools[key] = (set(by_source.index), singleton_sources)
    pos, neg = [], []
    for row in fit.itertuples():
        if row.label not in (1, 2) or row.fit_mixed_single or not row.behavior_enabled:
            pos.append(0); neg.append(0); continue
        sources, singletons = pools[(row.behavior, row.label)]
        positives = sources - singletons.get(row.single, set()) - {row.group}
        negatives = pools[(row.behavior, 3 - row.label)][0] - {row.group}
        pos.append(len(positives)); neg.append(len(negatives))
    fit['diverse_positive_sources'] = pos
    fit['negative_sources'] = neg
    fit['aux_eligible'] = (~fit.fit_mixed_single & fit.behavior_enabled &
                           fit.diverse_positive_sources.ge(2) & fit.negative_sources.ge(2))
    return fit, enabled


def conflict_summary(part, column):
    table = part.groupby([column, 'label']).size().unstack(fill_value=0)
    mixed = table.gt(0).sum(axis=1).gt(1)
    return {'mixed_views': int(mixed.sum()),
            'rows_at_mixed_views': int(table[mixed].to_numpy().sum()),
            'retrospective_minimum_errors': int((table.sum(axis=1) - table.max(axis=1)).sum())}


def summarize(frame, keys):
    return frame.groupby(keys, dropna=False).agg(
        rows=('label', 'size'), sources=('group', 'nunique'), views=('single', 'nunique')
    ).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError('Use a new output directory; do not overwrite prior audit evidence')
    data = ROOT / 'artifacts/v61_source_factorial_20260914_r2'
    old = ROOT / 'artifacts/v62_capacity_20260915'
    inputs = [data/'records.parquet', data/'configuration.json', old/'cache.npz',
              old/'rows.parquet', old/'cache_receipt.json', old/'cached_view_keys.parquet',
              old/'meanmax/fit.parquet', old/'meanmax/selection.parquet',
              ROOT/'training/v61_common.py']
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    conf = read(data/'configuration.json')
    assert sha(data/'records.parquet') == conf['data_bindings']['records.parquet']
    assert sha(ROOT/'training/v61_common.py') == conf['source_bindings']['training/v61_common.py']
    for name, digest in read(old/'cache_receipt.json')['bindings'].items():
        assert sha(old/name) == digest, name
    # Predicate reads avoid decoding old outer rows/labels for this audit.
    frame = pd.read_parquet(data/'records.parquet', filters=[('role', 'in', ['fit', 'selection'])])
    frame = frame.sort_values('row_position').reset_index(drop=True)
    assert frame.row_position.is_unique and set(frame.role) == {'fit', 'selection'}
    assert frame.groupby('group').role.nunique().max() == 1
    assert frame.groupby('body_group').role.nunique().max() == 1
    facts = frame[FIELDS].astype(str).to_numpy()
    encoded = encode_facts(facts, fit_vocab(facts[frame.role.eq('fit')]))
    view = pd.DataFrame(encoded, columns=FIELDS); view.insert(0, 'text', frame.text)
    frame['single'] = pd.factorize(pd.MultiIndex.from_frame(view), sort=False)[0]
    frame['unknown_fact'] = (encoded == 1).any(axis=1)
    frame['unknown_fields'] = ['|'.join(FIELDS[j] for j in np.flatnonzero(row == 1)) for row in encoded]
    frame['behavior'] = frame[BEHAVIOR].astype(str).agg('|'.join, axis=1)
    cache_rows = pd.read_parquet(old/'rows.parquet')
    keys = pd.read_parquet(old/'cached_view_keys.parquet')
    assert cache_rows[['row_position', 'group', 'role', 'label']].equals(
        keys[['row_position', 'group', 'role', 'label']])
    with np.load(old/'cache.npz') as cache:
        neighbors = cache['neighbors']
    assert len(neighbors) == len(cache_rows)
    valid = neighbors >= 0
    assert ((cache_rows.role.to_numpy()[neighbors.clip(min=0)] ==
             cache_rows.role.to_numpy()[:, None]) | ~valid).all()
    keys['neighbor_count'] = valid.sum(axis=1)
    frame = frame.merge(keys, on=['row_position', 'role', 'group', 'label'], validate='one_to_one')
    assert len(frame) == len(cache_rows)
    fit, enabled = auxiliary_pool(frame[frame.role.eq('fit')])
    pool = fit[fit.aux_eligible].drop_duplicates(['behavior', 'label', 'group', 'single'])
    # Pool IDs and decisions exist before selection-label or model-error diagnostics.
    pool_columns = ['row_position', 'group', 'label', 'behavior', 'single',
                    'diverse_positive_sources', 'negative_sources']
    pool = pool[pool_columns].sort_values('row_position')
    diagnostics = []
    fit_labels = fit.groupby('single').label.agg(lambda x: set(x)).to_dict()
    for role, part in [('fit', fit), ('selection', frame[frame.role.eq('selection')])]:
        part = part.copy()
        pred = pd.read_parquet(old/'meanmax'/(role+'.parquet')).sort_values('row_position')
        expected = part.sort_values('row_position')
        assert pred[['row_position', 'label', 'group']].reset_index(drop=True).equals(
            expected[['row_position', 'label', 'group']].reset_index(drop=True))
        prob = pred[['p_B', 'p_M', 'p_S']].to_numpy()
        assert np.isfinite(prob).all() and (prob >= 0).all() and np.allclose(prob.sum(1), 1, atol=2e-6)
        pred['error'] = prob.argmax(axis=1) != pred.label.to_numpy()
        part = part.merge(pred[['row_position', 'error']], on='row_position', validate='one_to_one')
        part['behavior_enabled'] = part.behavior.isin(enabled)
        part['fit_support'] = [
            'unseen_single' if s not in fit_labels else
            'fit_conflicting' if len(fit_labels[s]) > 1 else
            'fit_same_label' if y in fit_labels[s] else 'fit_opposite_label'
            for s, y in zip(part.single, part.label)]
        part['empty_context'] = part.neighbor_count.eq(0)
        diagnostics.append(part)
    both = pd.concat(diagnostics, ignore_index=True)
    behavior = summarize(fit, ['behavior', 'label'])
    for name, sub in [('pure', fit[~fit.fit_mixed_single]), ('eligible', fit[fit.aux_eligible])]:
        x = summarize(sub, ['behavior', 'label']).rename(columns={k: name+'_'+k for k in ['rows', 'sources', 'views']})
        behavior = behavior.merge(x, on=['behavior', 'label'], how='left').fillna(0)
    for col in behavior.columns[2:]:
        behavior[col] = behavior[col].astype(int)
    errors = summarize(both[both.error], ['role', 'label', 'behavior', 'behavior_enabled', 'fit_support', 'empty_context'])
    support = summarize(both, ['role', 'label', 'fit_support'])
    stages = {}
    for name, sub in [('raw_fit', fit), ('pure_fit', fit[~fit.fit_mixed_single]),
                      ('eligible_fit', fit[fit.aux_eligible]), ('unique_auxiliary_pool', pool)]:
        stages[name] = summarize(sub, ['label']).to_dict('records')
    summary = {
        'status': 'support_audit_completed_no_training', 'quality_acceptance': False,
        'training_decisions_use_fit_labels_only': True,
        'old_outer_rows_or_labels_decoded': False,
        'enabled_behaviors': sorted(enabled), 'stage_support': stages,
        'view_conflicts': {role: {name: conflict_summary(sub, column) for name, column in
                                [('single_event', 'single'), ('frozen_D_cached_view', 'key')]}
                           for role, sub in both.groupby('role')},
        'unknown_fact_rows': {role: int(sub.unknown_fact.sum()) for role, sub in both.groupby('role')},
        'selection_unknown_fields': both[both.role.eq('selection')].groupby(['label', 'unknown_fields']).agg(
            rows=('label', 'size'), errors=('error', 'sum'), sources=('group', 'nunique')).reset_index().to_dict('records'),
        'selection_S_error_support': summarize(both[both.role.eq('selection') & both.label.eq(2) & both.error],
                                                ['behavior_enabled', 'fit_support']).to_dict('records'),
        'selection_S_errors': int((both.role.eq('selection') & both.label.eq(2) & both.error).sum()),
        'limitations': [
            'Exact source symbols are isolation proxies, not verified actors or physical domains.',
            'Coarse behavior support and nonconflicting inputs do not prove a learnable causal M/S boundary.',
            'Selection labels diagnose historical errors only; they do not filter auxiliary samples.',
            'Conflict floors are retrospective row-error minima within a role, not achievable new-source performance.',
            'D keys describe the frozen v62 cached representation, not arbitrary contextual information.',
            'Unique auxiliary pool does not delete or downweight main-task training rows.',
            'All inspected selection data remain adaptive development data.'
        ],
        'input_sha256': bindings, 'audit_source_sha256': sha(__file__)
    }
    # Verify immutable inputs even after diagnostics.
    assert all(sha(ROOT/name) == digest for name, digest in bindings.items())
    a.out.mkdir(parents=True)
    behavior.to_csv(a.out/'behavior_support.csv', index=False)
    errors.to_csv(a.out/'historical_error_support.csv', index=False)
    support.to_csv(a.out/'single_view_fit_support.csv', index=False)
    pool.to_parquet(a.out/'auxiliary_pool.parquet', index=False)
    both[['row_position', 'role', 'label', 'group', 'behavior', 'single', 'key', 'error',
          'fit_support', 'empty_context', 'behavior_enabled']].to_parquet(a.out/'development_diagnostics.parquet', index=False)
    summary['output_sha256'] = {p.name: sha(p) for p in a.out.iterdir()}
    save(a.out/'support_audit.json', summary)
    print(__import__('json').dumps({k: v for k, v in summary.items() if k not in
                                  ['input_sha256', 'output_sha256', 'limitations']}, indent=2))


if __name__ == '__main__':
    main()
