"""Read saved development predictions; never fit, load or change a model.

Print JSON to stdout. Oracle cutoffs use evaluation labels deliberately and are
diagnostic frontiers only, not deployable thresholds or independent test scores.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
USED = {}


def read(relative):
    path = ROOT / relative
    USED[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_parquet(path)


def hard(frame):
    return (frame.label.isin([1, 2]) & frame.action.eq('deny') &
            frame.outcome.eq('blocked') & frame.transport_protocol.isin(['tcp', 'udp']) &
            frame.src_role.eq('outside') & frame.dst_role.eq('dmz'))


def frontier(d):
    # Each source has equal mass WITHIN each class in this protocol/fold.
    # Source symbols are grouping units, not verified independent real devices.
    d = d.reset_index(drop=True).copy()
    size = d.groupby(['label', 'group']).label.transform('size').to_numpy()
    group_count = d.groupby('label').group.nunique()
    w = 1.0 / size / d.label.map(group_count).to_numpy()
    y = d.label.eq(2).to_numpy()
    p = d.hard_score.to_numpy()
    assert np.isfinite(p).all() and y.any() and (~y).any()
    order = np.argsort(-p, kind='stable')
    end = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(d) - 1]
    # Only include an entire equal-score block. An all-M prediction is feasible.
    mr = np.cumsum((~y)[order])[end] / (~y).sum()
    ms = np.cumsum((w * ~y)[order])[end]
    feasible = np.flatnonzero((mr <= .01 + 1e-12) & (ms <= .01 + 1e-12))
    alert = np.zeros(len(d), dtype=bool)
    cutoff = None
    if len(feasible):
        cutoff = float(p[order[end[feasible[-1]]]])
        alert = p >= cutoff
    return {
        'M_rows': int((~y).sum()), 'S_rows': int(y.sum()),
        'M_source_symbols': int(group_count.loc[1]), 'S_source_symbols': int(group_count.loc[2]),
        'source_balanced_ROC_AUC': float(roc_auc_score(y, p, sample_weight=w)),
        'source_balanced_standardized_partial_AUC_at_1pct': float(roc_auc_score(y, p, sample_weight=w, max_fpr=.01)),
        'oracle_M_row_error': float(alert[~y].mean()),
        'oracle_M_source_mean_error': float(np.sum(w * (~y) * alert)),
        'oracle_S_row_recall': float(alert[y].mean()),
        'oracle_S_source_mean_recall': float(np.sum(w * y * alert)),
        'oracle_target_repairs': int((alert & d.target_578.to_numpy()).sum()),
        'diagnostic_only_cutoff_NOT_FOR_DEPLOYMENT': cutoff,
    }


def main():
    records = read('artifacts/v61_source_factorial_20260914_r2/records.parquet')
    manifest = read('artifacts/v67_targeted_20260920/manifest.parquet')
    ledger = read('artifacts/v67_delivery_tables_20260920/target_578_ledger.parquet')
    fields = records[['row_position', 'action', 'outcome', 'transport_protocol', 'src_role', 'dst_role']]
    full = manifest.merge(fields, on='row_position', validate='one_to_one')
    auxiliary = records[records.role.ne('fit') & hard(records) & records.label.eq(2)]
    groups = auxiliary.groupby('group').size().sort_values(ascending=False)
    dominant = int(groups.index[0])
    result = {
        'scope': 'Existing adaptive development results only; no fit, no new predictions, no model promotion.',
        'new_fits': 0,
        'oracle_warning': 'Uses outer answers to diagnose each fixed score function. Not independent validation; not a bound for other models or non-threshold rules.',
        'grouping_warning': 'Source symbols are not established real entities; no independence or external-domain guarantee.',
        'target_units': {'rows': len(ledger), 'source_symbols': int(ledger.group.nunique()),
                         'raw_hashes': int(ledger.raw_sha256.nunique()), 'normalized_texts': int(ledger.normalized_text.nunique())},
        'target_buckets': [],
        'auxiliary_hard_S': {'rows': len(auxiliary), 'source_symbols': int(auxiliary.group.nunique()),
                             'dominant_group': dominant, 'dominant_rows': int(groups.iloc[0]),
                             'dominant_fraction': float(groups.iloc[0] / len(auxiliary)),
                             'all_hard_S_rows_in_prepared_pool': int((hard(records) & records.label.eq(2)).sum())},
        'inner_training_priors': [], 'fold_protocol_frontiers': [], 'coarse_changed_S_sources': []}
    for bucket, d in ledger.groupby('evidence_bucket'):
        result['target_buckets'].append({'bucket': bucket, 'rows': len(d), 'sources': int(d.group.nunique()),
                                        'raw_hashes': int(d.raw_sha256.nunique())})
    for fold in range(3):
        im = read(f'artifacts/v67_expansion_20260920/fold{fold}/inner_manifest.parquet')
        result['inner_training_priors'].append({'outer_fold': fold,
            'dominant_group_inner_fold': sorted(im.loc[im.group.eq(dominant), 'inner_fold'].unique().tolist()),
            'inner_training_S_fractions': [float(im.loc[im.inner_fold.ne(k), 'label'].eq(2).mean()) for k in range(3)],
            'refit_training_S_fraction': float(im.label.eq(2).mean())})
    for arm in ['coarse', 'numeric']:
        saved = read(f'artifacts/v67_expansion_20260920/{arm}_oof.parquet')
        assert np.array_equal(saved.row_position, manifest.row_position)
        assert np.array_equal(saved.label, manifest.label)
        d = full.copy()
        d['hard_score'] = saved.hard_score.to_numpy()
        for fold in range(3):
            for protocol in ['tcp', 'udp']:
                cell = d[hard(d) & d.fold.eq(fold) & d.transport_protocol.eq(protocol)]
                result['fold_protocol_frontiers'].append({'arm': arm, 'fold': fold, 'protocol': protocol, **frontier(cell)})
        if arm == 'coarse':
            d['prediction'] = saved.pred.to_numpy()
            for group, s in d[d.label.eq(2)].groupby('group'):
                old = int(s.baseline_pred_with_normal_gate.eq(2).sum())
                new = int(s.prediction.eq(2).sum())
                if old != new:
                    result['coarse_changed_S_sources'].append({'group': int(group), 'rows': len(s),
                                                              'old_correct': old, 'new_correct': new})
            old_recall = d[d.label.eq(2)].groupby('group').baseline_pred_with_normal_gate.apply(lambda x: x.eq(2).mean()).mean()
            new_recall = d[d.label.eq(2)].groupby('group').prediction.apply(lambda x: x.eq(2).mean()).mean()
            result['coarse_S_source_recall'] = {'before': float(old_recall), 'after': float(new_recall),
                                               'gain_percentage_points': float((new_recall-old_recall)*100)}
    pure = records.groupby('group').label.nunique()
    result['prepared_pool_source_purity'] = {'groups': len(pure), 'single_label_groups': int(pure.eq(1).sum()),
                                           'mixed_label_groups': int(pure.gt(1).sum())}
    result['input_sha256'] = USED
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
