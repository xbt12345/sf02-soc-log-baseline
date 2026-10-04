"""Recount final OOF predictions and apply pre-declared development gates."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import v39_core as core
from run_v39_prepare import sha, save
from run_v39_train import metric


def main(a):
    root = Path(a.root); out = Path(a.output)
    if out.exists():
        raise FileExistsError('Preserve review')
    out.mkdir(parents=True)
    first = root / 'artifacts/v39_local_r1_20260913'
    final = root / 'artifacts/v39_local_r2_20260913'
    rows = pq.read_table(final / 'prepared/rows.parquet').to_pandas()
    rows = rows[rows.fold >= 0].reset_index(drop=True)
    positions = rows.row_position.to_numpy(); y = rows.label_index.to_numpy()
    record = {}; predictions = {}; reports = {}; model_bindings = []
    pfile = pq.read_table(final / 'prepared/projections.parquet', columns=['facts']).to_pandas()
    facts = [json.loads(z) for z in pfile.facts]
    icmp = np.array([f.get('transport_protocol') == 'icmp' for f in facts])[rows.projection_id.to_numpy()]
    route = rows.route.to_numpy(); source = rows['product'].to_numpy()
    slices = {str(r): route == r for r in sorted(set(route))}
    slices.update({'ASA_ICMP': (route == 'asa') & icmp,
                   'ASA_non_ICMP': (route == 'asa') & ~icmp,
                   'ASA_code13_865': rows.union_group.to_numpy() == 2054164,
                   'WAF_suspicious_unique_body': np.isin(positions, [1477712, 1477722])})
    for view in ('BASELINE_C', 'SEMANTIC', 'AVAILABILITY'):
        run = final if view == 'SEMANTIC' else first
        probabilities = np.empty((len(rows), 3)); covered = np.zeros(len(rows), dtype=np.uint8)
        per_fold = []
        for fold in range(3):
            directory = run / 'primary' / ('fold_' + str(fold)) / view
            receipt = json.loads((directory / 'complete.json').read_text(encoding='utf-8'))
            assert sha(directory / 'model.joblib') == receipt['model_sha256']
            assert sha(directory / 'evaluation.parquet') == receipt['predictions_sha256']
            d = pq.read_table(directory / 'evaluation.parquet').to_pandas()
            ix = np.searchsorted(positions, d.row_position.to_numpy())
            assert np.array_equal(positions[ix], d.row_position.to_numpy())
            assert np.array_equal(y[ix], d.label_index.to_numpy())
            assert np.all(rows.fold.to_numpy()[ix] == fold)
            p = d[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy()
            assert np.array_equal(np.asarray(['benign', 'malicious', 'suspicious'])[p.argmax(axis=1)], d.pred_label.to_numpy())
            probabilities[ix] = p; covered[ix] += 1
            report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            reports[view, fold] = report
            recomputed = metric(y[ix], p)
            assert recomputed['confusion_matrix'] == report['evaluation']['confusion_matrix']
            per_fold.append(recomputed)
            model_bindings.append({'view': view, 'fold': fold, 'path': directory.relative_to(root).as_posix(), **receipt})
        assert (covered == 1).all()
        predictions[view] = probabilities.argmax(axis=1)
        record[view] = {'pooled': metric(y, probabilities), 'folds': per_fold,
            'slices': {k: metric(y[m], probabilities[m]) for k, m in slices.items() if m.any()},
            'sources': {str(k): metric(y[source == k], probabilities[source == k]) for k in sorted(set(source))}}
        del probabilities
    baseline = predictions['BASELINE_C']; semantic = predictions['SEMANTIC']
    delta = {'fixed': int(((baseline != y) & (semantic == y)).sum()), 'regressed': int(((baseline == y) & (semantic != y)).sum()),
        'changed_wrong_subtype': int(((baseline != y) & (semantic != y) & (baseline != semantic)).sum()),
        'per_route': {str(k): {'fixed': int(((baseline != y) & (semantic == y) & (route == k)).sum()),
                             'regressed': int(((baseline == y) & (semantic != y) & (route == k)).sum())} for k in sorted(set(route))}}
    eligible = {}
    for fold in range(3):
        def keyed(view):
            return {(z['route'], z['mask'], tuple(z['classes'])): z for z in reports[view, fold]['conditional_comparisons']}
        b, s = keyed('BASELINE_C'), keyed('SEMANTIC')
        for k in set(b) & set(s):
            if not b[k]['supported_across_bodies'] or not s[k]['supported_across_bodies']:
                continue
            eligible.setdefault(str(k), []).append({'fold': fold,
                'baseline_errors': b[k]['three_class_errors'], 'semantic_errors': s[k]['three_class_errors'],
                'baseline_AUC': b[k]['conditional_AUC'], 'semantic_AUC': s[k]['conditional_AUC']})
    supported_gains = []
    for key, z in eligible.items():
        wins = sum(r['semantic_errors'] < r['baseline_errors'] for r in z)
        auc_delta = float(np.mean([r['semantic_AUC'] - r['baseline_AUC'] for r in z]))
        if wins >= 2 and auc_delta >= -1e-12:
            supported_gains.append({'cell': key, 'error_reduction_folds': wins, 'mean_auc_delta': auc_delta, 'folds': z})
    b, s = record['BASELINE_C']['pooled'], record['SEMANTIC']['pooled']
    recall_ok = all(s['class_recall'][c] >= b['class_recall'][c] - .02 for c in [1, 2])
    fpr_ok = s['false_alerts_per_10000_normal'] <= b['false_alerts_per_10000_normal'] + 10
    gain_folds = sum(record['SEMANTIC']['folds'][k]['errors'] < record['BASELINE_C']['folds'][k]['errors'] for k in range(3))
    gate = {'two_or_more_fold_error_reductions': gain_folds >= 2, 'minority_recall_regression_limit': recall_ok,
            'normal_fpr_regression_limit': fpr_ok, 'has_supported_conditional_improvement': bool(supported_gains)}
    pair = []
    for fold in range(3):
        path = final / 'primary' / ('fold_' + str(fold)) / 'SEMANTIC/pair_support.json'
        z = json.loads(path.read_text(encoding='utf-8'))
        pair.append({'fold': fold, 'eligible': z['eligible'], 'cells': len(z['candidate_cells']), 'excluded_conflicted_fit_rows': z['fit_rows_in_conflicted_keys']})
    result = {'scope': 'Reviewed-data cross-validation only; neither external validation nor a final deployed model.',
        'rows': len(rows), 'metrics': record, 'paired_decision_change': delta, 'conditional_gains': supported_gains,
        'development_gate_checks': gate, 'proceed_to_old_protocol_stress': all(gate.values()),
        'auxiliary_pair_support': pair, 'auxiliary_fit_executed': False,
        'actual_primary_fit_count': 12, 'retained_comparison_model_count': 9,
        'superseded_semantic_models': 3, 'reason_for_replacement': 'Failed partially redacted absolute-clock invariance; not score selection',
        'baseline_and_control_reused_after_exact_input_checks': True,
        'quality_accepted': False, 'platform_used': False, 'external_data_used': False,
        'model_bindings': model_bindings}
    save(out / 'primary_review.json', result)
    save(out / 'recount_binding.json', {'script_sha256': sha(Path(__file__)), 'prepared_sha256': sha(final / 'prepared/complete.json')})
    print(json.dumps({k: v for k, v in result.items() if k not in ['metrics', 'conditional_gains', 'model_bindings']}, ensure_ascii=False))
    print(json.dumps({k: v['pooled'] for k, v in record.items()}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    main(p.parse_args())
