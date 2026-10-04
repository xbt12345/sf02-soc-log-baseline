"""Postfit source-only OOF accounting; never reads V/inner/C/H labels."""
import hashlib
import json

import numpy as np
import pandas as pd

from run_v75 import ROOT, read, save, sha
from v100_prepare import DEST, V92


def table(truth, pred):
    cm = np.zeros((3, 3), np.int64)
    np.add.at(cm, (truth, pred), 1)
    classes = {}
    for c, name in ((1, 'M'), (2, 'S')):
        tp = int(cm[c, c]); support = int(cm[c].sum())
        called = int(cm[:, c].sum())
        precision = tp/called if called else 0.
        recall = tp/support if support else 0.
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        classes[name] = {'support': support, 'TP': tp, 'FN': support-tp,
            'FP': called-tp, 'precision': precision, 'recall': recall, 'F1': f1}
    return {'rows': int(len(truth)), 'errors': int(len(truth)-np.trace(cm)),
            'confusion_matrix': cm.tolist(), 'M_S_equal_F1':
            (classes['M']['F1']+classes['S']['F1'])/2, 'classes': classes}


def bootstrap_delta(truth, reference, candidate, components, n=1000):
    unique, group = np.unique(components, return_inverse=True)
    pieces = np.zeros((len(unique), 2, 3, 3), np.int64)
    np.add.at(pieces, (group, 0, truth, reference), 1)
    np.add.at(pieces, (group, 1, truth, candidate), 1)
    rng = np.random.default_rng(9901)
    deltas = np.empty(n)
    for i in range(n):
        take = rng.integers(len(unique), size=len(unique))
        cm = pieces[take].sum(0)
        scores = []
        for j in range(2):
            terms = []
            for c in (1, 2):
                tp = cm[j, c, c]
                den = cm[j, c].sum()+cm[j, :, c].sum()
                terms.append(2*tp/den if den else 0.)
            scores.append(sum(terms)/2)
        deltas[i] = scores[1]-scores[0]
    return {'cluster_units': len(unique), 'resamples': n,
        'F1_delta_95pct_interval': np.quantile(deltas, [.025, .975]).tolist(),
        'bootstrap_probability_improvement': float((deltas > 0).mean())}


def main():
    if (DEST/'selection.json').exists():
        raise FileExistsError(DEST/'selection.json')
    exported = read(DEST/'source_export.json')
    assert sha(DEST/'source_rows.parquet') == exported['source_rows_sha256']
    s = pd.read_parquet(DEST/'source_rows.parquet')
    s = s[s.is_ASA].reset_index(drop=True)
    assert len(s) == 31289 and set(s.label_index) == {1, 2}
    ids = np.load(V92/'ASA_input_ids.npy')
    lookup = np.full(int(ids.max())+1, -1, np.int32)
    lookup[ids] = np.arange(len(ids))
    local = lookup[s.fid.to_numpy()]
    assert (local >= 0).all()
    truth = s.label_index.to_numpy()
    fold = s.fold.to_numpy()
    columns = {'row_position': s.row_position.to_numpy(),
        'component': s.component.to_numpy(), 'fold': fold, 'truth_official': truth}
    fit_count = 0
    for k in range(3):
        ix = fold == k
        for view in ('R0', 'N1'):
            folder = DEST/f'fold{k}_{view}'
            tfit = read(folder/'teacher_fit.json')
            assert tfit['converged'] and tfit['heldout_gradient_rows'] == 0
            assert sha(folder/'teacher_ASA_scores.npy') == tfit['scores_sha256']
            teacher_pred = np.load(folder/'teacher_ASA_scores.npy')[local[ix]].argmax(1)
            key = f'{view}_teacher'
            columns.setdefault(key, np.empty(len(s), np.int8))[ix] = teacher_pred
            fit_count += 1
            for arm in ('ERM', 'MAG'):
                report = read(folder/f'{arm}_fit.json')
                assert report['heldout_gradient_rows'] == 0
                assert report['steps_executed'] >= 200
                assert report['teacher_model_sha256'] == tfit['model_sha256']
                fit_count += 1
                for tag in ('200', 'long'):
                    score_path = folder/f'{arm}_{tag}_ASA_scores.npy'
                    assert sha(score_path) == report[f'endpoint_{tag}_scores_sha256']
                    pred = np.load(score_path)[local[ix]].argmax(1)
                    key = f'{view}_{arm}_{tag}'
                    columns.setdefault(key, np.empty(len(s), np.int8))[ix] = pred
    assert fit_count == 18
    oof = pd.DataFrame(columns)
    oof.to_parquet(DEST/'oof_predictions.parquet', index=False)
    teachers = {v: table(truth, oof[f'{v}_teacher'].to_numpy()) for v in ('R0', 'N1')}
    arms = {}
    for view in ('R0', 'N1'):
        teach_pred = oof[f'{view}_teacher'].to_numpy()
        teach = teachers[view]
        for arm in ('ERM', 'MAG'):
            for tag in ('200', 'long'):
                name = f'{view}_{arm}_{tag}'
                pred = oof[name].to_numpy()
                metric = table(truth, pred)
                neg = (teach_pred == truth)&(pred != truth)
                pos = (teach_pred != truth)&(pred == truth)
                fold_metrics = {str(k): table(truth[fold == k], pred[fold == k])
                                for k in range(3)}
                train_negative = sum(read(DEST/f'fold{k}_{view}'/f'{arm}_fit.json')
                                     ['final_protected_negative_flips']
                                     if tag == 'long' else next(z['teacher_correct_negative_flips']
                                       for z in read(DEST/f'fold{k}_{view}'/f'{arm}_progress.json')
                                       if z['step'] == 200) for k in range(3))
                component_frame = pd.DataFrame({'component': s.component, 'truth': truth,
                                                'correct': pred == truth})
                component_summary = component_frame.groupby(['component', 'truth']).agg(
                    rows=('correct', 'size'), correct=('correct', 'sum')).reset_index()
                supported = component_summary[component_summary.rows >= 10]
                metric.update(fold_metrics=fold_metrics,
                    teacher_correct_negative_flips=int(neg.sum()),
                    teacher_wrong_positive_flips=int(pos.sum()),
                    negative_flips_by_class={name: int((neg&(truth == cl)).sum())
                                             for cl, name in ((1, 'M'), (2, 'S'))},
                    final_training_negative_flips_sum=int(train_negative),
                    independent_components=int(s.component.nunique()),
                    components_with_10_or_more_class_rows=int(len(supported)),
                    worst_supported_component_recall=float((supported.correct/supported.rows).min())
                         if len(supported) else None)
                metric['OOF_vs_matched_teacher_F1_delta'] = metric['M_S_equal_F1']-teach['M_S_equal_F1']
                metric['bootstrap_vs_teacher'] = bootstrap_delta(
                    truth, teach_pred, pred, s.component.to_numpy(), n=500)
                if view == 'N1':
                    raw_pred = oof[f'R0_{arm}_{tag}'].to_numpy()
                    raw = table(truth, raw_pred)
                    metric['OOF_vs_matched_raw_F1_delta'] = metric['M_S_equal_F1']-raw['M_S_equal_F1']
                    metric['bootstrap_vs_matched_raw'] = bootstrap_delta(
                        truth, raw_pred, pred, s.component.to_numpy(), n=500)
                    metric['raw_correct_lost'] = int(((raw_pred == truth)&(pred != truth)).sum())
                    metric['raw_wrong_repaired'] = int(((raw_pred != truth)&(pred == truth)).sum())
                # Preserve the old-correct requirement from the v9.7/v9.9
                # plan. A source-selected candidate is not a quality pass.
                class_guard = all(metric['classes'][name][measure] >= teach['classes'][name][measure]-1e-12
                    for name in ('M', 'S') for measure in ('precision', 'recall', 'F1'))
                metric['source_qualification'] = bool(
                    class_guard and metric['teacher_correct_negative_flips'] == 0
                    and metric['final_training_negative_flips_sum'] == 0
                    and metric['OOF_vs_matched_teacher_F1_delta'] > 0
                    and (view == 'R0' or metric['OOF_vs_matched_raw_F1_delta'] > 0))
                arms[name] = metric
    qualified = [name for name, value in arms.items() if value['source_qualification']]
    ranked = sorted(arms, key=lambda name:(-arms[name]['M_S_equal_F1'],
                       0 if name.endswith('_200') else 1, name))
    selected = next((name for name in ranked if name in qualified), None)
    matched_raw = f'R0_{selected.split("_")[1]}_{selected.split("_")[2]}' if selected and selected.startswith('N1_') else None
    if matched_raw and not arms[matched_raw]['source_qualification']:
        selected = None
    result = {'status': 'source_OOF_selection_frozen',
        'source_sha256': sha(__file__), 'source_only': True,
        'classifier_fits': fit_count, 'original_ASA_OOF_rows': len(s),
        'teacher_metrics': teachers, 'arms': arms,
        'ranked_by_primary_F1': ranked, 'qualified': qualified,
        'selected': selected, 'matched_raw_reference': matched_raw,
        'V_labels_read': False, 'historic_labels_read': False,
        'quality_acceptance': False, 'model_promoted': False,
        'oof_sha256': sha(DEST/'oof_predictions.parquet'),
        'scope': 'OOF is method/step selection, not unseen evaluation; source fold group proxy is not an independent enterprise domain.'}
    save(DEST/'selection.json', result)
    print(json.dumps({'stage': 'source_selection',
        'selected': selected, 'qualified': qualified,
        'ranked': [(a, round(arms[a]['M_S_equal_F1'], 6),
                    arms[a]['errors'], arms[a]['teacher_correct_negative_flips'])
                    for a in ranked], 'teacher_F1': {k:v['M_S_equal_F1'] for k,v in teachers.items()}},
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
