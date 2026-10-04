"""v10.4: fixed-fold, official-only phase-A population contrast.

The old A+B subset and all official rows use the same frozen three-fold
partition and the same R0 teacher objective.  Labels on held-out rows are
read only by the post-fit evaluation stage.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from run_v75 import ROOT, OUT, load_sparse, read, save, sha
from v89_common import LAST
from v85_protection import baseline_objective
from v101_select import table


DEST = ROOT / 'artifacts/v104_phase_a_20260928'
PLAN = ROOT / 'artifacts/v103_plan_preflight_20260928/revised_training_contract.json'
PREFLIGHT = ROOT / 'artifacts/v103_plan_preflight_20260928/preflight.json'
FOLDS = ROOT / 'artifacts/v103_plan_preflight_20260928/v102_proposed_full_population_folds.parquet'
ROWS = OUT / 'rows.parquet'
FID = LAST / 'row_feature_id.npy'
X = LAST / 'X'
OFFICIAL = ROOT / 'data/official/train.parquet'
N = 2056871
K = 3
CLASS = ('benign', 'malicious', 'suspicious')


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def identity():
    reg = read(DEST / 'registration.json')
    assert reg['status'] == 'v104_phase_a_registered_before_fit'
    assert reg['source_sha256'] == sha(__file__)
    for rel, digest in reg['input_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    return reg


def load_rows():
    f = pd.read_parquet(FOLDS)
    r = pd.read_parquet(ROWS, columns=['row_position', 'route', 'label_index'])
    fid = np.load(FID, mmap_mode='r')
    assert len(f) == len(r) == len(fid) == N
    assert np.array_equal(f.row_position.to_numpy(), np.arange(N))
    assert np.array_equal(r.row_position.to_numpy(), np.arange(N))
    assert f.proposed_fold.between(0, 2).all()
    assert set(r.label_index.unique()) == {0, 1, 2}
    return f, r, fid


def register():
    if DEST.exists():
        raise FileExistsError(DEST)
    plan, check = read(PLAN), read(PREFLIGHT)
    assert plan['version'] == 'v103' and plan['phase_A']['fits'] == 6
    assert plan['split']['seed'] == 10203
    assert plan['split']['manifest_sha256'] == sha(FOLDS)
    assert all(check['split_leakage_checks'].values())
    f, r, fid = load_rows()
    x = load_sparse(X)
    assert x.shape == (457566, 66287)
    assert len(np.unique(fid)) == x.shape[0]
    assert (f.old_source == (f.old_source.astype(bool))).all()
    assert int(f.old_source.sum()) == 753709
    assert int((r.route == 'asa').sum()) == 112807
    assert np.bincount(r.label_index, minlength=3).sum() == N
    paths = [PLAN, PREFLIGHT, FOLDS, ROWS, FID, OFFICIAL,
             ROOT / 'training/v85_protection.py', ROOT / 'training/v101_select.py']
    paths += [Path(str(X) + ext) for ext in ('.json', '.data', '.indices', '.indptr')]
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    manifest = {
        'status': 'v104_phase_a_registered_before_fit',
        'source_sha256': sha(__file__), 'input_sha256': inputs,
        'population': ['old_A_B_intersect_new_fold_train', 'all_official_new_fold_train'],
        'folds': [0, 1, 2], 'fit_order': ['small', 'full'],
        'objective': 'v101 R0 three-output independent logistic, regularization 1e-6',
        'optimizer': 'L-BFGS-B, maxiter=1000, maxcor=10, gtol=1e-6, ftol=1e-12',
        'same_evaluation_rows': 'all 112807 original ASA rows, OOF by frozen full-population fold',
        'selection_use': 'population total-effect diagnostic; no automatic model promotion',
        'row_count': N, 'fid_count': x.shape[0],
        'old_source_rows': int(f.old_source.sum()),
        'official_sha256': inputs[OFFICIAL.relative_to(ROOT).as_posix()],
        'created_unix': time.time(),
    }
    DEST.mkdir()
    save(DEST / 'registration.json', manifest)
    emit(stage='registered', fits=6, rows=N, official_hash=manifest['official_sha256'])


def fit(fold, population):
    reg = identity()
    assert fold in range(K) and population in ('small', 'full')
    folder = DEST / f'fold{fold}_{population}'
    if folder.exists():
        raise FileExistsError(folder)
    f, r, fid = load_rows()
    fold_array = f.proposed_fold.to_numpy()
    mask = fold_array != fold
    if population == 'small':
        mask &= f.old_source.to_numpy()
    y = r.label_index.to_numpy(dtype=np.int8)
    counts = np.bincount(np.asarray(fid[mask], dtype=np.int64) * 3 + y[mask],
                         minlength=457566 * 3).reshape(-1, 3)
    assert int(counts.sum()) == int(mask.sum())
    assert (counts.sum(0)[1:] > 0).all()
    used = np.flatnonzero(counts.sum(1))
    x = load_sparse(X)
    features = x[used]
    weights = counts[used].astype(np.float64)
    w = np.zeros((x.shape[1] + 1, 3), np.float64)
    w[-1] = np.log(np.maximum(weights.sum(0) / weights.sum(), 1e-9))
    started = time.monotonic()
    folder.mkdir()
    save(folder / 'started.json', {
        'status': 'fit_started', 'source_sha256': reg['source_sha256'],
        'fold': fold, 'population': population,
        'train_original_rows': int(mask.sum()),
        'train_class_rows': counts.sum(0).astype(int).tolist(),
        'train_ASA_class_rows': np.bincount(y[mask & r.route.eq('asa').to_numpy()],
                                              minlength=3).astype(int).tolist(),
        'heldout_gradient_rows': 0, 'used_input_ids': len(used),
        'train_nnz': int(features.nnz),
        'predetermined_fold': True,
    })
    progress = []
    def callback(theta):
        progress.append({'iteration': len(progress) + 1,
                         'seconds': time.monotonic() - started})
        if len(progress) % 25 == 0:
            loss, gradient = baseline_objective(theta, features, weights)
            emit(stage='fit_progress', fold=fold, population=population,
                 iteration=len(progress), objective=loss,
                 gradient_inf=float(np.abs(gradient).max()),
                 seconds=round(time.monotonic() - started, 1))
    with threadpool_limits(limits=4):
        result = minimize(baseline_objective, w.ravel(), args=(features, weights),
                          jac=True, method='L-BFGS-B', callback=callback,
                          options={'maxiter': 1000, 'maxcor': 10, 'gtol': 1e-6,
                                   'ftol': 1e-12, 'maxls': 30})
    ww = result.x.reshape(x.shape[1] + 1, 3)
    model = {'coef': ww[:-1].copy(), 'intercept': ww[-1].copy()}
    joblib.dump(model, folder / 'teacher.joblib')
    # Each file contains scores for all unique observed inputs, so OOF
    # selection later uses only the model trained without the row's fold.
    scores = np.asarray(x @ model['coef']) + model['intercept']
    assert scores.shape == (457566, 3) and np.isfinite(scores).all()
    np.save(folder / 'scores_all_input_ids.npy', scores)
    save(folder / 'progress.json', progress)
    report = {
        'status': 'teacher_fit_executed', 'fold': fold, 'population': population,
        'classifier_fits': 1, 'source_sha256': reg['source_sha256'],
        'train_original_rows': int(mask.sum()),
        'train_class_rows': counts.sum(0).astype(int).tolist(),
        'train_ASA_class_rows': read(folder / 'started.json')['train_ASA_class_rows'],
        'heldout_gradient_rows': 0, 'converged': bool(result.success and
                                                    np.abs(result.jac).max() <= 1e-5),
        'iterations': int(result.nit), 'objective': float(result.fun),
        'gradient_inf': float(np.abs(result.jac).max()),
        'solver_message': str(result.message),
        'seconds': time.monotonic() - started,
        'model_sha256': sha(folder / 'teacher.joblib'),
        'scores_sha256': sha(folder / 'scores_all_input_ids.npy'),
    }
    save(folder / 'fit.json', report)
    emit(stage='fit_complete', fold=fold, population=population,
         rows=report['train_original_rows'], converged=report['converged'],
         iterations=report['iterations'], seconds=round(report['seconds'], 1))
    if not report['converged']:
        raise RuntimeError('Teacher did not satisfy the frozen numerical gate')


def class_metrics(y, pred):
    cm = np.zeros((3, 3), np.int64)
    np.add.at(cm, (y, pred), 1)
    names = {}
    for c, label in enumerate(CLASS):
        support = int(cm[c].sum())
        called = int(cm[:, c].sum())
        tp = int(cm[c, c])
        names[label] = {
            'support': support, 'correct': tp, 'missed': support - tp,
            'false_called': called - tp,
            'precision': tp / called if called else None,
            'recall': tp / support if support else None,
            'f1': 2 * tp / (support + called) if support + called else None,
        }
    return {'rows': len(y), 'errors': int(len(y) - np.trace(cm)),
            'confusion_matrix_true_rows_pred_columns': cm.tolist(),
            'predicted_counts': cm.sum(0).astype(int).tolist(),
            'class': names,
            'MS_equal_F1': (names['malicious']['f1'] + names['suspicious']['f1']) / 2}


def group_class_recall(ledger, prediction, c):
    x = ledger[ledger.truth == c]
    recall = x.assign(correct=(prediction[x.index.to_numpy()] == c)).groupby('root').correct.mean()
    sizes = x.groupby('root').size()
    return {'groups': int(len(recall)), 'mean_recall': float(recall.mean()),
            'zero_recall_groups': int((recall == 0).sum()),
            'largest_group_rows': int(sizes.max()),
            'largest_group_recall': float(recall.loc[sizes.idxmax()])}


def evaluate():
    identity()
    target = DEST / 'phase_A_evaluation.json'
    if target.exists():
        raise FileExistsError(target)
    for fold in range(K):
        for pop in ('small', 'full'):
            folder = DEST / f'fold{fold}_{pop}'
            report = read(folder / 'fit.json')
            assert report['converged'] and report['heldout_gradient_rows'] == 0
            assert report['model_sha256'] == sha(folder / 'teacher.joblib')
            assert report['scores_sha256'] == sha(folder / 'scores_all_input_ids.npy')
    f, r, fid = load_rows()
    asa_mask = r.route.eq('asa').to_numpy()
    positions = np.flatnonzero(asa_mask)
    a = pd.DataFrame({'row_position': positions,
                      'root': f.root.to_numpy()[asa_mask],
                      'fold': f.proposed_fold.to_numpy()[asa_mask],
                      'truth': r.label_index.to_numpy()[asa_mask]})
    assert len(a) == 112807 and set(a.truth.unique()) == {1, 2}
    y, fold, aid = a.truth.to_numpy(), a.fold.to_numpy(), fid[asa_mask]
    oof = {}
    preds = {}
    for pop in ('small', 'full'):
        scores = np.empty((len(a), 3), np.float64)
        for k in range(K):
            take = fold == k
            scores[take] = np.load(DEST / f'fold{k}_{pop}' /
                                   'scores_all_input_ids.npy', mmap_mode='r')[aid[take]]
        oof[pop] = scores
        preds[pop] = scores.argmax(1).astype(np.int8)
        a[f'{pop}_pred'] = preds[pop]
        a[f'{pop}_M_score'] = scores[:, 1]
        a[f'{pop}_S_score'] = scores[:, 2]
    # Label-conditional support is attached only to this diagnostic ledger.
    support = pd.read_parquet(ROOT / 'artifacts/v103_plan_preflight_20260928/ASA_support_preflight.parquet',
                              columns=['row_position', 'small_exact_support', 'full_exact_support',
                                       'small_coarse_support', 'full_coarse_support'])
    a = a.merge(support, on='row_position', validate='1:1')
    a.to_parquet(DEST / 'phase_A_ASA_OOF_ledger.parquet', index=False)
    aa = a.reset_index(drop=True)
    y = aa.truth.to_numpy()
    small, full = preds['small'], preds['full']
    summary = {
        'status': 'phase_A_six_teacher_fits_evaluated',
        'classifier_fits': 6, 'calibration_fits': 0,
        'official_rows_available': N, 'same_ASA_OOF_rows': len(aa),
        'metric_scope': 'ASA M/S original rows under fixed grouped three-fold OOF; developmental, not official score',
        'small': class_metrics(y, small),
        'full': class_metrics(y, full),
        'by_fold': {},
        'group': {},
        'ranking': {},
        'flips': {},
        'support_transitions': {},
        'model_promoted': False,
    }
    for pop, pred in preds.items():
        summary['by_fold'][pop] = {str(k): class_metrics(y[fold == k], pred[fold == k])
                                   for k in range(K)}
        summary['group'][pop] = {'M': group_class_recall(aa, pred, 1),
                                 'S': group_class_recall(aa, pred, 2)}
        margin = oof[pop][:, 2] - oof[pop][:, 1]
        summary['ranking'][pop] = {'S_vs_M_auc': float(roc_auc_score(y == 2, margin)),
                                   'score_name': 'raw S minus M logit; not a calibrated probability'}
    repaired = (small != y) & (full == y)
    regressed = (small == y) & (full != y)
    summary['flips'] = {
        'old_wrong_repaired': int(repaired.sum()),
        'old_correct_regressed': int(regressed.sum()),
        'by_class': {CLASS[c]: {'repaired': int((repaired & (y == c)).sum()),
                               'regressed': int((regressed & (y == c)).sum())}
                     for c in (1, 2)},
        'prediction_changed': int((small != full).sum()),
        'M_to_S': int(((small == 1) & (full == 2)).sum()),
        'S_to_M': int(((small == 2) & (full == 1)).sum()),
    }
    buckets = {
        'strict_zero_to_supported': (aa.small_exact_support == 0) & (aa.full_exact_support > 0),
        'strict_still_zero': (aa.full_exact_support == 0),
        'strict_already_supported': (aa.small_exact_support > 0),
        'coarse_support_to_strict_support': (aa.small_coarse_support > 0) &
                                            (aa.small_exact_support == 0) &
                                            (aa.full_exact_support > 0),
    }
    for name, mask in buckets.items():
        summary['support_transitions'][name] = {
            CLASS[c]: {'rows': int((mask & (y == c)).sum()),
                       'small_correct': int((mask & (y == c) & (small == c)).sum()),
                       'full_correct': int((mask & (y == c) & (full == c)).sum()),
                       'repairs': int((mask & (y == c) & repaired).sum()),
                       'regressions': int((mask & (y == c) & regressed).sum())}
            for c in (1, 2)}
    focus = pd.read_parquet(ROOT / 'artifacts/v103_plan_preflight_20260928/113_target_support_after_split.parquet',
                            columns=['row_position'])
    target_rows = aa[aa.row_position.isin(focus.row_position)]
    assert len(target_rows) == 113
    summary['historic_113'] = {
        'total': 113,
        'small_correct': int((target_rows.small_pred == target_rows.truth).sum()),
        'full_correct': int((target_rows.full_pred == target_rows.truth).sum()),
        'newly_supported_101': {
            'rows': int(((target_rows.small_exact_support == 0) &
                         (target_rows.full_exact_support > 0)).sum()),
            'small_correct': int(((target_rows.small_exact_support == 0) &
                                  (target_rows.full_exact_support > 0) &
                                  (target_rows.small_pred == target_rows.truth)).sum()),
            'full_correct': int(((target_rows.small_exact_support == 0) &
                                 (target_rows.full_exact_support > 0) &
                                 (target_rows.full_pred == target_rows.truth)).sum()),
        },
    }
    summary['input_sha256'] = read(DEST / 'registration.json')['input_sha256']
    summary['fit_reports_sha256'] = {f'fold{k}_{pop}': sha(DEST / f'fold{k}_{pop}' / 'fit.json')
                                     for k in range(K) for pop in ('small', 'full')}
    summary['OOF_ledger_sha256'] = sha(DEST / 'phase_A_ASA_OOF_ledger.parquet')
    save(target, summary)
    emit(stage='phase_A_complete', errors={p: summary[p]['errors'] for p in ('small', 'full')},
         F1={p: summary[p]['MS_equal_F1'] for p in ('small', 'full')},
         flips=summary['flips'], target_101=summary['historic_113']['newly_supported_101'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['register', 'fit', 'evaluate'])
    parser.add_argument('--fold', type=int)
    parser.add_argument('--population', choices=['small', 'full'])
    args = parser.parse_args()
    if args.stage == 'register':
        register()
    elif args.stage == 'fit':
        fit(args.fold, args.population)
    else:
        evaluate()
