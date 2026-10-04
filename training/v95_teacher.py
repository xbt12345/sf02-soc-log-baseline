"""Fit one fresh A-only teacher on every A original row, without V labels."""
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, read, save, sha, load_sparse
from v89_common import LAST, data, raw_counts
from v85_protection import baseline_objective, cm_from_counts
from v95_prepare import DEST


def main():
    reg = read(DEST / 'registration.json')
    assert reg['source_sha256'] == sha(ROOT / 'training/v95_prepare.py')
    assert reg['manifest_sha256'] == sha(DEST / 'full_format_roles.parquet')
    assert not (DEST / 'teacher_fit.json').exists()
    r, y, fid, _, _, _, fit = data()
    roles = pd.read_parquet(DEST / 'full_format_roles.parquet', columns=['role']).role.to_numpy()
    assert np.all(np.isin(roles[fit], list('ABV')))
    x = load_sparse(LAST / 'X')
    mask = roles == 'A'
    cc = raw_counts(fid, y, mask, x.shape[0])
    used = np.flatnonzero(cc.sum(1))
    rows = cc[used].astype(np.float64)
    features = x[used]
    w = np.zeros((x.shape[1] + 1, 3), dtype=np.float64)
    w[-1] = np.log(np.maximum(rows.sum(0) / rows.sum(), 1e-9))
    flat = w.ravel()
    # One training-only directional gradient check before numerical optimization.
    rng = np.random.default_rng(reg['seed'])
    direction = rng.normal(size=len(flat)); direction /= np.linalg.norm(direction)
    eps = 1e-5
    smallx, smallc = features[:31], rows[:31]
    _, grad = baseline_objective(flat, smallx, smallc)
    fd = (baseline_objective(flat + eps * direction, smallx, smallc)[0] -
          baseline_objective(flat - eps * direction, smallx, smallc)[0]) / (2 * eps)
    gradient_check = float(abs(fd - grad @ direction))
    assert gradient_check < 1e-6
    start = time.monotonic()
    progress = []
    def callback(theta):
        if (len(progress) + 1) % 25 == 0:
            loss, g = baseline_objective(theta, features, rows)
            print(json.dumps({'stage': 'teacher', 'iteration': len(progress) + 1,
                              'loss': loss, 'gradient_inf': float(np.abs(g).max()),
                              'seconds': time.monotonic() - start}), flush=True)
        progress.append(time.monotonic() - start)
    with threadpool_limits(limits=4):
        result = minimize(baseline_objective, flat, args=(features, rows), jac=True, method='L-BFGS-B',
                          callback=callback, options={'maxiter': 1000, 'maxcor': 10,
                          'gtol': 1e-6, 'ftol': 1e-12, 'maxls': 30})
    ww = result.x.reshape(x.shape[1] + 1, 3)
    model = {'coef': ww[:-1].copy(), 'intercept': ww[-1].copy()}
    joblib.dump(model, DEST / 'teacher.joblib')
    scores = np.asarray(x @ model['coef']) + model['intercept']
    np.save(DEST / 'teacher_scores.npy', scores)
    pred = scores.argmax(1).astype(np.int8)
    np.save(DEST / 'teacher_prediction.npy', pred)
    role_metrics = {}
    for role in 'ABV':
        cnt = raw_counts(fid, y, roles == role, len(pred))
        cm = cm_from_counts(cnt, pred)
        role_metrics[role] = {'rows': int(cnt.sum()), 'cm': cm.tolist(),
                              'correct_by_class': np.diag(cm).astype(int).tolist(),
                              'errors': int(cnt.sum() - np.trace(cm))}
    report = {'stage': 'teacher_fit', 'actual_classifier_fits': 1, 'actual_calibration_fits': 0,
              'A_original_rows_used': int(mask.sum()), 'A_unique_inputs_used': len(used),
              'A_class_rows_used': rows.sum(0).astype(int).tolist(),
              'B_V_gradient_rows_used': 0, 'gradient_check_error': gradient_check,
              'converged': bool(result.success and np.abs(result.jac).max() <= 1e-5),
              'iterations': int(result.nit), 'objective': float(result.fun),
              'gradient_inf': float(np.abs(result.jac).max()), 'message': str(result.message),
              'seconds': time.monotonic() - start, 'role_metrics': role_metrics,
              'source_sha256': sha(__file__), 'model_sha256': sha(DEST / 'teacher.joblib'),
              'scores_sha256': sha(DEST / 'teacher_scores.npy')}
    save(DEST / 'teacher_fit.json', report)
    print(json.dumps({'stage': 'teacher_complete', 'converged': report['converged'],
                      'iterations': report['iterations'], 'gradient_inf': report['gradient_inf'],
                      'role_errors': {k: v['errors'] for k, v in role_metrics.items()},
                      'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
