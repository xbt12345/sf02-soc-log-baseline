"""Fixed off-the-shelf development probe, never an automatic promotion run."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

FIELDS = ['action', 'outcome', 'transport_protocol', 'src_role', 'dst_role',
          'src_port_fixed', 'dst_port_fixed', 'src_port_range', 'dst_port_range',
          'icmp_type', 'icmp_code', 'icmp_message', 'icmp_unreachable']
FINITE = {'src_port_fixed': 65536, 'dst_port_fixed': 65536,
          'icmp_type': 256, 'icmp_code': 256}
MISSING = '__UNOBSERVED__'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def encode(f):
    result = []
    for k in FIELDS:
        v = f.get(k)
        if v is None or v == '' or (k in FINITE and
                (not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < FINITE[k])):
            result.append(MISSING)
        else:
            result.append(str(v))
    return result


def data(root):
    prep = root / 'artifacts/v39_local_r2_20260913/prepared'
    r = pq.read_table(prep / 'rows.parquet').to_pandas()
    r = r[(r.fold >= 0) & (r.route == 'asa')].copy().reset_index(drop=True)
    p = pq.read_table(prep / 'projections.parquet').to_pandas()
    ff = [encode(json.loads(s)) for s in p.facts]
    x = pd.DataFrame([ff[int(i)] for i in r.projection_id], columns=FIELDS)
    b = pd.concat([pq.read_table(prep.parent / ('primary/fold_%s/SEMANTIC/evaluation.parquet' % f)).to_pandas()
                   for f in range(3)]).sort_values('row_position').reset_index(drop=True)
    ab = b[b.route == 'asa'].reset_index(drop=True)
    assert np.array_equal(r.row_position, ab.row_position)
    assert np.array_equal(r.label_index, ab.label_index)
    assert len(r) == 106953 and set(r.label_index) == {1, 2}
    assert x.drop_duplicates().shape[0] == 16874
    # v39 uses body groups; legacy union_group is a broader, deliberately
    # different pressure protocol and is not isolated by these folds.
    assert r.groupby('body_group').fold.nunique().max() == 1
    return r, x, b, ab


def prepare(root, out):
    assert not out.exists(), 'Do not overwrite an existing probe.'
    prep = root / 'artifacts/v39_local_r2_20260913/prepared'
    expected = {'rows.parquet': 'e1b5add85250184f4975935fcd912cde82d83abb700f1fe88f8efa6b8dfc4df5',
                'projections.parquet': 'db23bf2603b938e7b502ba0061a368eda26d38e8fbfe82c8feb75da0fb8d030e'}
    bindings = {}
    for n, h in expected.items():
        p = prep / n
        assert sha(p) == h
        bindings[p.relative_to(root).as_posix()] = h
    for fold in range(3):
        d = prep.parent / ('primary/fold_%s/SEMANTIC' % fold)
        c = json.loads((d / 'complete.json').read_text(encoding='utf-8'))
        assert sha(d / 'evaluation.parquet') == c['predictions_sha256']
        for n in ['model.joblib', 'evaluation.parquet']:
            bindings[(d / n).relative_to(root).as_posix()] = sha(d / n)
    r, x, _, _ = data(root)
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out / Path(__file__).name)
    config = {
        'version': 'v44-ready-methods-development-probe-1', 'fields': FIELDS,
        'label': '1=malicious, 0=suspicious internally; original labels unchanged',
        'scope': 'Previously inspected official development folds. Fixed configs, no tuning or early stopping; not blind or transfer validation.',
        'fit_rows': 'All ASA rows with fold>=0 and fold!=evaluation_fold; original multiplicity and labels',
        'lr': {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 1e-6},
        'cat': {'iterations': 500, 'depth': 5, 'learning_rate': 0.05, 'l2_leaf_reg': 10,
                'loss_function': 'Logloss', 'boosting_type': 'Ordered', 'max_ctr_complexity': 2,
                'random_seed': 20260914, 'thread_count': 4, 'has_time': False,
                'allow_writing_files': False, 'verbose': False},
        'selection': 'No automatic promotion. Compare both classes, all folds, unseen keys, availability groups and independent head versus protected overlay.',
        'mask_control': 'Fit-side class-frequency table of presence/absence of all 13 allowed fields; unseen mask uses fit class prior. Diagnostic only.',
        'port_removal': 'Evaluation-only information-loss stress; remove value and derived range together. No equality or transfer guarantee.',
        'source_bindings': bindings, 'script_sha256': sha(__file__),
        'rows': len(r), 'unique_observed_keys': x.drop_duplicates().shape[0],
        'body_groups_crossing_folds': 0,
        'legacy_union_groups_crossing_folds': int((r.groupby('union_group').fold.nunique() > 1).sum()),
        'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'pyarrow']},
        'python': sys.version
    }
    save(out / 'configuration.json', config)
    save(out / 'preregistered.json', {'configuration_sha256': sha(out / 'configuration.json'),
                                    'source_sha256': sha(out / Path(__file__).name),
                                    'model_fits_so_far': 0})
    print(json.dumps({'prepared': str(out), 'rows': len(r), 'new_fits': 0}), flush=True)


def stats(y, pred, q=None):
    ans = {'rows': len(y), 'errors': int(np.sum(y != pred)),
           'confusion_M_S': confusion_matrix(y, pred, labels=[1, 2]).tolist(),
           'macro_f1_M_S': float(f1_score(y, pred, labels=[1, 2], average='macro', zero_division=0))}
    if q is not None and len(y):
        ans['conditional_log_loss'] = float(log_loss(y == 1, np.c_[1-q, q], labels=[False, True]))
    return ans


def fit(root, out):
    start = time.monotonic()
    c = json.loads((out / 'configuration.json').read_text(encoding='utf-8'))
    receipt = json.loads((out / 'preregistered.json').read_text(encoding='utf-8'))
    assert sha(__file__) == receipt['source_sha256']
    assert sha(out / 'configuration.json') == receipt['configuration_sha256']
    for path, h in c['source_bindings'].items():
        assert sha(root / path) == h, path
    assert not (out / 'started.json').exists()
    save(out / 'started.json', {'configuration_sha256': receipt['configuration_sha256'], 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    r, x, full_b, ab = data(root)
    y = r.label_index.to_numpy()
    bprob = ab[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy()
    bpred = bprob.argmax(1)
    assert (bpred != 0).all()
    keys = np.array([json.dumps(v, separators=(',', ':')) for v in x.to_numpy().tolist()], dtype=object)
    masks = np.array([''.join('0' if a == MISSING else '1' for a in v) for v in x.to_numpy().tolist()])
    results = r[['row_position', 'fold', 'body_group', 'projection_id', 'label_index']].copy()
    results['B_pred'] = bpred
    for n in ['LR', 'CAT', 'MASK']:
        results[n + '_qM'] = np.nan
    results['seen_fit_key'] = False
    results['mixed_eval_key_diagnostic_only'] = False
    stress = []
    model_hashes = {}
    for fold in range(3):
        tr = r.fold.to_numpy() != fold
        ev = ~tr
        d = out / ('fold_%s' % fold)
        d.mkdir()
        results.loc[ev, 'seen_fit_key'] = np.isin(keys[ev], keys[tr])
        kd = pd.DataFrame({'key': keys[ev], 'y': y[ev]})
        mixed = kd.groupby('key').y.nunique()
        results.loc[ev, 'mixed_eval_key_diagnostic_only'] = kd.key.map(mixed).to_numpy() > 1
        mt = pd.DataFrame({'mask': masks[tr], 'y': y[tr] == 1}).groupby('mask').y.mean()
        results.loc[ev, 'MASK_qM'] = pd.Series(masks[ev]).map(mt).fillna(float((y[tr] == 1).mean())).to_numpy()
        save(d / 'mask_control.json', {'frequency_by_mask': mt.to_dict(), 'prior': float((y[tr] == 1).mean())})
        for name in ['LR', 'CAT']:
            t = time.monotonic()
            if name == 'LR':
                model = make_pipeline(OneHotEncoder(handle_unknown='ignore', dtype=np.float64), LogisticRegression(**c['lr']))
                model.fit(x.loc[tr], y[tr] == 1)
                assert np.max(model[-1].n_iter_) < c['lr']['max_iter'], 'LR did not converge'
                p = d / 'LR.joblib'
                joblib.dump(model, p)
                loaded = joblib.load(p)
            else:
                model = CatBoostClassifier(**c['cat'])
                model.fit(x.loc[tr], y[tr] == 1, cat_features=FIELDS)
                p = d / 'CAT.cbm'
                model.save_model(p)
                loaded = CatBoostClassifier().load_model(p)
            assert list(model.classes_) == [False, True] or list(model.classes_) == [0, 1]
            q = model.predict_proba(x.loc[ev])[:, 1]
            q2 = loaded.predict_proba(x.loc[ev])[:, 1]
            assert np.array_equal(q, q2)
            model_hashes[p.relative_to(out).as_posix()] = sha(p)
            results.loc[ev, name + '_qM'] = q
            for scope, cols in [('hide_source', ['src_port_fixed', 'src_port_range']),
                                ('hide_destination', ['dst_port_fixed', 'dst_port_range'])]:
                subset = (x.loc[ev, cols[0]] != MISSING).to_numpy()
                xx = x.loc[ev].copy()
                xx[cols] = MISSING
                qp = loaded.predict_proba(xx)[:, 1]
                yy = y[ev][subset]
                stress.append({'fold': fold, 'model': name, 'scope': scope,
                               'before': stats(yy, np.where(q[subset] >= .5, 1, 2), q[subset]),
                               'after': stats(yy, np.where(qp[subset] >= .5, 1, 2), qp[subset]),
                               'label_flips': int(np.sum((q[subset] >= .5) != (qp[subset] >= .5)))})
            print(json.dumps({'fold': fold, 'model': name, 'seconds': time.monotonic()-t,
                              **stats(y[ev], np.where(q >= .5, 1, 2), q)}), flush=True)
    for n in ['LR', 'CAT', 'MASK']:
        assert np.isfinite(results[n + '_qM']).all()
        q = results[n + '_qM'].to_numpy()
        results[n + '_head_pred'] = np.where(q >= .5, 1, 2)
        p = bprob.copy()
        mass = bprob[:, 1] + bprob[:, 2]
        p[:, 1] = mass * q
        p[:, 2] = mass * (1 - q)
        fallback = (bpred == 0) | (p.argmax(1) == 0)
        p[fallback] = bprob[fallback]
        assert np.array_equal(p[:, 0], bprob[:, 0])
        assert np.array_equal(p.argmax(1) == 0, bpred == 0)
        results[n + '_protected_pred'] = p.argmax(1)
        results[n + '_fallback'] = fallback
    results['source_observed'] = (x.src_port_fixed != MISSING).to_numpy()
    results['destination_observed'] = (x.dst_port_fixed != MISSING).to_numpy()
    results['protocol'] = x.transport_protocol.to_numpy()
    results.to_parquet(out / 'evaluation.parquet', index=False)
    save(out / 'information_removal_stress.json', stress)
    report = {}
    masks_eval = {'all_ASA': np.ones(len(y), dtype=bool), 'seen_key': results.seen_fit_key.to_numpy(),
                  'unseen_key': ~results.seen_fit_key.to_numpy(),
                  'mixed_eval_key_diagnostic_only': results.mixed_eval_key_diagnostic_only.to_numpy(),
                  'pure_eval_key_diagnostic_only': ~results.mixed_eval_key_diagnostic_only.to_numpy()}
    for fold in range(3):
        masks_eval['fold_%s' % fold] = r.fold.to_numpy() == fold
    for proto in sorted(results.protocol.unique()):
        for src in [False, True]:
            for dst in [False, True]:
                masks_eval['%s_src%s_dst%s' % (proto, int(src), int(dst))] = (
                    (results.protocol == proto) & (results.source_observed == src) & (results.destination_observed == dst)).to_numpy()
    for name, mask in masks_eval.items():
        if not mask.any():
            continue
        rr = {'B': stats(y[mask], bpred[mask])}
        for n in ['LR', 'CAT', 'MASK']:
            pred = results[n + '_head_pred'].to_numpy()
            rr[n] = stats(y[mask], pred[mask], results[n + '_qM'].to_numpy()[mask])
            rr[n]['fixes'] = int(np.sum(mask & (bpred != y) & (pred == y)))
            rr[n]['regressions'] = int(np.sum(mask & (bpred == y) & (pred != y)))
            rr[n]['changed_body_keys'] = int(r.loc[mask & (bpred != pred), 'body_group'].nunique())
        report[name] = rr
    overall = {}
    all_y = full_b.label_index.to_numpy()
    bp = full_b[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy().argmax(1)
    target = full_b.route.to_numpy() == 'asa'
    for n in ['B', 'LR', 'CAT', 'MASK']:
        pred = bp.copy()
        if n != 'B':
            pred[target] = results[n + '_protected_pred']
        assert np.array_equal(pred[~target], bp[~target])
        overall[n] = {'rows': len(all_y), 'errors': int(np.sum(pred != all_y)),
                      'macro_f1': float(f1_score(all_y, pred, labels=[0, 1, 2], average='macro')),
                      'normal_false_alerts': int(np.sum((all_y == 0) & (pred != 0)))}
    save(out / 'report.json', {'status': 'development_probe_not_promoted', 'new_classifier_fits': 6,
                              'mask_frequency_tables': 3, 'slices': report, 'protected_overall': overall,
                              'fallback_rows': {n: int(results[n + '_fallback'].sum()) for n in ['LR', 'CAT', 'MASK']},
                              'model_sha256': model_hashes, 'elapsed_seconds': time.monotonic()-start,
                              'source_bindings': c['source_bindings'],
                              'limits': ['All evaluation folds already inspected in earlier development.',
                                         'No raw-record inference replay yet; same-feature model reload replay completed.',
                                         'Presence and absence remain inferable. This is not a missingness-invariance guarantee.',
                                         'No fresh source holdout, external data or platform execution.',
                                         'No benign ASA records; only malicious/suspicious conditional head evaluated.']})
    print(json.dumps({'finished': True, 'overall': overall, 'ASA': report['all_ASA']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'fit'])
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    out = Path(a.out).resolve()
    (prepare if a.mode == 'prepare' else fit)(root, out)
