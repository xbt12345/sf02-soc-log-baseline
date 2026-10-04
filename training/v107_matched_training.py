"""V107: fixed N1/N2 x full teacher/ASA TabM25 on body-source-closed folds.

Official rows retain their original labels and frequencies.  No private answer
is read.  This is developmental OOF evidence, not an external test.
"""
import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from run_v75 import ROOT, OUT, load_sparse, save, sha
from v75_views import BYTE_FEATURES, byte_matrix, matrix_hashes
from v75_corrective import stable
from v99_normalization_feasibility import normalize
from v85_protection import baseline_objective
from v104_phase_b import (ASA_IDS, BATCH, DEVICE, FID, K, PREP,
                          ROWS, SEED, WIDTH, X, SparseTabM, csr_tensor,
                          predict_all)

V106 = ROOT / 'artifacts/v106_frozen_audit_20260928'
DEST = ROOT / 'artifacts/v107_matched_training_20260928'
FOLDS = V106 / 'proposed_body_closed_folds.parquet'
V92 = ROOT / 'artifacts/v92_evidence_training_20260928'


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def rel(path):
    return str(Path(path).relative_to(ROOT)).replace('\\', '/')


def prepare():
    assert not DEST.exists(), DEST
    contract = json.loads((V106 / 'next_training_contract.json').read_text(encoding='utf-8'))
    assert contract['status'] == 'revised_plan_not_trained'
    assert sha(FOLDS) == contract['split']['sha256']
    assert sha(ROOT / 'training/v106_frozen_wrapper_audit.py') == contract['normalizer_source_sha256']
    assert json.loads((V106 / 'final_verification.json').read_text())['all_checks_passed']
    r = pd.read_parquet(ROWS, columns=['row_position', 'route', 'new_text_id', 'label_index'])
    f = pd.read_parquet(FOLDS, columns=['row_position', 'proposed_fold', 'root'])
    assert len(r) == len(f) == 2056871
    assert np.array_equal(r.row_position.to_numpy(), np.arange(len(r)))
    assert np.array_equal(f.row_position.to_numpy(), r.row_position.to_numpy())
    assert set(f.proposed_fold.unique()) == {0, 1, 2}
    asa = r.route.eq('asa').to_numpy()
    pos = np.flatnonzero(asa)
    assert len(pos) == 112807
    ids = np.load(ASA_IDS)
    fid = np.load(FID, mmap_mode='r')
    lookup = np.full(int(fid.max()) + 1, -1, dtype=np.int32)
    lookup[ids] = np.arange(len(ids))
    local = lookup[fid[pos]]
    assert (local >= 0).all()
    dictionary = pd.read_parquet(OUT / 'text_dictionary.parquet').set_index('text_id').text
    spans = pd.read_parquet(V106 / 'variant_span_ledger.parquet')
    collapse = spans.loc[spans['mode'].eq('collapse'), ['before_N1', 'after_N1']].drop_duplicates()
    assert collapse.groupby('before_N1').after_N1.nunique().max() == 1
    replacement = dict(zip(collapse.before_N1, collapse.after_N1))
    pair = pd.DataFrame({'local': local, 'text_id': r.new_text_id.to_numpy()[pos]}).drop_duplicates()
    pair['before'] = [normalize(stable(dictionary.loc[int(t)]), 'placeholder_cluster')
                      for t in pair.text_id]
    pair['after'] = pair.before.map(lambda t: replacement.get(t, t))
    assert pair.groupby('local').before.nunique().max() == 1
    assert pair.groupby('local').after.nunique().max() == 1
    unique = pair.drop_duplicates('local').sort_values('local')
    assert np.array_equal(unique.local.to_numpy(), np.arange(len(ids)))
    old_asa = sparse.load_npz(V92 / 'ASA_R0.npz')
    n1_asa = sparse.load_npz(PREP / 'N1_ASA.npz')
    assert old_asa.shape == n1_asa.shape == (22546, WIDTH)
    n1_bytes = byte_matrix(unique.before)
    mismatch = n1_bytes - n1_asa[:, :BYTE_FEATURES]
    assert mismatch.nnz == 0 or abs(mismatch.data).max() < 1e-7
    n2_bytes = byte_matrix(unique.after)
    n2_asa = sparse.hstack([n2_bytes, n1_asa[:, BYTE_FEATURES:]], format='csr')
    assert (n2_asa[:, BYTE_FEATURES:] - n1_asa[:, BYTE_FEATURES:]).nnz == 0
    assert (n2_asa[:, BYTE_FEATURES:] - old_asa[:, BYTE_FEATURES:]).nnz == 0
    difference = n2_asa - old_asa
    aidx = np.repeat(ids, np.diff(difference.indptr))
    delta = sparse.csr_matrix((difference.data, (aidx, difference.indices)),
                              shape=(int(fid.max())+1, WIDTH))
    # Reproduce the separately registered all-ASA N2 exact-input groups.
    hashes = matrix_hashes(n2_asa)
    row_hashes = pd.Series(hashes, dtype=object).iloc[local].to_numpy()
    groups, _ = pd.factorize(row_hashes, sort=False)
    expected = pd.read_parquet(V106 / 'collapse_input_groups.parquet',
                               columns=['row_position', 'new_input'])
    assert np.array_equal(expected.row_position.to_numpy(), pos)
    assert np.array_equal(expected.new_input.to_numpy(), groups)
    # The same N2 input cannot straddle folds. V106 also checked non-ASA equality.
    assert pd.DataFrame({'group': groups, 'fold': f.proposed_fold.to_numpy()[pos]}).groupby('group').fold.nunique().max() == 1
    DEST.mkdir(parents=True)
    sparse.save_npz(DEST / 'N2_ASA.npz', n2_asa, compressed=True)
    sparse.save_npz(DEST / 'N2_delta.npz', delta, compressed=True)
    pair[['local', 'text_id', 'before', 'after']].to_parquet(DEST / 'N2_text_pairs.parquet', index=False)
    inputs = [FOLDS, V106 / 'next_training_contract.json',
              V106 / 'final_verification.json', V106 / 'variant_span_ledger.parquet',
              V106 / 'collapse_input_groups.parquet', ROWS, OUT / 'text_dictionary.parquet',
              FID, ASA_IDS, V92 / 'ASA_R0.npz', PREP / 'N1_ASA.npz', PREP / 'N1_delta.npz',
              ROOT / 'data/official/train.parquet']
    reg = {'status': 'prepared_before_any_fit', 'fits': 0,
           'source_sha256': sha(__file__),
           'v104_model_source_sha256': sha(ROOT / 'training/v104_phase_b.py'),
           'objective_source_sha256': sha(ROOT / 'training/v85_protection.py'),
           'input_sha256': {rel(p): sha(p) for p in inputs},
           'N2_ASA_sha256': sha(DEST / 'N2_ASA.npz'),
           'N2_delta_sha256': sha(DEST / 'N2_delta.npz'),
           'N2_pairs_sha256': sha(DEST / 'N2_text_pairs.parquet'),
           'official_rows': len(r), 'ASA_rows': len(pos),
           'N2_changed_unique_ASA_inputs': int(np.asarray((n2_asa != n1_asa).getnnz(axis=1) > 0).sum()),
           'fold_ASA_class_rows': {str(k): np.bincount(r.label_index.to_numpy()[pos][f.proposed_fold.to_numpy()[pos] == k], minlength=3).tolist()
                                   for k in (0, 1, 2)},
           'view_definitions': {'N1': rel(PREP / 'N1_ASA.npz'), 'N2': 'v106 collapse then old stable+N1'},
           'models': 'matched v104 full teacher and SparseTabM25',
           'environment': {'python': __import__('sys').version, 'torch': torch.__version__,
                           'tabm': importlib.metadata.version('tabm'), 'device': DEVICE}}
    save(DEST / 'registration.json', reg)
    emit(stage='prepared', changed_inputs=reg['N2_changed_unique_ASA_inputs'],
         fold_classes=reg['fold_ASA_class_rows'])


def check():
    reg = json.loads((DEST / 'registration.json').read_text(encoding='utf-8'))
    assert sha(__file__) == reg['source_sha256']
    assert sha(ROOT / 'training/v104_phase_b.py') == reg['v104_model_source_sha256']
    assert sha(ROOT / 'training/v85_protection.py') == reg['objective_source_sha256']
    for p, digest in reg['input_sha256'].items():
        assert sha(ROOT / p) == digest, p
    assert sha(DEST / 'N2_ASA.npz') == reg['N2_ASA_sha256']
    assert sha(DEST / 'N2_delta.npz') == reg['N2_delta_sha256']
    return reg


def rows():
    r = pd.read_parquet(ROWS, columns=['route', 'label_index'])
    f = pd.read_parquet(FOLDS, columns=['proposed_fold', 'root'])
    fid = np.load(FID, mmap_mode='r')
    assert len(r) == len(f) == len(fid) == 2056871
    return r, f, fid


def view_paths(view):
    assert view in ('N1', 'N2')
    if view == 'N1': return PREP / 'N1_ASA.npz', PREP / 'N1_delta.npz'
    return DEST / 'N2_ASA.npz', DEST / 'N2_delta.npz'


def teacher(fold, view):
    check()
    assert fold in (0, 1, 2)
    folder = DEST / f'fold{fold}_{view}_teacher'
    assert not folder.exists()
    r, f, fid = rows()
    train = f.proposed_fold.to_numpy() != fold
    y = r.label_index.to_numpy(dtype=np.int8)
    x = load_sparse(X)
    _, delta_path = view_paths(view)
    delta = sparse.load_npz(delta_path)
    assert delta.shape == x.shape
    counts = np.bincount(np.asarray(fid[train], dtype=np.int64)*3 + y[train],
                         minlength=x.shape[0]*3).reshape(-1, 3)
    assert int(counts.sum()) == int(train.sum())
    used = np.flatnonzero(counts.sum(1))
    feat = x[used] + delta[used]
    weights = counts[used].astype(np.float64)
    w = np.zeros((WIDTH+1, 3), np.float64)
    w[-1] = np.log(np.maximum(weights.sum(0)/weights.sum(), 1e-9))
    folder.mkdir()
    started = time.monotonic()
    save(folder / 'started.json', {'status': 'started', 'fold': fold, 'view': view,
         'train_original_rows': int(train.sum()), 'class_rows': counts.sum(0).astype(int).tolist(),
         'heldout_gradient_rows': 0, 'source_sha256': sha(__file__)})
    progress = []
    def callback(theta):
        progress.append({'iteration': len(progress)+1, 'seconds': time.monotonic()-started})
        if len(progress) % 25 == 0:
            loss, grad = baseline_objective(theta, feat, weights)
            emit(stage='teacher_progress', view=view, fold=fold, iteration=len(progress),
                 objective=float(loss), gradient_inf=float(np.abs(grad).max()),
                 seconds=round(time.monotonic()-started, 1))
    with threadpool_limits(limits=4):
        result = minimize(baseline_objective, w.ravel(), args=(feat, weights),
                          jac=True, method='L-BFGS-B', callback=callback,
                          options={'maxiter':1000, 'maxcor':10, 'gtol':1e-6,
                                   'ftol':1e-12, 'maxls':30})
    ww = result.x.reshape(WIDTH+1, 3)
    model = {'coef': ww[:-1].copy(), 'intercept': ww[-1].copy()}
    joblib.dump(model, folder / 'teacher.joblib')
    scores = np.empty((x.shape[0], 3), np.float64)
    for start in range(0, x.shape[0], 8192):
        stop = min(start+8192, x.shape[0])
        scores[start:stop] = np.asarray((x[start:stop]+delta[start:stop]) @ model['coef']) + model['intercept']
    assert np.isfinite(scores).all()
    np.save(folder / 'scores_all_input_ids.npy', scores)
    save(folder / 'progress.json', progress)
    report = {'status': 'fit_executed', 'fold': fold, 'view': view, 'arm': 'teacher',
              'classifier_fits': 1, 'train_original_rows': int(train.sum()),
              'train_class_rows': counts.sum(0).astype(int).tolist(),
              'heldout_gradient_rows': 0,
              'converged': bool(result.success and np.abs(result.jac).max() <= 1e-5),
              'iterations': int(result.nit), 'objective': float(result.fun),
              'gradient_inf': float(np.abs(result.jac).max()), 'message': str(result.message),
              'seconds': time.monotonic()-started, 'source_sha256': sha(__file__),
              'model_sha256': sha(folder / 'teacher.joblib'),
              'scores_sha256': sha(folder / 'scores_all_input_ids.npy')}
    save(folder / 'fit.json', report)
    emit(stage='teacher_complete', view=view, fold=fold, converged=report['converged'],
         iterations=report['iterations'], seconds=round(report['seconds'], 1))
    assert report['converged'], 'Teacher convergence failed'


def tabm(fold, view, seed=SEED):
    check()
    assert fold in (0, 1, 2)
    folder = DEST / f'fold{fold}_{view}_TabM25_seed{seed}'
    assert not folder.exists()
    r, f, fid = rows()
    asa = r.route.eq('asa').to_numpy()
    pos = np.flatnonzero(asa)
    ids = np.load(ASA_IDS)
    lookup = np.full(int(fid.max())+1, -1, dtype=np.int32)
    lookup[ids] = np.arange(len(ids))
    local = lookup[fid[pos]]
    assert (local >= 0).all()
    source_fold = f.proposed_fold.to_numpy()[pos]
    y = r.label_index.to_numpy(dtype=np.int8)[pos]
    train = source_fold != fold
    x_path, _ = view_paths(view)
    x = sparse.load_npz(x_path)
    assert x.shape == (len(ids), WIDTH)
    counts = np.bincount(local[train].astype(np.int64)*3+y[train],
                         minlength=x.shape[0]*3).reshape(-1, 3)
    assert int(counts.sum()) == int(train.sum())
    assert counts.sum(0)[0] == 0 and (counts.sum(0)[1:] > 0).all()
    used = np.flatnonzero(counts.sum(1))
    xtrain = x[used]
    cc = counts[used].astype(np.float32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SparseTabM().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0003)
    folder.mkdir()
    started = time.monotonic()
    save(folder / 'started.json', {'status': 'started', 'fold': fold, 'view': view,
         'seed': seed, 'train_ASA_original_rows': int(train.sum()),
         'train_ASA_classes': counts.sum(0).astype(int).tolist(),
         'heldout_gradient_rows': 0, 'train_unique_R0_ids': len(used),
         'batch_size': BATCH, 'epochs': 25, 'source_sha256': sha(__file__)})
    batches = math.ceil(len(used)/BATCH)
    normalizer = float(train.sum())/batches
    rng = np.random.default_rng(seed+fold)
    history = []
    for epoch in range(1, 26):
        model.train()
        order = rng.permutation(len(used))
        online_numerator = 0.
        for start in range(0, len(order), BATCH):
            batch = order[start:start+BATCH]
            block = xtrain[batch]
            fact = torch.as_tensor(block[:, BYTE_FEATURES:].toarray(), device=DEVICE, dtype=torch.float32)
            mass = torch.as_tensor(cc[batch], device=DEVICE, dtype=torch.float32)
            z = model(csr_tensor(block, DEVICE), fact)
            numerator = -(torch.log_softmax(z, -1)*mass[:, None, :]).sum()/K
            loss = numerator/normalizer
            assert bool(torch.isfinite(loss).item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            online_numerator += float(numerator.detach().item())
        record = {'epoch': epoch, 'online_original_row_CE': online_numerator/float(train.sum()),
                  'seconds': time.monotonic()-started}
        history.append(record)
        if epoch == 1 or epoch % 5 == 0:
            save(folder / 'progress.json', history)
            emit(stage='TabM_progress', view=view, fold=fold, seed=seed,
                 epoch=epoch, online_CE=round(record['online_original_row_CE'], 6),
                 seconds=round(record['seconds'], 1))
    prob = predict_all(model, 'TabM', x, DEVICE)
    np.save(folder / 'ASA_input_prob.npy', prob)
    torch.save({'state_dict': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'view': view, 'fold': fold, 'seed': seed, 'epoch': 25,
                'source_sha256': sha(__file__)}, folder / 'model.pt')
    pred = prob[used].argmax(1)
    train_errors = int((counts[used].sum(1)-counts[used, pred]).sum())
    report = {'status': 'fit_executed', 'fold': fold, 'view': view, 'arm': 'TabM25',
              'seed': seed, 'classifier_fits': 1, 'train_ASA_original_rows': int(train.sum()),
              'train_ASA_classes': counts.sum(0).astype(int).tolist(),
              'heldout_gradient_rows': 0, 'epochs': 25,
              'optimizer': 'AdamW', 'lr': .002, 'weight_decay': .0003,
              'batch_size': BATCH, 'K': K, 'train_original_errors': train_errors,
              'source_sha256': sha(__file__), 'seconds': time.monotonic()-started,
              'model_sha256': sha(folder / 'model.pt'),
              'prob_sha256': sha(folder / 'ASA_input_prob.npy')}
    save(folder / 'fit.json', report)
    emit(stage='TabM_complete', view=view, fold=fold, seed=seed,
         train_errors=train_errors, seconds=round(report['seconds'], 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'teacher', 'tabm', 'check'])
    p.add_argument('--fold', type=int)
    p.add_argument('--view', choices=['N1', 'N2'])
    p.add_argument('--seed', type=int, default=SEED)
    a = p.parse_args()
    if a.stage == 'prepare': prepare()
    elif a.stage == 'check': emit(stage='check', status=check()['status'])
    elif a.stage == 'teacher': teacher(a.fold, a.view)
    else: tabm(a.fold, a.view, a.seed)


if __name__ == '__main__': main()
