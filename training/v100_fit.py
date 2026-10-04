"""v9.9 source-only matched teachers and 64/16 residual trajectories."""
import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from run_v75 import ROOT, read, save, sha, load_sparse
from v89_common import LAST
from v85_protection import baseline_objective
from v92_train import Branch, csr, margin
from v100_prepare import DEST, V92


SEED = 9701
IDS = V92/'ASA_input_ids.npy'


def emit(**x):
    print(json.dumps(x, ensure_ascii=False), flush=True)


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def check_source():
    reg = read(DEST/'registration.json')
    assert reg['status'] == 'registered_before_fits'
    assert sha(ROOT/'training/v100_prepare.py') == reg['source_sha256']
    for p, digest in reg['input_sha256'].items():
        assert sha(ROOT/p) == digest, p
    for name, digest in reg['output_sha256'].items():
        assert sha(DEST/name) == digest, name
    exported = read(DEST/'source_export.json')
    assert exported['status'] == 'source_export_and_inference_equivalence_passed'
    assert sha(ROOT/'training/v100_export_source.py') == exported['source_sha256']
    assert sha(DEST/'source_rows.parquet') == exported['source_rows_sha256']
    return reg


def source_counts(fold):
    src = pd.read_parquet(DEST/'source_rows.parquet', columns=['fid', 'label_index', 'fold', 'is_ASA'])
    assert len(src) == 753709
    counts = np.bincount(src.fid.to_numpy()*3+src.label_index.to_numpy(),
                         weights=(src.fold.to_numpy()!=fold).astype(np.int32),
                         minlength=457566*3).reshape(-1, 3).astype(np.int32)
    hold = np.bincount(src.fid.to_numpy()*3+src.label_index.to_numpy(),
                       weights=(src.fold.to_numpy()==fold).astype(np.int32),
                       minlength=457566*3).reshape(-1, 3).astype(np.int32)
    assert int(counts.sum()+hold.sum()) == len(src)
    assert counts.sum(0)[1] and counts.sum(0)[2] and hold.sum(0)[1] and hold.sum(0)[2]
    return counts, hold


def teacher(fold, view):
    check_source()
    assert fold in (0, 1, 2) and view in ('R0', 'N1')
    folder = DEST/f'fold{fold}_{view}'
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir()
    started = time.monotonic()
    counts, hold = source_counts(fold)
    np.savez_compressed(folder/'counts.npz', train=counts, hold=hold)
    x = load_sparse(LAST/'X')
    used = np.flatnonzero(counts.sum(1))
    features = x[used]
    if view == 'N1':
        delta = sparse.load_npz(DEST/'N1_delta.npz')
        features = features+delta[used]
    rows = counts[used].astype(np.float64)
    w = np.zeros((x.shape[1]+1, 3), np.float64)
    w[-1] = np.log(np.maximum(rows.sum(0)/rows.sum(), 1e-9))
    flat = w.ravel()
    progress = []
    def callback(theta):
        progress.append({'iteration': len(progress)+1,
                         'seconds': time.monotonic()-started})
        if len(progress) % 25 == 0:
            loss, grad = baseline_objective(theta, features, rows)
            emit(stage='teacher_progress', fold=fold, view=view,
                 iteration=len(progress), loss=loss,
                 gradient_inf=float(np.abs(grad).max()),
                 seconds=round(time.monotonic()-started, 1))
    save(folder/'teacher_started.json', {'status': 'started',
        'source_sha256': sha(__file__), 'source_rows': int(rows.sum()),
        'holdout_rows_used_for_gradient': 0, 'historic_labels_loaded': False})
    with threadpool_limits(limits=4):
        result = minimize(baseline_objective, flat,
            args=(features, rows), jac=True, method='L-BFGS-B', callback=callback,
            options={'maxiter': 1000, 'maxcor': 10, 'gtol': 1e-6,
                     'ftol': 1e-12, 'maxls': 30})
    ww = result.x.reshape(x.shape[1]+1, 3)
    model = {'coef': ww[:-1].copy(), 'intercept': ww[-1].copy()}
    joblib.dump(model, folder/'teacher.joblib')
    ids = np.load(IDS)
    asa = sparse.load_npz(DEST/'N1_ASA.npz' if view == 'N1' else V92/'ASA_R0.npz')
    scores = np.asarray(asa @ model['coef'])+model['intercept']
    np.save(folder/'teacher_ASA_scores.npy', scores)
    save(folder/'teacher_progress.json', progress)
    report = {'status': 'teacher_fit_executed', 'fold': fold, 'view': view,
        'source_rows_supervised': int(rows.sum()),
        'source_class_rows_supervised': rows.sum(0).astype(int).tolist(),
        'heldout_rows': int(hold.sum()), 'heldout_gradient_rows': 0,
        'historic_labels_loaded': False, 'classifier_fits': 1,
        'converged': bool(result.success and np.abs(result.jac).max() <= 1e-5),
        'iterations': int(result.nit), 'objective': float(result.fun),
        'gradient_inf': float(np.abs(result.jac).max()),
        'solver_message': str(result.message), 'seconds': time.monotonic()-started,
        'ASA_ids': len(ids), 'source_sha256': sha(__file__),
        'model_sha256': sha(folder/'teacher.joblib'),
        'scores_sha256': sha(folder/'teacher_ASA_scores.npy')}
    save(folder/'teacher_fit.json', report)
    emit(stage='teacher_complete', fold=fold, view=view,
         converged=report['converged'], iterations=report['iterations'],
         source_rows=report['source_rows_supervised'],
         seconds=round(report['seconds'], 1))
    if not report['converged']:
        raise RuntimeError('Teacher did not satisfy registered numerical gate')


def _diagnostics(model, xt, teacher_logits, truth, mass, teacher_label,
                 protected, eps, denominator, coefficient):
    residual = model(xt).double()
    z = teacher_logits+residual
    ce = (-torch.log_softmax(z, 1)*truth).sum()/denominator
    v = (eps-margin(z, teacher_label)).clamp_min(0.)
    compat = z.new_zeros(())
    for cl in range(3):
        weights = protected*(teacher_label == cl)
        if float(weights.sum()) > 0:
            compat = compat+(weights*v.square()).sum()/weights.sum()
    compat = compat+v[protected > 0].max().square()
    mag = (mass*residual.square().sum(1)).sum()/denominator
    return residual, z, ce, compat, mag


def branch(fold, view, arm):
    check_source()
    assert fold in (0, 1, 2) and view in ('R0', 'N1') and arm in ('ERM', 'MAG')
    folder = DEST/f'fold{fold}_{view}'
    teacher_report = read(folder/'teacher_fit.json')
    assert teacher_report['converged']
    # The first R0 teacher was fitted by the bit-identical pilot source before
    # its branch-only plateau bug was corrected. Preserve that source receipt.
    accepted_teacher_sources = {sha(__file__)}
    if fold == 0 and view == 'R0':
        accepted_teacher_sources.add(sha(ROOT/'training/v100_fit_pilot.py'))
    assert teacher_report['source_sha256'] in accepted_teacher_sources
    assert sha(folder/'teacher.joblib') == teacher_report['model_sha256']
    assert sha(folder/'teacher_ASA_scores.npy') == teacher_report['scores_sha256']
    final = folder/f'{arm}_fit.json'
    if final.exists():
        raise FileExistsError(final)
    start = time.monotonic()
    with np.load(folder/'counts.npz') as packed:
        train_counts = packed['train']; hold_counts = packed['hold']
    ids = np.load(IDS)
    asa = sparse.load_npz(DEST/'N1_ASA.npz' if view == 'N1' else V92/'ASA_R0.npz')
    local_counts = train_counts[ids]
    used = np.flatnonzero(local_counts.sum(1))
    assert local_counts.sum(0)[0] == 0
    denominator = int(train_counts.sum())
    ASA_denominator = int(local_counts.sum())
    scores = np.load(folder/'teacher_ASA_scores.npy')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    model = Branch(asa.shape[1]).to(device)
    initial_hash = state_hash(model)
    xt = csr(asa[used], device)
    t = torch.as_tensor(scores[used], device=device, dtype=torch.float64)
    truth = torch.as_tensor(local_counts[used], device=device, dtype=torch.float64)
    mass = truth.sum(1)
    label = t.argmax(1)
    protected = truth.gather(1, label[:, None]).ravel()
    eps = torch.minimum(torch.full_like(protected, .001), margin(t, label).clamp_min(0.)/2)
    assert bool((eps[protected > 0] > 0).all().item())
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    coef = .01 if arm == 'MAG' else 0.
    save(folder/f'{arm}_started.json', {'status': 'started', 'fold': fold,
        'view': view, 'arm': arm, 'source_sha256': sha(__file__),
        'teacher_model_sha256': teacher_report['model_sha256'],
        'seed': SEED, 'initial_state_sha256': initial_hash,
        'train_rows': denominator, 'ASA_train_rows': ASA_denominator,
        'heldout_gradient_rows': 0, 'historic_labels_loaded': False})
    history = []
    plateau = 0
    stop = 'max2000_budget_exhausted'
    prev_objective = None
    prev_errors = None
    for step in range(1, 2001):
        model.train()
        opt.zero_grad(set_to_none=True)
        residual, z, ce, compat, mag = _diagnostics(
            model, xt, t, truth, mass, label, protected, eps, denominator, coef)
        objective = ce+.01*compat+coef*mag
        do_log = step == 1 or step % 25 == 0
        gradients = None
        if do_log:
            parameters = tuple(model.parameters())
            def vector(loss):
                gg = torch.autograd.grad(loss, parameters, retain_graph=True,
                                         allow_unused=True)
                return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                  for g, p in zip(gg, parameters)])
            gc = vector(ce)
            gp = vector(.01*compat)
            gm = vector(coef*mag) if coef else torch.zeros_like(gc)
            total = gc+gp+gm
            gradients = {'CE_gradient_l2': float(gc.norm().item()),
                'protection_gradient_l2': float(gp.norm().item()),
                'weighted_magnitude_gradient_l2': float(gm.norm().item()),
                'CE_magnitude_cosine': float((torch.dot(gc, gm)/(gc.norm()*gm.norm())).item())
                    if coef and gc.norm() > 0 and gm.norm() > 0 else None,
                'total_gradient_inf_full_units': float(total.abs().max().item()),
                'total_gradient_inf_ASA_units': float(total.abs().max().item()*denominator/ASA_denominator)}
            del gc, gp, gm, total
        objective.backward()
        opt.step()
        if do_log:
            model.eval()
            with torch.no_grad():
                updated = t+model(xt).double()
                pred = updated.argmax(1)
                errors = int((mass-truth.gather(1, pred[:, None]).ravel()).sum().item())
                neg = int(protected[(pred != label)&(protected > 0)].sum().item())
                # The post-update CE is used only for plateau detection;
                # held-out labels are not loaded here.
                updated_ce = float((-torch.log_softmax(updated, 1)*truth).sum().item()/ASA_denominator)
                record = {'step': step, 'source_fold': fold, 'view': view, 'arm': arm,
                    'ASA_train_errors': errors, 'teacher_correct_negative_flips': neg,
                    'CE_full_units_before_update': float(ce.item()),
                    'CE_ASA_units_before_update': float(ce.item()*denominator/ASA_denominator),
                    'CE_ASA_units_after_update': updated_ce,
                    'protection_loss_before_update': float(compat.item()),
                    'magnitude_loss_before_update': float(mag.item()),
                    'total_objective_before_update': float(objective.item()),
                    'full_denominator': denominator, 'ASA_denominator': ASA_denominator,
                    **gradients, 'seconds': time.monotonic()-start}
            history.append(record)
            save(folder/f'{arm}_progress.json', history)
            emit(stage='branch_progress', fold=fold, view=view, arm=arm,
                 step=step, ASA_train_errors=errors, negative_flips=neg,
                 CE=round(updated_ce, 5), seconds=round(record['seconds'], 1))
            if step == 200:
                checkpoint(model, folder, arm, 200, asa, scores, device)
            if step >= 200 and step % 25 == 0:
                if prev_objective is not None and prev_errors is not None:
                    relative = abs(float(objective.item())-prev_objective)/max(1e-12, abs(prev_objective))
                    plateau = plateau+1 if relative < 1e-4 and errors == prev_errors else 0
                prev_objective = float(objective.item())
                prev_errors = errors
                if plateau >= 5:
                    stop = 'registered_training_objective_plateau'
                    break
    assert step >= 200
    if step != 200:
        checkpoint(model, folder, arm, 'long', asa, scores, device)
    else:
        # An unusual immediate plateau still has a long endpoint identical
        # to the 200-step state; record the identity explicitly.
        checkpoint(model, folder, arm, 'long', asa, scores, device)
    last = history[-1]
    report = {'status': 'branch_fit_executed', 'fold': fold, 'view': view,
        'arm': arm, 'teacher_model_sha256': teacher_report['model_sha256'],
        'source_sha256': sha(__file__), 'classifier_fits': 1,
        'seed': SEED, 'initial_state_sha256': initial_hash,
        'source_rows_supervised': denominator,
        'ASA_rows_supervised': ASA_denominator,
        'ASA_class_rows_supervised': local_counts.sum(0).astype(int).tolist(),
        'heldout_gradient_rows': 0, 'historic_labels_loaded': False,
        'steps_executed': step, 'stop_reason': stop,
        'final_training_errors': last['ASA_train_errors'],
        'final_protected_negative_flips': last['teacher_correct_negative_flips'],
        'magnitude_coefficient': coef, 'protection_coefficient': .01,
        'final_gradient_diagnostics': {k: last[k] for k in gradients},
        'seconds': time.monotonic()-start,
        'endpoint_200_model_sha256': sha(folder/f'{arm}_200_model.pt'),
        'endpoint_long_model_sha256': sha(folder/f'{arm}_long_model.pt'),
        'endpoint_200_scores_sha256': sha(folder/f'{arm}_200_ASA_scores.npy'),
        'endpoint_long_scores_sha256': sha(folder/f'{arm}_long_ASA_scores.npy')}
    save(final, report)
    emit(stage='branch_complete', fold=fold, view=view, arm=arm,
         stop=stop, step=step, train_errors=report['final_training_errors'],
         negative_flips=report['final_protected_negative_flips'],
         seconds=round(report['seconds'], 1))


def checkpoint(model, folder, arm, tag, asa, teacher_scores, device):
    model.eval()
    predictions = np.empty((asa.shape[0], 3), np.float64)
    with torch.no_grad():
        for start in range(0, asa.shape[0], 4096):
            stop = min(start+4096, asa.shape[0])
            predictions[start:stop] = teacher_scores[start:stop]+model(
                csr(asa[start:stop], device)).double().cpu().numpy()
    np.save(folder/f'{arm}_{tag}_ASA_scores.npy', predictions)
    torch.save({'state_dict': {k: v.detach().cpu().clone()
              for k, v in model.state_dict().items()},
              'width': asa.shape[1], 'seed': SEED, 'step': tag,
              'arm': arm}, folder/f'{arm}_{tag}_model.pt')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['teacher', 'branch'])
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--view', choices=['R0', 'N1'], required=True)
    p.add_argument('--arm', choices=['ERM', 'MAG'])
    a = p.parse_args()
    with threadpool_limits(limits=4):
        teacher(a.fold, a.view) if a.stage == 'teacher' else branch(a.fold, a.view, a.arm)
