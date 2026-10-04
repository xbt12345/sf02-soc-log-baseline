"""Matched four-arm ASA residual trial with optional first-order cross-component update."""
import argparse
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, read, save, sha
from v89_common import data, raw_counts
from v85_protection import changes, cm_from_counts
from v81_training_contract import compare
from v92_train import Branch, csr, margin
from v92_common import DEST as V92
from v95_prepare import DEST

CHECKPOINTS = [0, 25, 50, 100, 200]


def state_hash(model):
    h = hashlib.sha256()
    for k, t in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def assert_ready():
    reg = read(DEST / 'registration.json')
    assert reg['source_sha256'] == sha(ROOT / 'training/v95_prepare.py')
    assert reg['manifest_sha256'] == sha(DEST / 'full_format_roles.parquet')
    teacher = read(DEST / 'teacher_fit.json')
    assert teacher['source_sha256'] == sha(ROOT / 'training/v95_teacher.py')
    assert teacher['model_sha256'] == sha(DEST / 'teacher.joblib')
    assert teacher['converged'], 'A-only teacher numerical convergence gate failed'
    return reg, teacher


def make_context():
    reg, teacher = assert_ready()
    r, y, fid, old_z, old_pred, _, fit = data()
    roles = pd.read_parquet(DEST / 'full_format_roles.parquet', columns=['role']).role.to_numpy()
    ids = np.load(V92 / 'ASA_input_ids.npy')
    x = sparse.load_npz(V92 / 'ASA_R0.npz')
    zz = np.load(DEST / 'teacher_scores.npy', mmap_mode='r')
    base = np.load(DEST / 'teacher_prediction.npy')
    assert x.shape == (len(ids), 66287)
    assert len(base) == len(zz) and np.all(base == np.asarray(zz).argmax(1))
    ab = np.isin(roles, ['A', 'B'])
    full_n = int(ab.sum())
    raw = raw_counts(fid, y, ab, len(base))
    cc = raw[ids]
    used = np.flatnonzero(cc.sum(1)); train_ids = ids[used]
    assert int(cc.sum()) == int((ab & r.route.eq('asa').to_numpy()).sum())
    assert int(raw.sum()) == full_n
    assert np.array_equal(raw.sum(0), np.bincount(y[ab], minlength=3))
    assert np.all(cc.sum(1)[used] > 0)
    eps = np.minimum(0.001, np.maximum(0., np.sort(np.asarray(zz[train_ids]), axis=1)[:, -1] -
                                      np.sort(np.asarray(zz[train_ids]), axis=1)[:, -2]) / 2)
    old_true = base[train_ids]
    pmass = cc[used, old_true].astype(np.float64)
    assert np.all(eps[pmass > 0] > 0)
    eligibility = read(ROOT / 'artifacts/v94_research_mechanism_20260928/auxiliary_eligibility.json')
    aux = pd.read_parquet(DEST / 'auxiliary_rows.parquet')
    assert set(aux.behavior) == set(eligibility['eligible_behaviors'])
    lookup = np.full(len(base), -1, dtype=np.int32); lookup[train_ids] = np.arange(len(train_ids))
    weights = {role: np.zeros((len(train_ids), 3), np.float64) for role in 'AB'}
    episodes = []
    for behavior in eligibility['eligible_behaviors']:
        item = {'behavior': behavior, 'roles': {}}
        for role in 'AB':
            for cls in [1, 2]:
                sl = aux[(aux.behavior == behavior) & (aux.role == role) & (aux.label_index == cls)]
                assert len(sl) > 0
                local = lookup[fid[sl.row_position.to_numpy()]]
                assert (local >= 0).all()
                np.add.at(weights[role][:, cls], local, 1. / (len(eligibility['eligible_behaviors']) * 2 * len(sl)))
                item['roles'][f'{role}_{cls}'] = {'rows': len(sl), 'components': int(sl.component.nunique())}
        episodes.append(item)
    for role in 'AB': assert np.isclose(weights[role].sum(), 1.)
    counts = {role: raw_counts(fid, y, roles == role, len(base)) for role in 'ABV'}
    counts.update({role: raw_counts(fid, y, r.fold.eq(k).to_numpy(), len(base)) for role, k in [('inner', 1), ('C', 2), ('H', 0)]})
    return dict(reg=reg, teacher=teacher, r=r, y=y, fid=fid, roles=roles, ids=ids, x=x,
                zz=zz, base=base, old=old_pred, old_z=old_z, cc=cc[used], used=used,
                train_ids=train_ids, full_n=full_n, pmass=pmass, epsilon=eps,
                aux_weights=weights, episodes=episodes, counts=counts)


def class_metrics(cm):
    diag = np.diag(cm); true = cm.sum(1); predicted = cm.sum(0)
    recall = np.divide(diag, true, out=np.zeros(3, float), where=true > 0)
    precision = np.divide(diag, predicted, out=np.zeros(3, float), where=predicted > 0)
    f1 = np.divide(2 * diag, true + predicted, out=np.zeros(3, float), where=true + predicted > 0)
    return dict(support=true.astype(int).tolist(), correct=diag.astype(int).tolist(),
                recall=recall.tolist(), precision=precision.tolist(), f1=f1.tolist(),
                errors=int(cm.sum() - diag.sum()), normal_false_alerts=int(cm[0, 1:].sum()),
                macro_f1=float(np.mean(f1)), confusion=cm.astype(int).tolist())


def run(arm):
    assert arm in ['E00', 'E10', 'E01', 'E11']
    assert not (DEST / f'{arm}_fit.json').exists()
    ctx = make_context()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(ctx['reg']['seed'])
    if device == 'cuda': torch.cuda.manual_seed_all(ctx['reg']['seed'])
    model = Branch(ctx['x'].shape[1]).to(device)
    initial_hash = state_hash(model)
    xt = csr(ctx['x'][ctx['used']], device)
    teacher = torch.as_tensor(np.asarray(ctx['zz'][ctx['train_ids']]), device=device, dtype=torch.float64)
    truth = torch.as_tensor(ctx['cc'], device=device, dtype=torch.float64)
    mass = torch.as_tensor(ctx['cc'].sum(1), device=device, dtype=torch.float64)
    protected_mass = torch.as_tensor(ctx['pmass'], device=device, dtype=torch.float64)
    teacher_label = torch.as_tensor(ctx['base'][ctx['train_ids']].astype(np.int64), device=device)
    epsilon = torch.as_tensor(ctx['epsilon'], device=device, dtype=torch.float64)
    aux = {s: torch.as_tensor(v, device=device, dtype=torch.float64) for s, v in ctx['aux_weights'].items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    history = []
    snapshots = []
    started = time.monotonic()
    teacher_v = class_metrics(cm_from_counts(ctx['counts']['V'], ctx['base']))
    def logits(network): return teacher + network(xt).double()
    def side_loss(z, side): return (-torch.log_softmax(z, 1) * aux[side]).sum()
    def snapshot(step):
        model.eval()
        with torch.no_grad():
            roi = logits(model)
            train_pred = roi.argmax(1).cpu().numpy().astype(np.int8)
            positive_margin = margin(roi, teacher_label)
            violation = (epsilon - positive_margin).clamp_min(0.)
            training_protection = {'negative_flips': int(ctx['pmass'][(train_pred != ctx['base'][ctx['train_ids']]) & (ctx['pmass'] > 0)].sum()),
                                   'violating_inputs': int(((violation > 1e-8) & (protected_mass > 0)).sum().item()),
                                   'max_margin_violation': float(violation[protected_mass > 0].max().item())}
            train_ce = float((-torch.log_softmax(roi, 1) * truth).sum().item() / int(truth.sum().item()))
            all_scores = np.empty((len(ctx['ids']), 3), np.float64)
            for start in range(0, len(ctx['ids']), 4096):
                end = min(start + 4096, len(ctx['ids']))
                all_scores[start:end] = np.asarray(ctx['zz'][ctx['ids'][start:end]]) + model(csr(ctx['x'][start:end], device)).double().cpu().numpy()
        pred = ctx['base'].copy(); pred[ctx['ids']] = all_scores.argmax(1).astype(np.int8)
        item = {'arm': arm, 'step': step, 'elapsed_seconds': time.monotonic() - started,
                'ASA_train_ce': train_ce, 'train_protection': training_protection,
                'roles': {}}
        for role, counts in ctx['counts'].items():
            base_stats = changes(counts, ctx['base'], pred)
            cm = np.asarray(base_stats['cm'])
            item['roles'][role] = {**class_metrics(cm), 'positive_flips': base_stats['positive_flips'],
                                  'negative_flips': base_stats['negative_flips'],
                                  'positive_flips_by_class': base_stats['positive_flips_by_class'],
                                  'negative_flips_by_class': base_stats['negative_flips_by_class']}
        v = item['roles']['V']
        guard = compare(np.asarray(teacher_v['confusion']), np.asarray(v['confusion']), True)
        item['V_eligibility'] = {'eligible': bool(guard['eligible'] and v['positive_flips'] > 0 and
                                                 v['negative_flips'] == 0 and training_protection['negative_flips'] == 0 and
                                                 training_protection['violating_inputs'] == 0),
                                 'guard': guard}
        filename = f'{arm}_step{step:03}'
        torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'width': ctx['x'].shape[1], 'seed': ctx['reg']['seed'], 'step': step, 'arm': arm}, DEST / (filename + '_model.pt'))
        np.save(DEST / (filename + '_prediction.npy'), pred)
        item['model_sha256'] = sha(DEST / (filename + '_model.pt'))
        item['prediction_sha256'] = sha(DEST / (filename + '_prediction.npy'))
        snapshots.append(item)
        save(DEST / (arm + '_snapshots.json'), snapshots)
        print(json.dumps({'stage': 'checkpoint', 'arm': arm, 'step': step,
                          'train_errors': item['roles']['A']['errors'] + item['roles']['B']['errors'],
                          'V_errors': v['errors'], 'V_M_correct': v['correct'][1], 'V_S_correct': v['correct'][2],
                          'V_positive': v['positive_flips'], 'V_negative': v['negative_flips'],
                          'P_negative': training_protection['negative_flips'], 'eligible': item['V_eligibility']['eligible'],
                          'seconds': round(item['elapsed_seconds'], 1)}), flush=True)
    snapshot(0)
    for step in range(1, 201):
        model.train(); optimizer.zero_grad(set_to_none=True)
        residual = model(xt).double()
        z = teacher + residual
        main = (-torch.log_softmax(z, 1) * truth).sum() / ctx['full_n']
        margin_violation = (epsilon - margin(z, teacher_label)).clamp_min(0.)
        compatibility = z.new_zeros(())
        for cl in range(3):
            w = protected_mass * (teacher_label == cl)
            if float(w.sum().item()) > 0:
                compatibility = compatibility + (w * margin_violation.square()).sum() / w.sum()
        compatibility = compatibility + margin_violation[protected_mass > 0].max().square()
        magnitude = (mass * residual.square().sum(1)).sum() / ctx['full_n']
        objective = main + .01 * compatibility
        if arm[1] == '1': objective = objective + .01 * magnitude
        objective.backward()
        meta = {'applied': False}
        if arm[2] == '1':
            inner_side, outer_side = ('A', 'B') if step % 2 else ('B', 'A')
            clone = copy.deepcopy(model)
            inner = side_loss(logits(clone), inner_side)
            inner.backward()
            with torch.no_grad():
                for p in clone.parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all()
                    p.add_(p.grad, alpha=-.01)
            clone.zero_grad(set_to_none=True)
            outer = side_loss(logits(clone), outer_side)
            outer.backward()
            with torch.no_grad():
                for p, q in zip(model.parameters(), clone.parameters()):
                    assert q.grad is not None and torch.isfinite(q.grad).all()
                    p.grad.add_(q.grad, alpha=.01)
            meta = {'applied': True, 'inner_role': inner_side, 'feedback_role': outer_side,
                    'inner_loss': float(inner.item()), 'feedback_loss_at_inner_updated_clone': float(outer.item()),
                    'method': 'first-order gradient, no Hessian'}
            del clone
        for p in model.parameters(): assert p.grad is not None and torch.isfinite(p.grad).all()
        optimizer.step()
        entry = {'step': step, 'main_loss_before_update': float(main.item()),
                 'compatibility_loss_before_update': float(compatibility.item()),
                 'magnitude_loss_before_update': float(magnitude.item()),
                 'objective_before_meta_gradient': float(objective.item()), 'meta': meta,
                 'seconds': time.monotonic() - started}
        history.append(entry)
        if step % 25 == 0:
            save(DEST / (arm + '_progress.json'), history)
            if step in CHECKPOINTS: snapshot(step)
    save(DEST / (arm + '_progress.json'), history)
    result = {'status': 'fit_executed', 'arm': arm, 'actual_classifier_fits': 1,
              'actual_calibration_fits': 0, 'teacher_model_sha256': ctx['teacher']['model_sha256'],
              'source_sha256': sha(__file__), 'initial_state_sha256': initial_hash,
              'steps_executed': 200, 'A_B_original_rows_supervised': ctx['full_n'],
              'A_B_ASA_original_rows_supervised': int(ctx['cc'].sum()),
              'A_B_all_M_S_supervised': True,
              'auxiliary_behavior_groups': ctx['episodes'], 'checkpoint_steps': CHECKPOINTS,
              'V_gradient_rows_used': 0, 'inner_C_H_gradient_rows_used': 0,
              'seconds': time.monotonic() - started,
              'snapshots': [s['step'] for s in snapshots],
              'eligible_checkpoints': [s['step'] for s in snapshots if s['V_eligibility']['eligible']],
              'quality_acceptance': False, 'model_promoted': False}
    save(DEST / (arm + '_fit.json'), result)
    print(json.dumps({'stage': 'arm_complete', 'arm': arm, 'seconds': result['seconds'],
                      'eligible_checkpoints': result['eligible_checkpoints']}), flush=True)
    del model, xt
    if device == 'cuda': torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=['E00', 'E10', 'E01', 'E11'])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.arm)
