"""Conditional v10.4 phase B: N1 teacher and matched independent ASA networks.

Every real fit uses the full official outer-train population.  Model selection
uses only fixed 25/50/100 checkpoints after all folds finish.  The sparse
BatchEnsemble first layer is a custom adapter, not a claim of reproducing the
TabM paper.  Run ``selftest`` before registration and fitting.
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
from torch import nn
from threadpoolctl import threadpool_limits
import tabm

from run_v75 import ROOT, OUT, load_sparse, read, save, sha
from v89_common import LAST
from v85_protection import baseline_objective
from v75_views import BYTE_FEATURES
from v104_phase_a import DEST as ADEST, FOLDS, OFFICIAL, PLAN, PREFLIGHT, ROWS, FID, X, load_rows


DEST = ROOT / 'artifacts/v104_phase_b_20260928'
PREP = ROOT / 'artifacts/v101_full_input_group_n1_20260928'
ASA_IDS = ROOT / 'artifacts/v92_evidence_training_20260928/ASA_input_ids.npy'
WIDTH = 66287
HIDDEN = 128
K = 16
SEED = 10201
BATCH = 256
CHECKPOINTS = (25, 50, 100)
FACTS = WIDTH - BYTE_FEATURES
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def emit(**x):
    print(json.dumps(x, ensure_ascii=False), flush=True)


def csr_tensor(x, device):
    x = x.tocsr()
    return torch.sparse_csr_tensor(
        torch.as_tensor(x.indptr.astype(np.int64), device=device),
        torch.as_tensor(x.indices.astype(np.int64), device=device),
        torch.as_tensor(x.data.astype(np.float32), device=device),
        size=x.shape, device=device)


class SparseFirstBatchEnsemble(nn.Module):
    """Exact dense BatchEnsemble formula, evaluated through sparse x @ W_eff."""
    def __init__(self, d_in=WIDTH, d_out=HIDDEN, k=K):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.r = nn.Parameter(torch.empty(k, d_in))
        self.s = nn.Parameter(torch.ones(k, d_out))
        self.bias = nn.Parameter(torch.empty(k, d_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.r)
        bound = 1 / math.sqrt(d_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        d_out, d_in = self.weight.shape
        k = self.r.shape[0]
        effective = (self.weight.T[:, None, :] * self.r.T[:, :, None]).reshape(d_in, k * d_out)
        y = torch.sparse.mm(x, effective).reshape(x.shape[0], k, d_out)
        return y * self.s + self.bias


class IndependentMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(WIDTH, HIDDEN)
        self.second = nn.Linear(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, 3)
        self.facts_direct = nn.Linear(FACTS, 3, bias=False)

    def forward(self, x, facts):
        h = torch.sparse.mm(x, self.first.weight.T) + self.first.bias
        h = torch.relu(h)
        h = torch.relu(self.second(h))
        return self.head(h) + self.facts_direct(facts)


class SparseTabM(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = SparseFirstBatchEnsemble()
        self.second = tabm.LinearBatchEnsemble(HIDDEN, HIDDEN, k=K, scaling_init='ones')
        self.head = tabm.LinearEnsemble(HIDDEN, 3, k=K)
        self.facts_direct = nn.Linear(FACTS, 3, bias=False)

    def forward(self, x, facts):
        h = torch.relu(self.first(x))
        h = torch.relu(self.second(h))
        return self.head(h) + self.facts_direct(facts)[:, None, :]


def selftest():
    rng = np.random.default_rng(3001)
    dense = rng.normal(size=(5, 13)).astype('f4')
    dense[rng.random(dense.shape) < .65] = 0
    x = sparse.csr_matrix(dense)
    for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
        torch.manual_seed(SEED)
        model = SparseFirstBatchEnsemble(13, 7, 3).to(device)
        a = csr_tensor(x, device)
        b = torch.as_tensor(dense, device=device)
        output = model(a)
        reference = ((b[:, None, :] * model.r[None, :, :]) @ model.weight.T) * model.s + model.bias
        assert torch.allclose(output, reference, atol=2e-5, rtol=2e-5), device
        sparse_grads = torch.autograd.grad(output.square().mean(), tuple(model.parameters()), retain_graph=True)
        dense_grads = torch.autograd.grad(reference.square().mean(), tuple(model.parameters()))
        for lhs, rhs in zip(sparse_grads, dense_grads):
            assert torch.allclose(lhs, rhs, atol=2e-5, rtol=2e-5), (device, (lhs-rhs).abs().max())
        assert float(model.r.detach().std(dim=0).mean()) > .1
    emit(stage='adapter_selftest', cpu=True, cuda=torch.cuda.is_available(),
         forward_and_gradient_match=True, distinct_members=True)


def registration():
    if DEST.exists():
        raise FileExistsError(DEST)
    phase_a = read(ADEST / 'phase_A_evaluation.json')
    ext = read(ADEST / 'phase_A_extended_diagnosis.json')
    assert phase_a['classifier_fits'] == 6 and phase_a['historic_113']['newly_supported_101']['full_correct'] == 0
    assert phase_a['full']['errors'] >= phase_a['small']['errors']
    assert ext['full_task_OOF']['full']['errors'] < ext['full_task_OOF']['small']['errors']
    selftest()
    plan = read(PLAN)
    assert plan['version'] == 'v103' and plan['phase_B']['fits'] == 9
    assert plan['phase_B']['candidate_input']['ensemble_first_layer'].startswith('Create distinct')
    assert plan['phase_B']['common_candidate_config']['batch_size'] == BATCH
    assert plan['phase_B']['common_candidate_config']['checkpoints_epochs'] == list(CHECKPOINTS)
    inputs = [PLAN, PREFLIGHT, FOLDS, ROWS, FID, OFFICIAL, ASA_IDS,
              PREP / 'N1_ASA.npz', PREP / 'N1_delta.npz',
              PREP / 'source_variant_audit.json',
              ADEST / 'phase_A_evaluation.json', ADEST / 'phase_A_extended_diagnosis.json',
              ROOT / 'training/v85_protection.py', ROOT / 'training/v104_phase_a.py',
              ROOT / 'training/v104_phase_b_evaluate.py']
    inputs += [Path(str(X)+ext) for ext in ('.json', '.data', '.indices', '.indptr')]
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    tabm_source = Path(tabm.__file__)
    reg = {'status': 'v104_phase_B_registered_before_fits',
           'phase_A_result': {'ASA_errors_small': phase_a['small']['errors'],
                              'ASA_errors_full': phase_a['full']['errors'],
                              'newly_supported_101_full_correct': 0,
                              'full_task_errors_small': ext['full_task_OOF']['small']['errors'],
                              'full_task_errors_full': ext['full_task_OOF']['full']['errors']},
           'remaining_question': 'Can N1 and independent MLP/TabM recover S across groups without M negative flips?',
           'source_sha256': sha(__file__), 'input_sha256': bindings,
           'tabm_version': importlib.metadata.version('tabm'),
           'tabm_source_sha256': sha(tabm_source),
           'pytorch_version': torch.__version__, 'device': DEVICE,
           'same_split_all_arms': True, 'original_labels_unchanged': True,
           'candidate_loss': 'original-row count-weighted three-class CE, equal row weights; mean per member for TabM',
           'candidate_minibatch': BATCH,
           'candidate_minibatch_normalizer': 'global training row count divided by number of batches; original row frequencies retained',
           'candidate_epochs': list(CHECKPOINTS), 'max_epochs': 100,
           'fit_order': ['N1_teacher_fold0..2', 'C_MLP_fold0..2', 'C_TabM_fold0..2'],
           'holdout_gradient_rows': 0,
           'tabm_custom_sparse_adapter': True,
           'tabm_is_official_package_backbone_not_paper_reproduction': True,
           'full_coordinate_width': WIDTH, 'fact_direct_width': FACTS,
           'checkpoint_selection': plan['phase_B']['common_candidate_config']['checkpoint_selection'],
           'model_promoted': False, 'created_unix': time.time()}
    DEST.mkdir()
    save(DEST / 'registration.json', reg)
    emit(stage='phase_B_registered', question=reg['remaining_question'],
         tabm_version=reg['tabm_version'], device=DEVICE)


def check():
    reg = read(DEST / 'registration.json')
    assert reg['status'] == 'v104_phase_B_registered_before_fits'
    assert sha(__file__) == reg['source_sha256']
    for rel, digest in reg['input_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    assert importlib.metadata.version('tabm') == reg['tabm_version']
    assert sha(Path(tabm.__file__)) == reg['tabm_source_sha256']
    return reg


def teacher(fold):
    check()
    assert fold in (0, 1, 2)
    folder = DEST / f'fold{fold}_N1_teacher'
    if folder.exists():
        raise FileExistsError(folder)
    f, r, fid = load_rows()
    train = f.proposed_fold.to_numpy() != fold
    y = r.label_index.to_numpy(dtype=np.int8)
    counts = np.bincount(np.asarray(fid[train], dtype=np.int64)*3+y[train],
                         minlength=457566*3).reshape(-1, 3)
    assert int(counts.sum()) == int(train.sum())
    used = np.flatnonzero(counts.sum(1))
    x = load_sparse(X)
    delta = sparse.load_npz(PREP / 'N1_delta.npz')
    assert delta.shape == x.shape
    feat = x[used] + delta[used]
    weights = counts[used].astype(np.float64)
    w = np.zeros((WIDTH+1, 3), np.float64)
    w[-1] = np.log(np.maximum(weights.sum(0)/weights.sum(), 1e-9))
    folder.mkdir()
    started = time.monotonic()
    save(folder / 'started.json', {'status': 'started', 'fold': fold,
        'train_original_rows': int(train.sum()), 'class_counts': counts.sum(0).astype(int).tolist(),
        'heldout_gradient_rows': 0, 'source_sha256': sha(__file__)})
    progress = []
    def callback(theta):
        progress.append({'iteration': len(progress)+1, 'seconds': time.monotonic()-started})
        if len(progress) % 25 == 0:
            loss, grad = baseline_objective(theta, feat, weights)
            emit(stage='N1_teacher_progress', fold=fold, iteration=len(progress),
                 objective=loss, gradient_inf=float(np.abs(grad).max()),
                 seconds=round(time.monotonic()-started, 1))
    with threadpool_limits(limits=4):
        result = minimize(baseline_objective, w.ravel(), args=(feat, weights),
                          jac=True, method='L-BFGS-B', callback=callback,
                          options={'maxiter':1000,'maxcor':10,'gtol':1e-6,'ftol':1e-12,'maxls':30})
    ww = result.x.reshape(WIDTH+1,3)
    model = {'coef': ww[:-1].copy(), 'intercept': ww[-1].copy()}
    joblib.dump(model, folder / 'teacher.joblib')
    # Full unique-input scores retain the same all-format teacher output
    # interface as phase A. N1 changes only rows with nonzero delta.
    scores = np.empty((x.shape[0], 3), np.float64)
    for start in range(0, x.shape[0], 8192):
        stop = min(start+8192, x.shape[0])
        scores[start:stop] = np.asarray((x[start:stop]+delta[start:stop])@model['coef'])+model['intercept']
    np.save(folder / 'scores_all_input_ids.npy', scores)
    save(folder / 'progress.json', progress)
    report = {'status':'N1_teacher_fit_executed', 'classifier_fits':1,
        'fold':fold, 'train_original_rows':int(train.sum()),
        'train_class_rows':counts.sum(0).astype(int).tolist(),
        'heldout_gradient_rows':0,
        'converged':bool(result.success and np.abs(result.jac).max() <= 1e-5),
        'iterations':int(result.nit), 'objective':float(result.fun),
        'gradient_inf':float(np.abs(result.jac).max()), 'message':str(result.message),
        'seconds':time.monotonic()-started,
        'source_sha256':sha(__file__),
        'model_sha256':sha(folder/'teacher.joblib'),
        'scores_sha256':sha(folder/'scores_all_input_ids.npy')}
    save(folder / 'fit.json', report)
    emit(stage='N1_teacher_complete', fold=fold, converged=report['converged'],
         iterations=report['iterations'], seconds=round(report['seconds'],1))
    if not report['converged']:
        raise RuntimeError('N1 teacher numerical convergence gate failed')


def local_asa():
    r = pd.read_parquet(ROWS, columns=['route','label_index'])
    f = pd.read_parquet(FOLDS, columns=['proposed_fold'])
    fid = np.load(FID, mmap_mode='r')
    ids = np.load(ASA_IDS)
    x = sparse.load_npz(PREP / 'N1_ASA.npz')
    assert x.shape == (len(ids), WIDTH) and len(ids) == 22546
    positions = np.flatnonzero(r.route.eq('asa').to_numpy())
    lookup = np.full(457566, -1, np.int32)
    lookup[ids] = np.arange(len(ids))
    local = lookup[fid[positions]]
    assert (local >= 0).all()
    return x, positions, local, f.proposed_fold.to_numpy()[positions], r.label_index.to_numpy()[positions]


@torch.no_grad()
def predict_all(model, arm, x, device):
    model.eval()
    scores = np.empty((x.shape[0], 3), np.float32)
    for start in range(0, len(scores), 512):
        stop = min(start+512, len(scores))
        block = x[start:stop]
        fact = torch.as_tensor(block[:, BYTE_FEATURES:].toarray(), device=device, dtype=torch.float32)
        z = model(csr_tensor(block, device), fact)
        prob = torch.softmax(z, -1)
        if arm == 'TabM':
            prob = prob.mean(dim=1)
        scores[start:stop] = prob.float().cpu().numpy()
    assert np.isfinite(scores).all()
    return scores


def candidate(fold, arm):
    check()
    assert fold in (0,1,2) and arm in ('MLP','TabM')
    folder = DEST / f'fold{fold}_C_{arm}'
    if folder.exists():
        raise FileExistsError(folder)
    x, positions, local, source_fold, y = local_asa()
    train = source_fold != fold
    counts = np.bincount(local[train].astype(np.int64)*3+y[train],
                         minlength=x.shape[0]*3).reshape(-1,3)
    assert int(counts.sum()) == int(train.sum())
    assert counts.sum(0)[0] == 0 and (counts.sum(0)[1:] > 0).all()
    used = np.flatnonzero(counts.sum(1))
    xtrain = x[used]
    cc = counts[used].astype(np.float32)
    device = DEVICE
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = (IndependentMLP() if arm=='MLP' else SparseTabM()).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    folder.mkdir()
    started = time.monotonic()
    save(folder/'started.json', {'status':'started','fold':fold,'arm':arm,
        'source_sha256':sha(__file__),'train_ASA_original_rows':int(train.sum()),
        'train_ASA_classes':counts.sum(0).astype(int).tolist(),
        'heldout_gradient_rows':0,'train_unique_R0_ids':len(used),
        'train_N1_input_rows':len(used),
        'batch_size':BATCH,'max_epochs':100,'checkpoints':list(CHECKPOINTS),
        'seed':SEED,'device':device,'fact_direct':True})
    batches = math.ceil(len(used)/BATCH)
    normalizer = float(train.sum()) / batches
    history = []
    rng = np.random.default_rng(SEED+fold)
    for epoch in range(1,101):
        model.train()
        order = rng.permutation(len(used))
        online_numerator = 0.
        for start in range(0,len(order),BATCH):
            batch = order[start:start+BATCH]
            block = xtrain[batch]
            fact = torch.as_tensor(block[:, BYTE_FEATURES:].toarray(),device=device,dtype=torch.float32)
            mass = torch.as_tensor(cc[batch],device=device,dtype=torch.float32)
            z = model(csr_tensor(block,device),fact)
            if arm=='TabM':
                numerator = -(torch.log_softmax(z,-1)*mass[:,None,:]).sum()/K
            else:
                numerator = -(torch.log_softmax(z,-1)*mass).sum()
            loss = numerator / normalizer
            assert bool(torch.isfinite(loss).item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            online_numerator += float(numerator.detach().item())
        record={'epoch':epoch,'online_original_row_CE':online_numerator/float(train.sum()),
                'seconds':time.monotonic()-started}
        if epoch in CHECKPOINTS:
            prob = predict_all(model,arm,x,device)
            np.save(folder/f'epoch{epoch}_ASA_input_prob.npy',prob)
            torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'arm':arm,'fold':fold,'epoch':epoch,'source_sha256':sha(__file__)},
                       folder/f'epoch{epoch}_model.pt')
            train_pred = prob[used].argmax(1)
            record['checkpoint_train_original_errors'] = int((counts[used].sum(1)-counts[used,np.asarray(train_pred)]).sum())
            record['checkpoint_score_sha256'] = sha(folder/f'epoch{epoch}_ASA_input_prob.npy')
            record['checkpoint_model_sha256'] = sha(folder/f'epoch{epoch}_model.pt')
            emit(stage='candidate_checkpoint',fold=fold,arm=arm,epoch=epoch,
                 train_errors=record['checkpoint_train_original_errors'],
                 seconds=round(record['seconds'],1))
        history.append(record)
        if epoch==1 or epoch%10==0:
            save(folder/'progress.json',history)
            emit(stage='candidate_progress',fold=fold,arm=arm,epoch=epoch,
                 online_CE=round(record['online_original_row_CE'],6),
                 seconds=round(record['seconds'],1))
    save(folder/'progress.json',history)
    report={'status':'independent_candidate_fit_executed','classifier_fits':1,
        'fold':fold,'arm':arm,'train_ASA_original_rows':int(train.sum()),
        'train_ASA_classes':counts.sum(0).astype(int).tolist(),
        'heldout_gradient_rows':0,'epochs':100,'optimizer':'AdamW',
        'lr':.002,'weight_decay':.0003,'batch_size':BATCH,
        'K':K if arm=='TabM' else 1,
        'source_sha256':sha(__file__),'seconds':time.monotonic()-started,
        'checkpoint_sha256':{str(e):{'model':sha(folder/f'epoch{e}_model.pt'),
                                    'prob':sha(folder/f'epoch{e}_ASA_input_prob.npy')}
                             for e in CHECKPOINTS}}
    save(folder/'fit.json',report)
    emit(stage='candidate_complete',fold=fold,arm=arm,seconds=round(report['seconds'],1))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['selftest','register','teacher','candidate'])
    p.add_argument('--fold',type=int)
    p.add_argument('--arm',choices=['MLP','TabM'])
    args=p.parse_args()
    if args.stage=='selftest':selftest()
    elif args.stage=='register':registration()
    elif args.stage=='teacher':teacher(args.fold)
    else:candidate(args.fold,args.arm)
