"""V113 matched N1 versus ASA interface-case canonicalization, 25 epochs."""
import argparse
import importlib.metadata
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from run_v75 import ROOT, save, sha
from v75_views import BYTE_FEATURES, byte_matrix, matrix_hashes
from v104_phase_b import BATCH, DEVICE, FID, K, PREP, ROWS, SEED, WIDTH, SparseTabM, csr_tensor, predict_all
from v107_matched_training import DEST as OLD, FOLDS, check as check_old

DEST = ROOT/'artifacts/v113_case_training_20260929'
AUDIT = ROOT/'artifacts/v112_fine_supervision_review_20260929'
VIEW = DEST/'case_ASA.npz'


def rel(p):
    return p.relative_to(ROOT).as_posix()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def register():
    assert not DEST.exists()
    check_old()
    assert read(AUDIT/'verification.json')['all_checks_passed']
    assert read(AUDIT/'factor_audit.json')['variants'][0]['observed_floor']['observed_min_errors'] == 26
    baseline = []
    old_reg = read(OLD/'registration.json')
    for fold in range(3):
        p = OLD/f'fold{fold}_N1_TabM25_seed{SEED}'
        f = read(p/'fit.json')
        assert f['epochs'] == 25 and f['seed'] == SEED and f['view'] == 'N1'
        assert f['model_sha256'] == sha(p/'model.pt')
        assert f['prob_sha256'] == sha(p/'ASA_input_prob.npy')
        baseline.append({'fold': fold, 'fit': f, 'fit_sha256': sha(p/'fit.json')})
    view = pd.read_parquet(AUDIT/'interface_views.parquet').sort_values('local')
    original = sparse.load_npz(PREP/'N1_ASA.npz')
    assert original.shape == (22546, WIDTH)
    assert np.array_equal(view.local, np.arange(len(view)))
    changed = 0
    strings = []
    for row in view.itertuples(index=False):
        s = row.before
        spans = json.loads(row.name_spans)
        if not row.interface_parse_verified:
            assert not spans
        else:
            assert len(spans) == 2
            for a,b,role in reversed(spans):
                token = s[a:b]
                assert token and role in ('src','dst')
                s = s[:a] + token.lower() + s[b:]
        strings.append(s)
        changed += s != row.before
    assert changed == 11273
    matrix = sparse.hstack([byte_matrix(strings), original[:, BYTE_FEATURES:]], format='csr')
    assert (matrix[:, BYTE_FEATURES:] != original[:, BYTE_FEATURES:]).nnz == 0
    factors = pd.read_parquet(AUDIT/'factor_hashes.parquet', columns=['local','lowercase_only_keep_suffix'])
    assert factors.groupby('local').lowercase_only_keep_suffix.nunique().max() == 1
    expected = factors.drop_duplicates('local').sort_values('local').lowercase_only_keep_suffix.to_numpy()
    assert np.array_equal(np.asarray(matrix_hashes(matrix), dtype=object), expected)
    # No new equal-input leakage; the 3 outer roles and full original row
    # frequencies can be reused exactly as registered in V107.
    r = pd.read_parquet(ROWS, columns=['route', 'label_index'])
    f = pd.read_parquet(FOLDS, columns=['proposed_fold', 'root'])
    fid = np.load(FID, mmap_mode='r')
    ids = np.load(ROOT/'artifacts/v92_evidence_training_20260928/ASA_input_ids.npy')
    pos = np.flatnonzero(r.route.eq('asa').to_numpy())
    lookup = np.full(int(fid.max())+1, -1, np.int32)
    lookup[ids] = np.arange(len(ids))
    local = lookup[fid[pos]]
    assert len(pos) == 112807 and (local >= 0).all()
    hashes = np.asarray(matrix_hashes(matrix), dtype=object)[local]
    role = f.proposed_fold.to_numpy()[pos]
    assert pd.DataFrame({'h':hashes, 'role':role}).groupby('h').role.nunique().max() == 1
    assert old_reg['fold_ASA_class_rows'] == {str(k): np.bincount(r.label_index.to_numpy()[pos][role==k], minlength=3).tolist() for k in range(3)}
    DEST.mkdir()
    sparse.save_npz(VIEW, matrix, compressed=True)
    input_paths = [PREP/'N1_ASA.npz', AUDIT/'verification.json', AUDIT/'factor_hashes.parquet',
        AUDIT/'interface_views.parquet', ROWS, FOLDS, FID,
        ROOT/'artifacts/v92_evidence_training_20260928/ASA_input_ids.npy',
        ROOT/'data/official/train.parquet', OLD/'registration.json', OLD/'evaluation.json',
        ROOT/'training/v104_phase_b.py', ROOT/'training/v107_matched_training.py']
    reg = {'status': 'registered_before_any_new_fit', 'classifier_fits_at_registration': 0,
        'source_sha256': sha(__file__),
        'evaluation_source_sha256': sha(ROOT/'training/v113_case_evaluate.py'),
        'input_sha256': {rel(p): sha(p) for p in input_paths},
        'view_sha256': sha(VIEW), 'baseline': baseline, 'actual_baseline_fits_reused': 3,
        'planned_new_classifier_fits': 3, 'calibration_fits': 0,
        'view': 'lowercase only in verified ASA src/dst interface name spans; preserve suffixes, all other text and facts',
        'changed_unique_ASA_inputs': changed,
        'no_new_crossfold_equal_input': True,
        'old_fold_ASA_class_rows': old_reg['fold_ASA_class_rows'],
        'model': 'V107 SparseTabM25, original row-frequency 3-class CE',
        'environment': {'python':__import__('sys').version, 'torch':torch.__version__,
            'tabm':importlib.metadata.version('tabm'), 'device':DEVICE}}
    save(DEST/'registration.json', reg)
    emit(stage='registered', view=reg['view'], new_fits=3, reused_fits=3, changed_inputs=changed)


def check():
    check_old()
    reg = read(DEST/'registration.json')
    assert reg['status'] == 'registered_before_any_new_fit'
    assert sha(__file__) == reg['source_sha256']
    assert sha(ROOT/'training/v113_case_evaluate.py') == reg['evaluation_source_sha256']
    assert sha(VIEW) == reg['view_sha256']
    for path, h in reg['input_sha256'].items():
        assert sha(ROOT/path) == h, path
    for b in reg['baseline']:
        fold = b['fold']; p = OLD/f'fold{fold}_N1_TabM25_seed{SEED}'
        assert sha(p/'fit.json') == b['fit_sha256']
        assert sha(p/'model.pt') == b['fit']['model_sha256']
        assert sha(p/'ASA_input_prob.npy') == b['fit']['prob_sha256']
    return reg


def fit(fold):
    reg = check()
    assert fold in (0,1,2)
    folder = DEST/f'fold{fold}_case_TabM25_seed{SEED}'
    assert not folder.exists()
    r = pd.read_parquet(ROWS, columns=['route','label_index'])
    f = pd.read_parquet(FOLDS, columns=['proposed_fold'])
    fid = np.load(FID, mmap_mode='r')
    ids = np.load(ROOT/'artifacts/v92_evidence_training_20260928/ASA_input_ids.npy')
    pos = np.flatnonzero(r.route.eq('asa').to_numpy())
    lookup = np.full(int(fid.max())+1, -1, np.int32)
    lookup[ids] = np.arange(len(ids))
    local = lookup[fid[pos]]
    y = r.label_index.to_numpy(dtype=np.int8)[pos]
    train = f.proposed_fold.to_numpy()[pos] != fold
    x = sparse.load_npz(VIEW)
    counts = np.bincount(local[train].astype(np.int64)*3+y[train], minlength=x.shape[0]*3).reshape(-1,3)
    assert counts.sum(0).tolist() == (np.bincount(y, minlength=3)-
        np.asarray(reg['old_fold_ASA_class_rows'][str(fold)])).tolist()
    assert int(counts.sum()) == int(train.sum())
    used = np.flatnonzero(counts.sum(1))
    # This equality makes RNG permutations and minibatch membership match N1.
    old = sparse.load_npz(PREP/'N1_ASA.npz')
    old_counts = np.bincount(local[train].astype(np.int64)*3+y[train], minlength=old.shape[0]*3).reshape(-1,3)
    assert np.array_equal(counts, old_counts)
    cc = counts[used].astype(np.float32)
    xtrain = x[used]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = SparseTabM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0003)
    folder.mkdir()
    started = time.monotonic()
    save(folder/'started.json', {'status':'started', 'fold':fold, 'view':'case',
        'seed':SEED, 'train_ASA_original_rows':int(train.sum()),
        'train_ASA_classes':counts.sum(0).astype(int).tolist(),
        'heldout_gradient_rows':0, 'train_unique_input_ids':len(used),
        'batch_size':BATCH, 'epochs':25, 'source_sha256':sha(__file__)})
    batches = math.ceil(len(used)/BATCH)
    normalizer = float(train.sum())/batches
    rng = np.random.default_rng(SEED+fold)
    history=[]
    for epoch in range(1,26):
        model.train()
        order = rng.permutation(len(used))
        numerator_total=0.
        for start in range(0,len(order),BATCH):
            batch=order[start:start+BATCH]
            block=xtrain[batch]
            fact=torch.as_tensor(block[:,BYTE_FEATURES:].toarray(),device=DEVICE,dtype=torch.float32)
            mass=torch.as_tensor(cc[batch],device=DEVICE,dtype=torch.float32)
            z=model(csr_tensor(block,DEVICE),fact)
            numerator=-(torch.log_softmax(z,-1)*mass[:,None,:]).sum()/K
            loss=numerator/normalizer
            assert bool(torch.isfinite(loss).item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward();optimizer.step()
            numerator_total+=float(numerator.detach().item())
        history.append({'epoch':epoch, 'online_original_row_CE':numerator_total/float(train.sum()),
            'seconds':time.monotonic()-started})
        if epoch==1 or epoch%5==0:
            save(folder/'progress.json',history)
            emit(stage='training_progress',fold=fold,epoch=epoch,
                online_CE=round(history[-1]['online_original_row_CE'],6),
                seconds=round(history[-1]['seconds'],1))
    prob=predict_all(model,'TabM',x,DEVICE)
    np.save(folder/'ASA_input_prob.npy',prob)
    torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'view':'case','fold':fold,'seed':SEED,'epoch':25,'source_sha256':sha(__file__)}, folder/'model.pt')
    train_pred=prob[used].argmax(1)
    train_errors=int((counts[used].sum(1)-counts[used,train_pred]).sum())
    train_by_class={str(c):int((counts[used,c]*(train_pred!=c)).sum()) for c in (1,2)}
    report={'status':'fit_executed','fold':fold,'view':'case','arm':'TabM25','seed':SEED,
        'classifier_fits':1,'train_ASA_original_rows':int(train.sum()),
        'train_ASA_classes':counts.sum(0).astype(int).tolist(),
        'heldout_gradient_rows':0,'epochs':25,'optimizer':'AdamW','lr':.002,
        'weight_decay':.0003,'batch_size':BATCH,'K':K,
        'train_original_errors':train_errors,'train_errors_by_class':train_by_class,
        'source_sha256':sha(__file__),'seconds':time.monotonic()-started,
        'model_sha256':sha(folder/'model.pt'),
        'prob_sha256':sha(folder/'ASA_input_prob.npy')}
    save(folder/'fit.json',report)
    emit(stage='training_complete',fold=fold,seconds=round(report['seconds'],1),
        train_errors=train_errors, train_errors_by_class=train_by_class)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['register','fit','check'])
    parser.add_argument('--fold',type=int)
    a=parser.parse_args()
    if a.stage=='register':register()
    elif a.stage=='fit':fit(a.fold)
    else:emit(stage='check',status=check()['status'])
