"""Zero optimizer-step diagnosis of historical baseline drift and changed rows."""
import json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from v116_preflight import ROOT, DEST, VIEW, save, sha
from v104_phase_b import BATCH, DEVICE, FID, ROWS, K, SEED, SparseTabM, csr_tensor
from v75_views import BYTE_FEATURES
from v107_matched_training import FOLDS, ASA_IDS, DEST as OLD, check


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    check()
    manifest=pd.read_parquet(DEST/'inner_split_manifest.parquet')
    d=manifest[manifest.outer_fold.eq(0)].reset_index(drop=True)
    r=pd.read_parquet(ROWS,columns=['route','label_index'])
    f=pd.read_parquet(FOLDS,columns=['proposed_fold'])
    fid=np.load(FID,mmap_mode='r');ids=np.load(ASA_IDS)
    lookup=np.full(int(fid.max())+1,-1,np.int32);lookup[ids]=np.arange(len(ids))
    pos=np.flatnonzero(r.route.eq('asa'));local=lookup[fid[pos]]
    y=r.label_index.to_numpy()[pos];folds=f.proposed_fold.to_numpy()[pos]
    assert np.array_equal(local,d.local) and np.array_equal(y,d.truth) and np.array_equal(folds,d.fold)
    x=sparse.load_npz(VIEW); histories=[]
    for fold in range(3):
        train=folds!=fold
        old_counts=np.bincount(local[train].astype(np.int64)*3+y[train],minlength=x.shape[0]*3).reshape(-1,3)
        tr=d.fold.ne(fold)
        new_counts=np.bincount(d.loc[tr,'local'].to_numpy()*3+d.loc[tr,'truth'].to_numpy(),minlength=x.shape[0]*3).reshape(-1,3)
        assert np.array_equal(old_counts,new_counts)
        previous=read(OLD/f'fold{fold}_N1_TabM25_seed10201/progress.json')
        current=read(DEST/f'outer_fold{fold}/progress.json')
        histories.append({'fold':fold,'counts_equal':True,
            'first_epoch_CE_old':previous[0]['online_original_row_CE'],'first_epoch_CE_new':current[0]['online_row_CE'],
            'last_epoch_CE_old':previous[-1]['online_original_row_CE'],'last_epoch_CE_new':current[-1]['online_row_CE']})
    # Same initialized model, exact same first training minibatch, three backward
    # calls without any optimizer or parameter update. No additional fitting.
    train=folds!=0
    counts=np.bincount(local[train].astype(np.int64)*3+y[train],minlength=x.shape[0]*3).reshape(-1,3)
    used=np.flatnonzero(counts.sum(1));batch=np.random.default_rng(SEED).permutation(len(used))[:BATCH]
    block=x[used][batch]
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=SparseTabM().to(DEVICE)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    fact=torch.as_tensor(block[:,BYTE_FEATURES:].toarray(),device=DEVICE,dtype=torch.float32)
    mass=torch.as_tensor(counts[used][batch],device=DEVICE,dtype=torch.float32)
    reference=None;outputs=None;repeats=[]
    for rep in range(3):
        model.zero_grad(set_to_none=True)
        z=model(csr_tensor(block,DEVICE),fact)
        loss=-(torch.log_softmax(z,-1)*mass[:,None,:]).sum()/K
        loss.backward()
        grads={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters()}
        logits=z.detach().cpu().clone()
        if reference is None:
            reference=grads;outputs=logits
        else:
            repeats.append({'repeat':rep,'same_logits_bitwise':torch.equal(outputs,logits),
                'max_logit_abs_difference':float((outputs-logits).abs().max()),
                'gradient_differences':{k:{'changed_coordinates':int((g!=reference[k]).sum()),
                    'max_absolute_difference':float((g-reference[k]).abs().max())} for k,g in grads.items() if not torch.equal(g,reference[k])}})
    assert all(torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items())
    decisions=pd.read_parquet(DEST/'OOF_ASA_decisions.parquet')
    changes=decisions[decisions.A_epoch25!=decisions.B_selected].groupby(['root','truth']).agg(
        rows=('local','size'),repaired=('repaired','sum'),regressed=('regressed','sum')).reset_index()
    out={'status':'zero_fit_drift_and_error_audit','new_classifier_fits':0,'optimizer_steps':0,
        'population_and_history':histories,'same_minibatch_repeat':repeats,
        'unchanged_parameters_verified':True,'cuda_deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),
        'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name() if DEVICE=='cuda' else None,
        'changed_source_groups':changes.to_dict('records'),
        'interpretation':'Same raw training counts and algorithm settings verified. Any repeated-gradient differences demonstrate numerical nondeterminism in this runtime, but one minibatch does not causally explain every historical decision change. Use the fresh paired A for B attribution.',
        'source_sha256':sha(__file__),'evaluation_sha256':sha(DEST/'evaluation.json')}
    save(DEST/'baseline_drift_audit.json',out)
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
