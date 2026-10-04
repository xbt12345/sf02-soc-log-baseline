"""V125 registered A/B/C source-closed training. No outer-answer selection."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from v125_experiment_review import ROOT,sha,read,review_plan,seal_run,require_run_seal,require_checkpoint
from v125_model import Expert,probabilities
from v124_train import counts_for_fold
from v104_phase_b import K,DEVICE
from v107_matched_training import FOLDS,ROWS,FID,DEST as TEACHERS
from v116_preflight import VIEW

PLAN=ROOT/'training/review_policy/v125_next_training_plan.json'
DEST=ROOT/'artifacts/v125_order_trial_20260929'
MANIFEST=ROOT/'artifacts/v116_nested_selection_20260929/inner_split_manifest.parquet'
OFFICIAL=ROOT/'data/official/train.parquet'
TRACE=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
PREFLIGHT=DEST/'input_preflight.json'
ORDER=[(0,'A'),(0,'B'),(0,'C'),(1,'C'),(1,'A'),(1,'B'),(2,'B'),(2,'C'),(2,'A')]
CHECKPOINTS=(1,2,5,10,15,20,25)


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def tensor_hash(state):
    h=hashlib.sha256()
    for key,val in state.items():
        h.update(key.encode());h.update(val.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def body_for_arm(arm):
    if arm=='A':return None
    return np.load(DEST/('ordered_body_bytes.npy' if arm=='B' else 'shuffled_body_bytes.npy'),mmap_mode='r')


def sources():
    paths={Path(__file__),PLAN,VIEW,MANIFEST,OFFICIAL,TRACE,ROWS,FOLDS,FID,
           PREFLIGHT,DEST/'body_span_ledger.parquet',DEST/'body_lengths.npy',
           DEST/'ordered_body_bytes.npy',DEST/'shuffled_body_bytes.npy',
           ROOT/'training/v125_evaluate.py',ROOT/'training/v125_model.py',
           ROOT/'training/v125_input.py',ROOT/'training/v125_experiment_review.py',
           ROOT/'training/review_policy/v125_risk_actions.json',
           ROOT/'artifacts/v124_header_trial_20260929/delivery.json',
           ROOT/'artifacts/v125_review_20260929/verification.json'}
    for fold in range(3):
        folder=TEACHERS/f'fold{fold}_N1_teacher'
        paths.update((folder/'scores_all_input_ids.npy',folder/'fit.json'))
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.suffix=='.py' and p.is_file() and p.is_relative_to(ROOT) and not p.relative_to(ROOT).parts[0].startswith('.venv'):
                paths.add(p)
    return sorted(paths)


def batches_for_fold(manifest,n_input,fold,body_lengths):
    fit,counts=counts_for_fold(manifest,n_input,fold)
    used=np.flatnonzero(counts.sum(1));nb=math.ceil(len(used)/256)
    expected=[1825,800,1825][fold]
    if nb*25!=expected or len(fit)!=counts.sum() or len(body_lengths)!=n_input:raise ValueError('Registered ASA fit schedule changed')
    return fit,counts,used,nb


def make_model(arm,seed):
    model=Expert(arm,seed).to(DEVICE)
    if arm!='A':
        count=sum(p.numel() for p in model.branch.parameters())
        if count>250000:raise ValueError('Additional branch exceeds budget')
    return model


def fixed_input(model,x,body,lengths,ids):
    block=x[ids]
    if model.branch is None:return model(block)
    l=torch.as_tensor(lengths[ids],dtype=torch.int64,device=DEVICE)
    width=int(lengths[ids].max())
    b=torch.as_tensor(np.asarray(body[ids,:width]),dtype=torch.uint8,device=DEVICE)
    return model(block,b,l)


def diagnostic(model,x,body,lengths,counts,used,p):
    member_ce=np.zeros(3);ensemble_ce=np.zeros(3);nonconflict=np.zeros(3,dtype=np.int64)
    for start in range(0,len(used),256):
        ids=used[start:start+256]
        with torch.no_grad():logm=torch.log_softmax(fixed_input(model,x,body,lengths,ids),-1).double().mean(1).cpu().numpy()
        w=counts[ids].astype(np.float64);q=p[ids].astype(np.float64)
        mixed=(w[:,1]>0)&(w[:,2]>0);pred=q.argmax(1)
        for c in (1,2):
            member_ce[c]-=float((w[:,c]*logm[:,c]).sum())
            ensemble_ce[c]-=float((w[:,c]*np.log(np.clip(q[:,c],1e-12,1))).sum())
            nonconflict[c]+=int((w[:,c]*(pred!=c)*(~mixed)).sum())
    out={}
    pred=p[used].argmax(1)
    for c in (1,2):
        support=int(counts[:,c].sum())
        out[str(c)]={'support':support,'correct':int((counts[used,c]*(pred==c)).sum()),
            'mean_member_CE':float(member_ce[c]/support),'ensemble_CE':float(ensemble_ce[c]/support),
            'nonconflict_error_rows':int(nonconflict[c])}
    return out


def numeric_probe(x,lengths,body,plan):
    # No optimizer step. Replaying the exact same calculation checks this host,
    # not every later CUDA kernel or a second full fit.
    samples=np.array([0,1,2,3],dtype=np.int64)
    results=[]
    for arm in 'ABC':
        model=make_model(arm,plan['training']['seed']);model.train()
        view=body if arm=='B' else (np.load(DEST/'shuffled_body_bytes.npy',mmap_mode='r') if arm=='C' else None)
        run=[]
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            z=fixed_input(model,x,view,lengths,samples)
            loss=-torch.log_softmax(z,-1)[:,:,1].mean()
            loss.backward()
            run.append((z.detach().cpu().numpy(),float(loss.detach()),model.base.first.weight.grad.detach().cpu().numpy().copy()))
        dz=float(np.max(np.abs(run[0][0]-run[1][0])));dl=abs(run[0][1]-run[1][1]);dg=float(np.max(np.abs(run[0][2]-run[1][2])))
        if not np.isfinite(dz+dl+dg) or max(dz,dl,dg)>1e-5:raise ValueError('Repeated fixed numeric probe changed')
        results.append({'arm':arm,'forward_abs_diff':dz,'loss_abs_diff':dl,'base_gradient_abs_diff':dg,
                        'additional_parameters':sum(p.numel() for p in model.branch.parameters()) if model.branch else 0})
        del model
    return results


def register():
    if (DEST/'registration.json').exists() or (DEST/'run_seal.json').exists():raise FileExistsError('V125 already registered')
    allowed={'input_preflight.json','body_span_ledger.parquet','body_lengths.npy',
             'ordered_body_bytes.npy','shuffled_body_bytes.npy'}
    if {p.name for p in DEST.iterdir()}!=allowed:raise FileExistsError('Unexpected existing V125 files')
    plan=read(PLAN);preflight=read(PREFLIGHT);review=review_plan(plan,preflight)
    if DEVICE!='cuda' or not torch.cuda.is_available():raise RuntimeError('Registered CUDA unavailable')
    x=sparse.load_npz(VIEW);lengths=np.load(DEST/'body_lengths.npy');ordered=np.load(DEST/'ordered_body_bytes.npy',mmap_mode='r')
    manifest=pd.read_parquet(MANIFEST)
    schedule=[]
    for fold in range(3):
        fit,counts,used,nb=batches_for_fold(manifest,x.shape[0],fold,lengths)
        schedule.append({'fold':fold,'fit_rows':len(fit),'class_mass':counts.sum(0).tolist(),
                         'used_locals':len(used),'logical_batches':nb,'steps_per_arm':25*nb})
    reference=make_model('A',10201);baseline_hash=tensor_hash(reference.base.state_dict())
    both=[make_model(a,10201) for a in ('B','C')]
    branch_hash=[tensor_hash(m.branch.state_dict()) for m in both]
    if any(tensor_hash(m.base.state_dict())!=baseline_hash for m in both) or branch_hash[0]!=branch_hash[1]:
        raise ValueError('Paired base or branch initialization changed')
    first=x[:4]
    initial_output_differences={}
    with torch.no_grad():
        za=reference(first).detach().cpu().numpy()
        for arm,m in zip(('B','C'),both):
            bytes_=ordered if arm=='B' else np.load(DEST/'shuffled_body_bytes.npy',mmap_mode='r')
            zb=fixed_input(m,x,bytes_,lengths,np.arange(4)).detach().cpu().numpy()
            difference=float(np.max(np.abs(za-zb)))
            initial_output_differences[arm]=difference
            if not np.isfinite(difference) or difference>1e-6:
                raise ValueError('Nonzero new branch or unstable base calculation at initialization')
    probe=numeric_probe(x,lengths,ordered,plan)
    # All code and model identities are ready before sealing. No optimizer step has occurred.
    import tabm
    seal_run(PLAN,Path(__file__),sources(),DEST/'run_seal.json')
    reg={'status':'registered_before_any_optimizer_step','plan_sha256':sha(PLAN),
         'seal_sha256':sha(DEST/'run_seal.json'),'base_initial_sha256':baseline_hash,
         'B_C_branch_initial_sha256':branch_hash[0],
         'fit_order':ORDER,'schedule':schedule,'numeric_probe':probe,
         'initial_output_max_abs_differences':initial_output_differences,
         'package':{'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
                    'scipy':importlib.metadata.version('scipy'),'tabm':importlib.metadata.version('tabm'),
                    'tabm_sha256':sha(Path(tabm.__file__)),'cuda_runtime':torch.version.cuda,
                    'device':torch.cuda.get_device_name(0),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
         'plan_review_checks':review['checks'],'classifier_fits_at_registration':0,
         'model_promoted':False,'limitations':['Small-batch repeatability is not full-run reproducibility.','Existing source folds were inspected.']}
    save(DEST/'registration.json',reg)
    print(json.dumps({'stage':'registered','schedule':schedule,'numeric_probe':probe,
                      'base_initial_sha256':baseline_hash},ensure_ascii=False),flush=True)


def check_registration():
    plan=require_run_seal(DEST/'run_seal.json',Path(__file__))
    reg=read(DEST/'registration.json')
    if reg['plan_sha256']!=sha(PLAN) or reg['seal_sha256']!=sha(DEST/'run_seal.json'):
        raise ValueError('V125 registration changed')
    import tabm
    if sha(Path(tabm.__file__))!=reg['package']['tabm_sha256']:raise ValueError('Model package changed')
    return plan,reg


def fit(fold,arm,seed=10201):
    plan,reg=check_registration()
    if [fold,arm] not in reg['fit_order'] or seed!=plan['training']['seed']:
        raise ValueError('Unregistered fold, arm or seed')
    folder=DEST/f'fold{fold}_{arm}'
    if folder.exists():
        receipt=folder/'fit.json'
        if receipt.exists() and sha(folder/'epoch25_model.pt')==read(receipt)['model_sha256']:
            print(json.dumps({'stage':'already_completed','fold':fold,'arm':arm}),flush=True);return
        raise FileExistsError('Interrupted fit: preserve evidence and diagnose '+str(folder))
    x=sparse.load_npz(VIEW);lengths=np.load(DEST/'body_lengths.npy');body=body_for_arm(arm)
    manifest=pd.read_parquet(MANIFEST)
    fitrows,counts,used,nb=batches_for_fold(manifest,x.shape[0],fold,lengths)
    expectation=reg['schedule'][fold]
    if expectation!={'fold':fold,'fit_rows':len(fitrows),'class_mass':counts.sum(0).tolist(),
                      'used_locals':len(used),'logical_batches':nb,'steps_per_arm':25*nb}:
        raise ValueError('Schedule changed')
    model=make_model(arm,seed)
    if tensor_hash(model.base.state_dict())!=reg['base_initial_sha256'] or (arm!='A' and tensor_hash(model.branch.state_dict())!=reg['B_C_branch_initial_sha256']):
        raise ValueError('Initialization drift')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    rng=np.random.default_rng(seed+fold)
    folder.mkdir()
    save(folder/'started.json',{'status':'started','fold':fold,'arm':arm,'seed':seed,
         'train_rows':len(fitrows),'train_mass':counts.sum(0).tolist(),
         'base_initial_sha256':reg['base_initial_sha256'],'seal_sha256':sha(DEST/'run_seal.json'),
         'heldout_gradient_rows':0,'started_unix':time.time()})
    start=time.monotonic();steps=0;progress=[];checks=[]
    for epoch in range(1,26):
        require_run_seal(DEST/'run_seal.json',Path(__file__))
        model.train();seen=np.zeros_like(counts,dtype=np.float64);online=0.0
        order=used[rng.permutation(len(used))]
        batches=[order[i:i+256] for i in range(0,len(used),256)]
        if len(batches)!=nb:raise ValueError('Batch count changed')
        denominator=len(fitrows)/nb
        for ids in batches:
            ids=ids[np.argsort(lengths[ids],kind='stable')]
            mass=counts[ids].astype(np.float64)
            if len(ids)!=len(np.unique(ids)) or (mass<0).any():raise ValueError('Invalid original-row mass')
            seen[ids]+=mass
            logits=fixed_input(model,x,body,lengths,ids)
            weights=torch.as_tensor(mass,dtype=torch.float32,device=DEVICE)
            numerator=-(torch.log_softmax(logits,-1)*weights[:,None,:]).sum()/K
            loss=numerator/denominator
            if not bool(torch.isfinite(loss).item()):raise FloatingPointError('Nonfinite train loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();steps+=1
            online+=float(numerator.detach().item())
        err=float(np.max(np.abs(seen-counts)))
        if err>1e-8 or steps!=epoch*nb:raise ValueError('Original-row mass or optimizer steps changed')
        item={'epoch':epoch,'optimizer_steps_cumulative':steps,'fit_rows':len(fitrows),
              'M_mass':int(counts[:,1].sum()),'S_mass':int(counts[:,2].sum()),
              'mass_reconstruction_max_error':err,'online_original_row_CE':online/len(fitrows),
              'peak_GPU_allocated_bytes':int(torch.cuda.max_memory_allocated()),
              'elapsed_seconds':time.monotonic()-start}
        progress.append(item);save(folder/'progress.json',progress)
        if epoch in CHECKPOINTS:
            p=probabilities(model,x,body,lengths,DEVICE)
            np.save(folder/f'epoch{epoch}_prob.npy',p)
            state={'base':{k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()},
                   'branch':{k:v.detach().cpu().clone() for k,v in model.branch.state_dict().items()} if model.branch else None,
                   'fold':fold,'arm':arm,'seed':seed,'epoch':epoch,
                   'seal_sha256':sha(DEST/'run_seal.json')}
            torch.save(state,folder/f'epoch{epoch}_model.pt')
            diag=diagnostic(model,x,body,lengths,counts,used,p)
            checks.append({'epoch':epoch,'model_sha256':sha(folder/f'epoch{epoch}_model.pt'),
                           'prob_sha256':sha(folder/f'epoch{epoch}_prob.npy'),'fit_by_class':diag})
            save(folder/'checkpoints.json',checks)
            print(json.dumps({'stage':'checkpoint','fold':fold,'arm':arm,'epoch':epoch,
                              'steps':steps,'fit_M_correct':diag['1']['correct'],
                              'fit_S_correct':diag['2']['correct'],'seconds':round(time.monotonic()-start,1)},ensure_ascii=False),flush=True)
    require_checkpoint(plan,{'status':'fit_executed','completed_epochs':25,'prediction_epoch':25})
    save(folder/'fit.json',{'status':'fit_executed','fold':fold,'arm':arm,'seed':seed,
         'completed_epochs':25,'prediction_epoch':25,'optimizer_steps':steps,
         'fit_rows':len(fitrows),'train_mass':counts.sum(0).tolist(),
         'base_initial_sha256':reg['base_initial_sha256'],
         'branch_initial_sha256':reg['B_C_branch_initial_sha256'] if arm!='A' else None,
         'model_sha256':sha(folder/'epoch25_model.pt'),'prob_sha256':sha(folder/'epoch25_prob.npy'),
         'checkpoints_sha256':sha(folder/'checkpoints.json'),'progress_sha256':sha(folder/'progress.json'),
         'seal_sha256':sha(DEST/'run_seal.json'),'seconds':time.monotonic()-start,
         'quality_acceptance':False,'model_promoted':False})
    print(json.dumps({'stage':'fit_complete','fold':fold,'arm':arm,'steps':steps,'seconds':round(time.monotonic()-start,1)},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('register','run','fit'))
    parser.add_argument('--fold',type=int);parser.add_argument('--arm',choices=('A','B','C'))
    args=parser.parse_args()
    if args.stage=='register':register()
    elif args.stage=='fit':fit(args.fold,args.arm)
    else:
        for fold,arm in ORDER:fit(fold,arm)


if __name__=='__main__':main()
