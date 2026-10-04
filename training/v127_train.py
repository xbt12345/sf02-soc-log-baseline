"""Registered V127 K/W/P primary training; no outer-answer adaptation."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
import torch

from v127_experiment_review import ROOT, PLAN, check_plan, seal_run, require_run_seal, require_checkpoint
from v127_model import make_branch, branch_batch
from v125_model import Expert
from v125_train import tensor_hash
from v126_frozen_audit import PARENT, sha, read, save
from v125_evaluate import TRACE, OFFICIAL, MANIFEST

OUT = ROOT / 'artifacts/v127_frozen_branch_trial_20260929'
INPUT = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
BODY = PARENT / 'ordered_body_bytes.npy'
LENGTHS = PARENT / 'body_lengths.npy'
ORDER = [(0, 'K'), (0, 'W'), (0, 'P'), (1, 'P'), (1, 'W'), (1, 'K'),
         (2, 'W'), (2, 'K'), (2, 'P')]
CHECKPOINTS = (0, 1, 2, 5, 10, 15, 20, 25, 35, 50)
DEVICE = 'cuda'


def source_files():
    fixed = {Path(__file__), PLAN, ROOT/'training/review_policy/v127_risk_actions.json',
             OUT/'preflight.json', ROOT/'artifacts/v127_plan_review_20260929/verification.json',
             INPUT, BODY, LENGTHS, PARENT/'body_span_ledger.parquet',
             MANIFEST, OFFICIAL, TRACE, ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz',
             ROOT/'artifacts/v124_header_trial_20260929/input_preflight.json'}
    fixed.update(OUT/f'fold{f}_base_member_logits.npy' for f in range(3))
    fixed.update(PARENT/f'fold{f}_A/epoch25_model.pt' for f in range(3))
    fixed.update(ROOT/'artifacts/v127_plan_review_20260929'/f'fold{f}_A_header_probability.npy' for f in range(3))
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_file() and path.is_relative_to(ROOT):
                if not path.relative_to(ROOT).parts[0].startswith('.venv'):
                    fixed.add(path)
    return sorted(fixed)


def row_counts(manifest, fold, n):
    fit = manifest[(manifest.outer_fold == fold) & (manifest.fold != fold)]
    held = manifest[(manifest.outer_fold == fold) & (manifest.fold == fold)]
    if set(fit.root) & set(held.root) or fit.row_position.duplicated().any() or held.row_position.duplicated().any():
        raise ValueError('Source leakage or duplicate row')
    def collect(frame):
        count = np.bincount(frame.local.to_numpy(np.int64)*3+frame.truth.to_numpy(np.int64),
                            minlength=n*3).reshape(n, 3)
        if count[:, 0].any() or int(count.sum()) != len(frame):
            raise ValueError('Class mass does not cover original rows')
        return count
    return fit, held, collect(fit), collect(held)


def member_logits(fold, x):
    state = torch.load(PARENT/f'fold{fold}_A/epoch25_model.pt', map_location='cpu', weights_only=True)
    if (state['fold'], state['arm'], state['seed'], state['epoch']) != (fold, 'A', 10201, 25):
        raise ValueError('Wrong frozen base')
    model = Expert('A', 10201).to(DEVICE).eval()
    model.base.load_state_dict(state['base'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise ValueError('Base not frozen and eval')
    base_hash = tensor_hash(model.base.state_dict())
    chunks = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 256):
            chunks.append(model(x[start:start+256]).detach().cpu().numpy())
    logits = np.concatenate(chunks, axis=0).astype(np.float32)
    if logits.shape != (22546, 16, 3) or not np.isfinite(logits).all():
        raise ValueError('Invalid base member cache')
    expected = np.load(ROOT/'artifacts/v127_plan_review_20260929'/f'fold{fold}_A_header_probability.npy')
    prob = softmax(logits, axis=-1).mean(1)
    diff = float(np.max(np.abs(prob-expected)))
    if diff > 2e-6 or not np.array_equal(prob.argmax(1), expected.argmax(1)):
        raise ValueError('Canonical base probability replay changed')
    if tensor_hash(model.base.state_dict()) != base_hash:
        raise ValueError('Frozen base mutated while caching')
    del model
    return logits, {'fold': fold, 'base_state_sha256': base_hash, 'probability_replay_max_abs': diff}


def require_package_environment(reg):
    for name, value in reg['packages'].items():
        current = torch.__version__ if name == 'torch' else importlib.metadata.version(name)
        if current != value:
            raise ValueError('Runtime package changed: ' + name)
    if DEVICE != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Registered CUDA unavailable')


def small_numeric_probe(body, lengths, base):
    ids = np.arange(4)
    x = torch.as_tensor(base[ids], device=DEVICE)
    checks = []
    for arm in 'WP':
        model = make_branch(arm, 12701, DEVICE).train()
        observations = []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            extra = branch_batch(model, body, lengths, ids)
            loss = -torch.log_softmax(x+extra[:, None, :], -1)[:, :, 1].mean()
            loss.backward()
            observations.append((extra.detach().cpu().numpy(), float(loss.detach()),
                                 model.head.weight.grad.detach().cpu().numpy().copy()))
        gap = max(float(np.max(np.abs(observations[0][0]-observations[1][0]))),
                  abs(observations[0][1]-observations[1][1]),
                  float(np.max(np.abs(observations[0][2]-observations[1][2]))))
        if gap > 1e-5 or not np.isfinite(gap):
            raise ValueError('Numerical prefit probe changed')
        checks.append({'arm': arm, 'max_repeat_diff': gap,
                       'parameters': sum(p.numel() for p in model.parameters()),
                       'head_initially_zero': bool((model.head.weight == 0).all() and (model.head.bias == 0).all())})
    return checks


def register():
    if OUT.exists():
        raise FileExistsError('V127 output already exists: preserve and inspect '+str(OUT))
    plan = check_plan()
    if plan['training']['fit_order'] != [[f, a] for f, a in ORDER]:
        raise ValueError('Fit order disagrees with source')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    x = sparse.load_npz(INPUT)
    body = np.load(BODY, mmap_mode='r'); lengths = np.load(LENGTHS)
    trace = pd.read_parquet(TRACE, columns=['row_position', 'local', 'fold', 'root', 'truth'])
    official = pd.read_parquet(OFFICIAL, columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    manifest = pd.read_parquet(MANIFEST)
    ledger = pd.read_parquet(PARENT/'body_span_ledger.parquet')
    if (len(official),len(trace),x.shape,len(body),len(lengths)) != (2056871,112807,(22546,66287),22546,22546):
        raise ValueError('Population or input dimension changed')
    if not np.array_equal(official[trace.row_position], trace.truth) or not trace.row_position.equals(ledger.row_position):
        raise ValueError('Official truth or body ledger changed')
    if not np.array_equal(trace.local, ledger.local) or not np.array_equal(trace.fold, ledger.fold):
        raise ValueError('Input mapping changed')
    if int(lengths.min()) < 123 or int(lengths.max()) > 176 or (body.shape[1] < int(lengths.max())):
        raise ValueError('Body truncated or length changed')
    expected = read(PARENT/'input_preflight.json')
    for name, digest in expected['files_sha256'].items():
        if sha(PARENT/name) != digest:
            raise ValueError('V125 input changed: ' + name)
    head = read(ROOT/'artifacts/v124_header_trial_20260929/input_preflight.json')
    if sha(INPUT) != head['output_sha256']['B_header_ASA.npz'] or head['changed_rows'] != 682:
        raise ValueError('Header projection input changed')
    OUT.mkdir()
    schedule, bases = [], []
    for fold in range(3):
        fit, held, counts, _ = row_counts(manifest, fold, len(lengths))
        used = int(np.count_nonzero(counts.sum(1)))
        steps = math.ceil(used/256)*50
        if steps != plan['training']['expected_steps_per_arm'][fold]:
            raise ValueError('Logical-batch schedule changed')
        schedule.append({'fold':fold, 'fit_original_rows':len(fit), 'held_original_rows':len(held),
                         'fit_class_mass':counts.sum(0).tolist(), 'used_old_locals':used,
                         'steps_per_network_arm':steps})
        z, identity = member_logits(fold, x)
        np.save(OUT/f'fold{fold}_base_member_logits.npy', z)
        bases.append(identity)
    w = make_branch('W',12701,DEVICE);p = make_branch('P',12701,DEVICE)
    ws=w.state_dict();ps=p.state_dict();common={key for key in ws if key in ps}
    if not all(torch.equal(ws[key],ps[key]) for key in common):
        raise ValueError('W/P common tensor initializations differ')
    if not all(k.startswith('final_norm.') for k in set(ps)-set(ws)):
        raise ValueError('P differs beyond planned normalization')
    probe = small_numeric_probe(body,lengths,np.load(OUT/'fold0_base_member_logits.npy'))
    if not all(item['head_initially_zero'] for item in probe):
        raise ValueError('Residual head not zero')
    preflight = {'status':'inputs_and_contract_checked_before_any_update',
                 'latest_actual_training':'V125', 'classifier_fits':0, 'optimizer_steps':0,
                 'official_rows':len(official),'ASA_rows':len(trace), 'body_maxlen':int(lengths.max()),
                 'schedule':schedule,'bases':bases,'shared_initial_tensor_count':len(common),
                 'initial_branch_sha256':{a:tensor_hash(m.state_dict()) for a,m in [('W',w),('P',p)]},
                 'numeric_probe':probe, 'K_synthetic_gradient_max_abs_error':read(ROOT/'artifacts/v127_plan_review_20260929/verification.json')['synthetic_gradient_max_abs_error'],
                 'inputs_sha256':{q.relative_to(ROOT).as_posix():sha(q) for q in (INPUT,BODY,LENGTHS,MANIFEST,OFFICIAL,TRACE)},
                 'packages':{'torch':torch.__version__,'numpy':importlib.metadata.version('numpy'),
                             'scipy':importlib.metadata.version('scipy'),'pandas':importlib.metadata.version('pandas')},
                 'cuda_device':torch.cuda.get_device_name(0),
                 'scope':'Actual immutable input and model cache check, not training quality'}
    save(OUT/'preflight.json',preflight)
    seal = seal_run(Path(__file__), source_files(), OUT/'run_seal.json')
    registration = {'status':'registered_before_any_optimizer_update', 'seal_sha256':sha(OUT/'run_seal.json'),
                    'plan_sha256':sha(PLAN),'fit_order':ORDER,'schedule':schedule,
                    'packages':preflight['packages'],'quality_acceptance':False,'model_promoted':False}
    save(OUT/'registration.json',registration)
    print(json.dumps({'stage':'registered','schedule':schedule,'base_replay':bases,
                      'branch_init':preflight['initial_branch_sha256']},ensure_ascii=False),flush=True)


def registration():
    plan=require_run_seal(OUT/'run_seal.json', Path(__file__))
    reg=read(OUT/'registration.json')
    if reg['status']!='registered_before_any_optimizer_update' or reg['seal_sha256']!=sha(OUT/'run_seal.json'):
        raise ValueError('V127 registration invalid')
    require_package_environment(reg)
    return plan,reg


def fit_constant(fold, plan, reg):
    folder=OUT/f'fold{fold}_K'
    if folder.exists():
        raise FileExistsError('V127 K fit exists: '+str(folder))
    z=np.load(OUT/f'fold{fold}_base_member_logits.npy').astype(np.float64)
    manifest=pd.read_parquet(MANIFEST)
    _,_,counts,_=row_counts(manifest,fold,len(z))
    mass=counts.sum(1).astype(np.float64);n=float(mass.sum())
    lam=plan['constant_control']['l2_coefficient']
    def objective(uv):
        b=np.array([uv[0],uv[1],-uv.sum()]);v=z+b
        lse=logsumexp(v,axis=-1)
        member_logp=(v-lse[:,:,None]).mean(1)
        value=float(-(counts*member_logp).sum()/n+lam/2*np.dot(b,b))
        q=softmax(v,axis=-1).mean(1)
        gb=(q*mass[:,None]-counts).sum(0)/n+lam*b
        grad=np.array([gb[0]-gb[2],gb[1]-gb[2]])
        return value,grad
    initial=objective(np.zeros(2))[0]
    folder.mkdir()
    save(folder/'started.json',{'status':'started','fold':fold,'arm':'K','seal_sha256':sha(OUT/'run_seal.json'),
                                'fit_mass':counts.sum(0).tolist(),'initial_objective':initial})
    start=time.monotonic()
    result=minimize(objective,np.zeros(2),method='L-BFGS-B',jac=True,
                    options=plan['constant_control']['options'])
    value,grad=objective(result.x)
    b=np.array([result.x[0],result.x[1],-result.x.sum()])
    valid=bool(np.isfinite(value) and np.isfinite(b).all() and np.isfinite(grad).all()
               and np.max(np.abs(grad))<=1e-7 and value<=initial+1e-10)
    np.save(folder/'endpoint_prob.npy',softmax(z+b,axis=-1).mean(1).astype(np.float32))
    result_record={'status':'fit_executed' if valid else 'invalid_constant_control',
                   'fold':fold,'arm':'K','optimizer':'L-BFGS-B','objective':value,
                   'initial_objective':initial,'bias':b.tolist(),'uv_gradient':grad.tolist(),
                   'gradient_inf':float(np.max(np.abs(grad))),'success_reported':bool(result.success),
                   'solver_message':str(result.message),'nit':int(result.nit),'nfev':int(result.nfev),
                   'elapsed_seconds':time.monotonic()-start,'fit_mass':counts.sum(0).tolist(),
                   'prob_sha256':sha(folder/'endpoint_prob.npy'), 'seal_sha256':sha(OUT/'run_seal.json'),
                   'quality_acceptance':False,'model_promoted':False}
    save(folder/'fit.json',result_record)
    print(json.dumps({'stage':'constant_complete','fold':fold,'valid':valid,'gradient_inf':result_record['gradient_inf'],
                      'bias':result_record['bias'],'nit':result_record['nit']},ensure_ascii=False),flush=True)
    if not valid:
        raise ValueError('K failed registered stationarity check; preserve evidence')


def probs_from_residual(z, extra):
    return softmax(z+extra[:,None,:],axis=-1).mean(1).astype(np.float32)


def class_diagnostic(z, extra, fit_counts, held_counts):
    member_logp=(z+extra[:,None,:]-logsumexp(z+extra[:,None,:],axis=-1)[:,:,None]).mean(1)
    p=probs_from_residual(z,extra)
    predicted=p.argmax(1)
    result={}
    for role,counts in [('fit',fit_counts),('held',held_counts)]:
        classes={}
        mixed=(counts[:,1]>0)&(counts[:,2]>0)
        for c in (1,2):
            support=int(counts[:,c].sum())
            classes[str(c)]={'support':support,'errors':int((counts[:,c]*(predicted!=c)).sum()),
                             'nonconflict_errors':int((counts[:,c]*(predicted!=c)*(~mixed)).sum()),
                             'member_CE':float(-(counts[:,c]*member_logp[:,c]).sum()/support),
                             'ensemble_CE':float(-(counts[:,c]*np.log(np.maximum(p[:,c],1e-12))).sum()/support)}
        result[role]=classes
    return p,result


def branch_diagnostic(model, body, lengths, used):
    ids=np.asarray(used[:min(256,len(used))],dtype=np.int64)
    model.eval()
    with torch.no_grad():
        le=torch.as_tensor(lengths[ids],device=DEVICE,dtype=torch.int64)
        raw=torch.as_tensor(body[ids,:int(le.max())],device=DEVICE,dtype=torch.uint8)
        pooled=model.encode(raw,le).cpu().numpy()
    return {'fixed_fit_panel_unique_locals':len(ids),'pooled_mean_feature_std':float(pooled.std(0).mean()),
            'pooled_pairwise_distance_to_mean':float(np.linalg.norm(pooled-pooled.mean(0),axis=1).mean())}


def network_checkpoint(folder, model, arm, fold, epoch, z, body, lengths, fit_counts, held_counts, used, steps):
    model.eval();extra=np.empty((len(lengths),3),dtype=np.float32)
    with torch.no_grad():
        for start in range(0,len(lengths),256):
            ids=np.arange(start,min(start+256,len(lengths)))
            extra[ids]=branch_batch(model,body,lengths,ids,DEVICE).cpu().numpy()
    prob,classes=class_diagnostic(z,extra,fit_counts,held_counts)
    np.save(folder/f'epoch{epoch}_prob.npy',prob)
    np.save(folder/f'epoch{epoch}_residual.npy',extra)
    torch.save({'branch':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'fold':fold,'arm':arm,'seed':12701,'epoch':epoch,'optimizer_steps':steps,
                'seal_sha256':sha(OUT/'run_seal.json')},folder/f'epoch{epoch}_model.pt')
    fit_mass=fit_counts.sum(1).astype(np.float64)
    mean=(extra.astype(float)*fit_mass[:,None]).sum(0)/fit_mass.sum()
    const_p,const_class=class_diagnostic(z,np.broadcast_to(mean,extra.shape),fit_counts,held_counts)
    info={'epoch':epoch,'steps':steps,'fit_and_held':classes,'train_mean_residual':mean.tolist(),
          'train_mean_constant_comparison':const_class,
          'real_vs_train_mean_held_prediction_flips':int(((prob.argmax(1)!=const_p.argmax(1))*(held_counts.sum(1)>0)).sum()),
          'residual_M_minus_S_std':float((extra[:,1]-extra[:,2]).std()),
          'representation':branch_diagnostic(model,body,lengths,used),
          'model_sha256':sha(folder/f'epoch{epoch}_model.pt'),
          'prob_sha256':sha(folder/f'epoch{epoch}_prob.npy'),
          'residual_sha256':sha(folder/f'epoch{epoch}_residual.npy')}
    return info


def fit_network(fold, arm, plan, reg):
    folder=OUT/f'fold{fold}_{arm}'
    if folder.exists():
        raise FileExistsError('V127 network fit exists: '+str(folder))
    z=np.load(OUT/f'fold{fold}_base_member_logits.npy')
    body=np.load(BODY,mmap_mode='r');lengths=np.load(LENGTHS)
    manifest=pd.read_parquet(MANIFEST)
    fit,held,counts,held_counts=row_counts(manifest,fold,len(lengths))
    used=np.flatnonzero(counts.sum(1))
    nb=math.ceil(len(used)/256);total=nb*50
    if total!=reg['schedule'][fold]['steps_per_network_arm']:
        raise ValueError('Schedule changed')
    branch=make_branch(arm,plan['training']['seed'],DEVICE)
    if tensor_hash(branch.state_dict())!=read(OUT/'preflight.json')['initial_branch_sha256'][arm]:
        raise ValueError('Branch initialization changed')
    params=list(branch.parameters())
    optimizer=torch.optim.AdamW(params,lr=0.0003,weight_decay=0.0003)
    if any(not p.requires_grad for p in params):
        raise ValueError('Frozen parameter in optimizer')
    cache=torch.as_tensor(z,device=DEVICE,dtype=torch.float32)
    weight=torch.as_tensor(counts,device=DEVICE,dtype=torch.float32)
    if cache.requires_grad:
        raise ValueError('Base cache requires gradients')
    rng=np.random.default_rng(12701+fold)
    folder.mkdir()
    save(folder/'started.json',{'status':'started','fold':fold,'arm':arm,'seed':12701,
          'fit_rows':len(fit),'held_rows':len(held),'fit_mass':counts.sum(0).tolist(),
          'branch_initial_sha256':tensor_hash(branch.state_dict()),'seal_sha256':sha(OUT/'run_seal.json'),
          'base_cache_sha256':sha(OUT/f'fold{fold}_base_member_logits.npy'), 'start_unix':time.time()})
    start=time.monotonic();steps=0;progress=[];checks=[]
    checks.append(network_checkpoint(folder,branch,arm,fold,0,z,body,lengths,counts,held_counts,used,steps))
    save(folder/'checkpoints.json',checks)
    log=(folder/'steps.jsonl').open('w',encoding='utf-8')
    try:
        for epoch in range(1,51):
            if epoch in CHECKPOINTS:
                require_run_seal(OUT/'run_seal.json',Path(__file__))
            branch.train();seen=np.zeros_like(counts,dtype=np.float64)
            order=used[rng.permutation(len(used))]
            batch_m,batch_s=0.0,0.0
            for start_index in range(0,len(used),256):
                ids=order[start_index:start_index+256]
                ids=ids[np.argsort(lengths[ids],kind='stable')]
                if len(ids)!=len(np.unique(ids)):
                    raise ValueError('Repeated local in logical batch')
                seen[ids]+=counts[ids]
                residual=branch_batch(branch,body,lengths,ids,DEVICE)
                logits=cache[ids]+residual[:,None,:]
                logp=torch.log_softmax(logits,-1).mean(1)
                category=-(logp*weight[ids]).sum(0)
                normalizer=len(fit)/nb
                loss=category.sum()/normalizer
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError('Nonfinite training objective')
                steps+=1
                lr=0.0003*min(steps/math.ceil(.1*total),1)
                for group in optimizer.param_groups:
                    group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norms={}
                for name,module in [('embedding',branch.embed),('encoder',branch.encoder),
                                    ('final_norm',branch.final_norm),('head',branch.head)]:
                    squares=[p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
                    norms[name]=float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0
                if not all(np.isfinite(v) for v in norms.values()):
                    raise FloatingPointError('Nonfinite branch gradient')
                before={name:p.detach().clone() for name,p in branch.named_parameters()}
                optimizer.step()
                update_squares={name:(p.detach()-before[name]).square().sum()
                                for name,p in branch.named_parameters()}
                updates={key:float(torch.stack([value for name,value in update_squares.items()
                                 if name.startswith(key+'.')]).sum().sqrt().item())
                         if any(name.startswith(key+'.') for name in update_squares) else 0.0
                         for key in ('embed','encoder','final_norm','head')}
                if not all(np.isfinite(v) for v in updates.values()):
                    raise FloatingPointError('Nonfinite parameter update')
                batch_m+=float(category[1].detach());batch_s+=float(category[2].detach())
                log.write(json.dumps({'step':steps,'epoch':epoch,'lr':lr,
                    'M_original_rows':int(counts[ids,1].sum()),'S_original_rows':int(counts[ids,2].sum()),
                    'M_member_CE_numerator':float(category[1].detach()),
                    'S_member_CE_numerator':float(category[2].detach()),
                    'module_gradient_norm':norms,
                    'module_update_norm':updates,
                    'finite':True},ensure_ascii=False)+'\n')
            if not np.array_equal(seen,counts) or steps!=epoch*nb:
                raise ValueError('Original row mass or update count changed')
            item={'epoch':epoch,'steps':steps,'M_original_rows':int(counts[:,1].sum()),
                  'S_original_rows':int(counts[:,2].sum()),'M_member_CE_numerator':batch_m,
                  'S_member_CE_numerator':batch_s,'online_original_row_member_CE':(batch_m+batch_s)/len(fit),
                  'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),
                  'elapsed_seconds':time.monotonic()-start}
            progress.append(item);save(folder/'progress.json',progress);log.flush()
            if epoch in CHECKPOINTS:
                check=network_checkpoint(folder,branch,arm,fold,epoch,z,body,lengths,counts,held_counts,used,steps)
                checks.append(check);save(folder/'checkpoints.json',checks)
                print(json.dumps({'stage':'checkpoint','fold':fold,'arm':arm,'epoch':epoch,'steps':steps,
                       'fit_M_errors':check['fit_and_held']['fit']['1']['errors'],
                       'fit_S_errors':check['fit_and_held']['fit']['2']['errors'],
                       'held_M_errors':check['fit_and_held']['held']['1']['errors'],
                       'held_S_errors':check['fit_and_held']['held']['2']['errors'],
                       'seconds':round(time.monotonic()-start,1)},ensure_ascii=False),flush=True)
    finally:
        log.close()
    receipt={'status':'fit_executed','fold':fold,'arm':arm,'seed':12701,
             'completed_epochs':50,'prediction_epoch':50,'optimizer_steps':steps,
             'fit_rows':len(fit),'fit_mass':counts.sum(0).tolist(),'held_rows':len(held),
             'model_sha256':sha(folder/'epoch50_model.pt'),
             'prob_sha256':sha(folder/'epoch50_prob.npy'),
             'residual_sha256':sha(folder/'epoch50_residual.npy'),
             'steps_sha256':sha(folder/'steps.jsonl'),'progress_sha256':sha(folder/'progress.json'),
             'checkpoints_sha256':sha(folder/'checkpoints.json'),
             'base_cache_sha256':sha(OUT/f'fold{fold}_base_member_logits.npy'),
             'seal_sha256':sha(OUT/'run_seal.json'),'seconds':time.monotonic()-start,
             'quality_acceptance':False,'model_promoted':False}
    require_checkpoint(plan,receipt)
    save(folder/'fit.json',receipt)
    print(json.dumps({'stage':'network_complete','fold':fold,'arm':arm,'steps':steps,
                      'seconds':round(receipt['seconds'],1)},ensure_ascii=False),flush=True)


def fit(fold,arm):
    plan,reg=registration()
    if [fold,arm] not in plan['training']['fit_order'] or plan['training']['seed']!=12701:
        raise ValueError('Unregistered fit')
    if arm=='K':fit_constant(fold,plan,reg)
    else:fit_network(fold,arm,plan,reg)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('register','fit','run'))
    parser.add_argument('--fold',type=int,choices=(0,1,2))
    parser.add_argument('--arm',choices=('K','W','P'))
    args=parser.parse_args()
    if args.stage=='register':register()
    elif args.stage=='fit':fit(args.fold,args.arm)
    else:
        for fold,arm in ORDER:
            fit(fold,arm)


if __name__=='__main__':main()
