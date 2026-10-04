"""V128 registered nested-score-role trial. No endpoint selection or outer-answer adaptation."""
import argparse
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
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
import torch

from experiment_review import check_bindings, sha
from v128_experiment_review import ROOT, PLAN, REVIEW, ROLES, check_plan, seal_run, require_run_seal, require_checkpoint, read, save
from v125_model import Expert
from v125_train import tensor_hash
from v127_model import make_branch, branch_batch
from v127_train import row_counts, class_diagnostic

OUT = ROOT / 'artifacts/v128_nested_score_trial_20260929'
PREP = ROOT / 'artifacts/v128_mechanism_review_20260929/nested_role_preparation.json'
ORIGINAL = ROOT / 'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
CANON = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
PARENT = ROOT / 'artifacts/v125_order_trial_20260929'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
BODY = PARENT / 'ordered_body_bytes.npy'
LENGTHS = PARENT / 'body_lengths.npy'
OFFICIAL = ROOT / 'data/official/train.parquet'
MANIFEST = ROOT / 'artifacts/v116_nested_selection_20260929/inner_split_manifest.parquet'
BASE = ROOT / 'artifacts/v127_frozen_branch_trial_20260929'
ORDER = [(f,j,'teacher') for f in range(3) for j in range(3)] + [
    (f,-1,arm) for f in range(3) for arm in ('K_IS','K_CF','P_IS','P_CF')]
CHECKPOINTS = (0,1,2,5,10,15,20,25,35,50)
DEVICE = 'cuda'


def source_files():
    fixed = {Path(__file__), PLAN, REVIEW, ROLES, PREP, ORIGINAL, CANON, BODY, LENGTHS,
             OFFICIAL, TRACE, MANIFEST, ROOT/'training/review_policy/v128_risk_actions.json',
             ROOT/'artifacts/v124_header_trial_20260929/input_preflight.json',
             PARENT/'input_preflight.json', PARENT/'body_span_ledger.parquet',
             PARENT/'run_seal.json', BASE/'run_seal.json'}
    for f in range(3):
        fixed.update((PARENT/f'fold{f}_A/epoch25_model.pt', PARENT/f'fold{f}_A/epoch25_prob.npy',
                      BASE/f'fold{f}_base_member_logits.npy',
                      ROOT/f'artifacts/v127_plan_review_20260929/fold{f}_A_header_probability.npy'))
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_file() and path.is_relative_to(ROOT):
                if not path.relative_to(ROOT).parts[0].startswith('.venv'):
                    fixed.add(path)
    return sorted(fixed)


def counts(frame, n=22546):
    a = np.bincount(frame.local.to_numpy(np.int64)*3+frame.truth.to_numpy(np.int64),
                    minlength=n*3).reshape(n,3)
    if int(a[:,0].sum()) or int(a.sum()) != len(frame):
        raise ValueError('Original-row class mass mismatch')
    return a


def role_frame(fold):
    m = pd.read_parquet(ROLES)
    m = m[m.outer_fold == fold].copy()
    if len(m) != 112807 - int((pd.read_parquet(TRACE, columns=['fold']).fold == fold).sum()):
        raise ValueError('Nested role population changed')
    if m.row_position.duplicated().any() or m.groupby('local').inner_fold.nunique().max() != 1:
        raise ValueError('Duplicate or inconsistent local role')
    return m


def check_roles(roles, trace, fold):
    legal = trace[trace.fold != fold]
    held = trace[trace.fold == fold]
    if set(roles.row_position) != set(legal.row_position) or set(roles.root) & set(held.root):
        raise ValueError('Outer-held source contamination')
    joined = roles.merge(legal[['row_position','local','root','fold','truth']], on='row_position',
                         validate='one_to_one',suffixes=('','_truth'))
    for name in ('local','root','fold','truth'):
        if not np.array_equal(joined[name], joined[name+'_truth']):
            raise ValueError('Role/official trace mismatch: '+name)
    for j in range(3):
        fit = roles[roles.inner_fold != j]
        cf = roles[roles.crossfit_teacher == j]
        isin = roles[roles.matched_infit_teacher == j]
        if (set(fit.root) & set(cf.root) or not set(isin.root) <= set(fit.root)
                or set(fit.root) & set(held.root)):
            raise ValueError('Leaking CF/IS teacher role')
        if not all((fit.truth == c).any() for c in (1,2)):
            raise ValueError('Teacher has missing class')
    if not np.array_equal(roles.crossfit_teacher, roles.inner_fold) or not np.array_equal(
            roles.matched_infit_teacher, (roles.inner_fold+1)%3):
        raise ValueError('Teacher assignment was substituted')


def check_historical_counterexamples():
    # These are rejection examples, not thresholds inferred from the new run.
    from v127_evaluate import TOP3_S_ROOTS
    from experiment_review import class_counts
    if TOP3_S_ROOTS != (21702,20849,29):
        raise ValueError('Protected source scope changed')
    base = pd.read_parquet(BASE/'expert_ASA_predictions.parquet',
                           columns=['truth','pred_A0','pred_K','pred_P'])
    y=base.truth.to_numpy()
    actual={a:{c:int(((y==c)&(base['pred_'+a].to_numpy()!=c)).sum()) for c in (1,2)}
            for a in ('A0','K','P')}
    if actual != {'A0':{1:318,2:2074},'K':{1:326,2:1930},'P':{1:326,2:1980}}:
        raise ValueError('V127 M-regression/constant counterexample changed')
    # V116's early selected epoch must never satisfy the registered endpoint.
    try:
        require_checkpoint(check_plan(),{'status':'fit_executed','completed_epochs':50,'prediction_epoch':15},'branch')
    except ValueError:
        pass
    else:
        raise ValueError('Early checkpoint was accepted')
    return {'V127_class_errors':actual,'V116_early_endpoint_rejected':True,
            'largest_three_S_roots':TOP3_S_ROOTS}


def register():
    if OUT.exists():
        raise FileExistsError('Preserve existing V128 run: '+str(OUT))
    plan=check_plan()
    check_bindings(read(PARENT/'run_seal.json')['source_sha256'])
    check_bindings(read(BASE/'run_seal.json')['source_sha256'])
    if not torch.cuda.is_available():
        raise RuntimeError('Registered CUDA unavailable')
    x=sparse.load_npz(ORIGINAL);h=sparse.load_npz(CANON)
    body=np.load(BODY,mmap_mode='r');lengths=np.load(LENGTHS)
    trace=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    official=pd.read_parquet(OFFICIAL,columns=['label_binary']).label_binary.map(
        {'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    if (x.shape,h.shape,body.shape,len(lengths),len(trace),len(official)) != (
            (22546,66287),(22546,66287),(22546,176),22546,112807,2056871):
        raise ValueError('Data shape changed')
    if not np.array_equal(official[trace.row_position],trace.truth):
        raise ValueError('Independent official labels changed')
    if int(lengths.max())>176 or int(lengths.min())<123:
        raise ValueError('Body coverage changed')
    prep=read(PREP);schedule=[]
    for fold in range(3):
        roles=role_frame(fold);check_roles(roles,trace,fold)
        for j in range(3):
            r=roles[roles.inner_fold!=j];c=counts(r)
            steps=math.ceil(np.count_nonzero(c.sum(1))/256)*25
            expected=prep['teachers'][fold*3+j]
            if (len(r),steps,int(c[:,1].sum()),int(c[:,2].sum())) != (
                expected['fit_rows'], expected['planned_25epoch_steps'],expected['fit_M'],expected['fit_S']):
                raise ValueError('Teacher schedule/coverage changed')
            schedule.append({'fold':fold,'excluded_inner':j,'fit_rows':len(r),
                             'fit_M':int(c[:,1].sum()),'fit_S':int(c[:,2].sum()),
                             'fit_roots':int(r.root.nunique()),'steps':steps})
        _,_,fc,hc=row_counts(pd.read_parquet(MANIFEST),fold,22546)
        if int(fc.sum())!=len(roles) or int(hc.sum())!=int((trace.fold==fold).sum()):
            raise ValueError('Outer fit/held original-row mass changed')
        z=np.load(BASE/f'fold{fold}_base_member_logits.npy',mmap_mode='r')
        expected=np.load(ROOT/f'artifacts/v127_plan_review_20260929/fold{fold}_A_header_probability.npy')
        if z.shape!=(22546,16,3) or np.max(np.abs(softmax(z,axis=-1).mean(1)-expected))>2e-6:
            raise ValueError('Frozen canonical outer base changed')
    model=Expert('A',10201).to(DEVICE).train()
    branch=make_branch('P',12701,DEVICE).train()
    if not bool((branch.head.weight==0).all() and (branch.head.bias==0).all()):
        raise ValueError('Residual initialization changed')
    # A repeat forward/backward probe precedes every parameter update.
    probe=[]
    for _ in range(2):
        branch.zero_grad(set_to_none=True)
        extra=branch_batch(branch,body,lengths,np.arange(4))
        loss=-torch.log_softmax(torch.as_tensor(np.load(BASE/'fold0_base_member_logits.npy',mmap_mode='r')[:4],
                device=DEVICE)+extra[:,None,:],-1)[:,:,1].mean()
        loss.backward()
        probe.append((extra.detach().cpu().numpy(),branch.head.weight.grad.detach().cpu().numpy().copy()))
    gap=max(float(np.max(np.abs(probe[0][k]-probe[1][k]))) for k in (0,1))
    if not np.isfinite(gap) or gap>1e-5:
        raise ValueError('Numerical repeat probe failed')
    counter=check_historical_counterexamples()
    OUT.mkdir()
    preflight={'status':'checked_before_first_update','official_rows':len(official),
               'ASA_rows':len(trace),'role_rows':int(len(pd.read_parquet(ROLES))),
               'planned_order':ORDER,'teacher_schedule':schedule,
               'branch_initial_sha256':tensor_hash(branch.state_dict()),
               'teacher_initial_sha256':tensor_hash(model.state_dict()),
               'probe_repeat_max_abs':gap,'historical_counterexamples':counter,
               'packages':{'torch':torch.__version__,'numpy':importlib.metadata.version('numpy'),
                           'scipy':importlib.metadata.version('scipy'),'pandas':importlib.metadata.version('pandas')},
               'cuda_device':torch.cuda.get_device_name(0)}
    save(OUT/'preflight.json',preflight)
    seal_run(Path(__file__),source_files()+[OUT/'preflight.json'],OUT/'run_seal.json')
    reg={'status':'registered_before_any_update','plan_sha256':sha(PLAN),
         'seal_sha256':sha(OUT/'run_seal.json'),'fit_order':ORDER,
         'classifier_fits':0,'optimizer_steps':0,'quality_acceptance':False,'model_promoted':False}
    save(OUT/'registration.json',reg)
    print(json.dumps({'stage':'registered','schedule':schedule,'numeric_probe':gap,
                      'historical_cases':counter},ensure_ascii=False),flush=True)


def registration():
    plan=require_run_seal(OUT/'run_seal.json',Path(__file__))
    reg=read(OUT/'registration.json')
    if reg['fit_order']!=ORDER or reg['seal_sha256']!=sha(OUT/'run_seal.json'):
        raise ValueError('Fit order or seal changed')
    for name,value in read(OUT/'preflight.json')['packages'].items():
        current=torch.__version__ if name=='torch' else importlib.metadata.version(name)
        if current!=value:raise ValueError('Package changed: '+name)
    return plan,reg


def next_unfinished():
    for fold,j,kind in ORDER:
        folder=OUT/(f'teacher{fold}_{j}' if kind=='teacher' else f'fold{fold}_{kind}')
        if not (folder/'fit.json').exists():
            return fold,j,kind
    return None


def ensure_next(fold,j,kind):
    if next_unfinished()!=(fold,j,kind):
        raise ValueError('Unregistered fit order or an interrupted fit needs inspection')


def fit_teacher(fold,j):
    plan,_=registration();ensure_next(fold,j,'teacher')
    folder=OUT/f'teacher{fold}_{j}'
    if folder.exists():raise FileExistsError('Interrupted teacher; preserve and diagnose: '+str(folder))
    roles=role_frame(fold);trace=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    check_roles(roles,trace,fold)
    fit=roles[roles.inner_fold!=j];excluded=roles[roles.inner_fold==j]
    c=counts(fit);used=np.flatnonzero(c.sum(1));nb=math.ceil(len(used)/256)
    if nb*25!=plan['teacher_bank']['expected_steps'][fold*3+j]:raise ValueError('Teacher steps changed')
    x=sparse.load_npz(ORIGINAL);h=sparse.load_npz(CANON);lengths=np.load(LENGTHS)
    model=Expert('A',10201).to(DEVICE)
    if tensor_hash(model.state_dict())!=read(OUT/'preflight.json')['teacher_initial_sha256']:
        raise ValueError('Teacher init changed')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    rng=np.random.default_rng(10201+fold)
    folder.mkdir();save(folder/'started.json',{'fold':fold,'excluded_inner':j,
        'fit_rows':len(fit),'fit_M':int(c[:,1].sum()),'fit_S':int(c[:,2].sum()),
        'fit_root_count':int(fit.root.nunique()),'excluded_root_count':int(excluded.root.nunique()),
        'fit_role_rows_sha256':sha(ROLES),'seal_sha256':sha(OUT/'run_seal.json'),'started_unix':time.time()})
    start=time.monotonic();steps=0;progress=[]
    with (folder/'steps.jsonl').open('w',encoding='utf-8') as log:
        for epoch in range(1,26):
            require_run_seal(OUT/'run_seal.json',Path(__file__))
            model.train();seen=np.zeros_like(c);online=0.0
            order=used[rng.permutation(len(used))]
            for offset in range(0,len(used),256):
                ids=order[offset:offset+256];ids=ids[np.argsort(lengths[ids],kind='stable')]
                seen[ids]+=c[ids]
                logits=model(x[ids]);w=torch.as_tensor(c[ids],device=DEVICE,dtype=torch.float32)
                cat=-(torch.log_softmax(logits,-1)*w[:,None,:]).sum((0,1))/16
                loss=cat.sum()/(len(fit)/nb)
                if not bool(torch.isfinite(loss).item()):raise FloatingPointError('Teacher CE nonfinite')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();steps+=1
                online+=float(cat.sum().detach())
                log.write(json.dumps({'step':steps,'epoch':epoch,'M_original_rows':int(c[ids,1].sum()),
                    'S_original_rows':int(c[ids,2].sum()),'M_member_CE_numerator':float(cat[1].detach()),
                    'S_member_CE_numerator':float(cat[2].detach()),'finite':True})+'\n')
            if not np.array_equal(seen,c) or steps!=epoch*nb:raise ValueError('Teacher mass/steps mismatch')
            item={'epoch':epoch,'steps':steps,'fit_M':int(c[:,1].sum()),'fit_S':int(c[:,2].sum()),
                  'online_original_row_member_CE':online/len(fit),'seconds':time.monotonic()-start,
                  'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())}
            progress.append(item);save(folder/'progress.json',progress);log.flush()
            print(json.dumps({'stage':'teacher_epoch','fold':fold,'excluded_inner':j,
                              'epoch':epoch,'steps':steps,'CE':item['online_original_row_member_CE']},
                             ensure_ascii=False),flush=True)
    model.eval();z=np.empty((22546,16,3),np.float32)
    with torch.no_grad():
        for start_id in range(0,22546,256):
            z[start_id:start_id+256]=model(h[start_id:start_id+256]).cpu().numpy()
    if not np.isfinite(z).all():raise FloatingPointError('Teacher scores nonfinite')
    np.save(folder/'canonical_member_logits.npy',z)
    torch.save({'base':{k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()},
        'fold':fold,'excluded_inner':j,'arm':'A','seed':10201,'epoch':25,'optimizer_steps':steps,
        'seal_sha256':sha(OUT/'run_seal.json')},folder/'epoch25_model.pt')
    # The rows scored out-of-fit are disjoint from the teacher's fit roots.
    diag={}
    p=softmax(z,axis=-1).mean(1).argmax(1)
    for name,frame in [('fit',fit),('crossfit',excluded)]:
        cc=counts(frame);diag[name]={str(cl):{'support':int(cc[:,cl].sum()),
             'errors':int((cc[:,cl]*(p!=cl)).sum())} for cl in (1,2)}
    receipt={'status':'fit_executed','kind':'teacher','fold':fold,'excluded_inner':j,
        'completed_epochs':25,'prediction_epoch':25,'optimizer_steps':steps,
        'fit_rows':len(fit),'fit_M':int(c[:,1].sum()),'fit_S':int(c[:,2].sum()),
        'fit_root_count':int(fit.root.nunique()),'excluded_root_count':int(excluded.root.nunique()),
        'role_diagnostic':diag,'base_state_sha256':tensor_hash(model.base.state_dict()),
        'model_sha256':sha(folder/'epoch25_model.pt'),'logits_sha256':sha(folder/'canonical_member_logits.npy'),
        'steps_sha256':sha(folder/'steps.jsonl'),'progress_sha256':sha(folder/'progress.json'),
        'seal_sha256':sha(OUT/'run_seal.json'),'seconds':time.monotonic()-start}
    require_checkpoint(plan,receipt,'teacher');save(folder/'fit.json',receipt)
    print(json.dumps({'stage':'teacher_complete','fold':fold,'excluded_inner':j,
                      'steps':steps,'diagnostic':diag,'seconds':round(receipt['seconds'],1)},ensure_ascii=False),flush=True)


def score_role(fold,role):
    roles=role_frame(fold)
    trace=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    check_roles(roles,trace,fold)
    fit_counts=counts(roles);inner=np.full(22546,-1,dtype=np.int8)
    pairs=roles[['local','inner_fold']].drop_duplicates()
    inner[pairs.local.to_numpy()]=pairs.inner_fold.to_numpy()
    if np.count_nonzero(inner>=0)!=np.count_nonzero(fit_counts.sum(1)):
        raise ValueError('Every fit input needs exactly one teacher role')
    z=np.zeros((22546,16,3),dtype=np.float32);provenance=[]
    for j in range(3):
        source=(j if role=='CF' else (j+1)%3)
        receipt=read(OUT/f'teacher{fold}_{source}/fit.json')
        require_checkpoint(check_plan(),receipt,'teacher')
        teacher_path=OUT/f'teacher{fold}_{source}/canonical_member_logits.npy'
        if sha(teacher_path)!=receipt['logits_sha256']:
            raise ValueError('Teacher score changed')
        target_rows=roles[roles.inner_fold==j]
        teacher_fit=roles[roles.inner_fold!=source]
        if role=='CF' and set(target_rows.root)&set(teacher_fit.root):
            raise ValueError('Leaking crossfit teacher')
        if role=='IS' and not set(target_rows.root)<=set(teacher_fit.root):
            raise ValueError('Matched teacher was not in-fit')
        ids=pairs.loc[pairs.inner_fold==j,'local'].to_numpy(np.int64)
        z[ids]=np.load(teacher_path,mmap_mode='r')[ids]
        provenance.append({'inner':j,'teacher_excluded_inner':source,'role':role,
            'original_rows':len(target_rows),'unique_inputs':len(ids),'roots':int(target_rows.root.nunique()),
            'teacher_state_sha256':receipt['model_sha256'],'teacher_logits_sha256':receipt['logits_sha256']})
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite role scores')
    return z,fit_counts,provenance


def endpoint_prob(outer,extra):
    return softmax(outer+extra[:,None,:],axis=-1).mean(1).astype(np.float32)


def role_diagnostic(z,extra,c):
    p=endpoint_prob(z,extra)
    mixed=(c[:,1]>0)&(c[:,2]>0)
    lp=z+extra[:,None,:]
    lp=(lp-logsumexp(lp,axis=-1)[:,:,None]).mean(1)
    pred=p.argmax(1)
    return {str(cl):{'support':int(c[:,cl].sum()),
            'errors':int((c[:,cl]*(pred!=cl)).sum()),
            'nonconflict_errors':int((c[:,cl]*(pred!=cl)*(~mixed)).sum()),
            'member_CE':float(-(c[:,cl]*lp[:,cl]).sum()/c[:,cl].sum()),
            'ensemble_CE':float(-(c[:,cl]*np.log(np.maximum(p[:,cl],1e-12))).sum()/c[:,cl].sum())}
            for cl in (1,2)}


def fit_constant(fold,role):
    plan,_=registration();arm='K_'+role;ensure_next(fold,-1,arm)
    folder=OUT/f'fold{fold}_{arm}'
    if folder.exists():raise FileExistsError('Interrupted constant fit: '+str(folder))
    z,c,provenance=score_role(fold,role)
    ids=np.flatnonzero(c.sum(1));z=z[ids].astype(np.float64);w=c[ids].astype(np.float64)
    n=float(w.sum());lam=plan['matched_primary']['constant_solver']['l2_coefficient']
    def obj(uv):
        b=np.array([uv[0],uv[1],-uv.sum()]);v=z+b
        lp=v-logsumexp(v,axis=-1)[:,:,None]
        value=float(-(w*lp.mean(1)).sum()/n+lam/2*np.dot(b,b))
        q=softmax(v,axis=-1).mean(1)
        gb=(q*w.sum(1)[:,None]-w).sum(0)/n+lam*b
        return value,np.array([gb[0]-gb[2],gb[1]-gb[2]])
    initial=obj(np.zeros(2))[0]
    folder.mkdir();save(folder/'started.json',{'fold':fold,'arm':arm,'role_provenance':provenance,
        'fit_class_mass':w.sum(0).astype(int).tolist(),'initial_objective':initial,
        'seal_sha256':sha(OUT/'run_seal.json')})
    start=time.monotonic();k=plan['matched_primary']['constant_solver']
    res=minimize(obj,np.zeros(2),method=k['method'],jac=True,options={
        'maxiter':k['maxiter'],'maxfun':k['maxfun'],'gtol':k['gtol'],
        'ftol':k['ftol'],'maxls':k['maxls']})
    value,grad=obj(res.x);b=np.array([res.x[0],res.x[1],-res.x.sum()])
    valid=bool(np.isfinite(value) and np.isfinite(grad).all() and
               np.max(np.abs(grad))<=k['independent_gradient_inf_max'] and
               value<=initial+k['objective_le_initial_plus'])
    outer=np.load(BASE/f'fold{fold}_base_member_logits.npy')
    prob=endpoint_prob(outer,b);np.save(folder/'endpoint_prob.npy',prob)
    receipt={'status':'fit_executed' if valid else 'invalid_constant_control','fold':fold,
        'arm':arm,'score_role':role,'bias':b.tolist(),'objective':value,
        'initial_objective':initial,'gradient_inf':float(np.max(np.abs(grad))),
        'solver_success':bool(res.success),'solver_message':str(res.message),
        'nit':int(res.nit),'nfev':int(res.nfev),'fit_class_mass':w.sum(0).astype(int).tolist(),
        'role_provenance':provenance,'prob_sha256':sha(folder/'endpoint_prob.npy'),
        'seal_sha256':sha(OUT/'run_seal.json'),'seconds':time.monotonic()-start}
    save(folder/'fit.json',receipt)
    print(json.dumps({'stage':'constant_complete','fold':fold,'arm':arm,'valid':valid,
                      'gradient_inf':receipt['gradient_inf'],'bias':receipt['bias']},ensure_ascii=False),flush=True)
    if not valid:raise ValueError('Registered constant solver failed stationarity')


def network_checkpoint(folder,model,fold,arm,epoch,steps,outer,train_scores,body,lengths,c,held_counts,used):
    model.eval();extra=np.empty((len(lengths),3),np.float32)
    with torch.no_grad():
        for offset in range(0,len(lengths),256):
            ids=np.arange(offset,min(offset+256,len(lengths)))
            extra[ids]=branch_batch(model,body,lengths,ids).cpu().numpy()
    prob=endpoint_prob(outer,extra)
    np.save(folder/f'epoch{epoch}_prob.npy',prob)
    np.save(folder/f'epoch{epoch}_residual.npy',extra)
    torch.save({'branch':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'fold':fold,'arm':arm,'seed':12701,'epoch':epoch,'optimizer_steps':steps,
        'seal_sha256':sha(OUT/'run_seal.json')},folder/f'epoch{epoch}_model.pt')
    train_diag=role_diagnostic(train_scores,extra,c)
    held_diag=role_diagnostic(outer,extra,held_counts)
    mass=c.sum(1).astype(np.float64)
    mean=(extra.astype(np.float64)*mass[:,None]).sum(0)/mass.sum()
    mean_prob=endpoint_prob(outer,mean)
    held=held_counts.sum(1)>0
    item={'epoch':epoch,'steps':steps,'train_role':train_diag,
          'outer_held':held_diag,'train_mean_residual':mean.tolist(),
          'held_local_flips_vs_train_mean':int(((prob.argmax(1)!=mean_prob.argmax(1))&held).sum()),
          'model_sha256':sha(folder/f'epoch{epoch}_model.pt'),
          'prob_sha256':sha(folder/f'epoch{epoch}_prob.npy'),
          'residual_sha256':sha(folder/f'epoch{epoch}_residual.npy')}
    return item


def fit_network(fold,role):
    plan,_=registration();arm='P_'+role;ensure_next(fold,-1,arm)
    folder=OUT/f'fold{fold}_{arm}'
    if folder.exists():raise FileExistsError('Interrupted network fit: '+str(folder))
    z,c,provenance=score_role(fold,role)
    outer=np.load(BASE/f'fold{fold}_base_member_logits.npy')
    body=np.load(BODY,mmap_mode='r');lengths=np.load(LENGTHS)
    manifest=pd.read_parquet(MANIFEST)
    fit,held,fc,hc=row_counts(manifest,fold,len(lengths))
    if not np.array_equal(fc,c):raise ValueError('Nested role mass differs from outer-fit mass')
    used=np.flatnonzero(c.sum(1));nb=math.ceil(len(used)/256);total=nb*50
    if total!=plan['matched_primary']['branch_steps_per_fold_each_arm'][fold]:
        raise ValueError('Branch update budget changed')
    branch=make_branch('P',12701,DEVICE)
    if tensor_hash(branch.state_dict())!=read(OUT/'preflight.json')['branch_initial_sha256']:
        raise ValueError('Branch init changed')
    opt=torch.optim.AdamW(branch.parameters(),lr=.0003,weight_decay=.0003)
    cache=torch.as_tensor(z,device=DEVICE,dtype=torch.float32)
    weight=torch.as_tensor(c,device=DEVICE,dtype=torch.float32)
    if cache.requires_grad:raise ValueError('Frozen teacher cache has gradient')
    rng=np.random.default_rng(12701+fold)
    folder.mkdir();save(folder/'started.json',{'fold':fold,'arm':arm,'score_role':role,
        'fit_rows':len(fit),'held_rows':len(held),'fit_class_mass':c.sum(0).tolist(),
        'role_provenance':provenance,'branch_initial_sha256':tensor_hash(branch.state_dict()),
        'seal_sha256':sha(OUT/'run_seal.json'),'started_unix':time.time()})
    start=time.monotonic();steps=0;checks=[];progress=[]
    checks.append(network_checkpoint(folder,branch,fold,arm,0,0,outer,z,body,lengths,c,hc,used))
    save(folder/'checkpoints.json',checks)
    with (folder/'steps.jsonl').open('w',encoding='utf-8') as log:
        for epoch in range(1,51):
            require_run_seal(OUT/'run_seal.json',Path(__file__))
            branch.train();seen=np.zeros_like(c)
            order=used[rng.permutation(len(used))]
            online=np.zeros(3)
            for offset in range(0,len(used),256):
                ids=order[offset:offset+256];ids=ids[np.argsort(lengths[ids],kind='stable')]
                seen[ids]+=c[ids]
                extra=branch_batch(branch,body,lengths,ids)
                lp=torch.log_softmax(cache[ids]+extra[:,None,:],-1).mean(1)
                category=-(lp*weight[ids]).sum(0)
                loss=category.sum()/(len(fit)/nb)
                if not bool(torch.isfinite(loss).item()):raise FloatingPointError('Branch CE nonfinite')
                steps+=1;lr=.0003*min(steps/math.ceil(.1*total),1)
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True);loss.backward()
                grad={name:float(torch.stack([p.grad.detach().square().sum() for p in module.parameters()
                      if p.grad is not None]).sum().sqrt().item()) for name,module in
                      [('embed',branch.embed),('encoder',branch.encoder),('final_norm',branch.final_norm),('head',branch.head)]}
                if not all(np.isfinite(v) for v in grad.values()):raise FloatingPointError('Branch gradient nonfinite')
                before={n:p.detach().clone() for n,p in branch.named_parameters()}
                opt.step()
                updates={name:float(torch.stack([(p.detach()-before[n]).square().sum()
                         for n,p in branch.named_parameters() if n.startswith(name+'.')]).sum().sqrt().item())
                         for name in ('embed','encoder','final_norm','head')}
                if not all(np.isfinite(v) for v in updates.values()):raise FloatingPointError('Branch update nonfinite')
                online+=category.detach().cpu().numpy()
                log.write(json.dumps({'step':steps,'epoch':epoch,'lr':lr,
                    'M_original_rows':int(c[ids,1].sum()),'S_original_rows':int(c[ids,2].sum()),
                    'M_member_CE_numerator':float(category[1].detach()),
                    'S_member_CE_numerator':float(category[2].detach()),
                    'module_gradient_norm':grad,'module_update_norm':updates,'finite':True})+'\n')
            if not np.array_equal(seen,c) or steps!=epoch*nb:
                raise ValueError('Branch mass or step mismatch')
            item={'epoch':epoch,'steps':steps,'M_original_rows':int(c[:,1].sum()),
                  'S_original_rows':int(c[:,2].sum()),'online_original_row_member_CE':float(online.sum()/len(fit)),
                  'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'seconds':time.monotonic()-start}
            progress.append(item);save(folder/'progress.json',progress);log.flush()
            if epoch in CHECKPOINTS:
                check=network_checkpoint(folder,branch,fold,arm,epoch,steps,outer,z,body,lengths,c,hc,used)
                checks.append(check);save(folder/'checkpoints.json',checks)
                print(json.dumps({'stage':'branch_checkpoint','fold':fold,'arm':arm,'epoch':epoch,
                    'steps':steps,'train_M_errors':check['train_role']['1']['errors'],
                    'train_S_errors':check['train_role']['2']['errors'],
                    'held_M_errors':check['outer_held']['1']['errors'],
                    'held_S_errors':check['outer_held']['2']['errors'],
                    'seconds':round(time.monotonic()-start,1)},ensure_ascii=False),flush=True)
    receipt={'status':'fit_executed','kind':'branch','fold':fold,'arm':arm,'seed':12701,
        'completed_epochs':50,'prediction_epoch':50,'optimizer_steps':steps,
        'fit_rows':len(fit),'held_rows':len(held),'fit_class_mass':c.sum(0).tolist(),
        'role_provenance':provenance,'model_sha256':sha(folder/'epoch50_model.pt'),
        'prob_sha256':sha(folder/'epoch50_prob.npy'),'residual_sha256':sha(folder/'epoch50_residual.npy'),
        'steps_sha256':sha(folder/'steps.jsonl'),'progress_sha256':sha(folder/'progress.json'),
        'checkpoints_sha256':sha(folder/'checkpoints.json'),
        'seal_sha256':sha(OUT/'run_seal.json'),'seconds':time.monotonic()-start}
    require_checkpoint(plan,receipt,'branch');save(folder/'fit.json',receipt)
    print(json.dumps({'stage':'branch_complete','fold':fold,'arm':arm,'steps':steps,
                      'seconds':round(receipt['seconds'],1)},ensure_ascii=False),flush=True)


def run():
    while True:
        item=next_unfinished()
        if item is None:break
        f,j,kind=item
        if kind=='teacher':fit_teacher(f,j)
        elif kind.startswith('K_'):fit_constant(f,kind[-2:])
        else:fit_network(f,kind[-2:])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('register','run','next'))
    args=parser.parse_args()
    if args.stage=='register':register()
    elif args.stage=='run':run()
    else:
        item=next_unfinished()
        if item is not None:
            f,j,kind=item
            if kind=='teacher':fit_teacher(f,j)
            elif kind.startswith('K_'):fit_constant(f,kind[-2:])
            else:fit_network(f,kind[-2:])


if __name__=='__main__':main()
