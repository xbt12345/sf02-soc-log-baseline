"""Execute V137 round 1. Only original head/facts readout parameters update."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import gc
import time
from collections import deque
import numpy as np
import pandas as pd
import torch
from v138_runtime import ROOT,OUT,PLAN,CONTRACT,read,save,sha,review_execution,seal_run,require_run_seal,endpoint
from v135_runtime import load_data,fit_context,OUT as OLD,stats
from v135_model import Classifier,batch,tensor_hash
from v138_readout import Readout,full_gradient,probabilities,GradientBudgetExhausted,bounded_lbfgs_step
from v137_issue_guard import adversaries,protect_original_rows


def configure():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False


def emit(**data):
    import json
    print(json.dumps(data,ensure_ascii=False),flush=True)


def historical(fold):return OLD/f'fold{fold}_R_decay'


def cache_backbone(x,fold):
    state=torch.load(historical(fold)/'epoch100_model.pt',map_location='cpu',weights_only=True)
    model=Classifier(128).to('cuda');model.load_state_dict(state['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    h=[];q=[]
    with torch.no_grad():
        for start in range(0,x.shape[0],256):
            ids=np.arange(start,min(start+256,x.shape[0]));z,_,h2=batch(model,x,ids,True)
            h.append(h2.cpu().numpy());q.append(torch.softmax(z,-1).mean(1).cpu().numpy())
    hidden=np.concatenate(h);oldq=np.concatenate(q)
    del model;gc.collect();torch.cuda.empty_cache()
    return state,hidden,oldq


def cache_tensors(fold):
    hidden=torch.as_tensor(np.load(OUT/f'fold{fold}_hidden.npy'),device='cuda',dtype=torch.float64)
    facts=torch.as_tensor(np.load(OUT/'facts.npy'),device='cuda',dtype=torch.float64)
    state=torch.load(historical(fold)/'epoch100_model.pt',map_location='cpu',weights_only=True)
    model=Readout(state['model']).to('cuda');del state
    return hidden,facts,model


def register():
    p=review_execution();configure()
    if OUT.exists():raise FileExistsError('Preserve all prior results')
    x,d=load_data();x.sort_indices()
    # Independently verify the actual numerical input identities before caching.
    import hashlib
    keys=[hashlib.sha256(x.indices[x.indptr[i]:x.indptr[i+1]].astype('<i8').tobytes()+x.data[x.indptr[i]:x.indptr[i+1]].astype('<f4').tobytes()).hexdigest() for i in range(len(x.indptr)-1)]
    if not np.array_equal(d.canonical_key,d.local.map(dict(enumerate(keys)))):raise ValueError('Input identities changed')
    OUT.mkdir()
    # Use the authoritative feature boundary, never infer it from a magic count.
    from v75_views import BYTE_FEATURES
    facts=x[:,BYTE_FEATURES:].toarray()
    if facts.shape!=(22546,495):raise ValueError('Fact representation changed')
    np.save(OUT/'facts.npy',facts);reports=[];dependencies={OUT/'facts.npy',ROOT/'training/v138_evaluate.py',ROOT/'training/v138_feasibility.py',ROOT/'training/test_v138_readout.py'}
    for f in range(3):
        frame,c,pure,totals,ids=fit_context(d,f)
        frame[['row_position','local','root','fold','truth','canonical_key']].to_parquet(OUT/f'fold{f}_train_reference.parquet',index=False)
        state,h,oldq=cache_backbone(x,f);np.save(OUT/f'fold{f}_hidden.npy',h)
        m=Readout(state['model']).to('cuda');hh=torch.as_tensor(h,dtype=torch.float64,device='cuda');ff=torch.as_tensor(facts,dtype=torch.float64,device='cuda')
        q=probabilities(m,hh,ff,np.arange(len(c)));repeat=probabilities(m,hh,ff,np.arange(len(c)))
        saved=np.load(historical(f)/'sealed_all_prob.npy')
        gap=float(np.abs(q-saved).max())
        if gap>2e-6 or not np.array_equal(q.argmax(1),saved.argmax(1)):raise ValueError('Zero-step cache/cast changed decisions')
        if not np.array_equal(oldq,saved) or not np.array_equal(q,repeat):raise ValueError('Actual backbone or repeated readout not deterministic')
        mass=torch.as_tensor(c,dtype=torch.float64,device='cuda')
        l,seen=full_gradient(m,hh,ff,mass,ids);g=[v.grad.detach().clone() for v in m.parameters()]
        l2,seen2=full_gradient(m,hh,ff,mass,ids)
        if l!=l2 or seen!=seen2 or any(not torch.equal(a,b.grad) for a,b in zip(g,m.parameters())):raise ValueError('Repeated full-role gradient changed')
        oldrows=pd.read_parquet(historical(f)/'epochs/epoch100_rows.parquet')
        if not np.array_equal(frame.row_position,oldrows.row_position) or not np.array_equal(frame.truth,oldrows.truth):raise ValueError('Original reference changed')
        s=stats(frame,q,pure,oldrows.pred.to_numpy())
        np.save(OUT/f'fold{f}_zero_probability.npy',q)
        reports.append({'fold':f,'original_train_rows':len(frame),'original_class_mass':c.sum(0).tolist(),'pure_class_mass':totals.tolist(),
            'zero_step_probability_gap':gap,'zero_step_predictions_exact':True,'repeated_full_gradient_exact':True,'original_member_CE':l,'baseline_stats':s,
            'source_model_sha256':sha(historical(f)/'epoch100_model.pt'),'cache_dtype':'float32-original; float64-readout',
            'frozen_modules':['first','second'],'trainable_parameters':sum(v.numel() for v in m.parameters())})
        dependencies.update([historical(f)/'epoch100_model.pt',historical(f)/'sealed_all_prob.npy',historical(f)/'epochs/epoch100_rows.parquet',
            OUT/f'fold{f}_hidden.npy',OUT/f'fold{f}_zero_probability.npy',OUT/f'fold{f}_train_reference.parquet'])
        emit(stage='zero_step_verified',**reports[-1]);del m,hh,ff,mass,state,h;gc.collect();torch.cuda.empty_cache()
    registry=read(ROOT/p['retention']['registry_path'])
    for e in registry['engineering_contracts']:
        if sha(ROOT/e['evidence'])!=e['evidence_sha256']:raise ValueError('Old repair evidence changed')
    save(OUT/'preflight.json',{'status':'all_three_zero_steps_and_original_mass_verified','folds':reports,'negative_design_cases':adversaries(p),
        'new_fits':0,'new_updates':0,'device':torch.cuda.get_device_name(0),'deterministic_algorithms':True,
        'engineering_retention':'Actual immutable x/row mass and repeated active-dense backbone/readout, unchanged input/clock contract hashes',
        'runtime_api':'Shared experiment_review bindings plus new tested budgeted-closure endpoint adapter; old fixed-epoch API not misrepresented.'})
    seal_run(__file__,dependencies|{OUT/'preflight.json'})
    save(OUT/'registration.json',{'status':'registered_before_any_update','issue':'TRAIN-PURE-READOUT','round':1,'fits_max':6,
        'candidate':'H_L','seal_sha256':sha(OUT/'run_seal.json'),'new_fits':0,'new_updates':0})


def quick_stats(frame,q,pure,oldpred,floor):
    loc=frame.local.to_numpy();y=frame.truth.to_numpy();pred=q[loc].argmax(1);pr=pure[loc].astype(bool);wrong=pred!=y
    result={k:int(v.sum()) for k,v in {'M_errors':wrong&(y==1),'S_errors':wrong&(y==2),'pure_M_errors':wrong&pr&(y==1),
        'pure_S_errors':wrong&pr&(y==2),'old_correct_pure_regressions':wrong&pr&(oldpred==y),
        'repaired_vs_start':(oldpred!=y)&~wrong,'new_errors_vs_start':(oldpred==y)&wrong}.items()}
    result['empirical_minimum_errors']=floor
    result['mastered']=result['M_errors']==0 and result['S_errors']==floor and result['pure_M_errors']==result['pure_S_errors']==0
    return result


def run_fit(fold,arm):
    p=require_run_seal(__file__);configure()
    if fold not in [0,1,2] or arm not in ['H_A','H_L']:raise ValueError('Unregistered fit')
    folder=OUT/f'fold{fold}_{arm}'
    if folder.exists():raise FileExistsError('Do not resume or overwrite partial fits')
    x,d=load_data();frame,c,pure,_,ids=fit_context(d,fold);del x
    hidden,facts,model=cache_tensors(fold);mass=torch.as_tensor(c,device='cuda',dtype=torch.float64)
    originalq=np.load(OUT/f'fold{fold}_zero_probability.npy');oldpred=originalq[frame.local].argmax(1)
    floor=p['round1']['mastery']['full_TRAIN_S_errors_max_by_fold'][fold]
    folder.mkdir();start=time.monotonic();evaluations=accepted=0;states=deque(maxlen=5);history=[]
    save(folder/'started.json',{'fold':fold,'arm':arm,'seal_sha256':sha(OUT/'run_seal.json'),'original_class_mass':c.sum(0).tolist(),
        'start_state_sha256':tensor_hash(model.state_dict()),'held_labels_used_for_fitting_or_stopping':0})
    save(folder/'progress.json',history)
    closure_log=(folder/'closures.jsonl').open('w',encoding='utf-8',buffering=1)
    opt=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=0) if arm=='H_A' else torch.optim.LBFGS(model.parameters(),
        lr=1,max_iter=1,max_eval=200,tolerance_grad=1e-7,tolerance_change=1e-9,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        nonlocal evaluations
        if evaluations>=200:raise GradientBudgetExhausted('Registered full-gradient budget')
        loss,seen=full_gradient(model,hidden,facts,mass,ids);evaluations+=1
        norm=float(sum(v.grad.square().sum() for v in model.parameters()).sqrt())
        import json
        closure_log.write(json.dumps({'full_gradient_evaluation':evaluations,'original_member_CE':loss,'original_class_mass_seen':seen,
            'gradient_norm':norm,'seconds':time.monotonic()-start,'parameter_sha256':tensor_hash(model.state_dict()),'finite':True})+'\n')
        return torch.tensor(loss,device='cuda',dtype=torch.float64)
    termination='fixed_update_budget' if arm=='H_A' else 'gradient_budget'
    while evaluations<200 and accepted<200:
        if arm=='H_A':
            closure();opt.step();status='accepted_changed_state'
        else:
            status=bounded_lbfgs_step(opt,model,closure)
            if status!='accepted_changed_state':termination=status;break
        accepted+=1
        current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};identity=tensor_hash(current)
        if states and identity==states[-1]['parameter_sha256']:raise ValueError('No-op counted as learning state')
        q=probabilities(model,hidden,facts,np.arange(len(c)));s=quick_stats(frame,q,pure,oldpred,floor)
        item={'accepted_update':accepted,'full_gradient_evaluations':evaluations,'stats':s,'parameter_sha256':identity,'seconds':time.monotonic()-start}
        history.append(item);states.append(dict(item,state=current))
        save(folder/'progress.json',history)
        if accepted%20==0 or accepted==1:emit(stage='readout_update',fold=fold,arm=arm,**item)
    if arm=='H_L' and accepted==200:termination='accepted_update_budget'
    closure_log.close()
    q=probabilities(model,hidden,facts,np.arange(len(c)));s=quick_stats(frame,q,pure,oldpred,floor)
    fixed_window=len(states)==5 and all(v['stats']['mastered'] for v in states) and len({v['parameter_sha256'] for v in states})==5
    for z in states:torch.save({'readout':z['state'],'accepted_update':z['accepted_update'],'parameter_sha256':z['parameter_sha256']},folder/f"accepted{z['accepted_update']}_readout.pt")
    torch.save({'readout':model.state_dict(),'source_model_sha256':sha(historical(fold)/'epoch100_model.pt'),'fold':fold,'arm':arm,
        'accepted_updates':accepted,'full_gradient_evaluations':evaluations,'seal_sha256':sha(OUT/'run_seal.json')},folder/'endpoint_readout.pt')
    np.save(folder/'sealed_all_prob.npy',q)
    ledger=frame[['row_position','local','root','truth','canonical_key']].copy();ledger['training_role']=fold
    ledger['pure_TRAIN_input']=pure[frame.local].astype(bool);ledger['pred_start']=oldpred;ledger['pred']=q[frame.local].argmax(1)
    for cl in range(3):ledger[f'p{cl}']=q[frame.local,cl]
    ledger.to_parquet(folder/'endpoint_original_rows.parquet',index=False)
    protect=frame.loc[pure[frame.local].astype(bool)&(oldpred==frame.truth),['row_position']]
    gate=protect_original_rows(frame[['row_position','truth']],pd.DataFrame({'row_position':frame.row_position,'pred':oldpred}),ledger[['row_position','pred']],protect)
    receipt={'status':'fit_executed','round':1,'fold':fold,'arm':arm,'full_gradient_evaluations':evaluations,'accepted_updates':accepted,
        'termination':termination,'candidate_selected_by_score':False,'original_class_mass_per_full_evaluation':c.sum(0).tolist(),
        'original_row_exposures':[int(z*evaluations) for z in c.sum(0)],'endpoint_stats':s,'last_five_distinct_accepted_states_mastered':fixed_window,
        'all_old_correct_pure_input_guard':gate,'seconds':time.monotonic()-start,'model_sha256':sha(folder/'endpoint_readout.pt'),
        'probability_sha256':sha(folder/'sealed_all_prob.npy'),'closures_sha256':sha(folder/'closures.jsonl'),'progress_sha256':sha(folder/'progress.json'),
        'rows_sha256':sha(folder/'endpoint_original_rows.parquet'),'seal_sha256':sha(OUT/'run_seal.json')}
    endpoint(receipt);save(folder/'fit.json',receipt);emit(stage='fit_complete',**receipt)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','seal-prepared','fit']);parser.add_argument('--fold',type=int);parser.add_argument('--arm')
    args=parser.parse_args()
    if args.command=='register':register()
    elif args.command=='seal-prepared':
        # Recover only a complete zero-update preflight. Never resume a fit.
        review_execution()
        if (OUT/'run_seal.json').exists() or list(OUT.glob('*_H_*')):raise ValueError('Cannot reseal after fitting')
        pre=read(OUT/'preflight.json')
        if pre['new_fits'] or pre['new_updates'] or len(pre['folds'])!=3:raise ValueError('Incomplete preparation')
        extras={OUT/'facts.npy',OUT/'preflight.json',ROOT/'training/v138_evaluate.py',ROOT/'training/v138_feasibility.py',ROOT/'training/test_v138_readout.py'}
        for f in range(3):
            extras.update([historical(f)/'epoch100_model.pt',historical(f)/'sealed_all_prob.npy',historical(f)/'epochs/epoch100_rows.parquet',
                OUT/f'fold{f}_hidden.npy',OUT/f'fold{f}_zero_probability.npy',OUT/f'fold{f}_train_reference.parquet'])
        save(OUT/'preparation_issue.json',{'stage':'before_optimizer_or_seal','new_fits':0,'new_updates':0,
            'error':'PyTorch generated _classes.py module metadata is not an actual source file',
            'action':'Bind existing real source files; no missing explicit dependency is allowed; preserve complete preflight',
            'console_sha256':sha(ROOT/'artifacts/v138_registration_console.log')})
        extras.add(OUT/'preparation_issue.json');seal_run(__file__,extras)
        save(OUT/'registration.json',{'status':'registered_before_any_update','issue':'TRAIN-PURE-READOUT','round':1,'fits_max':6,
            'candidate':'H_L','seal_sha256':sha(OUT/'run_seal.json'),'new_fits':0,'new_updates':0})
    else:run_fit(args.fold,args.arm)


if __name__=='__main__':main()
