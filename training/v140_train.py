"""Round two: original-row independent versus ensemble CE, fixed readout."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import gc
import time
from collections import deque
import numpy as np
import pandas as pd
import torch
from v140_runtime import ROOT,OUT,OLD,PLAN,read,save,sha,review,seal_run,require_run_seal,endpoint,adversaries
from v138_train import configure,cache_tensors,cache_backbone,quick_stats,historical,emit
from v138_readout import probabilities,GradientBudgetExhausted,bounded_lbfgs_step
from v140_objective import full_gradient,losses
from v135_runtime import load_data,fit_context
from v135_model import tensor_hash
from v137_issue_guard import protect_original_rows


def start_model(fold):
    h,f,m=cache_tensors(fold)
    st=torch.load(OLD/f'fold{fold}_H_L/endpoint_readout.pt',map_location='cpu',weights_only=True)
    m.load_state_dict(st['readout'])
    return h,f,m


def register():
    p=review(); configure()
    if OUT.exists(): raise FileExistsError('Preserve existing preparation and fits')
    x,d=load_data(); OUT.mkdir(); reports=[]; deps={ROOT/'training/v140_evaluate.py',ROOT/'training/test_v140_objective.py'}
    for f in range(3):
        frame,c,pure,_,ids=fit_context(d,f)
        state,h,_=cache_backbone(x,f)
        if not np.array_equal(h,np.load(OLD/f'fold{f}_hidden.npy')): raise ValueError('Actual backbone differs from cache')
        del state,h; gc.collect(); torch.cuda.empty_cache()
        hidden,facts,m=start_model(f)
        q=probabilities(m,hidden,facts,np.arange(len(c)))
        oldq=np.load(OLD/f'fold{f}_H_L/sealed_all_prob.npy')
        if not np.array_equal(q,oldq): raise ValueError('Zero-step endpoint probability changed')
        oldrows=pd.read_parquet(OLD/f'fold{f}_H_L/endpoint_original_rows.parquet')
        if not np.array_equal(oldrows.row_position,frame.row_position) or not np.array_equal(oldrows.truth,frame.truth): raise ValueError('Role identities changed')
        mass=torch.as_tensor(c,device='cuda',dtype=torch.float64); checks={}
        for arm in ['C','E']:
            l,seen=full_gradient(m,hidden,facts,mass,ids,arm); g=[v.grad.detach().clone() for v in m.parameters()]
            l2,seen2=full_gradient(m,hidden,facts,mass,ids,arm)
            if l!=l2 or seen!=seen2 or any(not torch.equal(a,b.grad) for a,b in zip(g,m.parameters())): raise ValueError('Non-deterministic repeated gradient')
            checks[arm]={'risk':l,'gradient_norm':float(sum(v.square().sum() for v in g).sqrt()),'original_class_mass':seen,'repeat_exact':True}
        np.save(OUT/f'fold{f}_zero_probability.npy',q)
        frame[['row_position','local','root','fold','truth','canonical_key']].to_parquet(OUT/f'fold{f}_train_reference.parquet',index=False)
        reports.append({'fold':f,'original_train_rows':len(frame),'start_stats':quick_stats(frame,q,pure,oldrows.pred.to_numpy(),[22,6,28][f]),
                        'zero_step_exact':True,'actual_backbone_cache_exact':True,'objectives':checks,'parameters':sum(v.numel() for v in m.parameters())})
        deps.update({OLD/f'fold{f}_H_L/endpoint_readout.pt',OLD/f'fold{f}_H_L/sealed_all_prob.npy',OLD/f'fold{f}_H_L/endpoint_original_rows.parquet',
                     OUT/f'fold{f}_zero_probability.npy',OUT/f'fold{f}_train_reference.parquet'})
        emit(stage='new_objective_zero_step',**reports[-1]); del m,hidden,facts,mass;gc.collect();torch.cuda.empty_cache()
    save(OUT/'preflight.json',{'status':'zero_step_actual_backbone_and_both_gradients_verified','folds':reports,'new_fits':0,'new_updates':0,
                             'diagnostic_gradient_evaluations':12,'adversaries':adversaries(p),'device':torch.cuda.get_device_name(0)})
    deps.update({OUT/'preflight.json',ROOT/'training/v140_objective.py',ROOT/'training/v138_retention_check.py'})
    seal_run(__file__,deps)
    save(OUT/'registration.json',{'status':'registered_before_update','fits_max':6,'chain_old_fits':6,'chain_max':12,'seal_sha256':sha(OUT/'run_seal.json')})


def run_fit(fold,arm):
    p=require_run_seal(__file__); configure()
    if fold not in p['folds'] or arm not in p['arms']: raise ValueError('Unregistered fit')
    folder=OUT/f'fold{fold}_{arm}'
    if folder.exists(): raise FileExistsError('No overwrite or partial-fit resume')
    # Every started fit consumes a slot; no hidden retry budget.
    if len(list(OUT.glob('fold*_*/started.json')))>=6: raise ValueError('Fit budget exhausted')
    x,d=load_data();frame,c,pure,_,ids=fit_context(d,fold);del x
    hidden,facts,model=start_model(fold);mass=torch.as_tensor(c,device='cuda',dtype=torch.float64)
    oldq=np.load(OUT/f'fold{fold}_zero_probability.npy');oldpred=oldq[frame.local].argmax(1)
    floor=[22,6,28][fold]; folder.mkdir();start=time.monotonic();evaluations=accepted=0;states=deque(maxlen=5);history=[]
    save(folder/'started.json',{'fold':fold,'arm':arm,'seal_sha256':sha(OUT/'run_seal.json'),'start_state_sha256':tensor_hash(model.state_dict()),
                              'original_class_mass':c.sum(0).tolist(),'held_labels_used':0})
    log=(folder/'closures.jsonl').open('w',encoding='utf-8',buffering=1)
    opt=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=1,max_eval=200,tolerance_grad=1e-7,tolerance_change=1e-9,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        nonlocal evaluations
        if evaluations>=200: raise GradientBudgetExhausted('Registered gradient budget')
        loss,seen=full_gradient(model,hidden,facts,mass,ids,arm);evaluations+=1
        norm=float(sum(v.grad.square().sum() for v in model.parameters()).sqrt())
        import json
        log.write(json.dumps({'full_gradient_evaluation':evaluations,'objective':loss,'objective_arm':arm,'original_class_mass_seen':seen,
                              'gradient_norm':norm,'seconds':time.monotonic()-start,'parameter_sha256':tensor_hash(model.state_dict()),'finite':True})+'\n')
        return torch.tensor(loss,device='cuda',dtype=torch.float64)
    termination='gradient_budget'
    while evaluations<200 and accepted<200:
        status=bounded_lbfgs_step(opt,model,closure)
        if status!='accepted_changed_state': termination=status; break
        accepted+=1
        current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};identity=tensor_hash(current)
        q=probabilities(model,hidden,facts,np.arange(len(c)));s=quick_stats(frame,q,pure,oldpred,floor)
        item={'accepted_update':accepted,'full_gradient_evaluations':evaluations,'stats':s,'parameter_sha256':identity,'seconds':time.monotonic()-start}
        if states and states[-1]['parameter_sha256']==identity: raise ValueError('Repeated state counted')
        history.append(item);states.append(dict(item,state=current));save(folder/'progress.json',history)
        if accepted%50==0 or accepted==1: emit(stage='objective_fit_update',fold=fold,arm=arm,**item)
    if accepted==200: termination='accepted_update_budget'
    log.close();q=probabilities(model,hidden,facts,np.arange(len(c)));s=quick_stats(frame,q,pure,oldpred,floor)
    window=len(states)==5 and len({v['parameter_sha256'] for v in states})==5 and all(v['stats']['mastered'] for v in states)
    for z in states: torch.save({'readout':z['state'],'parameter_sha256':z['parameter_sha256']},folder/f"accepted{z['accepted_update']}_readout.pt")
    st={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial=torch.load(OLD/f'fold{fold}_H_L/endpoint_readout.pt',map_location='cpu',weights_only=True)['readout']
    delta={k:float((v-initial[k].cpu()).square().sum().sqrt()) for k,v in st.items()}
    torch.save({'readout':st,'fold':fold,'arm':arm,'seal_sha256':sha(OUT/'run_seal.json'),'source_model_sha256':sha(historical(fold)/'epoch100_model.pt')},folder/'endpoint_readout.pt')
    np.save(folder/'sealed_all_prob.npy',q)
    ledger=frame[['row_position','local','root','truth','canonical_key']].copy();ledger['training_role']=fold;ledger['pure_TRAIN_input']=pure[frame.local].astype(bool)
    ledger['pred_start']=oldpred;ledger['pred']=q[frame.local].argmax(1)
    for cl in range(3): ledger[f'p{cl}']=q[frame.local,cl]
    ledger.to_parquet(folder/'endpoint_original_rows.parquet',index=False)
    gate=protect_original_rows(frame[['row_position','truth']],pd.DataFrame({'row_position':frame.row_position,'pred':oldpred}),ledger[['row_position','pred']],
                               frame.loc[pure[frame.local].astype(bool)&(oldpred==frame.truth),['row_position']])
    r={'status':'fit_executed','round':2,'fold':fold,'arm':arm,'full_gradient_evaluations':evaluations,'accepted_updates':accepted,'termination':termination,
       'candidate_selected_by_score':False,'original_class_mass_per_full_evaluation':c.sum(0).tolist(),'endpoint_stats':s,'last_five_distinct_accepted_states_mastered':window,
       'all_old_correct_pure_input_guard':gate,'seconds':time.monotonic()-start,'parameter_delta_L2':delta,'endpoint_losses':losses(model,hidden,facts,mass,ids),
       'seal_sha256':sha(OUT/'run_seal.json')}
    for name,key in [('endpoint_readout.pt','model_sha256'),('sealed_all_prob.npy','probability_sha256'),('closures.jsonl','closures_sha256'),('progress.json','progress_sha256'),('endpoint_original_rows.parquet','rows_sha256')]: r[key]=sha(folder/name)
    endpoint(r);save(folder/'fit.json',r);emit(stage='fit_complete',**r)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['register','fit']);a.add_argument('--fold',type=int);a.add_argument('--arm');args=a.parse_args()
    if args.command=='register': register()
    else: run_fit(args.fold,args.arm)
