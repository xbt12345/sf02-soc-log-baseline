"""Only existing second-layer parameters update; all original TRAIN frequency."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import gc
import time
from collections import deque
import numpy as np
import pandas as pd
import torch
from v142_runtime import ROOT,OUT,BASE,PROBE,PLAN,read,save,sha,review,adversaries,seal_run,require_run_seal,endpoint
from v138_runtime import OUT as CACHE
from v138_train import configure,historical,quick_stats,emit
from v135_runtime import load_data,fit_context
from v135_model import tensor_hash
from v138_readout import full_gradient,probabilities,GradientBudgetExhausted,bounded_lbfgs_step
from v141_second_model import SecondRepresentation
from v137_issue_guard import protect_original_rows


def tensors(fold):
    st=torch.load(historical(fold)/'epoch100_model.pt',map_location='cpu',weights_only=True)['model']
    rs=torch.load(BASE/f'fold{fold}_C/endpoint_readout.pt',map_location='cpu',weights_only=True)['readout']
    h1=np.load(PROBE/f'fold{fold}_h1.npy');h2=np.load(CACHE/f'fold{fold}_hidden.npy')
    packed=torch.as_tensor(np.concatenate([h1,h2],axis=-1),device='cuda',dtype=torch.float64)
    facts=torch.as_tensor(np.load(CACHE/'facts.npy'),device='cuda',dtype=torch.float64)
    return packed,facts,SecondRepresentation(st,rs).to('cuda')


def register():
    p=review();configure()
    if OUT.exists():raise FileExistsError('No preparation overwrite')
    OUT.mkdir();x,d=load_data();extras={ROOT/'training/v142_evaluate.py',ROOT/'training/test_v142_second.py',ROOT/'training/v140_retention_check.py'};reports=[]
    for f in range(3):
        frame,c,pure,_,ids=fit_context(d,f);h,ff,m=tensors(f);q=probabilities(m,h,ff,np.arange(len(c)));old=np.load(BASE/f'fold{f}_C/sealed_all_prob.npy')
        if not np.array_equal(q,old):raise ValueError('Zero-step changed')
        mass=torch.as_tensor(c,device='cuda',dtype=torch.float64);l,s=full_gradient(m,h,ff,mass,ids);grads=[v.grad.clone() for v in m.parameters()]
        ll,ss=full_gradient(m,h,ff,mass,ids)
        if l!=ll or s!=ss or any(not torch.equal(a,b.grad) for a,b in zip(grads,m.parameters())):raise ValueError('Gradient not repeated exactly')
        # Construct solver before sealing to resolve its real lazy dependencies.
        torch.optim.LBFGS(m.parameters(),**p['solver'])
        frame[['row_position','local','root','fold','truth','canonical_key']].to_parquet(OUT/f'fold{f}_train_reference.parquet',index=False)
        np.save(OUT/f'fold{f}_zero_probability.npy',q)
        reports.append({'fold':f,'zero_step_probability_exact':True,'original_class_mass':s,'original_member_CE':l,'repeated_gradients_exact':True,
                        'trainable_existing_parameters':sum(v.numel() for v in m.parameters()),'baseline_stats':quick_stats(frame,q,pure,q[frame.local].argmax(1),[22,6,28][f])})
        extras.update({PROBE/f'fold{f}_h1.npy',BASE/f'fold{f}_C/endpoint_readout.pt',BASE/f'fold{f}_C/endpoint_original_rows.parquet',BASE/f'fold{f}_C/sealed_all_prob.npy',
                       OUT/f'fold{f}_zero_probability.npy',OUT/f'fold{f}_train_reference.parquet'})
        emit(stage='second_layer_zero_step',**reports[-1]);del h,ff,m,mass;gc.collect();torch.cuda.empty_cache()
    save(OUT/'preflight.json',{'folds':reports,'new_fits':0,'new_updates':0,'diagnostic_full_gradients':6,'negative_cases':adversaries(p)})
    # Resolve full evaluator source imports before capturing physical module files.
    import v142_evaluate
    extras.add(OUT/'preflight.json');seal_run(__file__,extras)
    save(OUT/'registration.json',{'status':'sealed_before_update','fits_max':3,'old_chain_fits':12,'old_chain_remaining':0,'seal_sha256':sha(OUT/'run_seal.json')})


def fit(fold):
    p=require_run_seal(__file__);configure()
    if fold not in p['folds']:raise ValueError('Unregistered fold')
    folder=OUT/f'fold{fold}_S2'
    if folder.exists() or len(list(OUT.glob('fold*_S2/started.json')))>=3:raise FileExistsError('No overwrite, resume or extra fitting')
    _,d=load_data();frame,c,pure,_,ids=fit_context(d,fold);h,facts,m=tensors(fold);mass=torch.as_tensor(c,device='cuda',dtype=torch.float64)
    before={k:v.detach().clone() for k,v in m.state_dict().items()};oldq=np.load(OUT/f'fold{fold}_zero_probability.npy');oldpred=oldq[frame.local].argmax(1)
    folder.mkdir();start=time.monotonic();evaluations=accepted=0;states=deque(maxlen=5);history=[]
    save(folder/'started.json',{'fold':fold,'arm':'S2','seal_sha256':sha(OUT/'run_seal.json'),'initial_state_sha256':tensor_hash(before),'held_labels_used':0})
    log=(folder/'closures.jsonl').open('w',encoding='utf-8',buffering=1);opt=torch.optim.LBFGS(m.parameters(),**p['solver'])
    def closure():
        nonlocal evaluations
        if evaluations>=200:raise GradientBudgetExhausted()
        loss,seen=full_gradient(m,h,facts,mass,ids);evaluations+=1
        import json
        log.write(json.dumps({'full_gradient_evaluation':evaluations,'original_member_CE':loss,'original_class_mass_seen':seen,'finite':True,
                              'gradient_norm':float(sum(v.grad.square().sum() for v in m.parameters()).sqrt()),'parameter_sha256':tensor_hash(m.state_dict()),'seconds':time.monotonic()-start})+'\n')
        return torch.tensor(loss,device='cuda',dtype=torch.float64)
    termination='gradient_budget'
    while evaluations<200 and accepted<200:
        status=bounded_lbfgs_step(opt,m,closure)
        if status!='accepted_changed_state':termination=status;break
        accepted+=1;q=probabilities(m,h,facts,np.arange(len(c)));s=quick_stats(frame,q,pure,oldpred,[22,6,28][fold])
        current={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};identity=tensor_hash(current)
        item={'accepted_update':accepted,'full_gradient_evaluations':evaluations,'stats':s,'parameter_sha256':identity,'seconds':time.monotonic()-start}
        states.append(dict(item,state=current));history.append(item);save(folder/'progress.json',history)
        if accepted==1 or accepted%20==0:emit(stage='second_layer_update',fold=fold,**item)
    if accepted==200:termination='accepted_update_budget'
    log.close();q=probabilities(m,h,facts,np.arange(len(c)));s=quick_stats(frame,q,pure,oldpred,[22,6,28][fold]);current={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    for key in before:
        if key.startswith(('reference_','head_')) and not torch.equal(current[key],before[key].cpu()):raise ValueError('Frozen state modified')
    for z in states:torch.save({'state':z['state'],'parameter_sha256':z['parameter_sha256']},folder/f"accepted{z['accepted_update']}.pt")
    torch.save({'state':current,'fold':fold,'arm':'S2','seal_sha256':sha(OUT/'run_seal.json')},folder/'endpoint.pt');np.save(folder/'sealed_all_prob.npy',q)
    rows=frame[['row_position','local','root','truth','canonical_key']].copy();rows['training_role']=fold;rows['pure_TRAIN_input']=pure[frame.local].astype(bool)
    rows['pred_start']=oldpred;rows['pred']=q[frame.local].argmax(1)
    for cl in range(3):rows[f'p{cl}']=q[frame.local,cl]
    rows.to_parquet(folder/'endpoint_original_rows.parquet',index=False)
    protected=frame.loc[pure[frame.local].astype(bool)&(oldpred==frame.truth),['row_position']]
    gate=protect_original_rows(frame[['row_position','truth']],pd.DataFrame({'row_position':frame.row_position,'pred':oldpred}),rows[['row_position','pred']],protected)
    delta={k:float((current[k]-before[k].cpu()).square().sum().sqrt()) for k in ['weight','r','s','bias']}
    r={'status':'fit_executed','fold':fold,'arm':'S2','full_gradient_evaluations':evaluations,'accepted_updates':accepted,'termination':termination,
       'candidate_selected_by_score':False,'original_class_mass_per_full_evaluation':c.sum(0).tolist(),'endpoint_stats':s,'parameter_delta_L2':delta,
       'last_five_distinct_accepted_states_mastered':len(states)==5 and len({z['parameter_sha256'] for z in states})==5 and all(z['stats']['mastered'] for z in states),
       'all_initial_correct_pure_guard':gate,'frozen_parameters_exact':True,'seconds':time.monotonic()-start,'seal_sha256':sha(OUT/'run_seal.json')}
    for name,key in [('endpoint.pt','model_sha256'),('sealed_all_prob.npy','probability_sha256'),('endpoint_original_rows.parquet','rows_sha256'),('closures.jsonl','closures_sha256'),('progress.json','progress_sha256')]:r[key]=sha(folder/name)
    endpoint(r);save(folder/'fit.json',r);emit(stage='second_layer_fit_complete',**r)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['register','fit']);a.add_argument('--fold',type=int);v=a.parse_args()
    if v.command=='register':register()
    else:fit(v.fold)
