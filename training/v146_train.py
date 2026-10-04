"""Fresh entry: CE vs restricted auxiliary, same guarded Armijo in all roles."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,time,json
from collections import deque
import numpy as np
import pandas as pd
import torch
from v146_runtime import ROOT,OUT,BASE,PLAN,SOLVER,read,save,sha,review,adversaries,seal,require,endpoint
from v142_train import tensors
from v138_train import configure,quick_stats,emit
from v135_runtime import load_data
from v135_model import tensor_hash
from v138_readout import full_gradient,probabilities
from v144_pair_objective import build_pairs,auxiliary
from v146_optimizer import assign,restore,acceptable


def initialize(f):
    h,ff,m=tensors(f);m.load_state_dict(torch.load(BASE/f'fold{f}_S2/endpoint.pt',map_location='cpu',weights_only=True)['state']);return h,ff,m


def objective_gradient(m,h,ff,c,ids,p,arm,lam):
    ce,seen=full_gradient(m,h,ff,c,ids);cegrad=[v.grad.detach().clone() for v in m.parameters()]
    if arm=='B':
        a=auxiliary(m,h,p);(lam*a).backward();aux=float(a.detach())
    else:
        with torch.no_grad():aux=float(auxiliary(m,h,p))
    if any(not torch.isfinite(v.grad).all() for v in m.parameters()):raise FloatingPointError('Nonfinite objective derivative')
    return dict(objective=ce+(lam*aux if arm=='B' else 0),original_member_CE=ce,auxiliary_loss=aux,original_class_mass_seen=seen,
        CE_gradient_norm=float(sum(v.square().sum() for v in cegrad).sqrt()),gradient_norm=float(sum(v.grad.square().sum() for v in m.parameters()).sqrt()),
        weighted_auxiliary_gradient_norm=0. if arm=='A' else float(sum((v.grad-a).square().sum() for v,a in zip(m.parameters(),cegrad)).sqrt()))


@torch.no_grad()
def values(m,h,ff,c,ids,p,arm,lam):
    prob=np.zeros((len(c),3),np.float64);class_losses=np.zeros(3);seen=np.zeros(3)
    for start in range(0,len(ids),2048):
        take=ids[start:start+2048];lp=torch.log_softmax(m(h[take],ff[take]),-1)
        prob[take]=lp.exp().mean(1).cpu().numpy();class_losses+=(-lp.mean(1).cpu().numpy()*c[take]).sum(0);seen+=c[take].sum(0)
    ce=float(class_losses.sum()/c.sum());aux=float(auxiliary(m,h,p))
    return dict(objective=ce+(lam*aux if arm=='B' else 0),original_member_CE=ce,auxiliary_loss=aux,
        per_class_member_CE=(class_losses/np.maximum(c.sum(0),1)).tolist(),original_class_mass_seen=seen.tolist()),prob


def register():
    p=review();configure()
    if OUT.exists():raise FileExistsError('No registration overwrite')
    _,d=load_data();facts=pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',columns=['facts_json']).facts_json
    evidence=read(ROOT/'artifacts/v144_aux_gradient_evidence_20261001/probe.json');OUT.mkdir();reports=[]
    extras={ROOT/'training/v146_evaluate.py',ROOT/'training/test_v146_guarded_optimizer.py',ROOT/'training/v142_retention_check.py'}
    for f in range(3):
        frame,c,pure,ids,pair,capacity=build_pairs(d,facts,f);h,ff,m=initialize(f);identity=tensor_hash(m.state_dict());q=probabilities(m,h,ff,np.arange(len(c)))
        if not np.array_equal(q,np.load(BASE/f'fold{f}_S2/sealed_all_prob.npy')):raise ValueError('V142 zero function changed')
        mass=torch.as_tensor(c,device='cuda',dtype=torch.float64);lam=evidence['folds'][f]['fixed_initial_auxiliary_coefficient'];checks=[]
        for arm in ['A','B']:
            a=objective_gradient(m,h,ff,mass,ids,pair,arm,lam);g=[v.grad.clone() for v in m.parameters()];b=objective_gradient(m,h,ff,mass,ids,pair,arm,lam)
            if a!=b or any(not torch.equal(x,v.grad) for x,v in zip(g,m.parameters())):raise ValueError('Preflight gradient not repeated exactly')
            val,qq=values(m,h,ff,c,ids,pair,arm,lam)
            if abs(val['objective']-a['objective'])>1e-12 or not np.array_equal(qq[ids].argmax(1),q[ids].argmax(1)):raise ValueError('Forward objective/gradient mismatched')
            checks.append(dict(arm=arm,gradient_exact=True,objective_forward_gap=abs(val['objective']-a['objective']),**a))
        if tensor_hash(m.state_dict())!=identity:raise ValueError('Preflight updated model')
        np.save(OUT/f'fold{f}_zero_probability.npy',q);pair['cells'].to_parquet(OUT/f'fold{f}_pair_reference.parquet',index=False)
        frame[['row_position','local','root','truth','canonical_key']].to_parquet(OUT/f'fold{f}_original_reference.parquet',index=False)
        save(OUT/f'fold{f}_fixed_lambda.json',dict(coefficient=lam,initial_parameter_sha256=identity,capacity=capacity))
        extras.update({BASE/f'fold{f}_S2/endpoint.pt',BASE/f'fold{f}_S2/sealed_all_prob.npy',OUT/f'fold{f}_zero_probability.npy',OUT/f'fold{f}_pair_reference.parquet',OUT/f'fold{f}_original_reference.parquet',OUT/f'fold{f}_fixed_lambda.json'})
        reports.append(dict(fold=f,zero_exact=True,capacity=capacity,objectives=checks));emit(stage='guarded_preflight',**reports[-1]);del h,ff,m,mass;gc.collect();torch.cuda.empty_cache()
    save(OUT/'preflight.json',dict(status='zero_updates_verified',folds=reports,new_fits=0,new_updates=0,CE_gradient_evaluations=12,auxiliary_gradient_evaluations=6,negative_cases=adversaries(p)))
    import v146_evaluate
    extras.add(OUT/'preflight.json');seal(__file__,extras);save(OUT/'registration.json',dict(status='sealed_before_update',fits_max=6,seal_sha256=sha(OUT/'run_seal.json')))


def fit(f,arm):
    plan=require(__file__);configure()
    if f not in plan['folds'] or arm not in plan['arms']:raise ValueError('Wrong arm/role')
    folder=OUT/f'fold{f}_{arm}'
    if folder.exists() or len(list(OUT.glob('fold*_*/started.json')))>=6:raise FileExistsError('No fitting overwrite/resume or extra fit')
    _,d=load_data();facts=pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',columns=['facts_json']).facts_json
    frame,c,pure,ids,pair,capacity=build_pairs(d,facts,f)
    if not pair['cells'].equals(pd.read_parquet(OUT/f'fold{f}_pair_reference.parquet')):raise ValueError('Pair reference changed')
    h,ff,m=initialize(f);parameters=tuple(m.parameters());mass=torch.as_tensor(c,device='cuda',dtype=torch.float64)
    before={k:v.detach().clone() for k,v in m.state_dict().items()};lam=read(OUT/f'fold{f}_fixed_lambda.json')['coefficient'];oldq=np.load(OUT/f'fold{f}_zero_probability.npy');oldpred=oldq[frame.local].argmax(1)
    folder.mkdir();save(folder/'started.json',dict(fold=f,arm=arm,seal_sha256=sha(OUT/'run_seal.json'),initial_parameter_sha256=tensor_hash(before),held_labels_used=0,lambda_fixed=lam))
    start=time.monotonic();gradients=proposals=accepted=0;history=[];last=deque(maxlen=5);step=SOLVER['initial_step'];termination='gradient_budget'
    log=(folder/'gradients.jsonl').open('w',encoding='utf-8',buffering=1);trials=(folder/'proposals.jsonl').open('w',encoding='utf-8',buffering=1)
    while gradients<SOLVER['max_gradients'] and accepted<SOLVER['max_updates']:
        if proposals>=SOLVER['max_proposals']:termination='proposal_budget';break
        info=objective_gradient(m,h,ff,mass,ids,pair,arm,lam);gradients+=1;norm=info['gradient_norm']
        log.write(json.dumps(dict(full_gradient_evaluation=gradients,finite=True,parameter_sha256=tensor_hash(m.state_dict()),**info))+'\n')
        if norm<=SOLVER['zero_gradient']:termination='zero_gradient';break
        direction=tuple(v.grad.detach().clone()/norm for v in parameters);base=tuple(v.detach().clone() for v in parameters);accepted_trial=False
        for backtrack in range(SOLVER['max_backtracks']):
            if proposals>=SOLVER['max_proposals']:termination='proposal_budget';break
            if step<SOLVER['min_step']:termination='no_feasible_step';break
            assign(parameters,base,direction,step);val,q=values(m,h,ff,c,ids,pair,arm,lam);proposals+=1
            s=quick_stats(frame,q,pure,oldpred,[22,6,28][f]);guard=s['mastered'] and s['new_errors_vs_start']==0
            accept=acceptable(info['objective'],val['objective'],step,norm,guard,SOLVER['armijo'])
            trials.write(json.dumps(dict(proposal=proposals,gradient=gradients,backtrack=backtrack,step=step,accepted=accept,classification_guard=bool(guard),
                stats=s,Armijo_bound=info['objective']-SOLVER['armijo']*step*norm,parameter_sha256=tensor_hash(m.state_dict()),**val))+'\n')
            if accept:
                accepted+=1;current={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};identity=tensor_hash(current)
                item=dict(accepted_update=accepted,full_gradient_evaluations=gradients,proposal_evaluations=proposals,step=step,stats=s,values=val,parameter_sha256=identity,seconds=time.monotonic()-start)
                history.append(item);last.append(dict(item,state=current));save(folder/'progress.json',history)
                if accepted==1 or accepted%25==0:emit(stage='guarded_update',fold=f,arm=arm,**item)
                step=min(SOLVER['max_step'],step*SOLVER['grow']);accepted_trial=True;break
            restore(parameters,base);step*=SOLVER['shrink']
        if not accepted_trial:
            restore(parameters,base)
            if termination not in ['proposal_budget','no_feasible_step']:termination='no_feasible_step'
            break
    if accepted>=SOLVER['max_updates']:termination='accepted_update_budget'
    elif gradients>=SOLVER['max_gradients'] and termination not in ['no_feasible_step','zero_gradient','proposal_budget']:termination='gradient_budget'
    log.close();trials.close();save(folder/'progress.json',history)
    q=probabilities(m,h,ff,np.arange(len(c)));s=quick_stats(frame,q,pure,oldpred,[22,6,28][f]);current={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    for k in before:
        if k.startswith(('reference_','head_')) and not torch.equal(current[k],before[k].cpu()):raise ValueError('Frozen state changed')
    for z in last:torch.save(dict(state=z['state']),folder/f"accepted{z['accepted_update']}.pt")
    torch.save(dict(state=current,fold=f,arm=arm,seal_sha256=sha(OUT/'run_seal.json')),folder/'endpoint.pt');np.save(folder/'sealed_all_prob.npy',q)
    rows=frame[['row_position','local','root','truth','canonical_key']].copy();rows['training_role']=f;rows['arm']=arm;rows['pure_TRAIN_input']=pure[frame.local].astype(bool);rows['pred_start']=oldpred;rows['pred']=q[frame.local].argmax(1)
    for y in range(3):rows[f'p{y}']=q[frame.local,y]
    rows.to_parquet(folder/'endpoint_original_rows.parquet',index=False)
    r=dict(status='fit_executed',fold=f,arm=arm,selected_by_score=False,full_gradient_evaluations=gradients,proposal_evaluations=proposals,accepted_updates=accepted,termination=termination,
        auxiliary_gradient_evaluations=gradients if arm=='B' else 0,original_class_mass=c.sum(0).tolist(),lambda_fixed=lam,endpoint_stats=s,
        stable_last_five=len(last)==5 and len({z['parameter_sha256'] for z in last})==5 and all(z['stats']['mastered'] and z['stats']['new_errors_vs_start']==0 for z in last),
        parameter_delta_L2={k:float((current[k]-before[k].cpu()).square().sum().sqrt()) for k in ['weight','r','s','bias']},seconds=time.monotonic()-start,seal_sha256=sha(OUT/'run_seal.json'))
    for name,key in [('endpoint.pt','model_sha256'),('sealed_all_prob.npy','probability_sha256'),('endpoint_original_rows.parquet','rows_sha256'),('gradients.jsonl','gradients_sha256'),('proposals.jsonl','proposals_sha256'),('progress.json','progress_sha256')]:r[key]=sha(folder/name)
    endpoint(r);save(folder/'fit.json',r);emit(stage='guarded_fit_complete',**r)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['register','fit']);a.add_argument('--fold',type=int);a.add_argument('--arm',choices=['A','B']);v=a.parse_args()
    if v.command=='register':register()
    else:fit(v.fold,v.arm)
