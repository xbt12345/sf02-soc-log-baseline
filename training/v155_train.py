"""Matched protected CE vs full-gradient first-order neighborhood risk."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,time
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from v155_runtime import ROOT,OUT,OLD,PLAN,SOLVER,read,save,sha,review,adversaries,physical,seal,require,endpoint
from v155_sam_objective import full_gradient_at,fixed_adversary,objective_gradient,values_at,acceptance,flat
from v135_runtime import load_data,fit_context
from v135_model import tensor_hash
from v138_train import configure,quick_stats,emit
from v138_readout import probabilities
from v142_train import tensors
from v142_retention_check import check as retention
from v146_optimizer import assign,restore

def initialize(f):
    h,ff,m=tensors(f);m.load_state_dict(torch.load(OLD/f'fold{f}_A/endpoint.pt',map_location='cpu',weights_only=True)['state'])
    return h,ff,m

def role_reference(d,f):
    frame,c,pure,_,ids=fit_context(d,f)
    r=frame[['row_position','local','root','fold','truth','canonical_key']].copy();r['pure_TRAIN_input']=pure[frame.local].astype(bool)
    return frame,c,pure,ids,r.reset_index(drop=True)

def brief(info):return {k:v for k,v in info.items() if k!='gradient'}

def register():
    p=review();configure();assert not OUT.exists()
    # Resolve lazy functional/autograd dependencies using one counted dummy.
    dummy=torch.nn.Linear(1,1,dtype=torch.float64).cuda();x=torch.ones((1,1),device='cuda',dtype=torch.float64)
    from torch.func import functional_call
    value=functional_call(dummy,dict(dummy.named_parameters()),(x,));torch.autograd.grad(value.sum(),tuple(dummy.parameters()))
    del dummy,x,value
    import v155_evaluate
    _,d=load_data();OUT.mkdir();extras={PLAN,ROOT/'training/v155_sam_objective.py',ROOT/'training/test_v155_sam_objective.py',ROOT/'training/v155_evaluate.py'}|{ROOT/z for z in p['evidence_sha256']}
    reports=[]
    for f in p['folds']:
        frame,c,pure,ids,r=role_reference(d,f);path=OUT/f'fold{f}_TRAIN_reference.parquet';r.to_parquet(path,index=False);extras.add(path)
        h,ff,m=initialize(f);norm=float(torch.cat([v.detach().reshape(-1) for v in m.parameters()]).norm());assert sum(v.numel() for v in m.parameters())==22528
        rp=OUT/f'fold{f}_fixed_radius.json';save(rp,dict(initial_parameter_L2=norm,radius_L2=.001*norm,initial_tensor_sha256=tensor_hash(m.state_dict())));extras.add(rp)
        reports.append(dict(fold=f,original_rows=len(frame),original_class_mass=c.sum(0).tolist(),initial_parameter_L2=norm,radius_L2=.001*norm))
        del h,ff,m;gc.collect();torch.cuda.empty_cache()
    seal(__file__,extras)
    save(OUT/'registration.json',dict(status='sealed_before_classifier_preflight',negative_cases=adversaries(p),folds=reports,
        classifier_fits=0,classifier_updates=0,classifier_registration_forwards=0,classifier_registration_gradients=0,
        setup_dummy_forward_calls=1,setup_dummy_gradients=1,preflight_gradients_max=9,fit_gradients_max=1800,
        fits_max=6,proposal_evaluations_max=3600,seal_sha256=sha(OUT/'run_seal.json')))
    emit(stage='prospectively_registered',fits_max=6,setup_dummy_gradients=1,preflight_gradients_max=9)

def preflight():
    require(__file__);configure();assert not (OUT/'preflight.json').exists()
    _,d=load_data();reports=[];role_rows=[];calls=[0]
    for f in range(3):
        frame,c,pure,ids,ref=role_reference(d,f);assert ref.equals(pd.read_parquet(OUT/f'fold{f}_TRAIN_reference.parquet'))
        h,ff,m=initialize(f);before=tensor_hash(m.state_dict());hook=m.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
        q=probabilities(m,h,ff,np.arange(len(c)));old=np.load(OLD/f'fold{f}_A/sealed_all_prob.npy');assert np.array_equal(q,old)
        mass=torch.as_tensor(c,device='cuda',dtype=torch.float64);radius=read(OUT/f'fold{f}_fixed_radius.json')['radius_L2']
        g=full_gradient_at(m,h,ff,mass,ids,pure=pure);g2=full_gradient_at(m,h,ff,mass,ids,pure=pure)
        assert g['member_CE']==g2['member_CE'] and all(torch.equal(g['gradient'][k],g2['gradient'][k]) for k in g['gradient'])
        shift=fixed_adversary(g['gradient'],radius);plus=full_gradient_at(m,h,ff,mass,ids,shift,pure)
        previous=read(ROOT/f'artifacts/v154_directional_neighborhood_v2_20261001/fold{f}_A/probe.json')
        assert np.isclose(g['member_CE'],previous['original_member_CE_gradient'],rtol=0,atol=1e-15)
        assert np.isclose(plus['member_CE'],previous['plus_member_CE_gradient'],rtol=0,atol=1e-15)
        stats=quick_stats(frame,q,pure,q[frame.local].argmax(1),[22,6,28][f]);assert stats['mastered'] and stats['new_errors_vs_start']==0
        assert before==tensor_hash(m.state_dict()) and all(v.grad is None for v in m.parameters());hook.remove()
        np.save(OUT/f'fold{f}_zero_probability.npy',q)
        role_rows.append(frame[['row_position','truth']].assign(training_role=f,pred=q[frame.local].argmax(1)))
        reports.append(dict(fold=f,base=brief(g),plus=brief(plus),radius_L2=radius,repeated_gradient_exact=True,baseline_probability_exact=True,
            V154_scalar_risk_reproduced=True,model_unchanged=True,baseline_stats=stats))
        emit(stage='preflight_fold',fold=f,baseline_member_CE=g['member_CE'],shifted_member_CE=plus['member_CE'],radius=radius)
        del m,h,ff,mass,g,g2,plus;gc.collect();torch.cuda.empty_cache()
    guard=retention(pd.concat(role_rows,ignore_index=True));assert guard['passed']
    result=dict(status='zero_step_and_first_order_objective_preflight_passed',full_classifier_gradients=9,classifier_forward_chunk_calls=calls[0],
        classifier_fits=0,classifier_updates=0,joint_TRAIN_retention=guard,folds=reports,seal_sha256=sha(OUT/'run_seal.json'))
    save(OUT/'preflight.json',result)
    paths=[OUT/'preflight.json',OUT/'run_seal.json']+[OUT/f'fold{f}_zero_probability.npy' for f in range(3)]
    save(OUT/'fit_activation.json',dict(status='preflight_passed_bound_before_first_fit',source_sha256={z.relative_to(ROOT).as_posix():sha(z) for z in paths}))
    emit(stage='preflight_complete',full_gradients=9,classifier_forward_chunks=calls[0],fits=0,updates=0)

def fit(f,arm):
    p=require(__file__);configure()
    from experiment_review import check_bindings
    activation=read(OUT/'fit_activation.json');assert activation['status']=='preflight_passed_bound_before_first_fit';check_bindings(activation['source_sha256'])
    assert f in p['folds'] and arm in p['arms'];folder=OUT/f'fold{f}_{arm}'
    assert not folder.exists() and len(list(OUT.glob('fold*_*/started.json')))<6
    _,d=load_data();frame,c,pure,ids,ref=role_reference(d,f);assert ref.equals(pd.read_parquet(OUT/f'fold{f}_TRAIN_reference.parquet'))
    h,ff,m=initialize(f);parameters=tuple(m.parameters());mass=torch.as_tensor(c,device='cuda',dtype=torch.float64)
    radius=read(OUT/f'fold{f}_fixed_radius.json')['radius_L2'];before={k:v.detach().clone() for k,v in m.state_dict().items()}
    old=np.load(OUT/f'fold{f}_zero_probability.npy');oldpred=old[frame.local].argmax(1);folder.mkdir()
    save(folder/'started.json',dict(fold=f,arm=arm,radius_L2=radius,initial_tensor_sha256=tensor_hash(before),seal_sha256=sha(OUT/'run_seal.json'),held_labels_used=0))
    start=time.monotonic();attempts=gradients=proposals=accepted=0;step=SOLVER['initial_step'];history=[];last=deque(maxlen=5);calls=[0]
    hook=m.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
    log=(folder/'gradients.jsonl').open('w',encoding='utf-8',buffering=1);trials=(folder/'proposals.jsonl').open('w',encoding='utf-8',buffering=1)
    termination='outer_gradient_budget'
    while attempts<200 and accepted<200:
        if proposals>=600:termination='proposal_budget';break
        info=objective_gradient(m,h,ff,mass,ids,arm,radius,pure);attempts+=1
        for location,ginfo in [('base',info['base'])]+([('adversarial',info['proxy'])] if arm=='B' else []):
            gradients+=1;log.write(json.dumps(dict(full_gradient_evaluation=gradients,outer_gradient_attempt=attempts,location=location,
                finite=True,parameter_sha256=tensor_hash(m.state_dict()),radius_L2=radius,**brief(ginfo)))+'\n')
        norm=info['proxy']['gradient_norm']
        if norm<=SOLVER['zero_gradient']:termination='zero_gradient';break
        direction=tuple(v/norm for v in info['proxy']['gradient'].values());base=tuple(v.detach().clone() for v in parameters);accepted_trial=False
        epsilon_hash=tensor_hash(info['shift']);base_proxy=info['proxy']['member_CE']
        for backtrack in range(SOLVER['max_backtracks']):
            if proposals>=600:termination='proposal_budget';break
            if step<SOLVER['min_step']:termination='no_feasible_step';break
            assign(parameters,base,direction,step)
            plain,q=values_at(m,h,ff,mass,ids,pure)
            proxy=values_at(m,h,ff,mass,ids,pure,info['shift'],False)[0] if arm=='B' else plain
            proposals+=1;s=quick_stats(frame,q,pure,oldpred,[22,6,28][f]);guard=s['mastered'] and s['new_errors_vs_start']==0
            accepted_here=acceptance(base_proxy,proxy['member_CE'],step,norm,guard,SOLVER['armijo'])
            bound=base_proxy-SOLVER['armijo']*step*norm
            trials.write(json.dumps(dict(proposal=proposals,outer_gradient_attempt=attempts,backtrack=backtrack,step=step,
                accepted=accepted_here,classification_guard=bool(guard),stats=s,frozen_epsilon_proxy_CE=proxy['member_CE'],
                frozen_epsilon_base_proxy_CE=base_proxy,proxy_Armijo_bound=bound,epsilon_sha256=epsilon_hash,
                ordinary_member_CE=plain['member_CE'],ordinary_CE_contributions=plain,shifted_CE_contributions=proxy,
                parameter_sha256=tensor_hash(m.state_dict()),moving_epsilon_proxy_monotonic_claimed=False))+'\n')
            if accepted_here:
                accepted+=1;current={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
                item=dict(accepted_update=accepted,outer_gradient_attempts=attempts,full_gradient_evaluations=gradients,proposal_evaluations=proposals,
                    step=step,stats=s,ordinary_member_CE=plain['member_CE'],frozen_epsilon_proxy_CE=proxy['member_CE'],
                    parameter_sha256=tensor_hash(current),seconds=time.monotonic()-start)
                history.append(item);last.append(dict(item,state=current,epsilon={k:v.detach().cpu().clone() for k,v in info['shift'].items()}));save(folder/'progress.json',history)
                if accepted==1 or accepted%25==0:emit(stage='neighborhood_update',fold=f,arm=arm,**item)
                step=min(SOLVER['max_step'],step*SOLVER['grow']);accepted_trial=True;break
            restore(parameters,base);step*=SOLVER['shrink']
        if not accepted_trial:
            restore(parameters,base)
            if termination not in ['proposal_budget','no_feasible_step']:termination='no_feasible_step'
            break
    if accepted>=200:termination='accepted_update_budget'
    elif attempts>=200 and termination not in ['no_feasible_step','zero_gradient','proposal_budget']:termination='outer_gradient_budget'
    log.close();trials.close();save(folder/'progress.json',history)
    q=probabilities(m,h,ff,np.arange(len(c)));s=quick_stats(frame,q,pure,oldpred,[22,6,28][f]);hook.remove()
    current={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    for k,v in before.items():
        if k.startswith(('reference_','head_')):assert torch.equal(current[k],v.cpu())
    assert all(v.grad is None for v in m.parameters())
    for z in last:torch.save(dict(state=z['state'],epsilon=z['epsilon']),folder/f"accepted{z['accepted_update']}.pt")
    torch.save(dict(state=current,fold=f,arm=arm,seal_sha256=sha(OUT/'run_seal.json')),folder/'endpoint.pt');np.save(folder/'sealed_all_prob.npy',q)
    rows=ref.copy();rows['training_role']=f;rows['arm']=arm;rows['pred_start']=oldpred;rows['pred']=q[frame.local].argmax(1)
    for cl in range(3):rows[f'p{cl}']=q[frame.local,cl]
    rows.to_parquet(folder/'endpoint_original_rows.parquet',index=False)
    r=dict(status='fit_executed',fold=f,arm=arm,selected_by_score=False,outer_gradient_attempts=attempts,full_gradient_evaluations=gradients,
        proposal_evaluations=proposals,accepted_updates=accepted,termination=termination,radius_L2=radius,endpoint_stats=s,
        stable_last_five=len(last)==5 and len({z['parameter_sha256'] for z in last})==5 and all(z['stats']['mastered'] and z['stats']['new_errors_vs_start']==0 for z in last),
        classifier_forward_chunk_calls=calls[0],ordinary_class_mass=c.sum(0).tolist(),first_order_ignore_shift_derivative=arm=='B',
        proxy_scope='current_epsilon_frozen_for_backtracks_not_reestimated_ball_maximum',new_independent_support_created=False,
        parameter_delta_L2={k:float((current[k]-before[k].cpu()).norm()) for k in ['weight','r','s','bias']},seconds=time.monotonic()-start,
        seal_sha256=sha(OUT/'run_seal.json'))
    for name,k in [('endpoint.pt','model_sha256'),('sealed_all_prob.npy','probability_sha256'),('endpoint_original_rows.parquet','rows_sha256'),
        ('gradients.jsonl','gradients_sha256'),('proposals.jsonl','proposals_sha256'),('progress.json','progress_sha256')]:r[k]=sha(folder/name)
    endpoint(r);save(folder/'fit.json',r);emit(stage='neighborhood_fit_complete',**r)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','preflight','fit']);ap.add_argument('--fold',type=int);ap.add_argument('--arm',choices=['A','B']);z=ap.parse_args()
    if z.mode=='register':register()
    elif z.mode=='preflight':preflight()
    else:fit(z.fold,z.arm)
