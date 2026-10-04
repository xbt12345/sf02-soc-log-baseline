"""Two further same-function restorations; no gradient or learning update."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,importlib.metadata,json,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import Counter,configure,CurrentInputBoundary,tensor_hash,probabilities,restore,stats
from v159_float64_repeat_policy_v2 import repeat_values
from v161_fixed_error_endpoint_diagnostic_v2 import endpoint_context,probe,store_scope,joint_check,PRIOR,save
from v161_fixed_pure_error_risk import error_risk
from v162_fixed_endpoint_finite_restoration import blocking_functions,margins,DIAG
from v162_multi_function_finite_restoration_v2 import propose
PREVIOUS=ROOT/'artifacts/v162_fixed_endpoint_finite_restoration_20261002/role2'
SOURCE=PREVIOUS/'restoration3'
OUT=ROOT/'artifacts/v162_cached_function_restoration_tail_20261002'
PLAN=ROOT/'training/review_policy/v162_cached_function_restoration_tail_contract.json'
PROTOCOL='V162-same-function-two-restoration-tail-v1'
CAPS=dict(heads=88,features=88,fixed_error_target_gradients=0,full_original_class_gradients=0,margin_gradients=0,finite_proposals=2,restoration_solves=2,QP_solves=0,fits=0,permanent_updates=0)

def require():
    seal=read(OUT/'run_seal.json');plan=read(PLAN)
    assert seal['protocol']==plan['protocol']==PROTOCOL and seal['allowed_entries']==['training/v162_cached_function_restoration_tail.py']
    assert sha(PLAN)==seal['plan_sha256'];check_bindings(seal['source_sha256'])
    assert sys.flags.optimize==0 and sys.version==seal['python_version'] and {n:importlib.metadata.version(n) for n in seal['package_versions']}==seal['package_versions']
    assert plan['new_caps']==CAPS and plan['prior_actual_costs']['heads']==17246 and plan['future_cumulative_actual_caps']['heads']==17334
    return plan

def run():
    plan=require();configure();folder=OUT/'role2';assert not folder.exists();folder.mkdir();ctx=endpoint_context(2)
    model=CurrentInputBoundary().cuda();model.load_state_dict(torch.load(PRIOR/'fold2_B/endpoint.pt',map_location='cpu',weights_only=True)['state'])
    initial=tensor_hash(model.state_dict());assert initial==plan['endpoint_parameter_sha256'];base=tuple(p.detach().clone() for p in model.parameters());counter=Counter(model,88,folder/'calls.jsonl',0)
    u=np.load(SOURCE/'displacement.npy');a=np.load(SOURCE/'active_complete_margin_normals.npy');b=np.load(SOURCE/'base_margins.npy');records=read(SOURCE/'active_functions.json');keys=list(records)
    assert len(keys)==len(a)==1 and all(r['base_parameter_sha256']==initial for r in records.values())
    gs=[np.load(DIAG/f'role2/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]]
    previous=read(SOURCE/'finite_probe/probe.json');assert not previous['accepted'] and previous['step']==1.
    blockers=pd.read_parquet(SOURCE/'finite_probe/actual_blocking_original_rows.parquet');current_logs={scope:np.load(SOURCE/f'finite_probe/{scope}_logq.npy') for scope in ['OOF','deployment']}
    failure=None;passed=False;proposals=0;solves=0;stop='two_restoration_tail_budget_stop'
    try:
        rv,qo,lo,_=error_risk(model,ctx,ctx['ids']);qd,ld=probabilities(model,ctx,'deployment',np.arange(22546),True);same={}
        for key,value in rv.items():same[key]=repeat_values(value,np.load(PREVIOUS/f'baseline_{key}.npy'),'risk');np.save(folder/f'baseline_{key}.npy',value)
        for scope,q,lp in [('OOF',qo,lo),('deployment',qd,ld)]:
            used=ctx['ids'] if scope=='OOF' else np.arange(22546)
            same[scope+'_q']=repeat_values(q[used],np.load(PREVIOUS/f'baseline/{scope}_q.npy')[used],'probability');same[scope+'_logq']=repeat_values(lp[used],np.load(PREVIOUS/f'baseline/{scope}_logq.npy')[used],'log_probability');store_scope(folder/'baseline',ctx,scope,q,lp)
        save(folder/'zero_step_review.json',same);assert all(v['passed'] for v in same.values())
        ctx['baseline_stats']=stats(ctx,qo,'OOF');assert ctx['baseline_stats']['protected_regressions']==0 and stats(ctx,qd,'deployment')['mastered'] and joint_check(ctx,qd,ld)['passed']
        assert np.array_equal(b,margins(records,keys,dict(OOF=lo,deployment=ld)))
        save(folder/'started.json',dict(parameter_sha256=initial,run_seal_sha256=sha(OUT/'run_seal.json'),saved_prior_candidate_sha256=sha(SOURCE/'displacement.npy'),old_four_proposals_not_repeated=True,zero_new_gradient_fit_or_permanent_update=True))
        for iteration in [4,5]:
            if not len(blockers):stop='no_actual_negative_protected_blocker_stop';break
            functions=blocking_functions(ctx,blockers)
            if not set(functions)<=set(records):stop='new_actual_protection_function_outside_cached_tail_scope_stop';break
            target=folder/f'restoration{iteration}';target.mkdir();c=margins(records,keys,current_logs)
            for name,v in [('current_displacement',u),('active_complete_margin_normals',a),('base_margins',b),('actual_current_margins',c)]:np.save(target/f'{name}.npy',v)
            save(target/'active_functions.json',records);solves+=1;result=propose(u,a,b,c,*gs)
            for key in ['displacement','correction']:
                if key in result:np.save(target/f'{key}.npy',result[key])
            save(target/'original_unit_restoration_review.json',{k:v for k,v in result.items() if k not in ['displacement','correction']})
            if result['status']!='linear_restoration_candidate_requires_full_actual_finite_guard':stop=result['status'];break
            u=result['displacement'];proposal=target/'finite_probe';proposal.mkdir();proposals+=1
            report,blockers=probe(model,base,ctx,proposal,u,1.,rv['fixed_pure_error_contribution'],[v['linear_change'] for v in result['class_reviews']])
            save(proposal/'finite_restoration_context.json',dict(complete_displacement_sha256=sha(target/'displacement.npy'),base_step=1/16,Armijo_step=1.,class_slopes_are_gradient_dot_complete_displacement=True,old_QP_optimality_not_claimed=True,all_original_guards_executed=True,new_fits=0,permanent_updates=0))
            if report['accepted']:passed=True;stop='same_cached_function_restoration_tail_actual_finite_pass_not_fit';break
            current_logs={scope:np.load(proposal/f'{scope}_logq.npy') for scope in ['OOF','deployment']}
    except Exception as error:
        failure=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure);stop='execution_exception_stop'
    finally:
        restore(model,base);restored=tensor_hash(model.state_dict());assert restored==initial
        rr,rq,rl,_=error_risk(model,ctx,ctx['ids']);rd,rdl=probabilities(model,ctx,'deployment',np.arange(22546),True)
        for scope,q,lp in [('OOF',rq,rl),('deployment',rd,rdl)]:store_scope(folder/'restored',ctx,scope,q,lp)
        joint=joint_check(ctx,rd,rdl);assert joint['passed'];counts=counter.counts();counter.close()
        summary=dict(status=stop,role=2,actual_finite_restoration_pass=passed,counts=counts,finite_proposals=proposals,restoration_solves=solves,new_full_original_class_gradients=0,new_fixed_error_target_gradients=0,new_margin_gradients=0,new_fits=0,permanent_updates=0,initial_parameter_sha256=initial,restored_parameter_sha256=restored,restored_joint_TRAIN_retention=joint,exception=failure,quality_acceptance=False)
        save(folder/'diagnostic.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);del model;gc.collect();torch.cuda.empty_cache()
    require()
    if failure:raise RuntimeError('Recorded failure; no automatic replay')

if __name__=='__main__':run()
