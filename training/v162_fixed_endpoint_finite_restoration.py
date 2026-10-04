"""Generic four-restoration finite diagnostic, all original guards, zero fit."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,importlib.metadata,json,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import Counter,configure,CurrentInputBoundary,tensor_hash,probabilities,restore,stats
from v159_float64_repeat_policy_v2 import repeat_values
from v160_fixed_endpoint_diagnostic_v3 import add_normals
from v160_margin_normal import input_identity
from v161_fixed_error_endpoint_diagnostic_v2 import endpoint_context,probe,store_scope,joint_check,PRIOR,save
from v161_fixed_pure_error_risk import error_risk
from v162_multi_function_finite_restoration_v2 import propose
DIAG=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002'
OLD=ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002'
OUT=ROOT/'artifacts/v162_fixed_endpoint_finite_restoration_20261002'
PLAN=ROOT/'training/review_policy/v162_fixed_endpoint_finite_restoration_contract.json'
PROTOCOL='V162-three-fixed-endpoints-original-unit-nonlinear-restoration-v1'

def require():
    seal=read(OUT/'run_seal.json');plan=read(PLAN)
    assert seal['protocol']==plan['protocol']==PROTOCOL and seal['allowed_entries']==['training/v162_fixed_endpoint_finite_restoration.py']
    assert sha(PLAN)==seal['plan_sha256'];check_bindings(seal['source_sha256'])
    assert sys.flags.optimize==0 and sys.version==seal['python_version'] and {n:importlib.metadata.version(n) for n in seal['package_versions']}==seal['package_versions']
    assert plan['new_caps']==dict(heads=494,features=494,fixed_error_target_gradients=0,full_original_class_gradients=0,margin_gradients=134,finite_proposals=12,restoration_solves=12,QP_solves=0,fits=0,permanent_updates=0)
    return plan

class RestorationCounter(Counter):
    def __init__(self,model,spec,path):
        super().__init__(model,spec['head_cap'],path,0);self.margin_cap=spec['fresh_margin_gradient_cap'];self.margin_attempts=self.margin_completed=0
    def margin_before(self,identity):
        if self.margin_attempts>=self.margin_cap:raise RuntimeError('Sealed paired actual margin-gradient cap exceeded')
        self.margin_attempts+=1;self.write(event='attempt',kind='full_parameter_margin_gradient',ordinal=self.margin_attempts,input_identity=identity)
    def margin_after(self,identity):
        self.margin_completed+=1;self.write(event='completed',kind='full_parameter_margin_gradient',ordinal=self.margin_completed,input_identity=identity)
    def counts(self):return dict(super().counts(),margin_attempts=self.margin_attempts,margin_completed=self.margin_completed)

def blocking_functions(ctx,blockers):
    result={}
    for (scope,local,truth,rival),group in blockers.groupby(['scope','local','truth','rival'],sort=True):
        ids=ctx['ids'] if scope=='OOF' else np.arange(22546);position=int(np.searchsorted(ids,local));assert ids[position]==local
        start=(position//2048)*2048;chunk=ids[start:start+2048];query=position-start
        identity=input_identity(ctx['fold'],scope,chunk,ctx['x'][chunk],np.asarray(ctx[scope][chunk],np.float64),query,int(truth),int(rival))
        result[identity]=dict(scope=scope,local=int(local),truth=int(truth),rival=int(rival),original_rows=len(group))
    return result

def margins(records,identities,probability_logs):
    return np.array([probability_logs[records[k]['scope']][records[k]['local'],records[k]['truth']]-probability_logs[records[k]['scope']][records[k]['local'],records[k]['rival']] for k in identities],np.float64)

def run(role):
    plan=require();configure();spec=plan['roles'][role];folder=OUT/f'role{role}';assert not folder.exists();folder.mkdir()
    ctx=endpoint_context(role);model=CurrentInputBoundary().cuda();model.load_state_dict(torch.load(PRIOR/f'fold{role}_B/endpoint.pt',map_location='cpu',weights_only=True)['state'])
    initial=tensor_hash(model.state_dict());assert initial==spec['endpoint_parameter_sha256'];base=tuple(p.detach().clone() for p in model.parameters());counter=RestorationCounter(model,spec,folder/'calls.jsonl')
    failure=None;passed=False;proposals=solves=0;stop='four_restoration_budget_stop';records={};normals=[];active=[]
    try:
        rv,qo,lo,_=error_risk(model,ctx,ctx['ids']);qd,ld=probabilities(model,ctx,'deployment',np.arange(22546),True)
        same={key:repeat_values(value,np.load(DIAG/f'role{role}/baseline_error_class1_repeat0/{key}.npy'),'risk') for key,value in rv.items()}
        for key,value in rv.items():np.save(folder/f'baseline_{key}.npy',value)
        for scope,q,lp,priorfolder in [('OOF',qo,lo,DIAG/f'role{role}/baseline_error_class1_repeat0'),('deployment',qd,ld,DIAG/f'role{role}/baseline_deployment')]:
            used=ctx['ids'] if scope=='OOF' else np.arange(22546)
            same[scope+'_q']=repeat_values(q[used],np.load(priorfolder/f'{scope}_q.npy')[used],'probability');same[scope+'_logq']=repeat_values(lp[used],np.load(priorfolder/f'{scope}_logq.npy')[used],'log_probability');store_scope(folder/'baseline',ctx,scope,q,lp)
        save(folder/'zero_step_review.json',same);assert all(v['passed'] for v in same.values())
        ctx['baseline_stats']=stats(ctx,qo,'OOF');assert ctx['baseline_stats']['protected_regressions']==0 and stats(ctx,qd,'deployment')['mastered'] and joint_check(ctx,qd,ld)['passed']
        baseline_logs=dict(OOF=lo,deployment=ld)
        gs=[np.load(DIAG/f'role{role}/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]]
        for norm in spec['cached_normals']:
            metadata=read(ROOT/norm['metadata']);assert metadata['base_parameter_sha256']==initial
            records[metadata['input_identity']]=metadata;normals.append(np.load(ROOT/norm['gradient']))
        # Same rule and registered step for every role. Reuse the actual saved
        # failed candidate; zero-step checks establish the unchanged origin.
        old=DIAG/f'role{role}/round0/probe4';previous=read(old/'probe.json');assert previous['step']==1/16 and not previous['accepted']
        u=np.load(DIAG/f'role{role}/round0/polished_direction.npy')/16
        np.save(folder/'base_certified_direction_displacement.npy',u)
        blockers=pd.read_parquet(old/'actual_blocking_original_rows.parquet');blockers.to_parquet(folder/'initial_actual_blocking_original_rows.parquet',index=False)
        current_logs={scope:np.load(old/f'{scope}_logq.npy') for scope in ['OOF','deployment']}
        save(folder/'started.json',dict(role=role,parameter_sha256=initial,run_seal_sha256=sha(OUT/'run_seal.json'),registered_base_step=1/16,new_fits=0,permanent_updates=0))
        for iteration in range(4):
            if not len(blockers):stop='no_actual_protected_blocker_restoration_scope_stop';break
            functions=blocking_functions(ctx,blockers);fresh=[k for k in functions if k not in records]
            if fresh:
                issue=add_normals(model,ctx,base,blockers,records,normals,counter,folder/'fresh_normals')
                if issue:stop=issue;break
            for identity in functions:
                if identity not in active:active.append(identity)
            assert len(records)==len(normals) and len(records)<=24 and all(k in records for k in active)
            keys=list(records);a=np.stack([normals[keys.index(k)] for k in active]);b=margins(records,active,baseline_logs);c=margins(records,active,current_logs)
            target=folder/f'restoration{iteration}';target.mkdir();np.save(target/'current_displacement.npy',u);np.save(target/'active_complete_margin_normals.npy',a);np.save(target/'base_margins.npy',b);np.save(target/'actual_current_margins.npy',c)
            save(target/'active_functions.json',{k:records[k] for k in active});save(target/'all_measured_functions.json',records)
            solves+=1;result=propose(u,a,b,c,*gs)
            for key in ['displacement','correction']:
                if key in result:np.save(target/f'{key}.npy',result[key])
            save(target/'original_unit_restoration_review.json',{k:v for k,v in result.items() if k not in ['displacement','correction']})
            if result['status']!='linear_restoration_candidate_requires_full_actual_finite_guard':stop=result['status'];break
            proposal=target/'finite_probe';proposal.mkdir();proposals+=1
            u=result['displacement'];slopes=[v['linear_change'] for v in result['class_reviews']]
            # Step=1 because u includes both the base step and full correction.
            report,blockers=probe(model,base,ctx,proposal,u,1.,rv['fixed_pure_error_contribution'],slopes)
            save(proposal/'finite_restoration_context.json',dict(complete_displacement_sha256=sha(target/'displacement.npy'),base_step=1/16,Armijo_step=1.,class_slopes_are_gradient_dot_complete_displacement=True,old_QP_optimality_not_claimed=True,all_original_guards_executed=True,new_fits=0,permanent_updates=0))
            if report['accepted']:passed=True;stop='actual_full_original_finite_restoration_pass_not_fit';break
            current_logs={scope:np.load(proposal/f'{scope}_logq.npy') for scope in ['OOF','deployment']}
    except Exception as error:
        failure=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure);stop='execution_exception_stop'
    finally:
        restore(model,base);restored=tensor_hash(model.state_dict());assert restored==initial
        rr,rq,rl,_=error_risk(model,ctx,ctx['ids']);rd,rdl=probabilities(model,ctx,'deployment',np.arange(22546),True)
        for scope,q,lp in [('OOF',rq,rl),('deployment',rd,rdl)]:store_scope(folder/'restored',ctx,scope,q,lp)
        joint=joint_check(ctx,rd,rdl);assert joint['passed'];counts=counter.counts();counter.close()
        summary=dict(status=stop,role=role,actual_finite_restoration_pass=passed,counts=counts,finite_proposals=proposals,restoration_solves=solves,margin_normals=len(normals),active_function_count=len(active),new_full_original_class_gradients=0,new_fixed_error_target_gradients=0,new_fits=0,permanent_updates=0,initial_parameter_sha256=initial,restored_parameter_sha256=restored,restored_joint_TRAIN_retention=joint,exception=failure,quality_acceptance=False)
        save(folder/'diagnostic.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);del model;gc.collect();torch.cuda.empty_cache()
    require()
    if failure:raise RuntimeError('Recorded failure, no automatic replay')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--role',type=int,choices=[0,1,2],required=True);run(ap.parse_args().role)
