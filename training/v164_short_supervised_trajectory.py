"""Three bounded supervised trajectories; only actual accepted commits count."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,importlib.metadata,json,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import Counter,configure,CurrentInputBoundary,tensor_hash,restore,stats,probabilities,rows
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient,SEGMENTS
from v160_fixed_endpoint_diagnostic_v3 import add_normals
from v160_active_margin_direction_v5 import solve_direction
from v160_independent_saved_direction_certificate import norm
from v161_fixed_error_endpoint_diagnostic_v2 import endpoint_context,probe,store_scope,joint_check,PRIOR,save
from v161_fixed_pure_error_risk import error_risk
from v163_fixed_endpoint_one_sided_restoration_v2 import blocking_functions,margins
from v163_one_sided_joint_restoration import propose
from v164_trajectory_state import commit
from v164_solver_trace import observed_propose
from v163_independent_readout_override_bound_review import review as readout_review
OUT=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
PLAN=ROOT/'training/review_policy/v164_short_supervised_trajectory_contract.json'
BUDGET=ROOT/'training/review_policy/v164_short_trajectory_prospective_budget.json'
PROTOCOL='V164-three-supervised-short-fixed-error-trajectories-v1'

def require():
    seal=read(OUT/'run_seal.json');plan=read(PLAN);budget=read(BUDGET)
    assert seal['protocol']==plan['protocol']==PROTOCOL and seal['allowed_entries']==['training/v164_short_supervised_trajectory.py']
    assert sha(PLAN)==seal['plan_sha256'];check_bindings(seal['source_sha256'])
    assert sys.flags.optimize==0 and sys.version==seal['python_version'] and {n:importlib.metadata.version(n) for n in seal['package_versions']}==seal['package_versions']
    assert plan['new_caps']==budget['new_caps'] and plan['roles']==budget['roles'] and plan['new_caps']['fits']==3 and plan['new_caps']['permanent_updates']==30
    return plan

class TrainingCounter(Counter):
    def __init__(self,model,spec,path,target_counts):
        super().__init__(model,spec['head_cap'],path,spec['fixed_error_target_gradient_cap']);self.target_mass=target_counts.sum(0);self.margin_cap=spec['margin_gradient_cap'];self.margin_attempts=self.margin_completed=self.point_margin_attempts=0;self.parameter_point=None
    def begin_point(self,parameter_sha256):self.parameter_point=parameter_sha256;self.point_margin_attempts=0
    def gradient_before(self,cls,mass):
        if self.gradient_attempts>=self.gradient_cap:raise RuntimeError('Sealed complete target gradient cap exceeded')
        self.gradient_attempts+=1;self.write(event='attempt',kind='fixed_error_target_gradient',ordinal=self.gradient_attempts,class_id=cls,complete_original_class_mass=int(mass[cls]),fixed_error_original_rows=int(self.target_mass[cls]),parameter_sha256=self.parameter_point)
    def gradient_after(self,cls,mass):self.gradient_completed+=1;self.write(event='completed',kind='fixed_error_target_gradient',ordinal=self.gradient_completed,class_id=cls,parameter_sha256=self.parameter_point)
    def margin_before(self,identity):
        if self.margin_attempts>=self.margin_cap or self.point_margin_attempts>=48:raise RuntimeError('Sealed actual margin cap exceeded')
        self.margin_attempts+=1;self.point_margin_attempts+=1;self.write(event='attempt',kind='full_parameter_margin_gradient',ordinal=self.margin_attempts,input_identity=identity,parameter_sha256=self.parameter_point)
    def margin_after(self,identity):self.margin_completed+=1;self.write(event='completed',kind='full_parameter_margin_gradient',ordinal=self.margin_completed,input_identity=identity,parameter_sha256=self.parameter_point)
    def counts(self):return dict(super().counts(),margin_attempts=self.margin_attempts,margin_completed=self.margin_completed)

def segments(vector):return {name:dict(L2=norm(vector[start:end]),infinity=float(np.max(np.abs(vector[start:end]))),nonzero=int(np.count_nonzero(vector[start:end]))) for name,start,end in SEGMENTS}

def gradients_at_state(model,ctx,counter,folder,previous_q,previous_lq):
    parameter=tensor_hash(model.state_dict());counter.begin_point(parameter);observed=[];gradients=[]
    for cls in [1,2]:
        values=[]
        for repetition in range(2):
            target=folder/f'class{cls}_repeat{repetition}';target.mkdir(parents=True);rv,q,lq,g=error_risk(model,ctx,ctx['ids'],counter,cls)
            np.save(target/'complete_fixed_error_target_gradient.npy',g)
            for key,value in rv.items():np.save(target/f'{key}.npy',value)
            store_scope(target,ctx,'OOF',q,lq);save(target/'parameter_point.json',dict(parameter_sha256=parameter,gradient_segments=segments(g),original_class_mass=ctx['mass'].tolist(),fixed_target_mass=ctx['target_counts'].sum(0).tolist(),old_point_values_not_reused=True))
            values.append((rv,q,lq,g));observed.append((rv,q,lq))
        repeat=repeat_gradient(values[0][3],values[1][3]);save(folder/f'class{cls}_gradient_repeat.json',repeat);assert repeat['passed'];gradients.append(values[0][3])
    first=observed[0];reviews=[]
    for rv,q,lq in observed:
        review={key:repeat_values(rv[key],first[0][key],'risk') for key in rv};review['q']=repeat_values(q[ctx['ids']],previous_q[ctx['ids']],'probability');review['logq']=repeat_values(lq[ctx['ids']],previous_lq[ctx['ids']],'log_probability');assert all(v['passed'] for v in review.values());reviews.append(review)
    save(folder/'same_accepted_parameter_point_review.json',dict(parameter_sha256=parameter,reviews=reviews));assert tensor_hash(model.state_dict())==parameter
    return gradients,first[0]

def run(role):
    plan=require();configure();spec=plan['roles'][role];folder=OUT/f'role{role}';assert not folder.exists();folder.mkdir();ctx=endpoint_context(role)
    model=CurrentInputBoundary().cuda();model.load_state_dict(torch.load(PRIOR/f'fold{role}_B/endpoint.pt',map_location='cpu',weights_only=True)['state']);initial=tensor_hash(model.state_dict());assert initial==plan['initial_parameter_sha256'][str(role)]
    last=tuple(p.detach().clone() for p in model.parameters());counter=TrainingCounter(model,spec,folder/'calls.jsonl',ctx['target_counts']);counter.begin_point(initial)
    accepted=proposals=direction_qp=restoration_qp=restoration_solves=optimizer_iterations=0;failure=None;stop='ten_actual_accepted_updates_completed';state_stats=[];first=ROOT/spec['first_cached_candidate']
    counter.write(event='attempt',kind='supervised_classifier_fit',ordinal=1,initial_parameter_sha256=initial)
    save(folder/'started.json',dict(role=role,initial_parameter_sha256=initial,run_seal_sha256=sha(OUT/'run_seal.json'),new_fits=1,supervised_development_training_not_external_validation=True))
    def candidate_tables(path):return tuple(np.load(path/name) for name in ['OOF_q.npy','OOF_logq.npy','deployment_q.npy','deployment_logq.npy'])
    def accepted_commit(path,direction,proof):
        nonlocal accepted,last,qo,lo,qd,ld
        candidate=candidate_tables(path);target=folder/f'accepted{accepted+1}';target.mkdir();counter.write(event='attempt',kind='permanent_update',ordinal=accepted+1,previous_parameter_sha256=tensor_hash(model.state_dict()))
        receipt=commit(model,last,ctx,target,direction,proof,qo,*candidate,accepted+1)
        accepted+=1;last=tuple(p.detach().clone() for p in model.parameters());qo,lo,qd,ld=candidate
        counter.write(event='committed',kind='permanent_update',ordinal=accepted,parameter_sha256=receipt['parameter_sha256']);os_=stats(ctx,qo,'OOF');ds=stats(ctx,qd,'deployment');state_stats.append(dict(state_index=accepted,parameter_sha256=receipt['parameter_sha256'],OOF=os_,deployment=ds));save(target/'state_review.json',dict(**state_stats[-1],joint_TRAIN_retention=joint_check(ctx,qd,ld),displacement_segments=segments(direction),supervised_development_training_not_external_validation=True));ctx['baseline_stats']=os_
        prior=np.log(np.maximum(np.asarray(ctx['OOF'],np.float64).mean(1),1e-12));readout=readout_review(rows(ctx,qo,lo,'OOF'),prior,model.output_weight.detach().cpu().numpy())
        for values in readout['classes'].values():values['current_errors_not_blocked_by_fixed_readout_bound']=values['current_errors']-values['fixed_readout_cannot_override_prior'];values['pure_errors_not_blocked_by_fixed_readout_bound']=values['pure_current_errors']-values['pure_errors_fixed_readout_cannot_override']
        save(target/'fixed_readout_override_diagnostic.json',dict(parameter_sha256=receipt['parameter_sha256'],review=readout,official_calls=0,no_permanent_capacity_or_learning_sufficiency_claim=True))
    try:
        rv,qo,lo,_=error_risk(model,ctx,ctx['ids']);qd,ld=probabilities(model,ctx,'deployment',np.arange(22546),True)
        previous=first.parent.parent/'baseline';same={}
        for scope,q,lq in [('OOF',qo,lo),('deployment',qd,ld)]:
            ids=ctx['ids'] if scope=='OOF' else np.arange(22546);same[scope+'_q']=repeat_values(q[ids],np.load(previous/f'{scope}_q.npy')[ids],'probability');same[scope+'_logq']=repeat_values(lq[ids],np.load(previous/f'{scope}_logq.npy')[ids],'log_probability');store_scope(folder/'baseline',ctx,scope,q,lq)
        save(folder/'zero_step_review.json',same);assert all(v['passed'] for v in same.values());ctx['baseline_stats']=stats(ctx,qo,'OOF');assert ctx['baseline_stats']['protected_regressions']==0 and stats(ctx,qd,'deployment')['mastered'] and joint_check(ctx,qd,ld)['passed']
        original=read(first/'probe.json');assert original['accepted'];direction=np.load(first/'direction.npy');target=folder/'initial_cached_candidate_replay';target.mkdir();proposals+=1
        proof,_=probe(model,last,ctx,target,direction,1.,rv['fixed_pure_error_contribution'],original['class_slopes'])
        same={}
        for scope in ['OOF','deployment']:
            ids=ctx['ids'] if scope=='OOF' else np.arange(22546)
            for suffix,kind in [('q','probability'),('logq','log_probability')]:same[scope+'_'+suffix]=repeat_values(np.load(target/f'{scope}_{suffix}.npy')[ids],np.load(first/f'{scope}_{suffix}.npy')[ids],kind)
        save(target/'cached_candidate_replay_review.json',same);assert all(v['passed'] for v in same.values()) and proof['probe_parameter_sha256']==original['probe_parameter_sha256']
        if not proof['accepted']:stop='initial_actual_candidate_replay_not_accepted_stop'
        else:
            accepted_commit(target,direction,proof)
            while accepted<=10:
                point=folder/f'parameter_point{accepted}';point.mkdir();gs,rv=gradients_at_state(model,ctx,counter,point,qo,lo)
                if accepted==10:break
                direction_qp+=1;assert direction_qp<=9;result=solve_direction(*gs,np.empty((0,1060832)));save(point/'base_direction_QP_review.json',{k:v for k,v in result.items() if k!='direction'});save(point/'parameter_identity.json',dict(parameter_sha256=tensor_hash(model.state_dict()),normal_cache_reset=True));np.save(point/'base_direction.npy',result['direction'])
                if result['status']!='local_QP_certified_requires_actual_finite_guard':stop=result['status'];break
                target=point/'base_finite_proposal';target.mkdir();proposals+=1;proof,blockers=probe(model,last,ctx,target,result['direction'],1/16,rv['fixed_pure_error_contribution'],result['class_slopes'])
                if proof['accepted']:accepted_commit(target,result['direction']/16,proof);continue
                u=result['direction']/16;records={};normals=[];active=[];origin_logs=dict(OOF=lo,deployment=ld);current_logs={scope:np.load(target/f'{scope}_logq.npy') for scope in ['OOF','deployment']};found=False
                for index in range(6):
                    if not len(blockers):stop='rejected_without_actual_protected_blocker_stop';break
                    functions=blocking_functions(ctx,blockers);fresh=[k for k in functions if k not in records]
                    if fresh:
                        issue=add_normals(model,ctx,last,blockers,records,normals,counter,point/'normals')
                        if issue:stop=issue;break
                    for identity in functions:
                        if identity not in active:active.append(identity)
                    assert all(v['base_parameter_sha256']==counter.parameter_point for v in records.values());keys=list(records);a=np.stack([normals[keys.index(k)] for k in active]);b=margins(records,active,origin_logs);c=margins(records,active,current_logs)
                    restoration=point/f'restoration{index}';restoration.mkdir();np.save(restoration/'current_displacement.npy',u);np.save(restoration/'base_margins.npy',b);np.save(restoration/'actual_current_margins.npy',c);save(restoration/'active_normal_references.json',{k:dict(metadata=records[k],gradient=(point/'normals'/k/'repeat0_gradient.npy').relative_to(ROOT).as_posix()) for k in active})
                    def optimizer_event(event,value):
                        nonlocal restoration_qp,optimizer_iterations
                        if event=='call':restoration_qp+=1;counter.write(event='attempt',kind='restoration_QP',ordinal=restoration_qp,parameter_sha256=counter.parameter_point)
                        else:
                            iterations=int(value.nit) if value is not None else 0;optimizer_iterations+=iterations;counter.write(event='returned',kind='restoration_QP',ordinal=restoration_qp,optimizer_iterations=iterations,exception_return=value is None,parameter_sha256=counter.parameter_point)
                    restoration_solves+=1;res=observed_propose(optimizer_event,u,a,b,c,*gs)
                    for key in ['displacement','correction']:
                        if key in res:np.save(restoration/f'{key}.npy',res[key])
                    save(restoration/'original_unit_restoration_review.json',{k:v for k,v in res.items() if k not in ['displacement','correction']});save(restoration/'restoration_solver_context.json',dict(entered_SLSQP_QP='optimizer_iterations' in res,optimizer_iterations=res.get('optimizer_iterations',0),parameter_sha256=counter.parameter_point))
                    if res['status']!='one_sided_joint_restoration_requires_full_actual_finite_guard':stop=res['status'];break
                    u=res['displacement'];target=restoration/'finite_probe';target.mkdir();proposals+=1;proof,blockers=probe(model,last,ctx,target,u,1.,rv['fixed_pure_error_contribution'],[r['linear_change'] for r in res['class_reviews']])
                    if proof['accepted']:accepted_commit(target,u,proof);found=True;break
                    current_logs={scope:np.load(target/f'{scope}_logq.npy') for scope in ['OOF','deployment']}
                if not found:
                    if stop=='ten_actual_accepted_updates_completed':stop='six_restorations_at_new_parameter_point_exhausted_stop'
                    break
    except Exception as error:
        failure=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure);stop='execution_exception_preserved_last_actual_accepted_state_stop'
    finally:
        restore(model,last);final_hash=tensor_hash(model.state_dict());final_risk,fq,fl,_=error_risk(model,ctx,ctx['ids']);fd,fdl=probabilities(model,ctx,'deployment',np.arange(22546),True)
        for scope,q,lq in [('OOF',fq,fl),('deployment',fd,fdl)]:store_scope(folder/'endpoint',ctx,scope,q,lq)
        joint=joint_check(ctx,fd,fdl);assert joint['passed'];assert accepted<=10 and proposals<=64 and direction_qp<=9 and restoration_qp<=54 and restoration_solves<=54
        assert stats(ctx,fq,'OOF')['protected_regressions']==0 and stats(ctx,fd,'deployment')['mastered'];counts=counter.counts();counter.write(event='completed',kind='supervised_classifier_fit',ordinal=1,accepted_updates=accepted,endpoint_parameter_sha256=final_hash,status=stop);counter.close()
        torch.save(dict(state=model.state_dict(),accepted_updates=accepted,parameter_sha256=final_hash,supervised_training_scope=True),folder/'endpoint.pt')
        last5=state_stats[-5:];mastery=accepted>=5 and len({s['parameter_sha256'] for s in last5})==5 and all(s['OOF']['mastered'] for s in last5)
        summary=dict(status=stop,role=role,new_fits=1,permanent_updates=accepted,actual_accepted_states=state_stats,counts=counts,finite_proposals=proposals,direction_QP_solves=direction_qp,restoration_QP_solves=restoration_qp,QP_solves=direction_qp+restoration_qp,restoration_solves=restoration_solves,restoration_optimizer_iterations=optimizer_iterations,initial_parameter_sha256=initial,endpoint_parameter_sha256=final_hash,endpoint_is_last_actual_accepted_not_best_state=True,endpoint_OOF_stats=stats(ctx,fq,'OOF'),endpoint_deployment_stats=stats(ctx,fd,'deployment'),endpoint_joint_TRAIN_retention=joint,last_five_actual_distinct_states_mastered=mastery,supervised_development_training_not_external_validation=True,exception=failure,quality_acceptance=False)
        save(folder/'fit.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);del model;gc.collect();torch.cuda.empty_cache()
    require()
    if failure:raise RuntimeError('Recorded training exception, no automatic replay')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--role',type=int,choices=[0,1,2],required=True);run(ap.parse_args().role)
