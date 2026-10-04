"""Single same-theta complete observed-blocker coverage treatment; zero fit."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha
from v159_boundary_train_v4 import Counter,configure,CurrentInputBoundary,tensor_hash,restore,stats
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient
from v161_fixed_error_endpoint_diagnostic_v2 import probe,store_scope,joint_check,save
from v163_fixed_endpoint_one_sided_restoration_v2 import margins
from v165_fixed_endpoint_decision_floor_diagnostic import load_context,measure,paired_progress,cached_treatment
from v166_observed_function_measurement import add_observed_functions
from v166_solver_trace import observed_propose
from v166_coverage_execution_review import require_run_seal

OUT=ROOT/'artifacts/v166_coverage_first_diagnostic_20261002'
PLAN=ROOT/'training/review_policy/v166_coverage_first_diagnostic_contract.json'
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
CONTROL=ROOT/'artifacts/v165_fixed_endpoint_decision_floor_diagnostic_20261002'
DRAFT=ROOT/'training/review_policy/v166_coverage_first_diagnostic_draft.json'
PROTOCOL='V166-three-fixed-V164-endpoints-complete-observed-blocker-coverage-v1'

def require():return require_run_seal(OUT/'run_seal.json',__file__)

class CoverageCounter(Counter):
    def __init__(self,model,spec,path,parameter):
        super().__init__(model,spec['head_cap'],path,0);self.margin_cap=spec['fresh_margin_gradient_cap'];self.margin_attempts=self.margin_completed=0;self.parameter_point=parameter
    def gradient_before(self,*args):raise RuntimeError('V166 permits no new target or original-class derivative')
    def margin_before(self,identity):
        if self.margin_attempts>=self.margin_cap:raise RuntimeError('Sealed new margin derivative budget exceeded')
        self.margin_attempts+=1;self.write(event='attempt',kind='full_parameter_margin_gradient',ordinal=self.margin_attempts,input_identity=identity,parameter_sha256=self.parameter_point)
    def margin_after(self,identity):self.margin_completed+=1;self.write(event='completed',kind='full_parameter_margin_gradient',ordinal=self.margin_completed,input_identity=identity,parameter_sha256=self.parameter_point)
    def counts(self):return dict(super().counts(),margin_attempts=self.margin_attempts,margin_completed=self.margin_completed)

def cached_origin(spec,initial):
    role=spec['role'];point=PRIOR/f"role{role}/parameter_point{spec['parameter_point']}";assert read(point/'parameter_identity.json')['parameter_sha256']==initial
    direction,_,_=cached_treatment(spec,initial);control=CONTROL/f'role{role}/treatment';assert np.array_equal(direction,np.load(control/'direction.npy'))
    records={};normals=[]
    for identity,ref in read(point/'restoration0/active_normal_references.json').items():
        meta=ref['metadata'];assert meta['input_identity']==identity and meta['base_parameter_sha256']==initial;p=ROOT/ref['gradient'];gradient=np.load(p);assert repeat_gradient(gradient,np.load(p.parent/'repeat1_gradient.npy'))['passed'];records[identity]=meta;normals.append(gradient)
    assert len(records)==spec['cached_functions']
    gradients=[]
    for cls in [1,2]:
        assert read(point/f'class{cls}_repeat0/parameter_point.json')['parameter_sha256']==initial
        gradient=np.load(point/f'class{cls}_repeat0/complete_fixed_error_target_gradient.npy');assert repeat_gradient(gradient,np.load(point/f'class{cls}_repeat1/complete_fixed_error_target_gradient.npy'))['passed'];gradients.append(gradient)
    return records,normals,gradients,direction

def execute_one(model,ctx,base,counter,folder,spec,records,normals,gradients,u):
    initial=tensor_hash(model.state_dict());role=spec['role'];point=PRIOR/f"role{role}/parameter_point{spec['parameter_point']}";failure=None;proof=None;qp_attempts=optimizer_iterations=restoration_attempts=proposal_attempts=0;status='execution_failure_preserves_V164_endpoint'
    def optimizer_event(event,value):
        nonlocal qp_attempts,optimizer_iterations
        if event=='call':
            qp_attempts+=1;assert qp_attempts<=1;counter.write(event='attempt',kind='restoration_QP',ordinal=qp_attempts,parameter_sha256=initial)
        else:
            count=int(value.nit) if value is not None else 0;optimizer_iterations+=count;counter.write(event='returned',kind='restoration_QP',ordinal=qp_attempts,optimizer_iterations=count,exception_return=value is None,parameter_sha256=initial)
    try:
        risk,q,lp,dq,dlp=measure(model,ctx)
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:store_scope(folder/'baseline',ctx,scope,values,logs)
        repeats={key:repeat_values(value,np.load(point/f'class1_repeat0/{key}.npy'),'risk') for key,value in risk.items()}
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:
            ids=ctx['ids'] if scope=='OOF' else np.arange(22546);previous=PRIOR/f'role{role}/endpoint'
            repeats[scope+'_q']=repeat_values(values[ids],np.load(previous/f'{scope}_q.npy')[ids],'probability');repeats[scope+'_logq']=repeat_values(logs[ids],np.load(previous/f'{scope}_logq.npy')[ids],'log_probability')
        save(folder/'zero_step_review.json',repeats);assert all(r['passed'] for r in repeats.values())
        ctx['baseline_stats']=stats(ctx,q,'OOF');assert ctx['baseline_stats']['protected_regressions']==0 and stats(ctx,dq,'deployment')['mastered'] and joint_check(ctx,dq,dlp)['passed']
        origin_logs=dict(OOF=lp,deployment=dlp);origin_q=dict(OOF=q,deployment=dq);control=CONTROL/f'role{role}/treatment';blockers=pd.read_parquet(control/'actual_blocking_original_rows.parquet')
        coverage=add_observed_functions(model,ctx,base,blockers,records,normals,counter,folder/'fresh_normals',spec['fresh_functions'],origin_logs,origin_q);assert len(records)==spec['joint_functions'];save(folder/'observed_coverage_measurement.json',coverage)
        identities=list(records);a=np.stack(normals);b=margins(records,identities,origin_logs);control_logs={scope:np.load(control/f'{scope}_logq.npy') for scope in ['OOF','deployment']};c=margins(records,identities,control_logs);target=folder/'joint_restoration';target.mkdir()
        cached_refs=read(point/'restoration0/active_normal_references.json');refs={}
        for identity in identities:
            path=ROOT/cached_refs[identity]['gradient'] if identity in cached_refs else folder/'fresh_normals'/identity/'repeat0_gradient.npy';refs[identity]=dict(metadata=records[identity],gradient=path.relative_to(ROOT).as_posix())
        save(target/'active_normal_references.json',refs);np.save(target/'current_displacement.npy',u);np.save(target/'actual_origin_margins.npy',b);np.save(target/'actual_current_margins.npy',c);restoration_attempts+=1
        result=observed_propose(optimizer_event,u,a,b,c,*gradients)
        for key in ['displacement','correction']:
            if key in result:np.save(target/f'{key}.npy',result[key])
        save(target/'original_unit_restoration_review.json',{k:v for k,v in result.items() if k not in ['displacement','correction']});save(target/'solver_context.json',dict(actual_minimize_calls=qp_attempts,optimizer_iterations=optimizer_iterations,parameter_sha256=initial))
        if result['status']!='one_sided_joint_restoration_requires_full_actual_finite_guard':status='joint_coverage_local_direction_unqualified_stop'
        else:
            proposal=folder/'treatment';proposal.mkdir();proposal_attempts+=1;counter.write(event='attempt',kind='finite_coverage_proposal',ordinal=1,origin_parameter_sha256=initial)
            proof,actual_blockers=probe(model,base,ctx,proposal,result['displacement'],1.,risk['fixed_pure_error_contribution'],[r['linear_change'] for r in result['class_reviews']]);counter.write(event='completed',kind='finite_coverage_proposal',ordinal=1,probe_parameter_sha256=proof['probe_parameter_sha256'],accepted=proof['accepted'])
            assert tensor_hash(model.state_dict())==initial;before=pd.read_parquet(folder/'baseline/OOF_original_rows.parquet');after=pd.read_parquet(proposal/'OOF_original_rows.parquet');save(folder/'paired_treatment_vs_V164_endpoint.json',paired_progress(before,after));save(folder/'matched_V165_control.json',dict(control=(control/'probe.json').relative_to(ROOT).as_posix(),same_theta=initial,control_probe=read(control/'probe.json'),treatment=proof,only_changed_factor='complete_observed_blocker_function_coverage',new_fits=0,permanent_updates=0));status='coverage_actual_finite_pass_not_fit' if proof['accepted'] else 'coverage_actual_finite_failed_preserved_endpoint'
    except Exception as error:
        failure=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure)
    finally:
        restore(model,base);assert tensor_hash(model.state_dict())==initial;risk,q,lp,dq,dlp=measure(model,ctx)
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:store_scope(folder/'endpoint',ctx,scope,values,logs)
        repeats={}
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:
            ids=ctx['ids'] if scope=='OOF' else np.arange(22546);previous=PRIOR/f'role{role}/endpoint'
            repeats[scope+'_q']=repeat_values(values[ids],np.load(previous/f'{scope}_q.npy')[ids],'probability');repeats[scope+'_logq']=repeat_values(logs[ids],np.load(previous/f'{scope}_logq.npy')[ids],'log_probability')
        joint=joint_check(ctx,dq,dlp);assert all(v['passed'] for v in repeats.values()) and joint['passed'];torch.save(dict(state=model.state_dict(),parameter_sha256=initial,new_fits=0,permanent_updates=0,restored_V164_last_actual_accepted=True),folder/'endpoint.pt')
        save(folder/'restoration_review.json',dict(initial_parameter_sha256=initial,restored_parameter_sha256=tensor_hash(model.state_dict()),all_parameter_tensors_exact=all(torch.equal(p,b) for p,b in zip(model.parameters(),base)),repeated_original_q_and_logq=repeats,joint_TRAIN_retention=joint,endpoint_OOF_stats=stats(ctx,q,'OOF'),endpoint_deployment_stats=stats(ctx,dq,'deployment')))
    counts=counter.counts();assert counts['gradient_attempts']==counts['gradient_completed']==0 and counts['head_attempts']<=spec['head_cap'] and counts['margin_attempts']<=spec['fresh_margin_gradient_cap']
    if failure is None:
        assert counts['head_attempts']==counts['head_completed']==counts['feature_attempts']==counts['feature_completed'] and counts['margin_attempts']==counts['margin_completed']==spec['fresh_margin_gradient_cap']
        assert counts['head_attempts']==(2+proposal_attempts)*(spec['OOF_chunks']+12)+counts['margin_attempts']
    summary=dict(status=status,role=role,counts=counts,finite_proposals=proposal_attempts,joint_restoration_attempts=restoration_attempts,actual_QP_solves=qp_attempts,optimizer_iterations=optimizer_iterations,candidate=proof,initial_parameter_sha256=initial,restored_parameter_sha256=tensor_hash(model.state_dict()),all_parameters_restored=True,exception=failure,new_fits=0,permanent_updates=0,quality_acceptance=False,no_training_or_promotion_permission=True);save(folder/'diagnostic.json',summary);return summary

def run(role):
    plan=require();configure();spec=plan['roles'][role];folder=OUT/f'role{role}';assert not folder.exists();folder.mkdir();ctx=load_context(role);model=CurrentInputBoundary().cuda();model.load_state_dict(torch.load(PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state']);initial=tensor_hash(model.state_dict());assert initial==spec['endpoint_parameter_sha256']
    records,normals,gradients,u=cached_origin(spec,initial);base=tuple(p.detach().clone() for p in model.parameters());counter=CoverageCounter(model,spec,folder/'calls.jsonl',initial);save(folder/'started.json',dict(role=role,initial_parameter_sha256=initial,run_seal_sha256=sha(OUT/'run_seal.json'),new_fits=0,permanent_updates=0))
    try:result=execute_one(model,ctx,base,counter,folder,spec,records,normals,gradients,u)
    finally:restore(model,base);counter.close();del model;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(result,ensure_ascii=False),flush=True);require()
    if result['exception']:raise RuntimeError('V166 recorded exception; no automatic replay')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role',type=int,choices=[0,1,2],required=True);run(parser.parse_args().role)
