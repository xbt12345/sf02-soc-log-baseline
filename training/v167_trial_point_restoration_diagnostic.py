"""Actual-guard conditioned bounded nonlinear restoration; never a fit."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,traceback
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha
from v159_boundary_train_v4 import Counter,configure,CurrentInputBoundary,tensor_hash,restore,assign,stats
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient,finite_step_review
from v161_fixed_error_endpoint_diagnostic_v2 import probe,store_scope,joint_check,save
from v161_independent_all_finite_results_review import parameter_hash
from v163_fixed_endpoint_one_sided_restoration_v2 import margins
from v165_fixed_endpoint_decision_floor_diagnostic import load_context,measure,paired_progress
from v166_solver_trace import observed_propose
from v167_trial_point_measurement import measure_all,blocker_identities
from v167_trial_point_execution_review import require_run_seal

OUT=ROOT/'artifacts/v167_trial_point_restoration_diagnostic_20261002'
PLAN=ROOT/'training/review_policy/v167_trial_point_restoration_contract.json'
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
CONTROL=ROOT/'artifacts/v166_coverage_first_diagnostic_20261002'
PROTOCOL='V167-V166-actual-guard-conditioned-trial-point-Jacobian-restoration-v1'

def require():return require_run_seal(OUT/'run_seal.json',__file__)

class TrialCounter(Counter):
    def __init__(self,model,spec,path,origin):
        super().__init__(model,spec['head_cap'],path,0);self.origin=origin;self.margin_cap=spec['margin_gradient_cap'];self.qp_cap=spec['QP_cap'];self.proposal_cap=spec['finite_proposal_cap'];self.margin_attempts=self.margin_completed=0;self.trial=None;self.stage=None
    def gradient_before(self,*args):raise RuntimeError('V167 authorizes no new fixed-target or original-class derivatives')
    def set_trial_point(self,trial,stage):
        assert trial!=self.origin and 0<=stage<2 and len(trial)==64;self.trial,self.stage=trial,stage;self.write(event='linearization_point',kind='temporary_parameter_assignment',origin_parameter_sha256=self.origin,linearization_parameter_sha256=trial,stage=stage,permanent_updates=0)
    def margin_before(self,identity):
        if self.margin_attempts>=self.margin_cap:raise RuntimeError('Sealed trial-point derivative cap exceeded')
        assert self.trial is not None;self.margin_attempts+=1;self.write(event='attempt',kind='full_parameter_margin_gradient',ordinal=self.margin_attempts,input_identity=identity,origin_parameter_sha256=self.origin,linearization_parameter_sha256=self.trial,stage=self.stage)
    def margin_after(self,identity):
        self.margin_completed+=1;self.write(event='completed',kind='full_parameter_margin_gradient',ordinal=self.margin_completed,input_identity=identity,origin_parameter_sha256=self.origin,linearization_parameter_sha256=self.trial,stage=self.stage)
    def counts(self):return dict(super().counts(),margin_attempts=self.margin_attempts,margin_completed=self.margin_completed)

def cached_origin(spec,state):
    role=spec['role'];origin=parameter_hash(state);assert origin==spec['endpoint_parameter_sha256'];point=PRIOR/f"role{role}/parameter_point{spec['parameter_point']}";assert read(point/'parameter_identity.json')['parameter_sha256']==origin;gradients=[]
    for cls in [1,2]:
        pair=[]
        for repeat in [0,1]:
            assert read(point/f'class{cls}_repeat{repeat}/parameter_point.json')['parameter_sha256']==origin;pair.append(np.load(point/f'class{cls}_repeat{repeat}/complete_fixed_error_target_gradient.npy'))
        assert repeat_gradient(*pair)['passed'];gradients.append(pair[0])
    control=CONTROL/f'role{role}/treatment';u=np.load(control/'direction.npy');proof=read(control/'probe.json');assert parameter_hash(state,u,1.)==proof['probe_parameter_sha256'];refs=read(CONTROL/f'role{role}/joint_restoration/active_normal_references.json');records={identity:ref['metadata'] for identity,ref in refs.items()};assert len(records)==spec['complete_functions']<=25 and all(m['base_parameter_sha256']==origin and m['input_identity']==identity for identity,m in records.items());return records,gradients,u,proof

def negative_margin_max(c):return max(0.,-float(np.min(c)))

def continuation_gate(ctx,records,blockers,risk_origin,proposal,previous_c,current_c):
    identities=blocker_identities(ctx,blockers);covered=bool(len(blockers)>0 and identities and identities<=set(records));proof=read(proposal/'probe.json');target=finite_step_review(risk_origin,np.load(proposal/'fixed_error_risk.npy'),proof['class_slopes'],*ctx['mass'][1:],'B',1.,True);previous,current=negative_margin_max(previous_c),negative_margin_max(current_c);decrease=previous>0 and current<.99*previous
    return dict(continue_second_correction=bool(not proof['accepted'] and covered and target['accepted'] and decrease),covered_only=covered,blocking_function_identities=sorted(identities),uncovered_function_identities=sorted(identities-set(records)),fixed_target_drop_and_Armijo_only=target,classification_acceptance_not_overridden=True,previous_max_negative_margin=previous,current_max_negative_margin=current,prospective_max_negative_margin_factor=.99,strict_residual_decrease_passed=decrease)

def compare_scope_repeat(folder,previous,ctx):
    result={}
    for scope in ['OOF','deployment']:
        ids=ctx['ids'] if scope=='OOF' else np.arange(22546)
        for suffix,kind in [('q','probability'),('logq','log_probability')]:result[f'{scope}_{suffix}']=repeat_values(np.load(folder/f'{scope}_{suffix}.npy')[ids],np.load(previous/f'{scope}_{suffix}.npy')[ids],kind)
    assert all(v['passed'] for v in result.values());return result

def execute_one(model,ctx,base,state,counter,folder,spec,records,gradients,u,control_proof):
    origin=tensor_hash(model.state_dict());role=spec['role'];point=PRIOR/f"role{role}/parameter_point{spec['parameter_point']}";control=CONTROL/f'role{role}/treatment';failure=None;proof=None;actual_qps=optimizer_iterations=restoration_attempts=proposals=0;status='execution_failure_preserves_V164_endpoint';stages=[]
    def optimizer_event(event,value):
        nonlocal actual_qps,optimizer_iterations
        if event=='call':
            actual_qps+=1;assert actual_qps<=counter.qp_cap;counter.write(event='attempt',kind='restoration_QP',ordinal=actual_qps,origin_parameter_sha256=origin,linearization_parameter_sha256=counter.trial,stage=counter.stage)
        else:
            nit=int(value.nit) if value is not None else 0;optimizer_iterations+=nit;counter.write(event='returned',kind='restoration_QP',ordinal=actual_qps,optimizer_iterations=nit,exception_return=value is None,linearization_parameter_sha256=counter.trial,stage=counter.stage)
    def actual_probe(delta,slopes):
        nonlocal proposals
        assert proposals<counter.proposal_cap;target=folder/f'probe{proposals}';target.mkdir();proposals+=1;counter.write(event='attempt',kind='finite_trial_point_proposal',ordinal=proposals,origin_parameter_sha256=origin);result,blocked=probe(model,base,ctx,target,delta,1.,risk['fixed_pure_error_contribution'],slopes);counter.write(event='completed',kind='finite_trial_point_proposal',ordinal=proposals,accepted=result['accepted'],probe_parameter_sha256=result['probe_parameter_sha256']);assert tensor_hash(model.state_dict())==origin;save(target/'paired_vs_V164.json',paired_progress(pd.read_parquet(folder/'baseline/OOF_original_rows.parquet'),pd.read_parquet(target/'OOF_original_rows.parquet')));save(target/'paired_vs_V166_control.json',paired_progress(pd.read_parquet(control/'OOF_original_rows.parquet'),pd.read_parquet(target/'OOF_original_rows.parquet')));return result,blocked,target
    try:
        risk,q,lp,dq,dlp=measure(model,ctx)
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:store_scope(folder/'baseline',ctx,scope,values,logs)
        repeats={key:repeat_values(value,np.load(point/f'class1_repeat0/{key}.npy'),'risk') for key,value in risk.items()};repeats.update(compare_scope_repeat(folder/'baseline',PRIOR/f'role{role}/endpoint',ctx));save(folder/'zero_step_review.json',repeats);assert all(v['passed'] for v in repeats.values());ctx['baseline_stats']=stats(ctx,q,'OOF');assert ctx['baseline_stats']['protected_regressions']==0 and stats(ctx,dq,'deployment')['mastered'] and joint_check(ctx,dq,dlp)['passed'];origin_logs=dict(OOF=lp,deployment=dlp)
        if control_proof['accepted']:
            assert spec['guard_condition']=='replay_V166_actual_safe_candidate_unchanged' and spec['margin_gradient_cap']==spec['QP_cap']==0;proof,blocked,target=actual_probe(u,control_proof['class_slopes']);assert proof['accepted'] and proof['probe_parameter_sha256']==control_proof['probe_parameter_sha256'];assert np.array_equal(np.load(target/'direction.npy'),np.load(control/'direction.npy'));save(folder/'safe_control_exact_replay.json',compare_scope_repeat(target,control,ctx));status='unchanged_V166_safe_candidate_actual_replay_pass_not_fit'
        else:
            assert spec['guard_condition']=='repair_V166_actual_covered_only_finite_failure';blockers=pd.read_parquet(control/'actual_blocking_original_rows.parquet');ids=blocker_identities(ctx,blockers);assert ids and ids<=set(records);trial_folder=control;current=u.copy();actual_trial_hash=control_proof['probe_parameter_sha256']
            for stage in range(2):
                temporary=folder/f'trial_point{stage}';temporary.mkdir();assert parameter_hash(state,current,1.)==actual_trial_hash;assign(model,base,current,1.);assert tensor_hash(model.state_dict())==actual_trial_hash;trial_logs={scope:np.load(trial_folder/f'{scope}_logq.npy') for scope in ['OOF','deployment']};trial_q={scope:np.load(trial_folder/f'{scope}_q.npy') for scope in trial_logs}
                try:measured,normals=measure_all(model,ctx,records,counter,temporary/'normals',origin,actual_trial_hash,stage,origin_logs,trial_logs,trial_q)
                finally:restore(model,base);assert tensor_hash(model.state_dict())==origin
                assert len(normals)==len(records)==25;identities=list(records);b=margins(records,identities,origin_logs);c=margins(records,identities,trial_logs);target=folder/f'correction{stage}';target.mkdir();save(target/'active_normal_references.json',{identity:dict(metadata=measured[identity],gradient=(temporary/'normals'/identity/'repeat0_gradient.npy').relative_to(ROOT).as_posix()) for identity in identities});np.save(target/'current_displacement.npy',current);np.save(target/'actual_origin_margins.npy',b);np.save(target/'actual_trial_margins.npy',c);restoration_attempts+=1;math=observed_propose(optimizer_event,current,np.stack(normals),b,c,*gradients)
                for key in ['displacement','correction']:
                    if key in math:np.save(target/f'{key}.npy',math[key])
                save(target/'original_unit_restoration_review.json',{k:v for k,v in math.items() if k not in ['displacement','correction']});save(target/'parameter_point_roles.json',dict(origin_parameter_sha256=origin,class_gradient_parameter_sha256=origin,margin_Jacobian_parameter_sha256=actual_trial_hash,stage=stage,complete_functions=len(records)));stages.append(dict(stage=stage,trial_parameter_sha256=actual_trial_hash,math_status=math['status']))
                if math['status']!='one_sided_joint_restoration_requires_full_actual_finite_guard':status='trial_point_local_direction_unqualified_stop';break
                proof,blockers,trial_folder=actual_probe(math['displacement'],[r['linear_change'] for r in math['class_reviews']])
                if proof['accepted']:status='trial_point_nonlinear_restoration_actual_finite_pass_not_fit';break
                current_logs={scope:np.load(trial_folder/f'{scope}_logq.npy') for scope in ['OOF','deployment']};current_c=margins(records,identities,current_logs);gate=continuation_gate(ctx,records,blockers,risk['fixed_pure_error_contribution'],trial_folder,c,current_c);save(target/'second_correction_gate.json',gate)
                if stage==1:status='bounded_second_correction_actual_failed_stop';break
                if not gate['continue_second_correction']:status='second_correction_gate_failed_stop';break
                current=math['displacement'];actual_trial_hash=proof['probe_parameter_sha256']
    except Exception as error:
        failure=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure)
    finally:
        restore(model,base);assert tensor_hash(model.state_dict())==origin;risk,q,lp,dq,dlp=measure(model,ctx)
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:store_scope(folder/'endpoint',ctx,scope,values,logs)
        repeats=compare_scope_repeat(folder/'endpoint',PRIOR/f'role{role}/endpoint',ctx);joint=joint_check(ctx,dq,dlp);assert joint['passed'];torch.save(dict(state=model.state_dict(),parameter_sha256=origin,new_fits=0,permanent_updates=0,restored_V164_last_actual_accepted=True),folder/'endpoint.pt');save(folder/'restoration_review.json',dict(origin_parameter_sha256=origin,restored_parameter_sha256=tensor_hash(model.state_dict()),all_parameter_tensors_exact=all(torch.equal(p,v) for p,v in zip(model.parameters(),base)),repeated_original_q_and_logq=repeats,joint_TRAIN_retention=joint,endpoint_OOF_stats=stats(ctx,q,'OOF'),endpoint_deployment_stats=stats(ctx,dq,'deployment')))
    counts=counter.counts();assert counts['gradient_attempts']==counts['gradient_completed']==0 and counts['head_attempts']<=spec['head_cap'] and counts['margin_attempts']<=spec['margin_gradient_cap'] and actual_qps<=spec['QP_cap'] and proposals<=spec['finite_proposal_cap']
    if failure is None:
        assert counts['head_attempts']==counts['head_completed']==counts['feature_attempts']==counts['feature_completed']==(2+proposals)*(spec['OOF_chunks']+12)+counts['margin_attempts'];assert counts['margin_attempts']==counts['margin_completed']==50*restoration_attempts;assert restoration_attempts<=2
    summary=dict(status=status,role=role,counts=counts,correction_stages=stages,finite_proposals=proposals,joint_restoration_attempts=restoration_attempts,actual_QP_solves=actual_qps,optimizer_iterations=optimizer_iterations,final_candidate=proof,actual_finite_accepted=bool(proof and proof['accepted'] and failure is None),origin_parameter_sha256=origin,restored_parameter_sha256=tensor_hash(model.state_dict()),all_parameters_restored=True,exception=failure,new_fits=0,permanent_updates=0,quality_acceptance=False,no_training_or_promotion_permission=True);save(folder/'diagnostic.json',summary);return summary

def run(role):
    plan=require();configure();spec=plan['roles'][role];folder=OUT/f'role{role}';assert not folder.exists();folder.mkdir();ctx=load_context(role);state=torch.load(PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state'];model=CurrentInputBoundary().cuda();model.load_state_dict(state);origin=tensor_hash(model.state_dict());records,gradients,u,control_proof=cached_origin(spec,state);base=tuple(p.detach().clone() for p in model.parameters());counter=TrialCounter(model,spec,folder/'calls.jsonl',origin);save(folder/'started.json',dict(role=role,origin_parameter_sha256=origin,run_seal_sha256=sha(OUT/'run_seal.json'),control_parameter_sha256=control_proof['probe_parameter_sha256'],new_fits=0,permanent_updates=0))
    try:result=execute_one(model,ctx,base,state,counter,folder,spec,records,gradients,u,control_proof)
    finally:restore(model,base);counter.close();del model;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(result,ensure_ascii=False),flush=True);require()
    if result['exception']:raise RuntimeError('V167 recorded exception; no automatic replay')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role',type=int,choices=[0,1,2],required=True);run(parser.parse_args().role)
