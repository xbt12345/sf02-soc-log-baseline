"""One real decision-floor treatment per last V164 endpoint; zero fit/update."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha
from v159_boundary_train_v4 import Counter,configure,CurrentInputBoundary,tensor_hash,restore,stats,probabilities
from v159_float64_repeat_policy_v2 import repeat_values,EPS,STEP_EPS
from v160_independent_saved_direction_certificate import dot,norm
from v161_fixed_error_endpoint_diagnostic_v2 import endpoint_context,probe,store_scope,joint_check,save
from v161_fixed_pure_error_risk import error_risk
from v165_execution_review import require_run_seal

OUT=ROOT/'artifacts/v165_fixed_endpoint_decision_floor_diagnostic_20261002'
PLAN=ROOT/'training/review_policy/v165_fixed_endpoint_decision_floor_diagnostic_contract.json'
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
COUNTERFACTUAL=ROOT/'artifacts/v164_independent_decision_floor_counterfactual_20261002'
PROTOCOL='V165-three-V164-endpoints-single-decision-floor-finite-diagnostic-v1'

def require():return require_run_seal(OUT/'run_seal.json',__file__)

class DiagnosticCounter(Counter):
    def __init__(self,model,cap,path):super().__init__(model,cap,path,0)
    def gradient_before(self,*args):raise RuntimeError('V165 authorizes no new complete derivative')
    def margin_before(self,*args):raise RuntimeError('V165 authorizes no new margin derivative')

def load_context(role):
    ctx=endpoint_context(role);frozen=pd.read_parquet(PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet')
    keys=['row_position','local','truth','root','pure_current_input']
    assert np.array_equal(ctx['OOF_rows'][keys].to_numpy(),frozen[keys].to_numpy())
    old=ctx['OOF_rows'].protected_correct.to_numpy(bool);mask=frozen.protected_correct.to_numpy(bool)
    assert np.all(~old|mask) and not np.any(mask&frozen.pred.ne(frozen.truth).to_numpy())
    ctx['OOF_rows']=ctx['OOF_rows'].copy();ctx['OOF_rows']['protected_correct']=mask
    return ctx

def cached_treatment(spec,initial):
    role=spec['role'];point=PRIOR/f"role{role}/parameter_point{spec['parameter_point']}";original=point/'restoration0'
    assert read(point/'parameter_identity.json')['parameter_sha256']==initial
    refs=read(original/'active_normal_references.json');assert 0<len(refs)<=24
    assert all(r['metadata']['base_parameter_sha256']==initial for r in refs.values())
    for cls in [1,2]:assert read(point/f'class{cls}_repeat0/parameter_point.json')['parameter_sha256']==initial
    cached=ROOT/spec['cached_treatment'];direction=np.load(cached/'displacement.npy');review=read(cached/'counterfactual_original_unit_review.json')
    assert review['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard' and direction.shape==(1060832,) and np.isfinite(direction).all()
    slopes=[];certificate=[]
    for cls,name in [(1,'M'),(2,'S')]:
        gradient=np.load(point/f'class{cls}_repeat0/complete_fixed_error_target_gradient.npy');value,error=dot(gradient,direction);resolution=STEP_EPS*EPS*norm(gradient)*norm(direction)
        assert value<-max(error,resolution)
        observed=dict(class_name=name,linear_change=value,arithmetic_error=error,resolution=resolution,resolved_negative=True)
        assert observed==review['class_reviews'][cls-1];slopes.append(value);certificate.append(observed)
    return direction,slopes,certificate

def measure(model,ctx):
    risk,q,lp,_=error_risk(model,ctx,ctx['ids']);dq,dlp=probabilities(model,ctx,'deployment',np.arange(22546),True)
    return risk,q,lp,dq,dlp

def paired_progress(before,after):
    assert np.array_equal(before[['row_position','local','truth','pure_current_input']].to_numpy(),after[['row_position','local','truth','pure_current_input']].to_numpy())
    oldwrong=before.pred.ne(before.truth);wrong=after.pred.ne(after.truth);pure=before.pure_current_input
    report={}
    for cls,name in [(1,'M'),(2,'S')]:
        rows=before.truth.eq(cls);repair=rows&oldwrong&~wrong;regress=rows&~oldwrong&wrong
        report[name]=dict(original_rows=int(rows.sum()),errors_before=int((rows&oldwrong).sum()),errors_after=int((rows&wrong).sum()),pure_errors_before=int((rows&oldwrong&pure).sum()),pure_errors_after=int((rows&wrong&pure).sum()),repairs_vs_V164_endpoint=int(repair.sum()),new_errors_vs_V164_endpoint_all_original_rows=int(regress.sum()),pure_repairs_vs_V164_endpoint=int((repair&pure).sum()),new_pure_errors_vs_V164_endpoint=int((regress&pure).sum()),new_mixed_errors_vs_V164_endpoint=int((regress&~pure).sum()),registered_cumulative_protection_regressions=int((rows&wrong&before.protected_correct).sum()))
    return report

def execute_one(model,ctx,base,counter,folder,spec,direction,slopes,fault=None):
    initial=tensor_hash(model.state_dict());role=spec['role'];point=PRIOR/f"role{role}/parameter_point{spec['parameter_point']}";failure=None;proof=None;restoration=None
    try:
        risk,q,lp,dq,dlp=measure(model,ctx)
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:store_scope(folder/'baseline',ctx,scope,values,logs)
        repeats={key:repeat_values(value,np.load(point/f'class1_repeat0/{key}.npy'),'risk') for key,value in risk.items()}
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:
            ids=ctx['ids'] if scope=='OOF' else np.arange(22546);previous=PRIOR/f'role{role}/endpoint'
            repeats[scope+'_q']=repeat_values(values[ids],np.load(previous/f'{scope}_q.npy')[ids],'probability');repeats[scope+'_logq']=repeat_values(logs[ids],np.load(previous/f'{scope}_logq.npy')[ids],'log_probability')
        save(folder/'zero_step_review.json',repeats);assert all(r['passed'] for r in repeats.values())
        ctx['baseline_stats']=stats(ctx,q,'OOF');assert ctx['baseline_stats']['protected_regressions']==0 and stats(ctx,dq,'deployment')['mastered'] and joint_check(ctx,dq,dlp)['passed']
        target=folder/'treatment';target.mkdir();counter.write(event='attempt',kind='finite_decision_floor_proposal',ordinal=1,origin_parameter_sha256=initial,permanent_updates=0)
        proof,blockers=probe(model,base,ctx,target,direction,1.,risk['fixed_pure_error_contribution'],slopes)
        counter.write(event='completed',kind='finite_decision_floor_proposal',ordinal=1,probe_parameter_sha256=proof['probe_parameter_sha256'],accepted=proof['accepted'],permanent_updates=0)
        assert tensor_hash(model.state_dict())==initial
        before=pd.read_parquet(folder/'baseline/OOF_original_rows.parquet');after=pd.read_parquet(target/'OOF_original_rows.parquet')
        save(folder/'paired_treatment_vs_V164_endpoint.json',paired_progress(before,after))
        save(folder/'matched_control.json',dict(control_original_restoration=(point/'restoration0/finite_probe').relative_to(ROOT).as_posix(),control=read(point/'restoration0/finite_probe/probe.json'),treatment=proof,only_changed_factor='local restoration goal b=0',same_theta_and_fixed_target_and_actual_guard=True,no_control_model_calls_repeated=True))
        if fault is not None:fault('after_treatment')
    except Exception as error:
        failure=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure)
    finally:
        restore(model,base);assert tensor_hash(model.state_dict())==initial
        risk,q,lp,dq,dlp=measure(model,ctx)
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:store_scope(folder/'endpoint',ctx,scope,values,logs)
        repeats={}
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:
            ids=ctx['ids'] if scope=='OOF' else np.arange(22546);previous=PRIOR/f'role{role}/endpoint'
            repeats[scope+'_q']=repeat_values(values[ids],np.load(previous/f'{scope}_q.npy')[ids],'probability');repeats[scope+'_logq']=repeat_values(logs[ids],np.load(previous/f'{scope}_logq.npy')[ids],'log_probability')
        joint=joint_check(ctx,dq,dlp);assert all(r['passed'] for r in repeats.values()) and joint['passed']
        restoration=dict(initial_parameter_sha256=initial,restored_parameter_sha256=tensor_hash(model.state_dict()),all_parameter_tensors_exact=all(torch.equal(p,b) for p,b in zip(model.parameters(),base)),repeated_original_q_and_logq=repeats,joint_TRAIN_retention=joint,endpoint_OOF_stats=stats(ctx,q,'OOF'),endpoint_deployment_stats=stats(ctx,dq,'deployment'))
        torch.save(dict(state=model.state_dict(),parameter_sha256=tensor_hash(model.state_dict()),initial_parameter_sha256=initial,new_fits=0,permanent_updates=0,restored_V164_last_actual_accepted=True),folder/'endpoint.pt')
        save(folder/'restoration_review.json',restoration)
    counts=counter.counts();assert counts['gradient_attempts']==counts['gradient_completed']==0 and counts['head_attempts']==counts['head_completed']==counts['feature_attempts']==counts['feature_completed']<=spec['head_cap']
    if failure is None:assert counts['head_attempts']==spec['head_cap']
    passed=bool(proof and proof['accepted'] and failure is None);summary=dict(status='decision_floor_actual_finite_pass_not_fit' if passed else 'decision_floor_actual_finite_failed_preserved_endpoint',role=role,counts=counts,finite_proposals=int((folder/'treatment').exists()),candidate=proof,restoration=restoration,exception=failure,new_fits=0,permanent_updates=0,new_complete_derivatives=0,new_CPU_QP_solves=0,all_parameters_restored=True,quality_acceptance=False,no_training_or_model_promotion_authority=True,supervised_development_not_external_validation=True)
    save(folder/'diagnostic.json',summary);return summary

def run(role):
    plan=require();configure();spec=plan['roles'][role];folder=OUT/f'role{role}';assert not folder.exists();folder.mkdir();ctx=load_context(role);model=CurrentInputBoundary().cuda()
    model.load_state_dict(torch.load(PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state']);initial=tensor_hash(model.state_dict());assert initial==spec['endpoint_parameter_sha256']
    direction,slopes,certificate=cached_treatment(spec,initial);save(folder/'cached_same_theta_class_direction_review.json',dict(origin_parameter_sha256=initial,class_reviews=certificate,cached_treatment=spec['cached_treatment'],new_gradient_calls=0,new_CPU_QP_solves=0))
    base=tuple(p.detach().clone() for p in model.parameters());counter=DiagnosticCounter(model,spec['head_cap'],folder/'calls.jsonl')
    save(folder/'started.json',dict(role=role,initial_parameter_sha256=initial,run_seal_sha256=sha(OUT/'run_seal.json'),new_fits=0,permanent_updates=0))
    try:result=execute_one(model,ctx,base,counter,folder,spec,direction,slopes)
    finally:restore(model,base);counter.close();del model;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(result,ensure_ascii=False),flush=True);require()
    if result['exception']:raise RuntimeError('Recorded V165 execution failure; no automatic replay')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role',type=int,choices=[0,1,2],required=True);run(parser.parse_args().role)
