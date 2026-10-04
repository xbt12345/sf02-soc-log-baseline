"""One correction at the actual V167 final tie, no fit or model promotion."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json,traceback
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import configure,CurrentInputBoundary,tensor_hash,restore,assign,stats
from v159_float64_repeat_policy_v2 import repeat_values,finite_step_review
from v161_fixed_error_endpoint_diagnostic_v2 import probe,store_scope,joint_check,save
from v161_independent_all_finite_results_review import parameter_hash
from v163_fixed_endpoint_one_sided_restoration_v2 import margins
from v165_fixed_endpoint_decision_floor_diagnostic import load_context,measure,paired_progress
from v167_trial_point_restoration_diagnostic import TrialCounter,cached_origin,compare_scope_repeat,PRIOR,CONTROL
from v167_trial_point_measurement import measure_all,blocker_identities
from v168_decision_floor_joint_restoration import decision_floors,shortfall,propose
from v168_decision_floor_execution_review import require_run_seal

OUT=ROOT/'artifacts/v168_decision_floor_diagnostic_20261002'
PLAN=ROOT/'training/review_policy/v168_decision_floor_contract.json'
OLD=ROOT/'artifacts/v167_trial_point_restoration_diagnostic_20261002'
PROTOCOL='V168-single-decision-aware-floor-restoration-at-actual-V167-tie-point-v1'

def require():return require_run_seal(OUT/'run_seal.json',__file__)

def safe_references(plan):
    audit=ROOT/'artifacts/v167_independent_actual_trial_point_review_v2_20261002';check_bindings(read(audit/'pre_review_bindings.json')['source_sha256']);assert read(audit/'review.json')['new_heads']==296
    results=[]
    for role in [0,2]:
        folder=ROOT/plan['actual_safe_role_references'][str(role)];proof=read(folder/'probe.json');control=CONTROL/f'role{role}/treatment';origin=torch.load(PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state'];u=np.load(folder/'direction.npy');assert proof['accepted'] and proof['probe_parameter_sha256']==read(control/'probe.json')['probe_parameter_sha256']==parameter_hash(origin,u,1.) and np.array_equal(u,np.load(control/'direction.npy'))
        for scope in ['OOF','deployment']:
            ids=load_context(role)['ids'] if scope=='OOF' else np.arange(22546)
            for suffix,kind in [('q','probability'),('logq','log_probability')]:assert repeat_values(np.load(folder/f'{scope}_{suffix}.npy')[ids],np.load(control/f'{scope}_{suffix}.npy')[ids],kind)['passed']
        results.append(dict(role=role,accepted_saved_candidate=True,parameter_sha256=proof['probe_parameter_sha256'],unchanged_V166_replay_not_new_gain=True,official_calls=0))
    return results

def cached_inputs(plan,state,ctx):
    old_spec=read(ROOT/'training/review_policy/v167_trial_point_restoration_contract.json')['roles'][1];records,gradients,_,_=cached_origin(old_spec,state);origin=parameter_hash(state);assert origin==plan['origin_parameter_sha256'];trial=ROOT/plan['actual_failed_trial_path'];u=np.load(trial/'direction.npy');proof=read(trial/'probe.json');assert not proof['accepted'] and parameter_hash(state,u,1.)==proof['probe_parameter_sha256']==plan['actual_failed_trial_parameter_sha256']
    blockers=pd.read_parquet(trial/'actual_blocking_original_rows.parquet');assert len(blockers)==2 and blocker_identities(ctx,blockers)<=set(records)
    goal=pd.read_parquet(ROOT/plan['additional_candidate_repair_goal']);assert len(goal)==4 and goal.truth.eq(1).all() and goal.pred.eq(1).all();baseline=pd.read_parquet(PRIOR/'role1/endpoint/OOF_original_rows.parquet').set_index('row_position');candidate=pd.read_parquet(trial/'OOF_original_rows.parquet').set_index('row_position');idx=goal.row_position.to_numpy()
    assert np.array_equal(baseline.loc[idx,['local','truth']].to_numpy(),goal[['local','truth']].to_numpy()) and np.array_equal(candidate.loc[idx,['local','truth','pred']].to_numpy(),goal[['local','truth','pred']].to_numpy()) and baseline.loc[idx].pred.ne(baseline.loc[idx].truth).all()
    assert np.array_equal(ctx['OOF_rows'].protected_correct.to_numpy(),baseline.reset_index().protected_correct.to_numpy());return records,gradients,u,proof,goal,blockers

def extra_guard(original_rows,goal,tie_rows,error_caps):
    original=original_rows.set_index('row_position',verify_integrity=True);g=original.loc[goal.row_position.to_numpy()];t=original.loc[tie_rows.row_position.to_numpy()]
    assert np.array_equal(g[['local','truth']].to_numpy(),goal[['local','truth']].to_numpy()) and np.array_equal(t[['local','truth']].to_numpy(),tie_rows[['local','truth']].to_numpy())
    counts={name:int((original.truth.eq(cls)&original.pred.ne(cls)).sum()) for name,cls in [('M',1),('S',2)]};repairs=bool(g.pred.eq(g.truth).all());ties=bool(t.pred.eq(t.truth).all());caps=all(counts[k]<=v for k,v in error_caps.items())
    return dict(passed=bool(repairs and ties and caps),all_four_observed_M_candidate_repairs_retained=repairs,all_two_actual_S_tie_rows_now_correct=ties,original_full_class_error_caps_passed=caps,actual_original_class_errors=counts,prospective_class_error_caps=error_caps,additional_goal_not_merged_into_origin_mask=True)

def execute_one(model,ctx,base,state,counter,folder,plan,records,gradients,u,trial_proof,goal,tie_rows):
    origin=tensor_hash(model.state_dict());spec=plan['role_spec'];point=PRIOR/f"role1/parameter_point{spec['parameter_point']}";failure=None;proof=None;qps=nit=proposals=0;status='decision_floor_execution_failure_stop';trial=ROOT/plan['actual_failed_trial_path']
    def optimizer_event(event,value):
        nonlocal qps,nit
        if event=='call':qps+=1;assert qps<=1;counter.write(event='attempt',kind='restoration_QP',ordinal=qps,origin_parameter_sha256=origin,linearization_parameter_sha256=counter.trial,stage=0)
        else:
            iterations=int(value.nit) if value is not None else 0;nit+=iterations;counter.write(event='returned',kind='restoration_QP',ordinal=qps,optimizer_iterations=iterations,exception_return=value is None)
    try:
        risk,q,lp,dq,dlp=measure(model,ctx)
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:store_scope(folder/'baseline',ctx,scope,values,logs)
        repeats={key:repeat_values(value,np.load(point/f'class1_repeat0/{key}.npy'),'risk') for key,value in risk.items()};repeats.update(compare_scope_repeat(folder/'baseline',PRIOR/'role1/endpoint',ctx));save(folder/'zero_step_review.json',repeats);assert all(v['passed'] for v in repeats.values());ctx['baseline_stats']=stats(ctx,q,'OOF');assert ctx['baseline_stats']['protected_regressions']==0 and stats(ctx,dq,'deployment')['mastered'] and joint_check(ctx,dq,dlp)['passed']
        origin_logs=dict(OOF=lp,deployment=dlp);trial_logs={scope:np.load(trial/f'{scope}_logq.npy') for scope in ['OOF','deployment']};trial_q={scope:np.load(trial/f'{scope}_q.npy') for scope in trial_logs}
        assign(model,base,u,1.);assert tensor_hash(model.state_dict())==trial_proof['probe_parameter_sha256']
        try:measured,normals=measure_all(model,ctx,records,counter,folder/'trial_point0/normals',origin,trial_proof['probe_parameter_sha256'],0,origin_logs,trial_logs,trial_q)
        finally:restore(model,base);assert tensor_hash(model.state_dict())==origin
        identities=list(records);assert len(normals)==25;c=margins(records,identities,trial_logs);b=margins(records,identities,origin_logs);floors=decision_floors(records,trial_logs);assert np.max(shortfall(c,floors))>0
        target=folder/'correction0';target.mkdir();save(target/'active_normal_references.json',{identity:dict(metadata=measured[identity],gradient=(folder/'trial_point0/normals'/identity/'repeat0_gradient.npy').relative_to(ROOT).as_posix()) for identity in identities})
        for name,value in [('current_displacement',u),('actual_origin_margins',b),('actual_trial_margins',c),('prospective_decision_floors',floors),('actual_trial_floor_shortfalls',shortfall(c,floors))]:np.save(target/f'{name}.npy',value)
        save(target/'parameter_point_roles.json',dict(origin_parameter_sha256=origin,class_gradient_parameter_sha256=origin,margin_Jacobian_parameter_sha256=trial_proof['probe_parameter_sha256'],complete_functions=25))
        math=propose(optimizer_event,u,np.stack(normals),b,c,*gradients,floors)
        for key in ['displacement','correction']:
            if key in math:np.save(target/f'{key}.npy',math[key])
        save(target/'original_unit_restoration_review.json',{k:v for k,v in math.items() if k not in ['displacement','correction']})
        if math['status']!='one_sided_joint_restoration_requires_full_actual_finite_guard':status='decision_floor_local_direction_unqualified_stop'
        else:
            proposal=folder/'probe0';proposal.mkdir();assert proposals==0;proposals+=1;counter.write(event='attempt',kind='finite_decision_floor_proposal',ordinal=1,origin_parameter_sha256=origin);slopes=[r['linear_change'] for r in math['class_reviews']];base_proof,blockers=probe(model,base,ctx,proposal,math['displacement'],1.,risk['fixed_pure_error_contribution'],slopes);assert tensor_hash(model.state_dict())==origin
            rows=pd.read_parquet(proposal/'OOF_original_rows.parquet');additional=extra_guard(rows,goal,tie_rows,plan['prospective_full_class_error_caps']);combined=bool(base_proof['classification_guard'] and additional['passed']);finite=finite_step_review(risk['fixed_pure_error_contribution'],np.load(proposal/'fixed_error_risk.npy'),slopes,*ctx['mass'][1:],'B',1.,combined)
            proof=dict(base_original_guard_review=base_proof,additional_prospective_guard_review=additional,classification_guard=combined,finite_error_target_review=finite,accepted=bool(finite['accepted'] and base_proof['actual_parameter_change']),probe_parameter_sha256=base_proof['probe_parameter_sha256'],base_probe_is_not_final_V168_acceptance=True,new_fits=0,permanent_updates=0)
            save(proposal/'v168_complete_probe_review.json',proof);save(proposal/'paired_vs_V164.json',paired_progress(pd.read_parquet(folder/'baseline/OOF_original_rows.parquet'),rows));save(proposal/'paired_vs_V167_final_tie.json',paired_progress(pd.read_parquet(trial/'OOF_original_rows.parquet'),rows));counter.write(event='completed',kind='finite_decision_floor_proposal',ordinal=1,accepted=proof['accepted'],probe_parameter_sha256=proof['probe_parameter_sha256']);status='decision_floor_single_actual_finite_pass_not_fit' if proof['accepted'] else 'decision_floor_single_actual_finite_failed_stop'
    except Exception as error:
        failure=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure)
    finally:
        restore(model,base);assert tensor_hash(model.state_dict())==origin;risk,q,lp,dq,dlp=measure(model,ctx)
        for scope,values,logs in [('OOF',q,lp),('deployment',dq,dlp)]:store_scope(folder/'endpoint',ctx,scope,values,logs)
        repeats=compare_scope_repeat(folder/'endpoint',PRIOR/'role1/endpoint',ctx);joint=joint_check(ctx,dq,dlp);assert joint['passed'];torch.save(dict(state=model.state_dict(),parameter_sha256=origin,new_fits=0,permanent_updates=0,restored_V164_last_actual_accepted=True),folder/'endpoint.pt');save(folder/'restoration_review.json',dict(origin_parameter_sha256=origin,restored_parameter_sha256=tensor_hash(model.state_dict()),all_parameter_tensors_exact=all(torch.equal(p,v) for p,v in zip(model.parameters(),base)),repeated_original_q_and_logq=repeats,joint_TRAIN_retention=joint,endpoint_OOF_stats=stats(ctx,q,'OOF'),endpoint_deployment_stats=stats(ctx,dq,'deployment')))
    counts=counter.counts();assert counts['gradient_attempts']==counts['gradient_completed']==0 and counts['head_attempts']<=98 and counts['margin_attempts']<=50 and qps<=1 and proposals<=1
    if failure is None:assert counts['head_attempts']==counts['head_completed']==counts['feature_attempts']==counts['feature_completed']==(2+proposals)*16+counts['margin_attempts'] and counts['margin_attempts']==counts['margin_completed']==50
    summary=dict(status=status,role=1,counts=counts,actual_QP_solves=qps,optimizer_iterations=nit,finite_proposals=proposals,final_candidate=proof,actual_finite_accepted=bool(proof and proof['accepted'] and failure is None),origin_parameter_sha256=origin,restored_parameter_sha256=tensor_hash(model.state_dict()),all_parameters_restored=True,exception=failure,new_fits=0,permanent_updates=0,quality_acceptance=False,no_training_or_promotion_permission=True);save(folder/'diagnostic.json',summary);return summary

def run():
    plan=require();configure();folder=OUT/'role1';assert not folder.exists();folder.mkdir();ctx=load_context(1);state=torch.load(PRIOR/'role1/endpoint.pt',map_location='cpu',weights_only=True)['state'];model=CurrentInputBoundary().cuda();model.load_state_dict(state);origin=tensor_hash(model.state_dict());records,gradients,u,proof,goal,tie_rows=cached_inputs(plan,state,ctx);base=tuple(p.detach().clone() for p in model.parameters());counter=TrialCounter(model,plan['role_spec'],folder/'calls.jsonl',origin);save(folder/'safe_saved_references_review.json',safe_references(plan));save(folder/'started.json',dict(role=1,origin_parameter_sha256=origin,trial_parameter_sha256=proof['probe_parameter_sha256'],run_seal_sha256=sha(OUT/'run_seal.json'),new_fits=0,permanent_updates=0))
    try:result=execute_one(model,ctx,base,state,counter,folder,plan,records,gradients,u,proof,goal,tie_rows)
    finally:restore(model,base);counter.close();del model;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(result,ensure_ascii=False),flush=True);require()
    if result['exception']:raise RuntimeError('V168 recorded exception; no automatic replay')

if __name__=='__main__':run()
