"""CPU assignment lifecycle and mocked derivative plumbing, no official fit."""
import json,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
import v164_short_supervised_trajectory as entry
from v164_trajectory_state import commit,repaired_protection
import v164_solver_trace as trace
from v161_saved_fixed_error_result_audit import risks
OUT=ROOT/'artifacts/v164_short_trajectory_lifecycle_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();budget=read(entry.BUDGET);first=ROOT/budget['roles'][1]['first_cached_candidate'];endpoint=entry.PRIOR/'fold1_B/endpoint.pt'
    files={Path(__file__).resolve(),ROOT/'training/v164_short_supervised_trajectory.py',ROOT/'training/v164_trajectory_state.py',ROOT/'training/v164_solver_trace.py',entry.BUDGET,endpoint,ROOT/'training/v163_one_sided_joint_restoration.py'}|{p for p in first.rglob('*') if p.is_file()}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0))
    ctx=entry.endpoint_context(1);model=entry.CurrentInputBoundary();model.load_state_dict(torch.load(endpoint,map_location='cpu',weights_only=True)['state']);initial=entry.tensor_hash(model.state_dict());origin=tuple(p.detach().clone() for p in model.parameters())
    previous=first.parent.parent/'baseline';q0=np.load(previous/'OOF_q.npy');candidate=tuple(np.load(first/name) for name in ['OOF_q.npy','OOF_logq.npy','deployment_q.npy','deployment_logq.npy']);direction=np.load(first/'direction.npy');proof=read(first/'probe.json')
    ctx['baseline_stats']=entry.stats(ctx,q0,'OOF');target=OUT/'mocked_fixture_not_official_training';target.mkdir();receipt=commit(model,origin,ctx,target,direction,proof,q0,*candidate,1)
    assert receipt['newly_repaired_original_rows']==8 and receipt['newly_repaired_mixed_original_rows']==0 and entry.tensor_hash(model.state_dict())==proof['probe_parameter_sha256']
    last=tuple(p.detach().clone() for p in model.parameters());last_hash=entry.tensor_hash(model.state_dict())
    try:repaired_protection(ctx,candidate[0],q0)
    except AssertionError:pass
    else:raise AssertionError('A newly committed repair was allowed to regress')
    # Mixed actual repair is protected even though it is outside pure targets.
    small=dict(OOF_rows=pd.DataFrame(dict(row_position=[0,1],local=[0,1],truth=[1,2],root=[7,8],pure_current_input=[False,True],protected_correct=[False,True])))
    older=np.array([[0.,.1,.9],[0.,.1,.9]]);new=np.array([[0.,.9,.1],[0.,.1,.9]]);mask,repaired=repaired_protection(small,older,new);assert mask.tolist()==[True,True] and len(repaired)==1 and not repaired.pure_current_input.iloc[0];small['OOF_rows']['protected_correct']=mask
    try:repaired_protection(small,new,older)
    except AssertionError:pass
    else:raise AssertionError('Newly repaired mixed row lost protection')
    # A rejected probe and partial commit restore the last true accepted state.
    changed=direction*.001;entry.assign=__import__('v159_boundary_train_v4',fromlist=['assign']).assign
    entry.assign(model,last,changed,1.);assert entry.tensor_hash(model.state_dict())!=last_hash;entry.restore(model,last);assert entry.tensor_hash(model.state_dict())==last_hash
    injected=OUT/'injected_commit_failure';injected.mkdir();fake=dict(proof,probe_parameter_sha256=last_hash)
    def fault(stage):raise RuntimeError('synthetic commit failure after actual full assignment')
    oldmask=ctx['OOF_rows'].protected_correct.to_numpy().copy()
    try:commit(model,last,ctx,injected,changed,fake,candidate[0],*candidate,2,fault=fault)
    except RuntimeError:pass
    else:raise AssertionError('Injected failure was hidden')
    assert entry.tensor_hash(model.state_dict())==last_hash and np.array_equal(oldmask,ctx['OOF_rows'].protected_correct) and not (injected/'commit.json').exists()
    # Real gradient measurement helper must call the backend four times and
    # bind every request to the current changed theta. All results here mocked.
    targets=pd.read_parquet(ROOT/'artifacts/v161_independent_frozen_error_cohort_review_20261002/role1/fixed_pure_error_targets.parquet');frame=pd.read_parquet(first/'OOF_original_rows.parquet');rv=risks(frame,targets);calls=[]
    counter=entry.TrainingCounter(model,budget['roles'][1],OUT/'mocked_counter.jsonl',ctx['target_counts'])
    def mocked_error_risk(active_model,active_ctx,ids,active_counter,cls):
        calls.append(dict(parameter_sha256=entry.tensor_hash(active_model.state_dict()),class_id=cls));active_counter.gradient_before(cls,active_ctx['mass']);active_counter.gradient_after(cls,active_ctx['mass']);g=np.full(1060832,float(cls)*1e-8);return rv,candidate[0],candidate[1],g
    point=OUT/'mocked_new_parameter_derivatives';point.mkdir()
    with patch.object(entry,'error_risk',mocked_error_risk):gradients,newrisk=entry.gradients_at_state(model,ctx,counter,point,candidate[0],candidate[1])
    assert len(calls)==4 and {r['parameter_sha256'] for r in calls}=={last_hash} and last_hash!=initial and counter.gradient_attempts==counter.gradient_completed==4 and all(g.shape==(1060832,) for g in gradients)
    counter.point_margin_attempts=48
    try:counter.margin_before('over_point_cap')
    except RuntimeError:pass
    else:raise AssertionError('Point margin budget bypassed')
    counter.gradient_attempts=40
    try:counter.gradient_before(1,ctx['mass'])
    except RuntimeError:pass
    else:raise AssertionError('Target gradient cap bypassed')
    counter.close()
    # Optimizer observer counts actual entry and no-entry early return.
    args=(np.array([1.,-1.,-1.]),np.array([[1.,0.,0.]]),np.array([.1]),np.array([-.1]),np.array([0.,0.,1.]),np.array([0.,1.,1.]));events=[]
    result=trace.observed_propose(lambda e,v:events.append(e),*args);assert events==['call','return'] and result['optimizer_iterations']>0
    zero=list(args);zero[1]=np.zeros((1,3));events=[];trace.observed_propose(lambda e,v:events.append(e),*zero);assert not events
    original=trace.solver.minimize
    def failed_optimizer(*a,**k):raise RuntimeError('synthetic optimizer exception')
    events=[]
    try:
        trace.solver.minimize=failed_optimizer
        try:trace.observed_propose(lambda e,v:events.append((e,v is None)),*args)
        except RuntimeError:pass
        else:raise AssertionError('Optimizer exception swallowed')
    finally:trace.solver.minimize=original
    assert events==[('call',True),('return',True)] and __import__('sys').getprofile() is None
    entry.restore(model,origin);assert entry.tensor_hash(model.state_dict())==initial;check_bindings(bindings)
    entry.save(OUT/'qualification.json',dict(status='new_short_trajectory_CPU_assignment_repair_and_mixed_repair_protection_point_derivative_plumbing_budget_and_exception_lifecycle_passed',real_full_parameter_assignment_and_hash=True,first_actual_candidate_CPU_replay_committed_only_inside_fixture=True,new_pure_and_mixed_repairs_protected=True,injected_commit_exception_restores_last_accepted=True,real_measurement_function_calls_backend_four_times_at_new_theta=True,all_derivative_and_forward_results_mocked_or_cached_not_official=True,point_and_total_gradient_caps_reject=True,optimizer_actual_entry_early_return_and_exception_counted=True,initial_CPU_parameters_restored=True,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,source_sha256=bindings));print(json.dumps(dict(status='V164_short_trajectory_lifecycle_qualification_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
