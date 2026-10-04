"""Real full CPU state lifecycle; cached outputs and synthetic gradients only."""
import copy,json,os,sys,traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import assign,restore,stats,tensor_hash
from v159_float64_repeat_policy_v2 import finite_step_review
from v166_coverage_execution_review import prospective_plan,review_plan
import v166_coverage_first_diagnostic as entry
import v166_observed_function_measurement as measurement
import v166_solver_trace as trace

OUT=ROOT/'artifacts/v166_entry_lifecycle_qualification_20261002'

def save(path,value):entry.save(path,value)

def main():
    assert not OUT.exists();OUT.mkdir();plan=prospective_plan();review_plan(plan)
    previous=read(ROOT/'artifacts/v166_entry_identity_qualification_20261002/qualification.json');check_bindings(previous['source_sha256'])
    paths={ROOT/p for p in previous['source_sha256']}|{Path(__file__).resolve(),ROOT/'artifacts/v166_entry_identity_qualification_20261002/qualification.json',ROOT/'training/v166_independent_actual_coverage_review.py',ROOT/'docs/V166_NEXT_TRAINING_DECISION_AND_PROGRESSIVE_REVIEW_20261002.md'}
    folder=ROOT/'artifacts/v166_next_round_decision_evidence_review_20261002';paths.update(p for p in folder.rglob('*') if p.is_file())
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0))
    measurement_cases=[];fixtures={}
    for spec in plan['roles']:
        role=spec['role'];ctx=entry.load_context(role);model=entry.CurrentInputBoundary().cpu();model.load_state_dict(torch.load(entry.PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state']);initial=tensor_hash(model.state_dict());base=tuple(p.detach().clone() for p in model.parameters());records,normals,gradients,u=entry.cached_origin(spec,initial)
        reference=entry.PRIOR/f'role{role}/endpoint';values={scope:np.load(reference/f'{scope}_q.npy') for scope in ['OOF','deployment']};logs={scope:np.load(reference/f'{scope}_logq.npy') for scope in values};target=OUT/f'role{role}_synthetic_measurement';target.mkdir();counter=entry.CoverageCounter(model,spec,target/'mock_calls.jsonl',initial)
        blockers=pd.read_parquet(entry.CONTROL/f'role{role}/treatment/actual_blocking_original_rows.parquet');synthetic=np.zeros(1060832);synthetic[-48:]=np.arange(1,49,dtype=np.float64)
        def mock_inputs(observed_ctx,scope,chunk):return scope,chunk
        def mock_margin(observed_model,scope,chunk,query,truth,rival):
            for kind in ['head','feature']:counter.before(kind);counter.after(kind)
            return dict(gradient=synthetic.copy(),q=values[scope][chunk].copy(),logq=logs[scope][chunk].copy(),margin=float(logs[scope][chunk[query],truth]-logs[scope][chunk[query],rival]))
        try:
            with patch.object(measurement,'inputs',mock_inputs),patch.object(measurement,'measure',mock_margin):
                report=measurement.add_observed_functions(model,ctx,base,blockers,records,normals,counter,target/'fresh_normals',spec['fresh_functions'],logs,values)
            assert report['fresh_functions']==spec['fresh_functions'] and len(normals)==spec['joint_functions'] and counter.margin_completed==spec['fresh_margin_gradient_cap'];assert tensor_hash(model.state_dict())==initial
            refused=False
            try:counter.margin_before('extra_unregistered_function')
            except RuntimeError:refused=True
            assert refused
            gate_calls=counter.margin_attempts
            with patch.object(measurement,'measure',side_effect=RuntimeError('must not measure duplicate')):
                duplicate=False
                try:measurement.add_observed_functions(model,ctx,base,blockers,records,normals,counter,target/'duplicate',spec['fresh_functions'],logs,values)
                except AssertionError:duplicate=True
            assert duplicate and counter.margin_attempts==gate_calls
            fixtures[role]=(records,normals,target/'fresh_normals',report)
            measurement_cases.append(dict(role=role,fresh_functions=report['fresh_functions'],joint_functions=len(records),mocked_margin_events=counter.margin_completed,duplicate_refused_before_measurement=True,margin_cap_refused=True,complete_origin_state_exact=True,gradients_are_full_width_synthetic_not_official=True))
        finally:counter.close()
    # The observer sees the unchanged SciPy function entry/return, including
    # an exception raised inside SciPy. No official data forward is involved.
    width=1060832;u=np.zeros(width);u[:25]=-.125;u[25]=-1.;a=np.zeros((25,width));a[np.arange(25),np.arange(25)]=1.;gm=np.zeros(width);gm[25]=1.;gs=2*gm;b=np.full(25,.75);c=np.full(25,-.125);events=[];profile=sys.getprofile()
    result=trace.observed_propose(lambda e,v:events.append((e,None if v is None else int(v.nit))),u,a,b,c,gm,gs)
    assert result['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard' and [e[0] for e in events]==['call','return'] and events[-1][1]==result['optimizer_iterations'] and sys.getprofile() is profile
    optimizer_cases=[dict(case='actual_25_function_SciPy_call',actual_calls=1,nit=events[-1][1])];events=[]
    result=trace.observed_propose(lambda e,v:events.append((e,v)),u,a,b,np.ones(25),gm,gs);assert not events and result['status']=='no_actual_negative_protected_margin_to_restore_stop'
    optimizer_cases.append(dict(case='early_stop_has_zero_actual_optimizer_calls',actual_calls=0));events=[]
    def injected_slsqp(*args,**kwargs):raise RuntimeError('Intentional qualification-only SciPy internal fault')
    caught=False
    with patch.dict(trace.solver.minimize.__globals__,_minimize_slsqp=injected_slsqp):
        try:trace.observed_propose(lambda e,v:events.append((e,None if v is None else int(v.nit))),u,a,b,c,gm,gs)
        except RuntimeError:caught=True
    assert caught and events==[('call',None),('return',None)] and sys.getprofile() is profile
    optimizer_cases.append(dict(case='actual_SciPy_entry_exception_return_observed',actual_calls=1,nit=0,profile_restored=True))
    del a,u,gm,gs
    # Exercise the actual entry orchestration against the saved unsafe V165
    # control, with real assignments to all 1,060,832 CPU parameters.
    spec=plan['roles'][1];ctx=entry.load_context(1);model=entry.CurrentInputBoundary().cpu();model.load_state_dict(torch.load(entry.PRIOR/'role1/endpoint.pt',map_location='cpu',weights_only=True)['state']);base=tuple(p.detach().clone() for p in model.parameters());initial=tensor_hash(model.state_dict());point=entry.PRIOR/'role1/parameter_point1';records0,normals0,gradients,direction=entry.cached_origin(spec,initial);reference=entry.PRIOR/'role1/endpoint';control=entry.CONTROL/'role1/treatment';risk={key:np.load(point/f'class1_repeat0/{key}.npy') for key in ['fixed_pure_error_contribution','full_original_class_CE']};values=[np.load(reference/f'{scope}_{suffix}.npy') for scope,suffix in [('OOF','q'),('OOF','logq'),('deployment','q'),('deployment','logq')]];cases=[]
    for mode in ['saved_unsafe_control_rejected','exception_after_temporary_assignment','exception_inside_joint_optimizer','local_unqualified_no_finite_probe']:
        target=OUT/mode;target.mkdir();counter=entry.CoverageCounter(model,spec,target/'mock_calls.jsonl',initial)
        def mocked_calls(count):
            with torch.no_grad():
                for _ in range(count):
                    for kind in ['head','feature']:counter.before(kind);counter.after(kind)
        def mocked_measure(observed_model,observed_ctx):
            assert tensor_hash(observed_model.state_dict())==initial;mocked_calls(spec['OOF_chunks']+12);return risk,*values
        def cached_synthetic_measurements(observed_model,observed_ctx,observed_base,blockers,records,normals,observed_counter,folder,expected,origin_logs,origin_q):
            full_records,full_normals,source,report=fixtures[1]
            for identity in report['fresh_function_identities']:
                records[identity]=copy.deepcopy(full_records[identity]);normals.append(full_normals[list(full_records).index(identity)]);dest=folder/identity;dest.mkdir(parents=True)
                for path in (source/identity).iterdir():os.link(path,dest/path.name)
                for _ in range(2):counter.margin_before(identity);mocked_calls(1);counter.margin_after(identity)
            return copy.deepcopy(report)
        def mocked_propose(callback,*args):
            callback('call',None)
            if mode=='exception_inside_joint_optimizer':
                assign(model,base,direction,1.);callback('return',None);raise RuntimeError('Intentional qualification-only optimizer orchestration fault')
            callback('return',SimpleNamespace(nit=3))
            if mode=='local_unqualified_no_finite_probe':return dict(status='local_inequality_or_common_descent_unqualified_stop')
            return dict(status='one_sided_joint_restoration_requires_full_actual_finite_guard',displacement=direction,correction=np.zeros_like(direction),class_reviews=[dict(linear_change=-1.),dict(linear_change=-1.)])
        def mocked_probe(observed_model,observed_base,observed_ctx,folder,delta,step,base_risk,slopes):
            assign(observed_model,observed_base,delta,step);assert tensor_hash(observed_model.state_dict())!=initial;mocked_calls(spec['OOF_chunks']+12)
            if mode=='exception_after_temporary_assignment':raise RuntimeError('Intentional qualification-only post-assignment fault')
            try:
                for scope in ['OOF','deployment']:entry.store_scope(folder,ctx,scope,np.load(control/f'{scope}_q.npy'),np.load(control/f'{scope}_logq.npy'))
                proof=copy.deepcopy(read(control/'probe.json'));assert not proof['accepted'] and not proof['classification_guard'];assert not finite_step_review(base_risk,np.load(control/'fixed_error_risk.npy'),slopes,*ctx['mass'][1:],'B',1.,False)['accepted'];proof['probe_parameter_sha256']=tensor_hash(observed_model.state_dict());save(folder/'probe.json',proof);return proof,pd.read_parquet(control/'actual_blocking_original_rows.parquet')
            finally:restore(observed_model,observed_base)
        try:
            with patch.object(entry,'measure',mocked_measure),patch.object(entry,'add_observed_functions',cached_synthetic_measurements),patch.object(entry,'observed_propose',mocked_propose),patch.object(entry,'probe',mocked_probe):result=entry.execute_one(model,ctx,base,counter,target,spec,copy.deepcopy(records0),list(normals0),gradients,direction)
            assert tensor_hash(model.state_dict())==initial and result['all_parameters_restored'] and result['joint_restoration_attempts']==result['actual_QP_solves']==1 and result['counts']['margin_completed']==18
            expected_head=66 if mode in ['saved_unsafe_control_rejected','exception_after_temporary_assignment'] else 50;assert result['counts']['head_attempts']==result['counts']['head_completed']==expected_head
            restored=torch.load(target/'endpoint.pt',map_location='cpu',weights_only=True)['state'];assert all(torch.equal(restored[name],value) for name,value in model.state_dict().items())
            assert result['new_fits']==result['permanent_updates']==0
            if mode.startswith('exception'):assert result['exception'] is not None
            else:assert result['exception'] is None
            if mode=='saved_unsafe_control_rejected':assert not result['candidate']['accepted']
            if mode=='local_unqualified_no_finite_probe':assert result['finite_proposals']==0 and not (target/'treatment').exists()
            refused=False
            try:counter.gradient_before()
            except RuntimeError:refused=True
            assert refused and counter.counts()['gradient_attempts']==0
            cases.append(dict(mode=mode,full_CPU_assignment_and_state_restore_passed=True,mocked_head_events=expected_head,all_saved_original_rows_scored=True,complete_PT_restoration_verified=True,exception=result['exception'],optimizer_events_are_mocked_here_actual_observer_tested_separately=True))
        finally:restore(model,base);counter.close()
    old=pd.read_parquet(entry.PRIOR/'role1/baseline/OOF_original_rows.parquet');endpoint=pd.read_parquet(reference/'OOF_original_rows.parquet');repaired=old.pred.ne(old.truth).to_numpy()&endpoint.pred.eq(endpoint.truth).to_numpy();assert repaired.sum()==8 and endpoint.loc[repaired,'protected_correct'].all();fake=values[0].copy();local=int(endpoint.loc[repaired,'local'].iloc[0]);truth=int(endpoint.loc[repaired,'truth'].iloc[0]);fake[local]=0;fake[local,(truth+1)%3]=1;assert stats(ctx,fake,'OOF')['protected_regressions']>0
    assert len(endpoint.loc[~endpoint.pure_current_input])>0;assert all(v['repairs_vs_V164_endpoint']==0 for v in entry.paired_progress(endpoint,endpoint).values())
    for key in ['new_fitting_permission','uniform_maximum_margin_functions']:
        bad=copy.deepcopy(plan);bad[key]=True if key=='new_fitting_permission' else 26;refused=False
        try:review_plan(bad)
        except AssertionError:refused=True
        assert refused
    check_bindings(bindings);save(OUT/'qualification.json',dict(status='V166_full_CPU_lifecycle_synthetic_measurement_actual_optimizer_observer_qualified',measurement_cases=measurement_cases,lifecycle_cases=cases,optimizer_cases=optimizer_cases,actual_CPU_optimizer_entries=2,successful_CPU_QPs=1,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,all_mock_events_separate_from_actual_official_ledger=True,all_33_full_input_functions_covered_in_synthetic_orchestration=True,eight_cumulative_repairs_retained_and_regression_refused=True,all_mixed_original_rows_scored=True,model_classification_gain_not_claimed=True,physical_run_seal_still_required=True,execution_authority=False,source_sha256=bindings));print(json.dumps(dict(status='V166_entry_lifecycle_qualification_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
