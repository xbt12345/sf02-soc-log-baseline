"""Real CPU parameter lifecycle, saved forward fixtures; no official calls."""
import copy,json,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import assign,restore,stats,tensor_hash
from v159_float64_repeat_policy_v2 import repeat_gradient,finite_step_review
from v160_margin_normal import input_identity
from v165_execution_review import review_plan
import v165_fixed_endpoint_decision_floor_diagnostic as entry

OUT=ROOT/'artifacts/v165_entry_lifecycle_qualification_20261002'
BUDGET=ROOT/'training/review_policy/v165_decision_floor_prospective_budget.json'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def plan_fixture():
    plan=read(BUDGET);registry=read(ROOT/plan['failure_constraints']);plan['failure_case_actions']={r['id']:r['required_action'] for r in registry['cases']};return plan

def main():
    assert not OUT.exists();OUT.mkdir();plan=plan_fixture();assert review_plan(plan)['status']=='V165_zero_fit_finite_plan_passed'
    paths={Path(__file__).resolve(),Path(entry.__file__).resolve(),ROOT/'training/v165_execution_review.py',BUDGET,ROOT/plan['failure_constraints'],ROOT/'training/v165_decision_floor_restoration.py',ROOT/'training/v165_decision_floor_saved_vector_qualification.py',ROOT/'artifacts/v165_decision_floor_saved_vector_qualification_20261002/qualification.json',ROOT/'docs/V164_RESULTS_AND_V165_DECISION_FLOOR_DIAGNOSTIC_PLAN_20261002.md'}
    for name in ['v164_independent_actual_short_trajectory_review','v164_independent_capacity_stop_review','v164_independent_decision_floor_counterfactual']:
        folder=ROOT/f'artifacts/{name}_20261002';paths.update(p for p in folder.rglob('*') if p.is_file());paths.add(ROOT/'training'/f'{name}.py')
    paths.update(p for p in entry.PRIOR.rglob('*') if p.is_file())
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0))
    origins=[]
    for spec in plan['roles']:
        role=spec['role'];ctx=entry.load_context(role);state=torch.load(entry.PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state'];model=entry.CurrentInputBoundary().cpu();model.load_state_dict(state);identity=tensor_hash(model.state_dict());assert identity==spec['endpoint_parameter_sha256']
        direction,slopes,certificate=entry.cached_treatment(spec,identity);point=entry.PRIOR/f"role{role}/parameter_point{spec['parameter_point']}";refs=read(point/'restoration0/active_normal_references.json')
        for ref in refs.values():
            meta=ref['metadata'];scope=meta['scope'];ids=ctx['ids'] if scope=='OOF' else np.arange(22546);pos=int(np.searchsorted(ids,meta['local']));chunk=ids[pos//2048*2048:pos//2048*2048+2048]
            assert input_identity(role,scope,chunk,ctx['x'][chunk],np.asarray(ctx[scope][chunk],np.float64),pos%2048,meta['truth'],meta['rival'])==meta['input_identity']
            gradientpath=ROOT/ref['gradient'];assert repeat_gradient(np.load(gradientpath),np.load(gradientpath.parent/'repeat1_gradient.npy'))['passed']
        for cls in [1,2]:assert repeat_gradient(np.load(point/f'class{cls}_repeat0/complete_fixed_error_target_gradient.npy'),np.load(point/f'class{cls}_repeat1/complete_fixed_error_target_gradient.npy'))['passed']
        invalid=copy.deepcopy(spec);invalid['parameter_point']=1 if spec['parameter_point']!=1 else 2
        refused=False
        try:entry.cached_treatment(invalid,identity)
        except (AssertionError,FileNotFoundError):refused=True
        assert refused
        origins.append(dict(role=role,parameter_sha256=identity,complete_functions_reproduced=len(refs),full_gradient_pairs_repeat=True,wrong_parameter_point_refused=True,cumulative_protection_matches_actual_last_endpoint=True))
    # Exercise the actual new orchestration with real full CPU parameters.
    # Head results are explicitly saved fixtures, not model computations.
    role=1;spec=plan['roles'][role];ctx=entry.load_context(role);model=entry.CurrentInputBoundary().cpu();model.load_state_dict(torch.load(entry.PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state']);base=tuple(p.detach().clone() for p in model.parameters());initial=tensor_hash(model.state_dict());direction,slopes,_=entry.cached_treatment(spec,initial)
    point=entry.PRIOR/f'role{role}/parameter_point1';reference=entry.PRIOR/f'role{role}/endpoint';control=point/'restoration0/finite_probe';risk={key:np.load(point/f'class1_repeat0/{key}.npy') for key in ['fixed_pure_error_contribution','full_original_class_CE']}
    values=[np.load(reference/f'{scope}_{suffix}.npy') for scope,suffix in [('OOF','q'),('OOF','logq'),('deployment','q'),('deployment','logq')]]
    fixture_results=[]
    for mode in ['actual_saved_unsafe_control_rejected','exception_after_temporary_assignment_restored']:
        folder=OUT/mode;folder.mkdir();counter=entry.DiagnosticCounter(model,spec['head_cap'],folder/'mock_calls.jsonl')
        def mocked_calls():
            with torch.no_grad():
                for _ in range(spec['OOF_chunks']+12):
                    for kind in ['head','feature']:counter.before(kind);counter.after(kind)
        def mocked_measure(observed_model,observed_ctx):
            assert tensor_hash(observed_model.state_dict())==initial;mocked_calls();return risk,*values
        def mocked_probe(observed_model,observed_base,observed_ctx,target,delta,step,base_risk,observed_slopes):
            assign(observed_model,observed_base,delta,step);mocked_calls()
            if mode.startswith('exception'):raise RuntimeError('Intentional qualification-only post-assignment fault')
            try:
                for scope in ['OOF','deployment']:
                    entry.store_scope(target,observed_ctx,scope,np.load(control/f'{scope}_q.npy'),np.load(control/f'{scope}_logq.npy'))
                proof=read(control/'probe.json');assert not proof['accepted'] and not proof['classification_guard']
                finite=finite_step_review(base_risk,np.load(control/'fixed_error_risk.npy'),observed_slopes,*ctx['mass'][1:],'B',1.,False);assert not finite['accepted']
                proof=copy.deepcopy(proof);proof['probe_parameter_sha256']=tensor_hash(observed_model.state_dict());save(target/'probe.json',proof);return proof,pd.read_parquet(control/'actual_blocking_original_rows.parquet')
            finally:restore(observed_model,observed_base)
        try:
            with patch.object(entry,'measure',mocked_measure),patch.object(entry,'probe',mocked_probe):result=entry.execute_one(model,ctx,base,counter,folder,spec,direction,slopes)
            assert tensor_hash(model.state_dict())==initial and result['all_parameters_restored'] and result['counts']['head_attempts']==48
            if result['candidate'] is not None:assert not result['candidate']['accepted']
            else:assert result['exception'] is not None
            restored=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)['state'];assert all(torch.equal(restored[name],value) for name,value in model.state_dict().items())
            assert result['new_fits']==result['permanent_updates']==result['new_complete_derivatives']==0
            rejected=0
            for method in [counter.gradient_before,counter.margin_before]:
                try:method()
                except RuntimeError:rejected+=1
            assert rejected==2 and counter.counts()['gradient_attempts']==0
            overcap=False
            try:counter.before('head')
            except RuntimeError:overcap=True
            assert overcap
            fixture_results.append(dict(mode=mode,real_full_CPU_parameter_assignment_and_restore=True,mocked_head_feature_events=48,original_class_tradeoff_guard_rejected=True,exception=result['exception'],zero_derivative_gate_and_call_cap_refused=True))
        finally:restore(model,base);counter.close()
    previous=pd.read_parquet(entry.PRIOR/'role1/baseline/OOF_original_rows.parquet');endpoint=pd.read_parquet(reference/'OOF_original_rows.parquet');oldwrong=previous.pred.ne(previous.truth).to_numpy();repaired=oldwrong&endpoint.pred.eq(endpoint.truth).to_numpy();assert repaired.sum()==8 and endpoint.loc[repaired,'protected_correct'].all()
    fake=values[0].copy();target_local=int(endpoint.loc[repaired,'local'].iloc[0]);truth=int(endpoint.loc[repaired,'truth'].iloc[0]);fake[target_local]=0;fake[target_local,(truth+1)%3]=1;assert stats(ctx,fake,'OOF')['protected_regressions']>0
    history=entry.paired_progress(endpoint,endpoint);assert all(r['repairs_vs_V164_endpoint']==0 for r in history.values())
    cap=read(ROOT/'artifacts/v164_independent_capacity_stop_review_20261002/review.json');assert 'capacity' in cap['status'] or 'verified' in cap['status']
    for field in ['new_fitting_permission','max_margin_functions_unchanged']:
        bad=copy.deepcopy(plan);bad[field]=True if field=='new_fitting_permission' else 32;refused=False
        try:review_plan(bad)
        except AssertionError:refused=True
        assert refused
    check_bindings(bindings);save(OUT/'qualification.json',dict(status='V165_entry_identity_original_constraints_mocked_lifecycle_and_budget_qualification_passed',origins=origins,lifecycle_fixtures=fixture_results,actual_new_eight_rows_cumulative_guard_retained=True,old_eight_not_recounted_as_endpoint_gain=True,exact_zero_local_margin_not_actual_argmax_permission=True,no_physical_fit_or_promotion=True,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,fixture_head_outputs_are_cached_not_new_forward=True,source_sha256=bindings));print(json.dumps(dict(status='V165_entry_lifecycle_qualification_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
