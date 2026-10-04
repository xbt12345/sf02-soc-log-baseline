"""Full CPU state and real orchestration, explicit synthetic head/QP fixtures."""
import copy,json,os,traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import assign,restore,tensor_hash,stats
from v159_float64_repeat_policy_v2 import finite_step_review
from v160_independent_saved_direction_certificate import dot
from v160_fixed_endpoint_diagnostic_v3 import actual_blockers
from v166_coverage_core_qualification_v3 import memory
from v167_trial_point_execution_review import prospective_plan,review_plan
import v167_trial_point_restoration_diagnostic as entry
import v167_trial_point_measurement as measurement

OUT=ROOT/'artifacts/v167_trial_point_lifecycle_qualification_20261002'

def save(path,value):entry.save(path,value)

def cached_values(folder):return [np.load(folder/f'{scope}_{suffix}.npy') for scope,suffix in [('OOF','q'),('OOF','logq'),('deployment','q'),('deployment','logq')]]

def array_risks(ctx,logs):
    ids=ctx['ids'];return dict(fixed_pure_error_contribution=(-(ctx['target_counts'][ids]*logs[ids]).sum(0))[1:]/ctx['mass'][1:],full_original_class_CE=(-(ctx['counts'][ids]*logs[ids]).sum(0))[1:]/ctx['mass'][1:])

def main():
    assert not OUT.exists();OUT.mkdir();before=memory();save(OUT/'initial_memory.json',before);plan=prospective_plan();review_plan(plan);identity_path=ROOT/'artifacts/v167_trial_point_identity_qualification_v2_20261002/qualification.json';identity=read(identity_path);check_bindings(identity['source_sha256']);root_counterexample=ROOT/'artifacts/v167_root_trial_point_nonlinear_counterexamples_20261002/review.json';root=read(root_counterexample);check_bindings(root['source_sha256']);assert root['actual_CPU_synthetic_QP_solves']==3 and root['cases'][0]['stages'][0]['actual_finite_constraint_passed'];assert all(not r['actual_finite_constraint_passed'] and r['nonlinear_violation_ratio']<.99 for r in root['cases'][1]['stages'])
    paths={ROOT/p for p in identity['source_sha256']}|{Path(__file__).resolve(),identity_path,root_counterexample};paths.update(ROOT/p for p in root['source_sha256']);bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0))
    # Two distinct, actual full CPU temporary states; 100 paired mock events
    # execute the actual measurement function. Shared hardlinks only avoid
    # duplicating an explicitly synthetic full-width fixture on disk.
    spec=plan['roles'][1];ctx=entry.load_context(1);state=torch.load(entry.PRIOR/'role1/endpoint.pt',map_location='cpu',weights_only=True)['state'];records,gradients,u,control_proof=entry.cached_origin(spec,state);model=entry.CurrentInputBoundary().cpu();model.load_state_dict(state);base=tuple(p.detach().clone() for p in model.parameters());origin=tensor_hash(model.state_dict());counter=entry.TrialCounter(model,spec,OUT/'measurement_mock_calls.jsonl',origin);synthetic=(np.arange(1060832,dtype=np.float64)%31)/1000.;fixture=OUT/'explicit_synthetic_full_width_gradient.npy';np.save(fixture,synthetic);actual_save=np.save;measurement_fixtures=[];origin_values=cached_values(entry.PRIOR/'role1/endpoint');control_values=cached_values(entry.CONTROL/'role1/treatment');origin_logs=dict(OOF=origin_values[1],deployment=origin_values[3]);trial_q=dict(OOF=control_values[0],deployment=control_values[2]);trial_logs=dict(OOF=control_values[1],deployment=control_values[3])
    def mock_inputs(observed_ctx,scope,chunk):return scope,chunk
    def mock_margin(observed_model,scope,chunk,query,truth,rival):
        for kind in ['head','feature']:counter.before(kind);counter.after(kind)
        return dict(gradient=synthetic.copy(),q=trial_q[scope][chunk].copy(),logq=trial_logs[scope][chunk].copy(),margin=float(trial_logs[scope][chunk[query],truth]-trial_logs[scope][chunk[query],rival]))
    def fixture_save(path,value,*args,**kwargs):
        if Path(path).name.endswith('_gradient.npy'):os.link(fixture,path)
        else:actual_save(path,value,*args,**kwargs)
    try:
        for stage,delta in enumerate([u,u*.5]):
            assign(model,base,delta,1.);trial=tensor_hash(model.state_dict());folder=OUT/f'actual_measurement_mock_stage{stage}'
            with patch.object(measurement,'inputs',mock_inputs),patch.object(measurement,'measure',mock_margin),patch.object(measurement.np,'save',fixture_save):new_records,normals=measurement.measure_all(model,ctx,records,counter,folder,origin,trial,stage,origin_logs,trial_logs,trial_q)
            assert len(new_records)==len(normals)==25 and all(r['base_parameter_sha256']==r['linearization_parameter_sha256']==trial and r['origin_parameter_sha256']==origin for r in new_records.values());assert tensor_hash(model.state_dict())==trial;measurement_fixtures.append(dict(records=new_records,folder=folder,trial=trial));restore(model,base)
        assert measurement_fixtures[0]['trial']!=measurement_fixtures[1]['trial'] and counter.margin_completed==100
        refused=0
        for operation in [lambda:counter.margin_before('extra'),lambda:counter.set_trial_point(measurement_fixtures[-1]['trial'],2),lambda:counter.gradient_before()]:
            try:operation()
            except (AssertionError,RuntimeError):refused+=1
        assert refused==3
        for _ in range(spec['head_cap']-counter.head_attempts):
            for kind in ['head','feature']:counter.before(kind);counter.after(kind)
        refused=False
        try:counter.before('head')
        except RuntimeError:refused=True
        assert refused and counter.head_attempts==164
    finally:restore(model,base);counter.close()
    # Candidate oracle fixtures have actual row/gold identities and internally
    # consistent q/logq/risk/guards, but are NOT newly computed model outputs.
    safe=[value.copy() for value in origin_values]
    frame=ctx['OOF_rows'];wrong=origin_values[0][frame.local].argmax(1)!=frame.truth.to_numpy()
    for cls in [1,2]:
        choices=frame[wrong&frame.pure_current_input&frame.truth.eq(cls)&~frame.protected_correct];local=next(int(local) for local in choices.local if ctx['target_counts'][local,cls]>0 and ctx['counts'][local].sum()==ctx['counts'][local,cls]);safe[0][local]=.01;safe[0][local,cls]=.98;safe[1][local]=np.log(safe[0][local])
    def failing_values(factor):
        values=[v.copy() for v in control_values]
        for meta in records.values():
            if meta['scope']!='OOF':continue
            local,truth,rival=[meta[k] for k in ['local','truth','rival']];gap=values[1][local,truth]-values[1][local,rival]
            if gap<0:
                vector=values[1][local].copy();vector[truth]-=(1-factor)*gap;vector-=vector.max();vector-=np.log(np.exp(vector).sum());values[1][local]=vector;values[0][local]=np.exp(vector)
        return values
    fixtures=[]
    modes=['safe_V166_role0_exact_replay','safe_V166_role2_exact_replay','first_correction_actual_guard_fixture_pass','two_corrections_still_actual_guard_failed_stop','second_correction_actual_guard_fixture_pass','residual_gate_failed_no_second_correction','exception_after_trial_assignment','exception_inside_optimizer','exception_after_probe_assignment']
    for mode in modes:
        role=0 if 'role0' in mode else 2 if 'role2' in mode else 1;current_spec=plan['roles'][role];current_ctx=entry.load_context(role);current_state=torch.load(entry.PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state'];current_records,current_gradients,delta,old_proof=entry.cached_origin(current_spec,current_state);current_model=entry.CurrentInputBoundary().cpu();current_model.load_state_dict(current_state);current_base=tuple(p.detach().clone() for p in current_model.parameters());current_origin=tensor_hash(current_state);target=OUT/mode;target.mkdir();current_counter=entry.TrialCounter(current_model,current_spec,target/'mock_calls.jsonl',current_origin);baseline=cached_values(entry.PRIOR/f'role{role}/endpoint');point=entry.PRIOR/f"role{role}/parameter_point{current_spec['parameter_point']}";risk={key:np.load(point/f'class1_repeat0/{key}.npy') for key in ['fixed_pure_error_contribution','full_original_class_CE']};probe_ordinal=0
        def mocked_calls(count):
            with torch.no_grad():
                for _ in range(count):
                    for kind in ['head','feature']:current_counter.before(kind);current_counter.after(kind)
        def mocked_measure(observed_model,observed_ctx):
            assert tensor_hash(observed_model.state_dict())==current_origin;mocked_calls(current_spec['OOF_chunks']+12);return risk,*baseline
        def mocked_trial_measure(observed_model,observed_ctx,old_records,observed_counter,folder,observed_origin,trial,stage,old_logs,logs,q):
            assert observed_origin==current_origin and tensor_hash(observed_model.state_dict())==trial;observed_counter.set_trial_point(trial,stage)
            if mode=='exception_after_trial_assignment':raise RuntimeError('Intentional qualification-only trial measurement fault')
            fixture_point=measurement_fixtures[stage];assert trial==fixture_point['trial'];new_records=copy.deepcopy(fixture_point['records']);normals=[]
            for identity in new_records:
                dest=folder/identity;dest.mkdir(parents=True)
                for path in (fixture_point['folder']/identity).iterdir():os.link(path,dest/path.name)
                normals.append(synthetic.copy())
                for _ in range(2):observed_counter.margin_before(identity);mocked_calls(1);observed_counter.margin_after(identity)
            return new_records,normals
        def mocked_propose(callback,displacement,normals,b,c,gm,gs):
            callback('call',None)
            if mode=='exception_inside_optimizer':callback('return',None);raise RuntimeError('Intentional qualification-only optimizer fault')
            callback('return',SimpleNamespace(nit=2));candidate=displacement*.5;return dict(status='one_sided_joint_restoration_requires_full_actual_finite_guard',displacement=candidate,correction=candidate-displacement,class_reviews=[dict(linear_change=dot(g,candidate)[0]) for g in [gm,gs]],math_is_explicitly_mocked_in_this_lifecycle_fixture=True)
        def mocked_probe(observed_model,observed_base,observed_ctx,folder,displacement,step,base_risk,slopes):
            nonlocal probe_ordinal
            assign(observed_model,observed_base,displacement,step);mocked_calls(current_spec['OOF_chunks']+12);probe_ordinal+=1
            if mode=='exception_after_probe_assignment':raise RuntimeError('Intentional qualification-only post-probe assignment fault')
            try:
                if role!=1:values=cached_values(entry.CONTROL/f'role{role}/treatment')
                elif mode=='first_correction_actual_guard_fixture_pass' or mode=='second_correction_actual_guard_fixture_pass' and probe_ordinal==2:values=safe
                else:values=failing_values(1. if mode=='residual_gate_failed_no_second_correction' else .5**probe_ordinal)
                rv=array_risks(current_ctx,values[1]);np.save(folder/'direction.npy',displacement);np.save(folder/'fixed_error_risk.npy',rv['fixed_pure_error_contribution']);np.save(folder/'full_original_class_risk.npy',rv['full_original_class_CE'])
                for scope,q,logs in [('OOF',values[0],values[1]),('deployment',values[2],values[3])]:entry.store_scope(folder,current_ctx,scope,q,logs)
                os_,ds=stats(current_ctx,values[0],'OOF'),stats(current_ctx,values[2],'deployment');joint=entry.joint_check(current_ctx,values[2],values[3]);full_count=all(os_[key]<=current_ctx['baseline_stats'][key] for key in ['M_errors','S_errors']);guard=os_['protected_regressions']==0 and ds['mastered'] and joint['passed'] and full_count;finite=finite_step_review(base_risk,rv['fixed_pure_error_contribution'],slopes,*current_ctx['mass'][1:],'B',1.,guard);proof=dict(step=1.,class_slopes=list(slopes),actual_parameter_change=True,classification_guard=guard,full_original_M_S_error_count_guard=full_count,OOF_stats=os_,deployment_stats=ds,joint_TRAIN_retention=joint,finite_error_target_review=finite,accepted=finite['accepted'],probe_parameter_sha256=tensor_hash(observed_model.state_dict()),new_fits=0,permanent_updates=0,head_outputs_are_explicit_qualification_oracle=True);save(folder/'probe.json',proof);blocked=pd.concat([actual_blockers(current_ctx,values[0],values[1],'OOF'),actual_blockers(current_ctx,values[2],values[3],'deployment')],ignore_index=True);blocked.to_parquet(folder/'actual_blocking_original_rows.parquet',index=False);return proof,blocked
            finally:restore(observed_model,observed_base)
        try:
            with patch.object(entry,'measure',mocked_measure),patch.object(entry,'measure_all',mocked_trial_measure),patch.object(entry,'observed_propose',mocked_propose),patch.object(entry,'probe',mocked_probe):result=entry.execute_one(current_model,current_ctx,current_base,current_state,current_counter,target,current_spec,current_records,current_gradients,delta,old_proof)
            assert tensor_hash(current_model.state_dict())==current_origin and result['all_parameters_restored'];restored=torch.load(target/'endpoint.pt',map_location='cpu',weights_only=True)['state'];assert all(torch.equal(restored[name],value) for name,value in current_state.items());assert result['new_fits']==result['permanent_updates']==0
            if mode.startswith('exception'):assert result['exception'] is not None
            else:
                assert result['exception'] is None
                if role!=1:assert result['actual_finite_accepted'] and result['counts']['head_attempts']==66 and result['actual_QP_solves']==result['counts']['margin_attempts']==0
                elif mode=='two_corrections_still_actual_guard_failed_stop':assert not result['actual_finite_accepted'] and result['status']=='bounded_second_correction_actual_failed_stop' and result['finite_proposals']==result['actual_QP_solves']==2 and result['counts']['head_attempts']==164 and not (target/'trial_point2').exists()
                elif mode=='second_correction_actual_guard_fixture_pass':assert result['actual_finite_accepted'] and result['finite_proposals']==2 and result['counts']['head_attempts']==164
                elif mode=='residual_gate_failed_no_second_correction':assert not result['actual_finite_accepted'] and result['status']=='second_correction_gate_failed_stop' and result['finite_proposals']==1 and result['counts']['head_attempts']==98
                else:assert result['actual_finite_accepted'] and result['finite_proposals']==1 and result['counts']['head_attempts']==98
            fixtures.append(dict(mode=mode,role=role,mocked_counts=result['counts'],real_full_CPU_parameters_exactly_restored=True,finite_proposals=result['finite_proposals'],mock_QP_events=result['actual_QP_solves'],actual_finite_guard_fixture_accepted=result['actual_finite_accepted'],exception=result['exception'],not_new_official_model_evidence=True))
        finally:restore(current_model,current_base);current_counter.close()
    after=memory();save(OUT/'final_memory.json',after);check_bindings(bindings);save(OUT/'qualification.json',dict(status='V167_two_actual_trial_point_identities_all_25_functions_bounded_lifecycle_and_recovery_qualified',lifecycle_fixtures=fixtures,full_parameter_count=1060832,two_distinct_full_CPU_linearization_points=True,complete_100_pair_measurement_events_are_mocked=True,gradient_arrays_are_explicitly_synthetic_hardlinks=True,head_outputs_and_QP_results_are_mocked_here=True,actual_root_full_width_three_QP_counterexamples_bound=True,real_math_and_new_CUDA_derivatives_not_claimed_by_this_fixture=True,two_failed_finite_corrections_stop_even_when_residual_decreases=True,head_margin_stage_caps_and_no_target_derivative_refused=True,all_original_and_cumulative_protection_retained=True,physical_memory_observed_before=before,physical_memory_observed_after=after,raw_bytes_for_actual_100_new_margin_arrays=100*1060832*8,official_heads=0,official_features=0,official_derivatives=0,actual_optimizer_calls_by_this_lifecycle_fixture=0,fits=0,permanent_updates=0,physical_run_seal_still_required=True,execution_authority=False,source_sha256=bindings));print(json.dumps(dict(status='V167_trial_point_lifecycle_qualification_passed',cases=len(fixtures),official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
