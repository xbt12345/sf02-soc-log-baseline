"""Actual final-tie CPU parameter lifecycle; explicit cached/synthetic head oracles."""
import copy,gc,json,os,traceback
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
from v167_trial_point_lifecycle_qualification import cached_values,array_risks
from v168_decision_floor_execution_review import prospective_plan,review_plan
import v168_decision_floor_diagnostic_v2 as entry
import v167_trial_point_measurement as measurement

OUT=ROOT/'artifacts/v168_decision_floor_lifecycle_qualification_v2_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();plan=prospective_plan();review_plan(plan);identity=ROOT/'artifacts/v168_decision_floor_identity_qualification_v2_20261002/qualification.json';iq=read(identity);check_bindings(iq['source_sha256']);assert iq['official_heads']==iq['official_derivatives']==0
    paths={Path(__file__).resolve(),identity,ROOT/'training/v168_decision_floor_lifecycle_qualification.py',ROOT/'artifacts/v168_decision_floor_lifecycle_qualification_original_console_20261002.txt',ROOT/'training/v167_trial_point_measurement.py',ROOT/'training/v167_trial_point_lifecycle_qualification.py'}|{ROOT/p for p in iq['source_sha256']};bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));before=memory();entry.save(OUT/'initial_memory.json',before)
    spec=plan['role_spec'];ctx=entry.load_context(1);state=torch.load(entry.PRIOR/'role1/endpoint.pt',map_location='cpu',weights_only=True)['state'];records,gradients,u,old_proof,goal,tie_rows=entry.cached_inputs(plan,state,ctx);origin=tensor_hash(state);trial=old_proof['probe_parameter_sha256'];model=entry.CurrentInputBoundary().cpu();model.load_state_dict(state);base=tuple(p.detach().clone() for p in model.parameters());counter=entry.TrialCounter(model,spec,OUT/'measurement_mock_calls.jsonl',origin);synthetic=(np.arange(1060832,dtype=np.float64)%31)/1000.;fixture=OUT/'explicit_synthetic_full_width_gradient.npy';np.save(fixture,synthetic);actual_save=np.save;baseline=cached_values(entry.PRIOR/'role1/endpoint');tie_values=cached_values(ROOT/plan['actual_failed_trial_path']);trial_q=dict(OOF=tie_values[0],deployment=tie_values[2]);trial_logs=dict(OOF=tie_values[1],deployment=tie_values[3]);origin_logs=dict(OOF=baseline[1],deployment=baseline[3]);fixture_folder=OUT/'actual_final_tie_measurement_mock'
    def mock_inputs(observed_ctx,scope,chunk):return scope,chunk
    def mock_margin(observed_model,scope,chunk,query,truth,rival):
        for kind in ['head','feature']:counter.before(kind);counter.after(kind)
        return dict(gradient=synthetic.copy(),q=trial_q[scope][chunk].copy(),logq=trial_logs[scope][chunk].copy(),margin=float(trial_logs[scope][chunk[query],truth]-trial_logs[scope][chunk[query],rival]))
    def fixture_save(path,value,*args,**kwargs):
        if Path(path).name.endswith('_gradient.npy'):os.link(fixture,path)
        else:actual_save(path,value,*args,**kwargs)
    try:
        assign(model,base,u,1.);assert tensor_hash(model.state_dict())==trial
        with patch.object(measurement,'inputs',mock_inputs),patch.object(measurement,'measure',mock_margin),patch.object(measurement.np,'save',fixture_save):new_records,normals=measurement.measure_all(model,ctx,records,counter,fixture_folder,origin,trial,0,origin_logs,trial_logs,trial_q)
        assert len(new_records)==len(normals)==25 and counter.margin_completed==50 and all(m['linearization_parameter_sha256']==trial and m['origin_parameter_sha256']==origin for m in new_records.values())
        refused=False
        try:counter.margin_before('extra')
        except RuntimeError:refused=True
        assert refused
        for _ in range(spec['head_cap']-counter.head_attempts):
            for kind in ['head','feature']:counter.before(kind);counter.after(kind)
        refused=False
        try:counter.before('head')
        except RuntimeError:refused=True
        assert refused and counter.head_attempts==98
        refused=False
        try:counter.gradient_before()
        except RuntimeError:refused=True
        assert refused
    finally:restore(model,base);counter.close()
    safe=[v.copy() for v in tie_values];local=21985;v=safe[1][local].copy();v[2]+=.001;v-=v.max();v-=np.log(np.exp(v).sum());safe[1][local]=v;safe[0][local]=np.exp(v)
    lost=[v.copy() for v in safe];lost[0][569]=baseline[0][569];lost[1][569]=baseline[1][569]
    risk={key:np.load(entry.PRIOR/'role1/parameter_point1/class1_repeat0'/f'{key}.npy') for key in ['fixed_pure_error_contribution','full_original_class_CE']};cases=[]
    modes=['single_candidate_guard_oracle_pass','exact_S_tie_still_refused','four_M_candidate_repairs_lost_refused','local_math_unqualified_no_finite_probe','exception_after_trial_assignment','exception_inside_optimizer','exception_after_probe_assignment']
    for mode in modes:
        target=OUT/mode;target.mkdir();current_model=entry.CurrentInputBoundary().cpu();current_model.load_state_dict(state);current_base=tuple(p.detach().clone() for p in current_model.parameters());current_counter=entry.TrialCounter(current_model,spec,target/'mock_calls.jsonl',origin);current_ctx=entry.load_context(1)
        def mocked_calls(count):
            for _ in range(count):
                for kind in ['head','feature']:current_counter.before(kind);current_counter.after(kind)
        def mocked_measure(observed_model,observed_ctx):assert tensor_hash(observed_model.state_dict())==origin;mocked_calls(16);return risk,*baseline
        def mocked_trial_measure(observed_model,observed_ctx,old_records,observed_counter,folder,observed_origin,advertised_trial,stage,old_logs,logs,q):
            assert observed_origin==origin and advertised_trial==trial and stage==0 and tensor_hash(observed_model.state_dict())==trial;observed_counter.set_trial_point(trial,stage)
            if mode=='exception_after_trial_assignment':raise RuntimeError('Intentional qualification-only trial measurement fault')
            out_records=copy.deepcopy(new_records);out_normals=[]
            for identity in out_records:
                dest=folder/identity;dest.mkdir(parents=True)
                for path in (fixture_folder/identity).iterdir():os.link(path,dest/path.name)
                out_normals.append(synthetic.copy())
                for _ in range(2):observed_counter.margin_before(identity);mocked_calls(1);observed_counter.margin_after(identity)
            return out_records,out_normals
        def mocked_propose(callback,displacement,normals,b,c,gm,gs,floors):
            callback('call',None)
            if mode=='exception_inside_optimizer':callback('return',None);raise RuntimeError('Intentional qualification-only optimizer fault')
            callback('return',SimpleNamespace(nit=2))
            if mode=='local_math_unqualified_no_finite_probe':return dict(status='local_inequality_or_common_descent_unqualified_stop',math_is_explicitly_mocked=True)
            candidate=displacement*.5;return dict(status='one_sided_joint_restoration_requires_full_actual_finite_guard',displacement=candidate,correction=candidate-displacement,class_reviews=[dict(linear_change=dot(g,candidate)[0]) for g in [gm,gs]],math_is_explicitly_mocked=True)
        def mocked_probe(observed_model,observed_base,observed_ctx,folder,displacement,step,base_risk,slopes):
            assign(observed_model,observed_base,displacement,step);mocked_calls(16)
            if mode=='exception_after_probe_assignment':raise RuntimeError('Intentional qualification-only post-probe assignment fault')
            try:
                values=tie_values if mode=='exact_S_tie_still_refused' else lost if mode=='four_M_candidate_repairs_lost_refused' else safe;rv=array_risks(observed_ctx,values[1]);np.save(folder/'direction.npy',displacement);np.save(folder/'fixed_error_risk.npy',rv['fixed_pure_error_contribution']);np.save(folder/'full_original_class_risk.npy',rv['full_original_class_CE'])
                for scope,q,logs in [('OOF',values[0],values[1]),('deployment',values[2],values[3])]:entry.store_scope(folder,observed_ctx,scope,q,logs)
                os_,ds=stats(observed_ctx,values[0],'OOF'),stats(observed_ctx,values[2],'deployment');joint=entry.joint_check(observed_ctx,values[2],values[3]);full_count=all(os_[key]<=observed_ctx['baseline_stats'][key] for key in ['M_errors','S_errors']);guard=os_['protected_regressions']==0 and ds['mastered'] and ds['new_errors_vs_initial']==0 and joint['passed'] and full_count;finite=finite_step_review(base_risk,rv['fixed_pure_error_contribution'],slopes,*observed_ctx['mass'][1:],'B',1.,guard);proof=dict(step=1.,class_slopes=list(slopes),actual_parameter_change=True,classification_guard=guard,full_original_M_S_error_count_guard=full_count,OOF_stats=os_,deployment_stats=ds,joint_TRAIN_retention=joint,finite_error_target_review=finite,accepted=finite['accepted'],probe_parameter_sha256=tensor_hash(observed_model.state_dict()),new_fits=0,permanent_updates=0,head_outputs_are_explicit_qualification_oracle=True);entry.save(folder/'probe.json',proof);blocked=pd.concat([actual_blockers(observed_ctx,values[0],values[1],'OOF'),actual_blockers(observed_ctx,values[2],values[3],'deployment')],ignore_index=True);blocked.to_parquet(folder/'actual_blocking_original_rows.parquet',index=False);return proof,blocked
            finally:restore(observed_model,observed_base)
        try:
            with patch.object(entry,'measure',mocked_measure),patch.object(entry,'measure_all',mocked_trial_measure),patch.object(entry,'propose',mocked_propose),patch.object(entry,'probe',mocked_probe):result=entry.execute_one(current_model,current_ctx,current_base,state,current_counter,target,plan,records,gradients,u,old_proof,goal,tie_rows)
            assert tensor_hash(current_model.state_dict())==origin and result['all_parameters_restored'];restored=torch.load(target/'endpoint.pt',map_location='cpu',weights_only=True)['state'];assert all(torch.equal(restored[name],value) for name,value in state.items());assert result['new_fits']==result['permanent_updates']==0 and result['actual_QP_solves']<=1 and result['finite_proposals']<=1 and not (target/'correction1').exists()
            if mode.startswith('exception'):assert result['exception'] is not None
            else:
                assert result['exception'] is None
                if mode=='single_candidate_guard_oracle_pass':assert result['actual_finite_accepted'] and result['counts']['head_attempts']==98 and result['actual_QP_solves']==1
                elif mode=='local_math_unqualified_no_finite_probe':assert not result['actual_finite_accepted'] and result['counts']['head_attempts']==82 and result['finite_proposals']==0
                else:
                    assert not result['actual_finite_accepted'] and result['counts']['head_attempts']==98
                    if mode=='four_M_candidate_repairs_lost_refused':assert not result['final_candidate']['additional_prospective_guard_review']['all_four_observed_M_candidate_repairs_retained']
            cases.append(dict(mode=mode,mocked_counts=result['counts'],full_CPU_checkpoint_exactly_restored=True,mock_QP_events=result['actual_QP_solves'],finite_proposals=result['finite_proposals'],oracle_guard_accepted=result['actual_finite_accepted'],exception=result['exception'],not_actual_model_quality_evidence=True))
        finally:restore(current_model,current_base);current_counter.close();del current_model;gc.collect()
    after=memory();entry.save(OUT/'final_memory.json',after);check_bindings(bindings);entry.save(OUT/'qualification.json',dict(status='V168_actual_final_tie_full_CPU_lifecycle_one_candidate_caps_extra_goals_and_exceptions_qualified',lifecycle_fixtures=cases,full_parameter_count=1060832,all25_final_tie_full_input_measurement_identities_replayed=True,all50_paired_margin_events_are_explicitly_mocked=True,full_width_synthetic_gradients_use_hardlinks=True,head_and_QP_outputs_are_oracles_not_actual_SOC_results=True,one_correction_limit_preserved_on_pass_failure_and_exception=True,all_original_protection_masks_unchanged=True,positive_and_negative_four_M_and_two_S_guard_fixtures_verified=True,actual_optimizer_calls_by_this_lifecycle_fixture=0,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,physical_memory_observed_before=before,physical_memory_observed_after=after,physical_seal_still_required=True,execution_authority=False,source_sha256=bindings));print(json.dumps(dict(status='V168_decision_floor_lifecycle_qualified',cases=len(cases),official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
