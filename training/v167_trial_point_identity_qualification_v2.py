"""Actual complete input/parameter roles and saved risk continuation fixtures."""
import ast,copy,json,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import assign,restore,tensor_hash,stats
from v160_margin_normal import input_identity
from v167_trial_point_execution_review import prospective_plan,review_plan
import v167_trial_point_restoration_diagnostic as entry

OUT=ROOT/'artifacts/v167_trial_point_identity_qualification_v2_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();plan=prospective_plan();review_plan(plan)
    trial_review=ROOT/'artifacts/v166_independent_actual_coverage_review_20261002';check_bindings(read(trial_review/'pre_review_bindings.json')['source_sha256']);runtime=ROOT/'artifacts/v166_observed_runtime_failure_replay_v2_20261002/replay.json';check_bindings(read(runtime)['source_sha256']);counterexample=ROOT/'artifacts/v167_root_trial_point_nonlinear_counterexamples_20261002/review.json';counterexample_review=read(counterexample);check_bindings(counterexample_review['source_sha256']);assert all(counterexample_review[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates'])
    active=[Path(__file__).resolve(),Path(entry.__file__).resolve(),ROOT/'training/v167_trial_point_measurement.py',ROOT/'training/v167_trial_point_execution_review.py'];paths=set(active)|{ROOT/'training/v167_trial_point_identity_qualification.py',ROOT/'artifacts/v167_trial_point_identity_qualification_20261002/failure.json',ROOT/'artifacts/v167_trial_point_identity_qualification_original_console_20261002.txt',runtime,counterexample,ROOT/'training/review_policy/v167_trial_point_restoration_draft.json',ROOT/'training/review_policy/v166_observed_runtime_boundaries_v2.json',ROOT/'docs/V166_ROOT_ACTUAL_REVIEW_AND_V167_TRIAL_POINT_RESTORATION_PLAN_20261002.md'}
    for folder in [entry.CONTROL,ROOT/'artifacts/v167_root_trial_point_nonlinear_counterexamples_20261002',ROOT/'artifacts/v166_independent_covered_margin_geometry_review_20261002']:paths.update(p for p in folder.rglob('*') if p.is_file())
    paths.update(ROOT/p for p in read(trial_review/'pre_review_bindings.json')['source_sha256']);paths.update(ROOT/p for p in read(runtime)['source_sha256']);paths.add(ROOT/'training/v167_root_trial_point_nonlinear_counterexamples.py')
    for p in active:ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));roles=[]
    for spec in plan['roles']:
        role=spec['role'];ctx=entry.load_context(role);state=torch.load(entry.PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state'];records,gradients,u,proof=entry.cached_origin(spec,state);origin=tensor_hash(state);model=entry.CurrentInputBoundary().cpu();model.load_state_dict(state);base=tuple(p.detach().clone() for p in model.parameters());assign(model,base,u,1.);trial=tensor_hash(model.state_dict());assert trial==proof['probe_parameter_sha256']!=origin
        for identity,meta in records.items():
            scope,local,truth,rival=[meta[k] for k in ['scope','local','truth','rival']];ids=ctx['ids'] if scope=='OOF' else np.arange(22546);pos=int(np.searchsorted(ids,local));assert ids[pos]==local;chunk=ids[pos//2048*2048:pos//2048*2048+2048];assert input_identity(role,scope,chunk,ctx['x'][chunk],np.asarray(ctx[scope][chunk],np.float64),pos%2048,truth,rival)==identity
        restore(model,base);assert tensor_hash(model.state_dict())==origin;blockers=pd.read_parquet(entry.CONTROL/f'role{role}/treatment/actual_blocking_original_rows.parquet');identities=entry.blocker_identities(ctx,blockers);assert identities<=set(records);assert proof['accepted']==(len(blockers)==0)
        assert spec['head_cap']==(2+spec['finite_proposal_cap'])*(spec['OOF_chunks']+12)+spec['margin_gradient_cap'];assert spec['margin_gradient_cap']==50*spec['QP_cap'];assert spec['finite_proposal_cap']==1 if proof['accepted'] else spec['finite_proposal_cap']==2
        if not proof['accepted']:
            assert len(records)==25 and len(identities)==5 and len(blockers)==10;point=entry.PRIOR/f"role{role}/parameter_point{spec['parameter_point']}";risk=np.load(point/'class1_repeat0/fixed_pure_error_contribution.npy');logs={scope:np.load(entry.CONTROL/f'role{role}/treatment/{scope}_logq.npy') for scope in ['OOF','deployment']};c=entry.margins(records,list(records),logs);proposal=entry.CONTROL/f'role{role}/treatment'
            gate=entry.continuation_gate(ctx,records,blockers,risk,proposal,c,c*.5);assert gate['continue_second_correction'] and gate['classification_acceptance_not_overridden'] and not proof['accepted'];assert gate['fixed_target_drop_and_Armijo_only']['accepted']
            for factor in [.99,1.,1.01]:assert not entry.continuation_gate(ctx,records,blockers,risk,proposal,c,c*factor)['continue_second_correction']
            extra=blockers.copy();extra.loc[extra.index[0],'rival']=0;bad=entry.continuation_gate(ctx,records,extra,risk,proposal,c,c*.5);assert not bad['continue_second_correction'] and bad['uncovered_function_identities']
            actual_load=entry.np.load
            def failed_risk(path,*args,**kwargs):return risk if Path(path).name=='fixed_error_risk.npy' else actual_load(path,*args,**kwargs)
            with patch.object(entry.np,'load',failed_risk):assert not entry.continuation_gate(ctx,records,blockers,risk,proposal,c,c*.5)['continue_second_correction']
            entry.save(OUT/'role1_second_stage_gate_fixtures.json',dict(real_V166_fixed_target_drop_Armijo_passed=True,real_classification_failure_not_overridden=True,half_negative_margin_synthetic_fixture_continues=True,ratio_at_099_or_higher_refused=True,new_function_refused=True,no_target_descent_refused=True,fixture_negative_margins_are_synthetic_not_actual_new_candidates=True))
        previous=pd.read_parquet(entry.PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet');assert np.array_equal(previous.protected_correct,ctx['OOF_rows'].protected_correct);roles.append(dict(role=role,origin_parameter_sha256=origin,actual_V166_trial_parameter_sha256=trial,complete_functions=len(records),actual_protected_blocking_functions=len(identities),actual_V166_finite_accepted=proof['accepted'],real_full_CPU_temporary_parameter_assignment_and_restore=True,complete_function_input_identities_verified=True,old_class_gradients_remain_at_actual_V164_origin=True,original_cumulative_protection_preserved=True))
    for field,value in [('uniform_maximum_margin_functions',26),('maximum_corrections_on_actual_failed_role',3),('second_correction_max_negative_margin_factor',1.),('new_fitting_permission',True)]:
        bad=copy.deepcopy(plan);bad[field]=value;refused=False
        try:review_plan(bad)
        except AssertionError:refused=True
        assert refused
    check_bindings(bindings);entry.save(OUT/'qualification.json',dict(status='V167_complete_function_actual_trial_origin_identity_and_second_stage_gate_qualified',roles=roles,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,actual_zero_CPU_optimizers_by_this_identity_review=True,root_full_width_nonlinear_success_and_twice_failure_fixture_bound=True,complete_measurement_and_entry_lifecycle_still_required=True,physical_run_seal_still_required=True,execution_authority=False,source_sha256=bindings));print(json.dumps(dict(status='V167_trial_point_identity_qualification_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
