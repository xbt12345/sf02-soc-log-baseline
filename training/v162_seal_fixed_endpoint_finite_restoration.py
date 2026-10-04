"""Bind generic source/actual cached origins, new method and complete budget."""
import ast,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v162_fixed_endpoint_finite_restoration import OUT,PLAN,OLD,DIAG,PRIOR,PROTOCOL,save,require

def main():
    assert not OUT.exists() and not PLAN.exists()
    previous=ROOT/'artifacts/v161_cached_direction_backtrack_tail_20261002';seal=read(previous/'run_seal.json');check_bindings(seal['source_sha256'])
    qualifiers=['v162_multi_restoration_synthetic_qualification_v2','v162_finite_restoration_entry_qualification']
    independents=['v162_independent_restoration_function_review','v162_independent_multi_restoration_review','v161_independent_all_finite_results_review','v161_independent_learning_bottleneck_review']
    files={Path(__file__).resolve(),ROOT/'training/v162_fixed_endpoint_finite_restoration.py',ROOT/'training/v162_multi_function_finite_restoration_v2.py',ROOT/'docs/V162_FIXED_ENDPOINT_FINITE_RESTORATION_EXECUTION_PLAN_20261002.md',ROOT/'docs/V161_INDEPENDENT_TRAINING_DECISION_AND_TARGETED_NEXT_PLAN_20261002.md'}
    files|={ROOT/k for k in seal['source_sha256']}|{p for p in DIAG.rglob('*') if p.is_file()}|{p for p in previous.rglob('*') if p.is_file()}
    for name in qualifiers:
        folder=ROOT/f'artifacts/{name}_20261002';q=read(folder/'qualification.json');assert 'passed' in q['status'] and q['official_heads']==q['official_gradients']==q['official_fits']==0;check_bindings(q['source_sha256'])
        files|={ROOT/k for k in q['source_sha256']}|{p for p in folder.rglob('*') if p.is_file()}
    for name in independents:
        folder=ROOT/f'artifacts/{name}_20261002';assert folder.exists();files|={p for p in folder.rglob('*') if p.is_file()}|{ROOT/'training'/f'{name}.py'}
    roles=[]
    for role,k,round_number in [(0,10,0),(1,4,1),(2,10,0)]:
        target=OLD/f'role{role}/round{round_number}';records=read(target/'normal_records.json');a=np.load(target/'raw_margin_normals.npy');cached=[]
        for i,(identity,metadata) in enumerate(records.items()):
            f=OLD/f'role{role}/normals/{identity}';assert read(f/'input_binding.json')==metadata and np.array_equal(np.load(f/'repeat0_gradient.npy'),a[i])
            assert all(v['passed'] for v in read(f/'measurement_repeat_review.json').values())
            cached.append(dict(metadata=(f/'input_binding.json').relative_to(ROOT).as_posix(),gradient=(f/'repeat0_gradient.npy').relative_to(ROOT).as_posix()))
        for cls in [1,2]:assert all(v['passed'] for v in read(DIAG/f'role{role}/class{cls}_actual_gradient_repeat_review.json').values())
        fit=read(PRIOR/f'fold{role}_B/fit.json');margin_cap=2*(24-len(cached))
        roles.append(dict(role=role,OOF_chunks=k,deployment_chunks=12,head_cap=6*(k+12)+margin_cap,fresh_margin_gradient_cap=margin_cap,restoration_cap=4,finite_proposal_cap=4,endpoint_parameter_sha256=fit['endpoint_parameter_sha256'],cached_normals=cached,fixed_base_step=1/16,initial_saved_actual_failure=(DIAG/f'role{role}/round0/probe4').relative_to(ROOT).as_posix(),target_gradients_already_repeated_at_same_endpoint=True))
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    plan=dict(protocol=PROTOCOL,activation_entries=['training/v162_fixed_endpoint_finite_restoration.py'],roles=roles,new_caps=dict(heads=494,features=494,fixed_error_target_gradients=0,full_original_class_gradients=0,margin_gradients=134,finite_proposals=12,restoration_solves=12,QP_solves=0,fits=0,permanent_updates=0),prior_actual_costs=dict(heads=16908,features=16908,full_original_class_gradients=368,fixed_error_target_gradients=12,margin_gradients=10,complete_parameter_derivatives_all_types=390),future_cumulative_actual_caps=dict(heads=17402,features=17402,full_original_class_gradients=368,fixed_error_target_gradients=12,margin_gradients=144,complete_parameter_derivatives_all_types=524),preserved_technical_head_cap=82174,preserved_technical_complete_derivative_cap=2426,single_factor='actual nonlinear protected-margin residual restoration using measured original-unit same-origin functions',same_algorithm_and_step_for_all_roles=True,no_role_specific_algorithm=True,original_classification_and_numeric_gates_unchanged=True,full_displacement_Armijo_step=1.,no_old_QP_optimality_claim_for_corrected_displacement=True,normal_cap_per_role=24,paired_all_new_actual_blocker_normals=True,normal_gradients_at_unchanged_origin=True,all_active_functions_retained=True,all_original_mixed_rows_scored=True,full_original_M_S_each_error_count_guard=True,full_deployment_and_joint_TRAIN_retention=True,fixed_endpoint_pure_repairs_protected=True,zero_step_full_q_logq_risk_and_exact_argmax=True,all_candidate_outputs_saved_before_acceptance=True,restore_each_candidate_and_exit=True,no_new_target_gradient_fit_or_permanent_update=True,max_restorations_per_role=4,source_sha256=bindings,quality_acceptance=False)
    assert sum(r['head_cap'] for r in roles)==494 and sum(r['fresh_margin_gradient_cap'] for r in roles)==134 and 17402<=82174 and 524<=2426
    OUT.mkdir();save(OUT/'pre_registration_bindings.json',dict(source_sha256=bindings,official_calls=0));save(PLAN,plan);bindings[PLAN.relative_to(ROOT).as_posix()]=sha(PLAN)
    save(OUT/'run_seal.json',dict(status='generic_nonlinear_finite_restoration_sealed_before_calls',protocol=PROTOCOL,allowed_entries=plan['activation_entries'],plan_sha256=sha(PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings))
    require();save(OUT/'registration.json',dict(status='V162_generic_three_role_finite_restoration_registered',physical_sources=len(bindings),new_caps=plan['new_caps'],official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,run_seal_sha256=sha(OUT/'run_seal.json')))
    print(json.dumps(dict(status='V162_generic_finite_restoration_sealed',physical_sources=len(bindings),new_caps=plan['new_caps'])))

if __name__=='__main__':main()
