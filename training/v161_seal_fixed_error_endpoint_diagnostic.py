"""Seal fixed error objectives, unchanged actual input and prospective budget."""
import ast,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v161_fixed_error_endpoint_diagnostic_v2 import OUT,OLD,COHORT,PRIOR,PLAN,PROTOCOL,save,require

def main():
    assert not OUT.exists() and not PLAN.exists()
    previous=ROOT/'artifacts/v160_cached_polished_direction_finite_probe_20261002'
    oldseal=read(previous/'run_seal.json');check_bindings(oldseal['source_sha256'])
    qualifiers=[ROOT/'artifacts/v161_error_risk_synthetic_qualification_20261002',ROOT/'artifacts/v161_fixed_error_entry_qualification_20261002']
    files={Path(__file__).resolve(),ROOT/'training/v161_fixed_error_endpoint_diagnostic_v2.py',ROOT/'training/v161_fixed_pure_error_risk.py',ROOT/'docs/V161_FIXED_ERROR_ENDPOINT_DIAGNOSTIC_EXECUTION_PLAN_20261002.md'}
    files|={ROOT/k for k in oldseal['source_sha256']}|{p for p in previous.rglob('*') if p.is_file()}|{p for p in COHORT.rglob('*') if p.is_file()}
    for folder in qualifiers:
        q=read(folder/'qualification.json');assert 'passed' in q['status'] and q['official_heads']==q['official_error_gradients']==q['official_fits']==0
        check_bindings(q['source_sha256']);files|={ROOT/k for k in q['source_sha256']}|{p for p in folder.rglob('*') if p.is_file()}
    reviews=['v161_independent_saved_input_identity_review','v161_independent_actual_objective_counterexample_replay','v161_synthetic_accuracy_vs_mean_CE_review','v161_synthetic_cumulative_correct_guard_review','v161_independent_CPU_error_risk_function_review_v3']
    for name in reviews:
        folder=ROOT/f'artifacts/{name}_20261002';assert folder.exists();files|={p for p in folder.rglob('*') if p.is_file()}
        source=ROOT/'training'/f'{name}.py'
        if source.exists():files.add(source)
    roles=[]
    for role,k,round_number in [(0,10,0),(1,4,1),(2,10,0)]:
        target=OLD/f'role{role}/round{round_number}';records=read(target/'normal_records.json');a=np.load(target/'raw_margin_normals.npy');cached=[]
        for i,(identity,metadata) in enumerate(records.items()):
            folder=OLD/f'role{role}/normals/{identity}';assert read(folder/'input_binding.json')==metadata
            assert np.array_equal(np.load(folder/'repeat0_gradient.npy'),a[i])
            assert all(v['passed'] for v in read(folder/'measurement_repeat_review.json').values())
            cached.append(dict(metadata=(folder/'input_binding.json').relative_to(ROOT).as_posix(),gradient=(folder/'repeat0_gradient.npy').relative_to(ROOT).as_posix()))
        assert len(cached)==[1,3,1][role]
        fit=read(PRIOR/f'fold{role}_B/fit.json');cohort=read(COHORT/'review.json')['roles'][role]
        roles.append(dict(role=role,OOF_chunks=k,deployment_chunks=12,head_cap=4*k+12+60*(k+12)+2*(24-len(cached))+k+12,fixed_error_target_gradient_cap=4,fresh_margin_gradient_cap=2*(24-len(cached)),finite_proposal_cap=60,QP_solve_cap=3,endpoint_parameter_sha256=fit['endpoint_parameter_sha256'],cached_normals=cached,fixed_error_original_rows=cohort['fixed_targets'],fixed_pure_correct_original_rows=cohort['protected_pure_correct_rows'],complete_original_mass=[0]+[r['complete_original_class_mass'] for r in cohort['classes']]))
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    plan=dict(protocol=PROTOCOL,activation_entries=['training/v161_fixed_error_endpoint_diagnostic_v2.py'],roles=roles,new_caps=dict(heads=3926,features=3926,fixed_error_target_gradients=12,full_original_class_gradients=0,margin_gradients=134,finite_proposals=180,QP_solves=9,fits=0,permanent_updates=0),prior_actual_costs=dict(heads=15728,features=15728,full_original_class_gradients=368,fixed_error_target_gradients=0,margin_gradients=10),future_cumulative_actual_caps=dict(heads=19654,features=19654,full_original_class_gradients=368,fixed_error_target_gradients=12,margin_gradients=144),preserved_technical_head_cap=82174,preserved_technical_margin_gradient_cap=144,single_factor='fixed pure old-error CE original-frequency contribution over complete original class masses',fixed_targets_role_records=6824,fixed_target_unique_original_rows=4614,fixed_target_cohort_never_reselected=True,all_original_rows_forward_and_scored=True,mixed_original_rows_scored=True,full_CE_reported_not_hard_nonincrease_gate=True,full_original_M_S_errors_each_not_above_fixed_endpoint=True,protected_original_and_fixed_endpoint_pure_correct=True,full_deployment_joint_retention=True,paired_same_class_gradient_repeat='original four segment 8eps policy',finite_gate='original 16eps drop and Armijo',step_sequence='2**(-j), j=0..19',active_rounds_per_role=3,normal_cap_per_role=24,all_new_actual_blockers_collected=True,zero_fit_no_permanent_update=True,restore_every_candidate_and_exit=True,runtime_package_match_before_every_entry=True,quality_acceptance=False,source_sha256=bindings)
    assert sum(r['head_cap'] for r in roles)==3926 and sum(r['fresh_margin_gradient_cap'] for r in roles)==134
    assert plan['future_cumulative_actual_caps']['heads']<=82174 and 10+134==144
    OUT.mkdir();save(OUT/'pre_registration_bindings.json',dict(status='before_official_fixed_error_gradients_or_probes',source_sha256=bindings));save(PLAN,plan);bindings[PLAN.relative_to(ROOT).as_posix()]=sha(PLAN)
    save(OUT/'run_seal.json',dict(status='sealed_new_error_target_fixed_endpoint_diagnostic_before_calls',protocol=PROTOCOL,allowed_entries=plan['activation_entries'],plan_sha256=sha(PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings))
    require();save(OUT/'registration.json',dict(status='V161_three_fixed_endpoints_registered',physical_sources=len(bindings),new_caps=plan['new_caps'],official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,quality_acceptance=False,run_seal_sha256=sha(OUT/'run_seal.json')))
    print(json.dumps(dict(status='V161_fixed_error_diagnostic_sealed',physical_sources=len(bindings),new_caps=plan['new_caps'])))

if __name__=='__main__':main()
