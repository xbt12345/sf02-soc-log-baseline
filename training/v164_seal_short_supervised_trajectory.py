"""Seal three actual short fits and all derivative/restore/protection costs."""
import ast,importlib.metadata,json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
import v164_short_supervised_trajectory as entry

def main():
    assert not entry.OUT.exists() and not entry.PLAN.exists()
    previous=ROOT/'artifacts/v163_fixed_endpoint_one_sided_restoration_20261002';old=read(previous/'run_seal.json');check_bindings(old['source_sha256']);budget=read(entry.BUDGET)
    results=ROOT/'artifacts/v163_results_20261002/actual_result_summary.json';r=read(results);assert r['all_three_actual_finite_pass'] and r['cumulative_heads']==17662 and r['cumulative_all_complete_derivatives']==422 and r['new_fits']==r['permanent_updates']==0;check_bindings(r['source_sha256'])
    files={ROOT/k for k in old['source_sha256']}|{p for p in previous.rglob('*') if p.is_file()}|{ROOT/k for k in r['source_sha256']}
    files|={Path(__file__).resolve(),entry.BUDGET,results,ROOT/'training/v164_short_supervised_trajectory.py',ROOT/'training/v164_trajectory_state.py',ROOT/'training/v164_solver_trace.py',ROOT/'docs/V164_SHORT_SUPERVISED_TRAJECTORY_EXECUTION_PLAN_20261002.md',ROOT/'docs/V163_INDEPENDENT_ACTUAL_RESULTS_AND_SHORT_TRAINING_DECISION_20261002.md',ROOT/'docs/V163_READOUT_OVERRIDE_BOUND_DIAGNOSIS_20261002.md'}
    for name in ['v164_short_trajectory_lifecycle_qualification','v164_short_entry_qualification']:
        folder=ROOT/f'artifacts/{name}_20261002';q=read(folder/'qualification.json');assert 'passed' in q['status'] and q['official_heads']==q['official_gradients']==q['official_fits']==q['permanent_updates']==0;check_bindings(q['source_sha256']);files|={ROOT/k for k in q['source_sha256']}|{p for p in folder.rglob('*') if p.is_file()}
    for name in ['v163_saved_joint_restoration_actual_audit','v163_independent_actual_restoration_review','v163_independent_readout_override_bound_review']:
        folder=ROOT/f'artifacts/{name}_20261002';assert folder.exists();files|={p for p in folder.rglob('*') if p.is_file()}|{ROOT/'training'/f'{name}.py'}
    initial={str(role):read(previous/f'role{role}/diagnostic.json')['initial_parameter_sha256'] for role in range(3)}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).is_file():files.add(Path(name).resolve())
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    free=shutil.disk_usage(ROOT).free;assert free>=20*1024**3,'Keep all new physical evidence; do not start without bounded storage'
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    plan=dict(protocol=entry.PROTOCOL,allowed_entries=['training/v164_short_supervised_trajectory.py'],roles=budget['roles'],initial_parameter_sha256=initial,new_caps=budget['new_caps'],prior_actual_costs=budget['prior_actual_costs'],future_cumulative_actual_caps=budget['future_cumulative_actual_caps'],technical_head_cap=82174,technical_complete_derivative_cap=2426,source_sha256=bindings,uniform_algorithm_per_role=True,max_actual_accepted_updates_per_role=10,first_update_replays_registered_actual_candidate=True,fixed_original_error_targets_and_class_mass=True,full_new_target_gradient_repeats_at_each_accepted_point_including_terminal=True,new_normal_values_at_each_theta_only=True,max_actual_functions_per_point=24,max_restoration_proposals_per_point=6,max_total_proposals_per_new_direction_point=7,all_new_actual_repairs_including_mixed_cumulatively_protected=True,initial_correct_protection_scope='registered original pure correct plus earlier accepted protected rows; initial mixed correct remains fully scored',all_original_row_fixed_endpoint_regressions_reported=True,original_M_S_error_count_and_16eps_Armijo_unchanged=True,full_deployment_and_joint_TRAIN_retention=True,failed_probe_or_commit_restores_last_true_accepted_state=True,endpoint_is_last_actual_accepted_not_best_state=True,last_five_mastery_requires_real_distinct_accepted_states=True,all_recorded_QP_entries_include_exception=True,normal_matrix_references_avoid_duplicate_storage=True,preseal_free_bytes=free,fixed_output_override_bound_reported_every_accepted_state=True,supervised_development_training_not_external_validation=True,no_extra_update_or_model_promotion_authority=True,quality_acceptance=False)
    assert plan['future_cumulative_actual_caps']['heads']==23878<=82174 and plan['future_cumulative_actual_caps']['complete_parameter_derivatives_all_types']==1838<=2426
    entry.OUT.mkdir();entry.save(entry.OUT/'pre_registration_bindings.json',dict(source_sha256=bindings,official_calls=0));entry.save(entry.PLAN,plan);bindings[entry.PLAN.relative_to(ROOT).as_posix()]=sha(entry.PLAN)
    entry.save(entry.OUT/'run_seal.json',dict(status='three_supervised_short_trajectories_sealed_before_official_fit',protocol=entry.PROTOCOL,allowed_entries=plan['allowed_entries'],plan_sha256=sha(entry.PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings))
    entry.require();entry.save(entry.OUT/'registration.json',dict(status='V164_three_short_supervised_trajectories_registered',physical_sources=len(bindings),new_caps=plan['new_caps'],prior_actual_costs=plan['prior_actual_costs'],future_cumulative_actual_caps=plan['future_cumulative_actual_caps'],official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,run_seal_sha256=sha(entry.OUT/'run_seal.json')));print(json.dumps(dict(status='V164_short_supervised_trajectories_sealed',physical_sources=len(bindings),new_caps=plan['new_caps'])))

if __name__=='__main__':main()
