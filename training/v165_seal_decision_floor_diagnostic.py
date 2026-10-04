"""Seal the new zero-fit diagnostic and complete physical source closure."""
import ast,importlib.metadata,json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v165_execution_review import review_plan
import v165_fixed_endpoint_decision_floor_diagnostic as entry

BUDGET=ROOT/'training/review_policy/v165_decision_floor_prospective_budget.json'

def main():
    assert not entry.OUT.exists() and not entry.PLAN.exists()
    previous=read(entry.PRIOR/'run_seal.json');check_bindings(previous['source_sha256'])
    prior=ROOT/'artifacts/v164_results_20261002/actual_result_summary.json';actual=read(prior);check_bindings(actual['source_sha256'])
    assert actual['cumulative_heads']==18358 and actual['cumulative_all_complete_derivatives']==552 and actual['new_fits']==3 and actual['permanent_updates']==5
    files={ROOT/p for p in previous['source_sha256']}|{p for p in entry.PRIOR.rglob('*') if p.is_file()}|{ROOT/p for p in actual['source_sha256']}
    files.update([Path(__file__).resolve(),Path(entry.__file__).resolve(),BUDGET,prior,ROOT/'training/v165_execution_review.py',ROOT/'docs/V164_RESULTS_AND_V165_DECISION_FLOOR_DIAGNOSTIC_PLAN_20261002.md',ROOT/'training/review_policy/v164_root_failure_constraints.json'])
    for name in ['v165_decision_floor_saved_vector_qualification','v165_entry_lifecycle_qualification']:
        folder=ROOT/f'artifacts/{name}_20261002';q=read(folder/'qualification.json');assert 'qualified' in q['status'] or 'passed' in q['status'];check_bindings(q['source_sha256'])
        assert q.get('official_model_calls',q.get('official_heads'))==0 and q.get('fits',q.get('official_fits'))==q['permanent_updates']==0
        files.update(ROOT/p for p in q['source_sha256']);files.update(p for p in folder.rglob('*') if p.is_file());files.add(ROOT/f'artifacts/{name}_original_console_20261002.txt')
    for name in ['v164_independent_actual_short_trajectory_review','v164_independent_capacity_stop_review','v164_independent_decision_floor_counterfactual']:
        folder=ROOT/f'artifacts/{name}_20261002';assert (folder/'review.json').exists();files.update(p for p in folder.rglob('*') if p.is_file());files.add(ROOT/'training'/f'{name}.py')
    for role in range(3):files.add(ROOT/f'artifacts/v164_short_supervised_role{role}_original_console_20261002.txt')
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).is_file():files.add(Path(path).resolve())
    for path in files:
        if path.suffix=='.py' and path.is_relative_to(ROOT/'training'):ast.parse(path.read_text(encoding='utf-8-sig'))
    # No new derivative matrix or trajectory is saved. Three fixed proposals,
    # six scope tables per role, three full checkpoints and direction arrays
    # fit within a conservative 512MiB evidence estimate; reserve 1GiB.
    free=shutil.disk_usage(ROOT).free;assert free>=1024**3,'Retain all zero-fit evidence with at least one GiB free'
    plan=read(BUDGET);constraints=read(ROOT/plan['failure_constraints']);plan['failure_case_actions']={r['id']:r['required_action'] for r in constraints['cases']}
    assert review_plan(plan)['status']=='V165_zero_fit_finite_plan_passed'
    for spec in plan['roles']:
        role=spec['role'];fit=read(entry.PRIOR/f'role{role}/fit.json')
        assert fit['permanent_updates']==spec['parameter_point'] and fit['endpoint_parameter_sha256']==spec['endpoint_parameter_sha256'] and fit['exception'] is None
        entry.cached_treatment(spec,spec['endpoint_parameter_sha256'])
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    plan.update(source_sha256=bindings,allowed_entries=['training/v165_fixed_endpoint_decision_floor_diagnostic.py'],preseal_free_bytes=free,estimated_max_new_artifact_bytes=512*1024**2,cached_vectors_recomputed_only_without_new_official_derivatives=True,actual_checkpoints_saved_after_final_restoration=True,all_failed_candidates_and_true_argmax_protected_rows_recorded=True,supervised_development_not_independent_source_validation=True)
    entry.OUT.mkdir();entry.save(entry.OUT/'pre_registration_bindings.json',dict(source_sha256=bindings,official_calls=0));entry.save(entry.PLAN,plan);bindings[entry.PLAN.relative_to(ROOT).as_posix()]=sha(entry.PLAN)
    entry.save(entry.OUT/'run_seal.json',dict(status='V165_sealed_before_actual_finite_diagnostic',protocol=entry.PROTOCOL,entry_path='training/v165_fixed_endpoint_decision_floor_diagnostic.py',plan_path=entry.PLAN.relative_to(ROOT).as_posix(),plan_sha256=sha(entry.PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings))
    entry.require();entry.save(entry.OUT/'registration.json',dict(status='V165_zero_fit_decision_floor_diagnostic_registered',physical_sources=len(bindings),new_caps=plan['new_caps'],prior_actual_costs=plan['prior_actual_costs'],future_cumulative_actual_caps=plan['future_cumulative_actual_caps'],official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,run_seal_sha256=sha(entry.OUT/'run_seal.json')))
    print(json.dumps(dict(status='V165_zero_fit_decision_floor_diagnostic_sealed',physical_sources=len(bindings),new_caps=plan['new_caps'])))

if __name__=='__main__':main()
