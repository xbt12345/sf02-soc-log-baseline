"""Register one prospective coverage treatment per fixed endpoint, zero fit."""
import ast,importlib.metadata,json,shutil,sys
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v166_coverage_execution_review import prospective_plan,review_plan
from v166_coverage_core_qualification_v3 import memory
import v166_coverage_first_diagnostic as entry

def main():
    assert not entry.OUT.exists() and not entry.PLAN.exists()
    previous=read(entry.CONTROL/'run_seal.json');check_bindings(previous['source_sha256'])
    actual_path=ROOT/'artifacts/v165_results_20261002/actual_result_summary.json';actual=read(actual_path);check_bindings(actual['source_sha256']);assert actual['cumulative_heads']==18538 and actual['cumulative_all_complete_derivatives']==552 and actual['new_fits']==actual['permanent_updates']==0 and actual['all_restored_full_tensors_exact']
    files={ROOT/p for p in previous['source_sha256']}|{ROOT/p for p in actual['source_sha256']}|{actual_path,Path(__file__).resolve()}
    for folder in [entry.CONTROL,entry.PRIOR]:files.update(p for p in folder.rglob('*') if p.is_file())
    qualifications=[]
    for name in ['v166_coverage_core_qualification_v3','v166_entry_identity_qualification','v166_entry_lifecycle_qualification']:
        folder=ROOT/f'artifacts/{name}_20261002';q=read(folder/'qualification.json');check_bindings(q['source_sha256']);assert 'qualified' in q['status'];assert q['official_heads']==q['official_features']==q['official_derivatives']==q['fits']==q['permanent_updates']==0
        files.update(ROOT/p for p in q['source_sha256']);files.update(p for p in folder.rglob('*') if p.is_file());files.add(ROOT/f'artifacts/{name}_original_console_20261002.txt');qualifications.append(dict(path=(folder/'qualification.json').relative_to(ROOT).as_posix(),sha256=sha(folder/'qualification.json'),status=q['status']))
    for name in ['v165_independent_actual_decision_floor_review','v165_independent_blocker_coverage_review','v166_next_round_decision_evidence_review']:
        folder=ROOT/f'artifacts/{name}_20261002';assert (folder/'review.json').is_file();files.update(p for p in folder.rglob('*') if p.is_file());files.add(ROOT/'training'/f'{name}.py')
    files.update([ROOT/'training/v166_independent_actual_coverage_review.py',ROOT/'docs/V166_NEXT_TRAINING_DECISION_AND_PROGRESSIVE_REVIEW_20261002.md',ROOT/'docs/V165_ACTUAL_RESULTS_AND_COVERAGE_FIRST_NEXT_PLAN_20261002.md',entry.DRAFT,ROOT/'training/review_policy/v165_observed_boundaries.json',ROOT/'training/review_policy/v164_root_failure_constraints.json'])
    executable=set()
    for module in list(sys.modules.values()):
        source=getattr(module,'__file__',None)
        if source and Path(source).is_file():
            path=Path(source).resolve();files.add(path)
            if path.suffix=='.py' and path.is_relative_to(ROOT/'training'):executable.add(path)
    # Archive SyntaxErrors remain hash-bound. Only this actual executable
    # closure is parsed; an archived failed preparation is never imported.
    executable.add(Path(__file__).resolve())
    for path in executable:ast.parse(path.read_text(encoding='utf-8-sig'))
    mem=memory();free=shutil.disk_usage(ROOT).free;assert free>=4*1024**3 and mem['physical_available_bytes']>=3*1024**3
    gpu_free,gpu_total=torch.cuda.mem_get_info();assert gpu_free>=512*1024**2
    plan=prospective_plan();review_plan(plan)
    for spec in plan['roles']:
        fit=read(entry.PRIOR/f"role{spec['role']}/fit.json");assert fit['exception'] is None and fit['permanent_updates']==spec['parameter_point'] and fit['endpoint_parameter_sha256']==spec['endpoint_parameter_sha256'];entry.cached_origin(spec,spec['endpoint_parameter_sha256'])
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    inherited=read(ROOT/'training/review_policy/v164_root_failure_constraints.json')
    plan.update(status='qualified_prospectively_registered_V166_coverage_finite_diagnostic',execution_authority=True,allowed_entries=['training/v166_coverage_first_diagnostic.py'],source_sha256=bindings,qualifications=qualifications,inherited_failure_case_actions={r['id']:r['required_action'] for r in inherited['cases']},preseal_free_bytes=free,preseal_memory=mem,preseal_gpu_free_bytes=gpu_free,preseal_gpu_total_bytes=gpu_total,estimated_max_new_artifact_bytes=2*1024**3,raw_fresh_gradient_arrays_bytes=66*1060832*8,largest_full_joint_matrix_bytes=27*1060832*8,all_failed_preparation_sources_preserved_and_bound=True,AST_scope='actual_executable_dependency_closure_only',complete_original_population_scored_at_each_finite_proposal=True,old_206_mixed_correct_rows_not_retroactively_frozen=True,all_cumulative_new_repairs_protected=True,no_new_fit_or_permanent_update_or_budget_extension=True)
    entry.OUT.mkdir();entry.save(entry.OUT/'pre_registration_bindings.json',dict(source_sha256=bindings,official_calls=0,AST_checked_executable_paths=[p.relative_to(ROOT).as_posix() for p in sorted(executable)]));entry.save(entry.PLAN,plan);bindings[entry.PLAN.relative_to(ROOT).as_posix()]=sha(entry.PLAN)
    entry.save(entry.OUT/'run_seal.json',dict(status='V166_sealed_before_actual_coverage_diagnostic',protocol=entry.PROTOCOL,entry_path='training/v166_coverage_first_diagnostic.py',plan_path=entry.PLAN.relative_to(ROOT).as_posix(),plan_sha256=sha(entry.PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings))
    entry.require();entry.save(entry.OUT/'registration.json',dict(status='V166_coverage_diagnostic_registered_before_official_calls',physical_sources=len(bindings),new_caps=plan['new_caps'],prior_actual_costs=plan['prior_actual_costs'],future_cumulative_caps=plan['future_cumulative_caps'],official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,run_seal_sha256=sha(entry.OUT/'run_seal.json')));print(json.dumps(dict(status='V166_coverage_diagnostic_physically_sealed',physical_sources=len(bindings),new_caps=plan['new_caps'])))

if __name__=='__main__':main()
