"""Prospectively seal actual-guard conditioned trial-point restoration."""
import ast,importlib.metadata,json,shutil,sys
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v166_coverage_core_qualification_v3 import memory
from v167_trial_point_execution_review import prospective_plan,review_plan
import v167_trial_point_restoration_diagnostic as entry

def main():
    assert not entry.OUT.exists() and not entry.PLAN.exists();previous=read(entry.CONTROL/'run_seal.json');check_bindings(previous['source_sha256']);actual_path=ROOT/'artifacts/v166_results_20261002/actual_result_summary.json';actual=read(actual_path);check_bindings(actual['source_sha256']);assert actual['cumulative_heads']==18784 and actual['cumulative_all_complete_derivatives']==618 and actual['new_fits']==actual['permanent_updates']==0 and actual['all_restored_full_tensors_exact'];files={ROOT/p for p in previous['source_sha256']}|{ROOT/p for p in actual['source_sha256']}|{Path(__file__).resolve(),actual_path}
    for folder in [entry.CONTROL,entry.PRIOR]:files.update(p for p in folder.rglob('*') if p.is_file())
    qualifications=[]
    for name in ['v167_trial_point_identity_qualification_v2','v167_trial_point_lifecycle_qualification','v167_trial_point_adversarial_qualification']:
        folder=ROOT/f'artifacts/{name}_20261002';q=read(folder/'qualification.json');check_bindings(q['source_sha256']);assert 'qualified' in q['status'] and all(q[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates']);files.update(ROOT/p for p in q['source_sha256']);files.update(p for p in folder.rglob('*') if p.is_file());files.add(ROOT/f'artifacts/{name}_original_console_20261002.txt');qualifications.append(dict(path=(folder/'qualification.json').relative_to(ROOT).as_posix(),sha256=sha(folder/'qualification.json'),status=q['status']))
    counterexample=ROOT/'artifacts/v167_root_trial_point_nonlinear_counterexamples_20261002/review.json';counter=read(counterexample);check_bindings(counter['source_sha256']);assert counter['actual_CPU_synthetic_QP_solves']==3 and all(counter[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates']);assert all(not s['actual_finite_constraint_passed'] and s['nonlinear_violation_ratio']<.99 for s in counter['cases'][1]['stages'])
    for name in ['v166_independent_actual_coverage_review','v166_independent_covered_margin_geometry_review','v166_saved_nonlinear_margin_residual_review','v166_saved_residual_blocker_identity_review','v166_observed_runtime_failure_replay_v2','v167_root_trial_point_nonlinear_counterexamples','v166_direction']:
        folder=ROOT/f'artifacts/{name}_20261002';assert folder.exists();files.update(p for p in folder.rglob('*') if p.is_file())
    # The immutable direction snapshot binds the current published plan. Its
    # previous mutable snapshots are evidence; current README/MCP are not
    # runtime dependencies and must not freeze future honest publications.
    files.update([ROOT/'training/v167_independent_actual_trial_point_review.py',ROOT/'training/v167_root_trial_point_nonlinear_counterexamples.py',ROOT/'training/review_policy/v167_trial_point_restoration_draft.json',ROOT/'docs/V166_ROOT_ACTUAL_REVIEW_AND_V167_TRIAL_POINT_RESTORATION_PLAN_20261002.md',ROOT/'training/review_policy/v166_observed_runtime_boundaries_v2.json',ROOT/'artifacts/v167_prospective_direction_MCP_tests_original_console_20261002.txt'])
    test_log=(ROOT/'artifacts/v167_prospective_direction_MCP_tests_original_console_20261002.txt').read_text(encoding='utf-8');assert 'Ran 19 tests' in test_log and test_log.rstrip().endswith('OK');assert (ROOT/'training/v167_independent_actual_trial_point_review.py').is_file()
    executable={Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        source=getattr(module,'__file__',None)
        if source and Path(source).is_file():
            path=Path(source).resolve();files.add(path)
            if path.suffix=='.py' and path.is_relative_to(ROOT/'training'):executable.add(path)
    for path in executable:ast.parse(path.read_text(encoding='utf-8-sig'))
    mem=memory();free=shutil.disk_usage(ROOT).free;assert free>=4*1024**3 and mem['physical_available_bytes']>=3*1024**3;gpu_free,gpu_total=torch.cuda.mem_get_info();assert gpu_free>=512*1024**2
    plan=prospective_plan();review_plan(plan)
    for spec in plan['roles']:
        state=torch.load(entry.PRIOR/f"role{spec['role']}/endpoint.pt",map_location='cpu',weights_only=True)['state'];records,gradients,u,proof=entry.cached_origin(spec,state);assert proof['accepted']==(spec['QP_cap']==0)
        if not proof['accepted']:
            ctx=entry.load_context(spec['role']);blockers=__import__('pandas').read_parquet(entry.CONTROL/f"role{spec['role']}/treatment/actual_blocking_original_rows.parquet");assert entry.blocker_identities(ctx,blockers)<=set(records)
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)};plan.update(status='qualified_prospectively_registered_V167_trial_point_finite_diagnostic',execution_authority=True,allowed_entries=['training/v167_trial_point_restoration_diagnostic.py'],source_sha256=bindings,qualifications=qualifications,additional_synthetic_counterexample=counterexample.relative_to(ROOT).as_posix(),additional_synthetic_counterexample_required_action=counter['required_action'],synthetic_counterexample_CPU_QPs=3,preseal_free_bytes=free,preseal_memory=mem,preseal_gpu_free_bytes=gpu_free,preseal_gpu_total_bytes=gpu_total,estimated_max_new_artifact_bytes=2*1024**3,raw_max_fresh_gradient_array_bytes=100*1060832*8,largest_full_joint_matrix_bytes=27*1060832*8,AST_scope='actual_executable_dependency_closure_only',all_old_failed_sources_and_identity_v1_failure_preserved=True,no_mutable_publication_state_used_as_model_runtime_dependency=True)
    entry.OUT.mkdir();entry.save(entry.OUT/'pre_registration_bindings.json',dict(source_sha256=bindings,official_calls=0,AST_checked_executable_paths=[p.relative_to(ROOT).as_posix() for p in sorted(executable)]));entry.save(entry.PLAN,plan);bindings[entry.PLAN.relative_to(ROOT).as_posix()]=sha(entry.PLAN);entry.save(entry.OUT/'run_seal.json',dict(status='V167_sealed_before_actual_trial_point_diagnostic',protocol=entry.PROTOCOL,entry_path='training/v167_trial_point_restoration_diagnostic.py',plan_path=entry.PLAN.relative_to(ROOT).as_posix(),plan_sha256=sha(entry.PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings));entry.require();entry.save(entry.OUT/'registration.json',dict(status='V167_trial_point_diagnostic_registered_before_official_calls',physical_sources=len(bindings),new_caps=plan['new_caps'],prior_actual_costs=plan['prior_actual_costs'],future_cumulative_caps=plan['future_cumulative_caps'],official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,run_seal_sha256=sha(entry.OUT/'run_seal.json')));print(json.dumps(dict(status='V167_trial_point_diagnostic_physically_sealed',physical_sources=len(bindings),new_caps=plan['new_caps'])))

if __name__=='__main__':main()
