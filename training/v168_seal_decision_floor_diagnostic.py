"""Physical pre-registration only after qualified independent preseal review."""
import argparse,ast,importlib.metadata,json,shutil,sys
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v166_coverage_core_qualification_v3 import memory
from v168_decision_floor_execution_review import prospective_plan,review_plan
import v168_decision_floor_diagnostic_v2 as entry

def main(root_review_path):
    assert not entry.OUT.exists() and not entry.PLAN.exists()
    root_review_path=Path(root_review_path).resolve(strict=True);assert root_review_path.is_relative_to(ROOT/'artifacts')
    root_review=read(root_review_path);assert root_review['supports_physical_seal'] is True and root_review['execution_authority'] is False
    assert root_review['official_heads']==root_review['official_features']==root_review['official_derivatives']==root_review['fits']==root_review['permanent_updates']==0
    check_bindings(root_review['source_sha256']);assert root_review['source_sha256']['training/v168_decision_floor_diagnostic_v2.py']==sha(Path(entry.__file__).resolve())
    previous=read(entry.OLD/'run_seal.json');check_bindings(previous['source_sha256']);bundle_path=ROOT/'artifacts/v168_decision_floor_qualification_bundle_v2_20261002/bundle.json';bundle=read(bundle_path);check_bindings(bundle['source_sha256']);assert bundle['status']=='V168_entry_and_four_qualifications_ready_for_independent_preseal_root_review'
    files={ROOT/p for p in previous['source_sha256']}|{ROOT/p for p in bundle['source_sha256']}|{Path(__file__).resolve(),bundle_path,root_review_path,ROOT/'training/v168_independent_actual_decision_floor_review.py'};files.update(ROOT/p for p in root_review['source_sha256'])
    actual=ROOT/'artifacts/v167_results_20261002/actual_result_summary.json';summary=read(actual);check_bindings(summary['source_sha256']);assert summary['cumulative_heads']==19080 and summary['cumulative_all_complete_derivatives']==718 and summary['all_restored_full_tensors_exact'] and summary['latest_actual_training']=='V164' and summary['latest_complete_quality_delivery']=='V159';files.add(actual);files.update(ROOT/p for p in summary['source_sha256'])
    for folder in [entry.OLD,entry.PRIOR,ROOT/'artifacts/v167_direction_20261002',ROOT/'artifacts/v168_decision_floor_qualification_bundle_20261002']:
        files.update(p for p in folder.rglob('*') if p.is_file())
    for item in bundle['qualifications']:
        qpath=ROOT/item['path'];q=read(qpath);assert sha(qpath)==item['sha256'];check_bindings(q['source_sha256']);assert all(q[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates']);files.update(ROOT/p for p in q['source_sha256']);files.update(p for p in qpath.parent.rglob('*') if p.is_file())
    replay=ROOT/'artifacts/v167_observed_runtime_failure_replay_20261002/replay.json';check_bindings(read(replay)['source_sha256']);assert len(read(replay)['cases'])==6;files.add(replay);files.update(ROOT/p for p in read(replay)['source_sha256'])
    plan=prospective_plan();review_plan(plan);assert plan['new_caps']==bundle['new_caps'];plan['applicable_failure_registry_chain']=bundle['failure_registry_chain'];assert plan['applicable_failure_registry_chain'][1]['path']=='training/review_policy/v166_observed_runtime_boundaries_v2.json'
    for row in plan['applicable_failure_registry_chain']:assert sha(ROOT/row['path'])==row['sha256'];files.add(ROOT/row['path'])
    plan['actual_failure_constraint_replay']=replay.relative_to(ROOT).as_posix();plan['actual_failure_constraint_replay_sha256']=sha(replay)
    # Honest publication snapshots are evidence; live README/catalog/tests
    # remain mutable, and are never a dependency of model execution.
    current_direction=ROOT/'artifacts/v167_direction_20261002/validation.json';check_bindings(read(current_direction)['source_sha256']);testlog=ROOT/'artifacts/v168_prospective_direction_MCP_tests_original_console_20261002.txt';log=testlog.read_text(encoding='utf-8');assert 'Ran 23 tests' in log and log.rstrip().endswith('OK');files.add(testlog)
    executable={Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        source=getattr(module,'__file__',None)
        if source and Path(source).is_file():
            path=Path(source).resolve();files.add(path)
            if path.suffix=='.py' and path.is_relative_to(ROOT/'training'):executable.add(path)
    for path in executable:ast.parse(path.read_text(encoding='utf-8-sig'))
    mem=memory();free=shutil.disk_usage(ROOT).free;assert free>=4*1024**3 and mem['physical_available_bytes']>=3*1024**3;gpu_free,gpu_total=torch.cuda.mem_get_info();assert gpu_free>=512*1024**2
    state=torch.load(entry.PRIOR/'role1/endpoint.pt',map_location='cpu',weights_only=True)['state'];ctx=entry.load_context(1);records,gradients,u,proof,goal,ties=entry.cached_inputs(plan,state,ctx);assert len(records)==25 and len(goal)==4 and len(ties)==2;safe=entry.safe_references(plan);assert [r['role'] for r in safe]==[0,2]
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    plan.update(status='qualified_prospectively_registered_V168_single_decision_floor_diagnostic',execution_authority=True,allowed_entries=['training/v168_decision_floor_diagnostic_v2.py'],source_sha256=bindings,qualifications=bundle['qualifications'],independent_preseal_review=root_review_path.relative_to(ROOT).as_posix(),independent_preseal_review_sha256=sha(root_review_path),preseal_free_bytes=free,preseal_memory=mem,preseal_gpu_free_bytes=gpu_free,preseal_gpu_total_bytes=gpu_total,estimated_max_new_artifact_bytes=1024**3,AST_scope='actual_executable_dependency_closure_only',all_original_failed_sources_and_logs_preserved=True,no_mutable_publication_used_as_model_runtime_dependency=True)
    entry.OUT.mkdir();entry.save(entry.OUT/'pre_registration_bindings.json',dict(source_sha256=bindings,official_calls=0,AST_checked_executable_paths=[p.relative_to(ROOT).as_posix() for p in sorted(executable)]));entry.save(entry.PLAN,plan);bindings[entry.PLAN.relative_to(ROOT).as_posix()]=sha(entry.PLAN)
    entry.save(entry.OUT/'run_seal.json',dict(status='V168_sealed_before_actual_decision_floor_diagnostic',protocol=entry.PROTOCOL,entry_path='training/v168_decision_floor_diagnostic_v2.py',plan_path=entry.PLAN.relative_to(ROOT).as_posix(),plan_sha256=sha(entry.PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings));entry.require();entry.save(entry.OUT/'registration.json',dict(status='V168_single_role_decision_floor_registered_before_official_calls',physical_sources=len(bindings),new_caps=plan['new_caps'],prior_actual_costs=plan['prior_actual_costs'],future_cumulative_caps=plan['future_cumulative_caps'],applicable_failure_registry_chain=plan['applicable_failure_registry_chain'],explicit_six_V167_actions=plan['failure_case_actions'],official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,run_seal_sha256=sha(entry.OUT/'run_seal.json')));print(json.dumps(dict(status='V168_single_role_decision_floor_physically_sealed',physical_sources=len(bindings),new_caps=plan['new_caps'])))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root-review',required=True);main(parser.parse_args().root_review)
