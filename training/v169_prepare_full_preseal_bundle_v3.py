"""Full inherited physical dependency and current qualification preparation."""
import ast,importlib.metadata,json,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
import v169_prior_pair_training_entry_v12 as entry
from v169_pair_execution_review_v3 import review_plan

OUT=ROOT/'artifacts/v169_full_preseal_bundle_v3_20261002'
CANDIDATE=ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v2.json'

def module_closure(start):
    pending=list(start);seen=set()
    while pending:
        path=pending.pop().resolve()
        if path in seen:continue
        seen.add(path);tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                candidate=ROOT/'training'/((name.split('.')[0])+'.py')
                if candidate.is_file() and candidate not in seen:pending.append(candidate)
    return seen

def main():
    assert not OUT.exists() and not entry.OUT.exists() and not entry.PLAN.exists()
    plan=read(CANDIDATE);review_plan(plan);assert not plan['execution_authority'] and not plan['new_fit_permission']
    old_path=ROOT/'artifacts/v168_decision_floor_diagnostic_20261002/run_seal.json';old=read(old_path)
    check_bindings(old['source_sha256']);files={ROOT/p for p in old['source_sha256']}|{old_path,Path(__file__).resolve(),CANDIDATE}
    executable=module_closure([Path(entry.__file__),ROOT/'training/v169_pair_execution_review_v3.py',Path(__file__)])
    files.update(executable);files.update(ROOT/p for p in plan['source_sha256']);qualifications=[]
    names=['v169_pair_model_storage_qualification','v169_pair_lifecycle_qualification_v2','v169_saved_state_quality_qualification','v169_working_solver_full_size_qualification','v169_factorized_storage_qualification_v2','v169_measurement_references_qualification','v169_dataframe_and_current_guard_qualification','v169_actual_backend_synthetic_qualification_v3','v169_saved_correction_storage_qualification','v169_execution_contract_candidate_v2']
    for name in names:
        path=ROOT/f'artifacts/{name}_20261002/qualification.json';q=read(path);check_bindings(q['source_sha256']);assert all(q[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates'])
        qualifications.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),status=q['status']));files.update(ROOT/p for p in q['source_sha256']);files.update(p for p in path.parent.rglob('*') if p.is_file())
    for name in ['v169_initial_current_correct_protection','v169_learnable_prior_pair_plan','v169_factorized_resource_budget_v5','v168_decision_floor_diagnostic','v167_trial_point_restoration_diagnostic','v164_short_supervised_trajectory']:
        directory=ROOT/f'artifacts/{name}_20261002';assert directory.is_dir();files.update(p for p in directory.rglob('*') if p.is_file())
    for path in (ROOT/'training').glob('v169*.py'):files.add(path)
    for path in (ROOT/'artifacts').glob('v169*original_console_20261002.txt'):files.add(path)
    for directory in (ROOT/'artifacts').glob('v169_root*20261002'):
        files.update(p for p in directory.rglob('*') if p.is_file())
    for path in (ROOT/'training').glob('v169_root*.py'):files.add(path)
    # Keep all failed, replaced and accepted qualification artifacts immutable.
    failed=ROOT/'artifacts/v169_factorized_storage_qualification_20261002';files.update(p for p in failed.rglob('*') if p.is_file())
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value and Path(value).is_file():files.add(Path(value).resolve())
    files.discard(ROOT/'artifacts/v169_prepare_full_preseal_bundle_v3_original_console_20261002.txt')
    forbidden=[ROOT/'README.md',ROOT/'HANDOFF.md',ROOT/'mcp_readonly/catalog.json',ROOT/'mcp_readonly/tests/test_readonly_mcp.py']
    assert not any(p in files for p in forbidden)
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    assert all(bindings[k]==v for k,v in old['source_sha256'].items())
    report=dict(status='V169_final_v12_full_dependency_and_qualification_bundle_ready_for_independent_preseal_review',entry_path=Path(entry.__file__).resolve().relative_to(ROOT).as_posix(),entry_sha256=sha(Path(entry.__file__)),candidate_contract_path=CANDIDATE.relative_to(ROOT).as_posix(),candidate_contract_sha256=sha(CANDIDATE),qualifications=qualifications,resource_review_path=plan['resource_review_path'],resources=plan['resources'],applicable_failure_registry_chain=plan['applicable_failure_registry_chain'],new_caps=plan['new_caps'],prior_actual_costs=plan['prior_actual_costs'],future_cumulative_caps=plan['future_cumulative_caps'],AST_recursive_training_module_paths=[p.relative_to(ROOT).as_posix() for p in sorted(executable)],inherited_V168_physical_bindings_preserved=len(old['source_sha256']),physical_dependency_files=len(bindings),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},no_mutable_publication_runtime_dependency=True,actual_official_zero_step_and_full_runtime_storage_support_still_required=True,RAM_workspace_estimate_and_whole_storage_accounting_require_independent_review=True,supports_physical_seal=False,execution_authority=False,new_fit_permission=False,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,source_sha256=bindings)
    OUT.mkdir();(OUT/'bundle.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],physical_sources=len(bindings),recursive_training_modules=len(executable),qualifications=len(qualifications),official_calls=0)))

if __name__=='__main__':main()
