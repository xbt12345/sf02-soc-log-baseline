"""Preserve complete v5 bindings; add resource-only v14 and actual qualifications."""
import ast,json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v169_pair_execution_review_v5 import review_plan

OUT=ROOT/'artifacts/v169_full_preseal_bundle_v6_20261002'
OLD=ROOT/'artifacts/v169_full_preseal_bundle_v5_20261002/bundle.json'
ENTRY=ROOT/'training/v169_prior_pair_training_entry_v14.py'
SEALER=ROOT/'training/v169_seal_prior_pair_training_v5.py'
CANDIDATE=ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v4.json'

def closure(starts):
    seen=set();pending=list(starts)
    while pending:
        path=pending.pop().resolve()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                target=ROOT/'training'/(name.split('.')[0]+'.py')
                if target.is_file() and target.resolve() not in seen:pending.append(target)
    return seen

def main():
    assert not OUT.exists() and not (ROOT/'artifacts/v169_prior_pair_training').exists() and not (ROOT/'training/review_policy/v169_prior_pair_execution_contract.json').exists()
    old=read(OLD);check_bindings(old['source_sha256'])
    plan=read(CANDIDATE);review_plan(plan);assert not plan['execution_authority'] and not plan['new_fit_permission']
    bindings=dict(old['source_sha256']);added={OLD,ENTRY,SEALER,CANDIDATE,Path(__file__).resolve()}
    modules=closure([ENTRY,SEALER,ROOT/'training/v169_pair_execution_review_v3.py',Path(__file__).resolve()]);added.update(modules)
    qualifications=list(old['qualifications'])
    new_qualifications=[ROOT/f'artifacts/{name}_20261002/qualification.json' for name in ['v169_execution_contract_candidate_v4','v169_resource_policy_v5_qualification','v169_actual_backend_synthetic_qualification_v5']]+[ROOT/f'artifacts/v169_full_live_new_solver_resource_qualification_v2_20261002/arm{a}/qualification.json' for a in ['A','B']]
    for path in new_qualifications:
        q=read(path);check_bindings(q['source_sha256'])
        assert all(q[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates']) and not q['execution_authority']
        qualifications.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),status=q['status']))
        added.update(ROOT/p for p in q['source_sha256']);added.update(p for p in path.parent.rglob('*') if p.is_file())
    added.update(ROOT/p for p in plan['source_sha256'])
    root_path=ROOT/'artifacts/v169_root_full_live_resource_review_20261002/review.json';root=read(root_path);check_bindings(root['source_sha256'])
    assert root['all_checks_passed'] and root['supports_prospective_resource_only_revision'] and root['resources']==plan['resources']
    added.add(root_path);added.update(ROOT/p for p in root['source_sha256'])
    for name in ['v169_full_live_new_solver_resource_qualification','v169_full_live_new_solver_resource_qualification_v2','v169_factorized_resource_budget_v8','v169_factorized_resource_budget_v9','v169_factorized_resource_budget_v10','v169_root_full_matrix_page_touch_review','v169_actual_preseal_resource_shortfall','v169_root_disk_headroom_compression','v169_disk_headroom_current_status']:
        directory=ROOT/f'artifacts/{name}_20261002'
        assert directory.is_dir();added.update(p for p in directory.rglob('*') if p.is_file())
    for pattern in ['v169_full_live*original_console_20261002.txt','v169_prepare_full_live_resource_candidate*original_console_20261002.txt','v169_resource_policy_v5_qualification_original_console_20261002.txt','v169_actual_backend_synthetic_qualification_v5_original_console_20261002.txt','v169_prepare_resource_policy_upgrade_original_console_20261002.txt','v169_seal_prior_pair_training_v4_original_console_20261002.txt']:
        added.update(ROOT/'artifacts'/p.name for p in (ROOT/'artifacts').glob(pattern))
    # Active independent final reviewer is owned by root and bound by its
    # subsequent actual report, then added to the final physical run seal.
    assert ROOT/'training/v169_root_final_preseal_review_v2.py' not in added
    forbidden=[ROOT/'README.md',ROOT/'HANDOFF.md',ROOT/'mcp_readonly/catalog.json',ROOT/'mcp_readonly/tests/test_readonly_mcp.py']
    assert not any(p in added for p in forbidden)
    for path in sorted(added):
        key=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        value=sha(path)
        if key in bindings:assert bindings[key]==value
        bindings[key]=value
    assert all(bindings[k]==v for k,v in old['source_sha256'].items())
    assert 'training/v169_root_final_preseal_review_v2.py' not in bindings
    report=dict(old)
    report.update(status='V169_v14_resource_only_full_dependency_bundle_v6_ready_for_independent_final_review',entry_path=ENTRY.relative_to(ROOT).as_posix(),entry_sha256=sha(ENTRY),candidate_contract_path=CANDIDATE.relative_to(ROOT).as_posix(),candidate_contract_sha256=sha(CANDIDATE),qualifications=qualifications,resource_review_path=plan['resource_review_path'],resources=plan['resources'],source_sha256=bindings,physical_dependency_files=len(bindings),AST_recursive_training_module_paths=[p.relative_to(ROOT).as_posix() for p in sorted(modules)],previous_immutable_bundle_path=OLD.relative_to(ROOT).as_posix(),previous_immutable_bundle_sha256=sha(OLD),all_previous_v5_bindings_literal_preserved=len(old['source_sha256']),sealer_path=SEALER.relative_to(ROOT).as_posix(),sealer_sha256=sha(SEALER),root_resource_only_review_path=root_path.relative_to(ROOT).as_posix(),root_resource_only_review_sha256=sha(root_path),entry_only_resource_policy_import_changed=True,scientific_contract_unchanged_except_resource_keys=True,original6GiB_failed_preseal_and_BLAS4_confound_preserved=True,active_root_final_reviewer_not_bound_until_its_actual_report=True,supports_physical_seal=False,execution_authority=False,new_fit_permission=False,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0)
    OUT.mkdir();(OUT/'bundle.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],physical_sources=len(bindings),previous_v5_sources_preserved=len(old['source_sha256']),qualifications=len(qualifications),recursive_modules=len(modules),bundle_path=(OUT/'bundle.json').relative_to(ROOT).as_posix(),bundle_sha256=sha(OUT/'bundle.json'),entry_sha256=sha(ENTRY),candidate_sha256=sha(CANDIDATE),sealer_sha256=sha(SEALER),official_calls=0)),flush=True)

if __name__=='__main__':main()
