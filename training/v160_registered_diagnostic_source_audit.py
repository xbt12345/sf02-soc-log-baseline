"""Read-only postseal audit; no model, feature or gradient execution."""
import ast,json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v160_registered_diagnostic_source_audit_20261002'

def main():
    assert not OUT.exists();OUT.mkdir()
    trial=ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002';plan_path=ROOT/'training/review_policy/v160_fixed_endpoint_diagnostic_contract.json';plan=read(plan_path);seal=read(trial/'run_seal.json')
    assert sha(plan_path)==seal['plan_sha256'];check_bindings(seal['source_sha256'])
    costs=read(ROOT/'artifacts/v159_complete_result_records_20261002/actual_result_summary.json')['costs']
    assert costs['cumulative_actual_heads']==plan['prior_actual_V159_lineage']['heads']==14586 and costs['cumulative_actual_full_class_gradients']==plan['prior_actual_V159_lineage']['class_gradients']==362
    assert len(plan['roles'])==3 and [r['head_cap'] for r in plan['roles']]==[1444,1060,1444]
    for r in plan['roles']:
        k=r['OOF_chunks'];assert r['head_cap']==2*k+12+k+12+60*(k+12)+48+k+12
        assert r['class_gradient_cap']==2 and r['margin_gradient_cap']==48 and r['finite_proposal_cap']==61 and r['QP_solve_cap']==3
    assert plan['new_caps']['fits']==plan['new_caps']['permanent_updates']==0
    entry=ROOT/plan['activation_entries'][0];source=entry.read_text(encoding='utf-8');tree=ast.parse(source)
    prohibited=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
            if node.func.value.id in ['torch','optimizer'] and node.func.attr in ['save','step']:prohibited.append(node.lineno)
    assert not prohibited
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    probe=funcs['probe'];t=next(n for n in ast.walk(probe) if isinstance(n,ast.Try));assert t.finalbody and 'restore(model,base)' in ast.unparse(t.finalbody[0])
    assert "probabilities(model,ctx,'deployment',np.arange(22546),True)" in source and 'joint_check(ctx,qd,ld)' in source and "os_['protected_regressions']==0" in source
    assert "for round_number in range(3)" in source and 'for backtrack in range(20)' in source and 'len(normal_records)+len(pending)>24' in source
    qpaths=[ROOT/'artifacts/v160_active_direction_numeric_qualification_v5_20261002/qualification.json',ROOT/'artifacts/v160_margin_normal_synthetic_qualification_20261002/qualification.json',ROOT/'artifacts/v160_diagnostic_entry_synthetic_qualification_v3_20261002/qualification.json']
    for p in qpaths:
        q=read(p);check_bindings(q['source_sha256']);assert 'passed' in q['status']
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),entry,plan_path,trial/'run_seal.json',trial/'registration.json',*qpaths]}
    assert not any(trial.glob('role*'))
    result=dict(status='registered_source_and_exact_cost_restoration_guard_audit_passed_zero_official_calls',source_sha256=bindings,sealed_physical_sources=len(seal['source_sha256']),exact_per_role_head_caps=[1444,1060,1444],cumulative_V159_actual_costs_verified=True,full_deployment_not_only_role_ids=True,original_OOF_pure_correct_and_joint_deployment_guard=True,probe_finally_restores=True,no_model_save_or_optimizer_step=True,synthetic_qualifications_verified=True,official_heads=0,official_features=0,official_class_gradients=0,official_margin_gradients=0,official_fits=0,permanent_updates=0,quality_acceptance=False,scope='Own source and saved-qualification audit. Does not claim independent actor review, actual finite probe safety, mastery, transfer, or live acceptance.')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],official_calls=0)))

if __name__=='__main__':main()
