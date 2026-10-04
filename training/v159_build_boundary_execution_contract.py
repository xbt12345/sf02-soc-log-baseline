"""Bind real entry identity and conflict-qualified retention before activation."""
from pathlib import Path
import ast,copy,json
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_runtime import PLAN,review,save

OUT=ROOT/'artifacts/v159_execution_contract_preparation_20261002'

def main():
    assert not PLAN.exists() and not OUT.exists();OUT.mkdir()
    qualification=ROOT/'training/review_policy/v159_candidate_qualification_contract.json';p=copy.deepcopy(read(qualification))
    p.update(version='V159-six-fixed-current-input-class-boundary-execution',candidate_module='training/v159_current_input_boundary_v3.py',instrumentation_only_refinement_of_v2=True,
        activation_entries=['training/v159_boundary_train.py','training/v159_boundary_evaluate.py'],formal_runtime_ready=False,new_training_entry_registered=False,
        allowed_activation_entries=[],preflight_required=True,official_runtime_sealed=False,quality_acceptance=False)
    p['solver']['B_finite_guard']='both actual stable class risks meet Armijo and initially correct current-X-pure OOF rows retained'
    p['OOF_retention_refinement']=dict(before_any_official_fit=True,evidence='artifacts/v159_joint_input_retention_feasibility_audit_20261002/audit.json',
        old_all_initial_correct_forced_floors=[22,72,110],complete_visible_input_floors=[22,4,26],current_X_floors=[22,6,28],
        selected_protected_OOF='initially correct original rows in pure current complete X input groups',all_mixed_rows_still_in_loss_and_score=True,
        all_accepted_deployment_correct_and_joint_scopes_unchanged=True,mastery='all current-X-pure original errors0, protected pure OOF regressions0, mixed total <=current-X22/6/28',
        tuple_float_splits_are_not_new_behavior_evidence=True)
    p['total_future_caps']['opinion_feature_blocks']=p['total_future_caps']['classifier_forward_chunks']
    p['total_future_caps']['new_base_classifier_forward_chunks']=0;p['total_future_caps']['new_base_feature_blocks']=0
    p['task_quality']=read(ROOT/'training/review_policy/v137_single_issue_plan.json')['task_adoption_quality']
    p['risk_actions']={r['id']:r['required_action'] for r in read(ROOT/'training/review_policy/history_cases.json')['risks']}
    paths=[Path(__file__).resolve(),qualification]+[ROOT/s for s in [
        'training/v159_boundary_runtime.py','training/v159_prepare_boundary_inputs.py','training/v159_boundary_train.py','training/v159_boundary_evaluate.py',
        'training/v159_current_input_boundary_v3.py','training/v159_current_input_boundary_v2.py','training/v159_current_input_boundary.py','training/v159_class_direction.py',
        'training/v159_mgda_synthetic_qualification.py','training/v159_boundary_torch_synthetic_qualification_v4.py','training/v159_boundary_chunk_gradient_qualification.py',
        'artifacts/v159_boundary_torch_synthetic_qualification_v4_20261001/qualification.json','artifacts/v159_boundary_chunk_gradient_qualification_20261002/qualification.json',
        'artifacts/v159_joint_input_retention_feasibility_audit_20261002/audit.json','training/v159_joint_input_retention_feasibility_audit.py',
        'docs/V159_CONFLICT_RETENTION_AND_EXECUTION_SUPPLEMENT.md','training/review_policy/history_cases.json','training/review_policy/v137_single_issue_plan.json']]
    # Reuse every current label/input/fold/control/joint-guard physical source.
    old=read(ROOT/'training/review_policy/v158_fusion_execution_contract.json')['source_sha256']
    paths.extend(ROOT/k for k in old)
    # Include direct generic loss/MGDA helper, qualification sources and history.
    paths.extend(ROOT/k for k in read(qualification)['source_sha256'])
    paths.extend(ROOT/k for k in read(ROOT/'artifacts/v159_boundary_chunk_gradient_qualification_20261002/qualification.json')['source_sha256'])
    for file in paths:
        if file.suffix=='.py':ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
    p['source_sha256']={q.relative_to(ROOT).as_posix():sha(q) for q in sorted(set(paths))}
    save(OUT/'pre_contract_bindings.json',dict(status='all_real_entries_and_conflict_evidence_bound_before_official_calls',source_sha256=p['source_sha256']))
    # Tested entry source must match its actual prior synthetic receipt.
    check_bindings(read(ROOT/'artifacts/v159_boundary_chunk_gradient_qualification_20261002/qualification.json')['source_sha256'])
    save(PLAN,p);review()
    save(OUT/'qualification.json',dict(status='real_counted_entry_sources_contract_bound_pending_independent_review_input_prepare_and_real_seal',
        plan_sha256=sha(PLAN),contract_path=PLAN.relative_to(ROOT).as_posix(),candidate_module=p['candidate_module'],official_classifier_calls=0,official_features_calls=0,official_gradients=0,official_fits=0,official_updates=0,
        new_training_entry_registered=False,official_runtime_sealed=False,official_zero_step_replay=False,quality_acceptance=False,
        total_future_caps=p['total_future_caps'],source_sha256={Path(__file__).resolve().relative_to(ROOT).as_posix():sha(__file__),PLAN.relative_to(ROOT).as_posix():sha(PLAN)}))
    print(json.dumps(dict(status='formal_entry_contract_bound_not_executed',candidate=p['candidate_module'],future_caps=p['total_future_caps'],official_calls=0),ensure_ascii=False))

if __name__=='__main__':main()
