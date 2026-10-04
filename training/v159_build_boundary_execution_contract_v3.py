"""Prospective full actual OOF window replay budget; no extra training."""
import ast,copy,json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from experiment_review_v159_v2 import PROTOCOL,review_plan
from v159_boundary_runtime_v3 import PLAN,save,review

OUT=ROOT/'artifacts/v159_execution_protocol_contract_v3_20261002'

def main():
    assert not PLAN.exists() and not OUT.exists();OUT.mkdir()
    previous=ROOT/'training/review_policy/v159_boundary_execution_contract_v2.json';p=copy.deepcopy(read(previous))
    p.update(protocol=PROTOCOL,activation_entries=['training/v159_boundary_train_v3.py','training/v159_boundary_evaluate_v3.py'],previous_pre_execution_contract=previous.relative_to(ROOT).as_posix(),experiment_review_adapter='training/experiment_review_v159_v2.py')
    for b in p['role_call_budgets']:b['evaluation_classifier_cap_per_arm']=72+6*b['chunks']
    p['total_future_caps'].update(evaluation_classifier_forward_chunks=720,classifier_forward_chunks=78072,opinion_feature_blocks=78072)
    p['forward_count_derivation']['evaluation']='six actual endpoint/window full deploy72 +6K actual endpoint/last5 complete OOF classifier replay, no gradients'
    p['pre_registration_budget_refinement']=dict(reason='saved OOF window rows alone did not prove actual window model output; add5K per role/arm before any official call',extra_evaluation_chunks=240,previous_total_head_chunks=77832,new_total_head_chunks=78072,training_fit_and_gradient_caps_unchanged=True,all_last5_actual_OOF_replay=True)
    paths=[Path(__file__).resolve(),previous]+[ROOT/k for k in p['source_sha256']]+[ROOT/s for s in [
        'training/experiment_review_v159_v2.py','training/v159_boundary_runtime_v3.py','training/v159_boundary_train_v3.py','training/v159_boundary_evaluate_v3.py','training/v159_protocol_synthetic_qualification_v2.py',
        'artifacts/v159_protocol_synthetic_qualification_v2_20261002/qualification.json','docs/V159_FULL_OOF_WINDOW_EXECUTION_SUPPLEMENT.md','docs/V159_LAST_ROUND_DECISION_AND_TARGETED_PLAN_20261002.md']]
    for path in paths:
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    p['source_sha256']={q.relative_to(ROOT).as_posix():sha(q) for q in sorted(set(paths))}
    save(OUT/'pre_contract_bindings.json',dict(status='full_actual_OOF_window_protocol_sources_bound_before_official_calls',source_sha256=p['source_sha256']))
    check_bindings(read(ROOT/'artifacts/v159_protocol_synthetic_qualification_v2_20261002/qualification.json')['source_sha256'])
    r=review_plan(p);assert r['plan_review_passed'];save(PLAN,p);review()
    save(OUT/'qualification.json',dict(status='full_OOF_window_actual_replay_v3_entries_and_78072_caps_bound_no_official_execution',protocol=PROTOCOL,
        contract_path=PLAN.relative_to(ROOT).as_posix(),contract_sha256=sha(PLAN),candidate_module=p['candidate_module'],review=r,
        official_classifier_calls=0,official_features_calls=0,official_gradients=0,official_fits=0,official_updates=0,official_runtime_sealed=False,official_zero_step_replay=False,quality_acceptance=False,total_future_caps=p['total_future_caps'],source_sha256={Path(__file__).resolve().relative_to(ROOT).as_posix():sha(__file__),PLAN.relative_to(ROOT).as_posix():sha(PLAN)}))
    print(json.dumps(dict(status='full_actual_OOF_window_contract_bound_no_official_calls',entries=p['activation_entries'],caps=p['total_future_caps'],official_calls=0),ensure_ascii=False))

if __name__=='__main__':main()
