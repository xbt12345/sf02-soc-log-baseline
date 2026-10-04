"""Bind actual numerical qualifications and unchanged fit caps before execution."""
import ast,copy,json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from experiment_review_v159_v3 import PROTOCOL,review_plan
from v159_boundary_runtime_v4 import PLAN,save,review
from v159_float64_repeat_policy_v2 import repeat_gradient,repeat_direction
OUT=ROOT/'artifacts/v159_numeric_execution_contract_preparation_20261002'
def main():
    assert not OUT.exists() and not PLAN.exists();OUT.mkdir()
    previous=ROOT/'training/review_policy/v159_boundary_execution_contract_v3.json';p=copy.deepcopy(read(previous))
    p.update(protocol=PROTOCOL,activation_entries=['training/v159_boundary_train_v4.py','training/v159_boundary_evaluate_v4.py'],experiment_review_adapter='training/experiment_review_v159_v3.py',previous_execution_contract=previous.relative_to(ROOT).as_posix())
    p['historical_technical_cost']=dict(classifier_forward_chunks=124,opinion_feature_blocks=124,full_class_gradients=8,fits=0,updates=0,original_failed_preflight_head=84,original_failed_preflight_gradients=4,separate_diagnostic_head=40,separate_diagnostic_gradients=4,evidence='artifacts/v159_gradient_diagnostic_saved_vector_review_v2_20261002/review.json')
    p['total_cumulative_caps']=copy.deepcopy(p['total_future_caps']);p['total_cumulative_caps'].update(classifier_forward_chunks=78196,opinion_feature_blocks=78196,full_class_gradients=2420,preflight_and_diagnostic_classifier_forward_chunks=460,preflight_and_diagnostic_full_class_gradients=20)
    p['numeric_repeat_policy']=dict(module='training/v159_float64_repeat_policy_v2.py',repeat_eps=8,step_eps=16,argmax_exact=True,gradient_absolute_floor=0,Armijo_relaxation=False,per_parameter_segment_gradient_scale_and_relative_L2=True,resolved_direction_or_stop=True,strict_finite_drop_and_Armijo_slack_above_resolution=True,operational_empirical_scale_not_universal_theoretical_error_bound=True)
    p['technical_budget_revision']=dict(prior_total_head=78072,prior_total_gradients=2412,new_cumulative_head=78196,new_cumulative_gradients=2420,extra_head=124,extra_gradients=8,reason='actual saved repeated gradient failure and bounded diagnostic; fresh full preflight required after qualified fixed policy',all_prior_attempts_conserved=True,training_fit_gradient_proposal_update_caps_unchanged=True,no_new_candidate_or_seed=True)
    qualified=['artifacts/v159_nonzero_real_dimension_numeric_qualification_v2_20261002/qualification.json','artifacts/v159_numeric_batch_replay_qualification_v2_20261002/qualification.json','artifacts/v159_independent_numeric_policy_saved_state_audit_20261002/audit.json','artifacts/v159_protocol_synthetic_qualification_v3_20261002/qualification.json']
    for name in qualified:
        q=read(ROOT/name);check_bindings(q['source_sha256'])
    numeric=read(ROOT/qualified[0]);assert all(numeric['fixtures'].values())
    batch=read(ROOT/qualified[1]);assert batch['synthetic_real_dimension_finite_B_probe']['accepted']
    protocol=read(ROOT/qualified[3]);assert all(protocol['cases'].values()) and len(protocol['cases'])==20
    paths=[Path(__file__).resolve(),previous]+[ROOT/k for k in p['source_sha256']]+[ROOT/name for name in qualified]+[ROOT/name for name in ['training/experiment_review_v159_v3.py','training/v159_boundary_runtime_v4.py','training/v159_boundary_train_v4.py','training/v159_boundary_evaluate_v4.py','training/v159_float64_repeat_policy_v2.py','training/v159_build_numeric_execution_sources.py','training/v159_build_numeric_protocol_qualification.py','training/v159_protocol_synthetic_qualification_v3.py','training/v159_nonzero_real_dimension_numeric_qualification_v2.py','training/v159_numeric_batch_replay_qualification_v2.py','docs/V159_FIXED_NUMERIC_POLICY_AND_TECHNICAL_BUDGET_CANDIDATE.md','artifacts/v159_gradient_diagnostic_saved_vector_review_v2_20261002/review.json','artifacts/v159_independent_gradient_repeat_diagnostic_audit_20261002/audit.json','artifacts/v159_class_boundary_trial_20261002/run_seal.json','artifacts/v159_class_boundary_trial_20261002/initial.pt','artifacts/v159_bound_gradient_repeat_diagnostic_20261002/run_seal.json']]
    parsed={}
    for path in set(paths):
        if path.suffix=='.py':parsed[path.name]=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    unchanged=['Counter','context','inputs','model_for','probabilities','risk','stats','rows','record','assign','restore']
    def functions(name):return {n.name:ast.dump(n,include_attributes=False) for n in parsed[name].body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before,after=functions('v159_boundary_train_v3.py'),functions('v159_boundary_train_v4.py');assert all(before[n]==after[n] for n in unchanged)
    p['source_sha256']={path.relative_to(ROOT).as_posix():sha(path) for path in sorted(set(paths))}
    save(OUT/'pre_contract_bindings.json',dict(status='before_new_official_registration',source_sha256=p['source_sha256']))
    result=review_plan(p);assert result['plan_review_passed'];save(PLAN,p);review()
    save(OUT/'qualification.json',dict(status='new_numeric_v4_entries_bound_with_real_dimension_and_saved_state_qualification_no_new_official_calls',protocol=PROTOCOL,contract_path=PLAN.relative_to(ROOT).as_posix(),contract_sha256=sha(PLAN),review=result,unchanged_actual_training_functions=unchanged,protocol_cases=20,total_future_caps=p['total_future_caps'],total_cumulative_caps=p['total_cumulative_caps'],historical_technical_cost=p['historical_technical_cost'],new_official_heads=0,new_official_gradients=0,new_official_fits=0,new_official_updates=0,quality_acceptance=False,goal_status='active',source_sha256=p['source_sha256']))
    print(json.dumps(dict(status='numeric_v4_contract_bound_before_official_calls',future=p['total_future_caps'],cumulative=p['total_cumulative_caps'],review=result),ensure_ascii=False))
if __name__=='__main__':main()
