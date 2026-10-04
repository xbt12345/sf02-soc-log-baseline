"""Concrete false-authority contract for independent preseal review."""
import copy,json
from pathlib import Path
from experiment_review import ROOT,read,sha
from v169_pair_execution_review_v3 import review_plan,registry_chain

PLAN=ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v1.json'
OUT=ROOT/'artifacts/v169_execution_contract_candidate_20261002'

def main():
    assert not PLAN.exists() and not OUT.exists();budget_path=ROOT/'artifacts/v169_factorized_resource_budget_v4_20261002/review.json';budget=read(budget_path)
    old=read(ROOT/'artifacts/v169_saved_budget_scope_and_callgraph_v2_20261002/review.json')
    chain=registry_chain();source=[Path(__file__).resolve(),budget_path,ROOT/'training/v169_prior_pair_training_entry_v11.py',ROOT/'training/v169_pair_execution_review.py',ROOT/'training/v169_pair_execution_review_v2.py',ROOT/'training/v169_pair_execution_review_v3.py',ROOT/'docs/EXPERIMENT_REVIEW_RULES.md',ROOT/'training/review_policy/history_cases.json',ROOT/'docs/V169_INCREMENTAL_RESEARCH_AND_TRAINING_REVIEW_20261002.md',ROOT/'docs/V169_PRETRAINING_INDEPENDENT_FINDINGS_20261002.md',ROOT/'training/review_policy/v169_learnable_prior_pair_draft.json']+[ROOT/r['path'] for r in chain]
    plan=dict(status='V169_concrete_execution_contract_candidate_not_authority',protocol='V169-three-role-paired-learnable-prior-bounded-training-v1',execution_authority=False,new_fit_permission=False,entry='training/v169_prior_pair_training_entry_v11.py',schedule=dict(accepted_updates=20,corrections=2,working_functions=64,backtracks=8),complete_parameter_widths=dict(A=1060832,B=1060833),objective_policy='strict_individual_M_S_actual_16eps_Armijo_and_full_original_row_protection',roles=[0,1,2],arms=['A','B'],candidate='B',same_safe_initialization=True,bootstrap_in20=True,all_original_frequency_and_denominators_preserved=True,permanent_full_guard_ledger=True,full_guard_each_candidate=True,fresh_derivatives_at_actual_parameter_point=True,no_outer_selection=True,no_new_confirmations_or_model_promotion=True,unmatched_terminal_result='inconclusive',low_prior_slice_no_repair_cannot_claim_deep_error_solved=True,prior_actual_costs=old['historical_categories'],inherited_observed_actions=chain[0]['actions'],applicable_failure_registry_chain=chain,role_caps=old['per_fit_role_caps'],new_caps=old['new_prepared_entry_caps'],future_cumulative_caps=old['combined_future_if_executed'],full_task_quality_acceptance=False,model_promoted=False,resource_review_path=budget_path.relative_to(ROOT).as_posix(),resources=budget['resources'],batch_global_fault_stops_remaining_fits=True,same_point_final_replay_required=True,training_issue_mastery=dict(all_roles_pure_errors_zero=True,last_five_distinct_accepted_states_mastered=True,all_registered_old_and_accepted_repairs_preserved=True),B_support_requires=dict(fixed20_matched_all_three_roles=True,no_more_M_S_errors_each_role=True,extra_S_repair_each_role=True,all_initial_low_prior_S_slices_reported=True,no_low_prior_repair_means_deep_problem_unresolved=True),full_N_M_S_quality_and_unseen_source_generalization_separate=True,viewed_roles_are_development_not_blind=True,zero_step_is_registered_targets_point0_before_bootstrap_no_extra_heads=True,no_beta_clipping_or_exponential_fallback=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source})
    review=review_plan(plan);negative=[]
    for name,update in [('old_mean_objective',dict(objective_policy='mean')),('missing_beta_parameter',dict(complete_parameter_widths=dict(A=1060832,B=1060832))),('wrong_counts',dict(prior_actual_costs=dict(heads=0))),('longer_schedule',dict(schedule=dict(accepted_updates=21,corrections=2,working_functions=64,backtracks=8))),('missing_history',dict(applicable_failure_registry_chain=chain[:1])),('unmatched_endpoint_best_selection',dict(unmatched_terminal_result='best')),('no_frequency',dict(all_original_frequency_and_denominators_preserved=False)),('full_quality_claim',dict(full_task_quality_acceptance=True)),('missing_global_stop',dict(batch_global_fault_stops_remaining_fits=False)),('no_final_same_point',dict(same_point_final_replay_required=False))]:
        bad=copy.deepcopy(plan);bad.update(update)
        try:review_plan(bad)
        except (ValueError,KeyError):negative.append(name)
        else:raise AssertionError('Expected refusal '+name)
    from v169_prior_pair_training_entry_v11 import require
    try:require('initial')
    except RuntimeError as e:assert 'not sealed' in str(e)
    else:raise AssertionError('No official seal may grant execution')
    PLAN.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');OUT.mkdir()
    report=dict(status='V169_concrete_contract_and_dedicated_negative_scope_checks_passed_not_execution',candidate_path=PLAN.relative_to(ROOT).as_posix(),candidate_sha256=sha(PLAN),mechanical_review=review,negative_cases_refused=negative,unsealed_actual_entry_refused=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,new_fit_permission=False,source_sha256={**plan['source_sha256'],PLAN.relative_to(ROOT).as_posix():sha(PLAN)})
    (OUT/'qualification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],negative_cases=len(negative),official_calls=0)))

if __name__=='__main__':main()
