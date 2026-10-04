"""Prospective bounded trial-point correction and immutable physical checker."""
import importlib.metadata,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

def prospective_plan():
    plan=read(ROOT/'training/review_policy/v167_trial_point_restoration_draft.json');assert plan['execution_authority'] is False;plan['protocol']=plan['proposed_protocol'];plan['new_caps']=plan['proposed_caps'];old=read(ROOT/'training/review_policy/v166_coverage_first_diagnostic_contract.json')
    for spec,prior in zip(plan['roles'],old['roles']):
        assert spec['role']==prior['role'] and spec['OOF_chunks']==prior['OOF_chunks']
        for key in ['endpoint_parameter_sha256','parameter_point']:spec[key]=prior[key]
    registry=read(ROOT/plan['failure_constraints']);plan['failure_case_actions']={r['id']:r['required_action'] for r in registry['cases']};return plan

def review_plan(plan):
    assert plan['protocol']=='V167-V166-actual-guard-conditioned-trial-point-Jacobian-restoration-v1';assert plan['new_caps']==dict(heads=296,features=296,margin_gradients=100,fixed_error_target_gradients=0,full_original_class_gradients=0,joint_QP_solves=2,finite_proposals=4,fits=0,permanent_updates=0)
    assert [s['head_cap'] for s in plan['roles']]==[66,164,66] and [s['margin_gradient_cap'] for s in plan['roles']]==[0,100,0] and [s['QP_cap'] for s in plan['roles']]==[0,2,0] and [s['finite_proposal_cap'] for s in plan['roles']]==[1,2,1];assert [s['complete_functions'] for s in plan['roles']]==[24,25,14]
    assert plan['uniform_maximum_margin_functions']==25 and plan['maximum_corrections_on_actual_failed_role']==2 and plan['second_correction_max_negative_margin_factor']==.99 and plan['local_decision_floor']==0
    assert plan['second_correction_requires_actual_covered_only_blockers'] and plan['second_correction_requires_original_finite_target_drop_and_Armijo'] and plan['new_Jacobians_pair_measured_at_each_actual_trial_parameter'] and plan['original_fixed_target_gradients_remain_at_V164_origin'];assert plan['actual_argmax_and_all_original_guards_authoritative'] and plan['all_original_inputs_labels_frequencies_class_denominators_targets_and_protections_unchanged'] and plan['safe_V166_role0_and_role2_candidates_not_reoptimized'] and plan['finally_restore_complete_V164_parameters_and_original_predictions']
    assert plan['prior_actual_costs']['heads']==18784 and plan['prior_actual_costs']['all_complete_parameter_derivatives']==618 and plan['future_cumulative_caps']['heads']==19080<=82174 and plan['future_cumulative_caps']['all_complete_parameter_derivatives']==718<=2426
    assert plan['new_fitting_permission'] is False and plan['quality_acceptance'] is False and plan['model_promotion'] is False and plan['no_automatic_third_correction_capacity_increase_seed_change_step_change_or_fit'];registry=read(ROOT/plan['failure_constraints']);assert plan['failure_case_actions']=={r['id']:r['required_action'] for r in registry['cases']};return dict(status='V167_trial_point_finite_plan_reviewed',quality_acceptance=False)

def require_run_seal(path,entry):
    seal=read(path);assert seal['status']=='V167_sealed_before_actual_trial_point_diagnostic' and (ROOT/seal['entry_path']).resolve()==Path(entry).resolve();assert sys.flags.optimize==0 and sys.version==seal['python_version'] and {n:importlib.metadata.version(n) for n in seal['package_versions']}==seal['package_versions'];check_bindings(seal['source_sha256']);plan=read(ROOT/seal['plan_path']);assert plan['execution_authority'] is True and sha(ROOT/seal['plan_path'])==seal['plan_sha256'];review_plan(plan);return plan
