"""Prospective runtime review; use the immutable shared physical binding gate."""
import importlib.metadata,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

def prospective_plan():
    plan=read(ROOT/'training/review_policy/v166_coverage_first_diagnostic_draft.json')
    assert plan['execution_authority'] is False
    plan['protocol']=plan['proposed_protocol'];plan['new_caps']=plan['proposed_caps'];plan['execution_authority']=False
    prior=read(ROOT/'training/review_policy/v165_fixed_endpoint_decision_floor_diagnostic_contract.json')
    for spec,old in zip(plan['roles'],prior['roles']):
        assert spec['role']==old['role'] and spec['parameter_point']==old['parameter_point']
        for key in ['OOF_chunks','endpoint_parameter_sha256','cached_treatment']:spec[key]=old[key]
    registry=read(ROOT/plan['failure_constraints']);plan['failure_case_actions']={r['id']:r['required_action'] for r in registry['cases']}
    return plan

def review_plan(plan):
    assert plan['protocol']=='V166-three-fixed-V164-endpoints-complete-observed-blocker-coverage-v1'
    assert plan['new_caps']==dict(heads=246,features=246,fixed_error_target_gradients=0,full_original_class_gradients=0,margin_gradients=66,finite_proposals=3,joint_QP_solves=3,fits=0,permanent_updates=0)
    assert [r['head_cap'] for r in plan['roles']]==[106,66,74] and [r['fresh_margin_gradient_cap'] for r in plan['roles']]==[40,18,8]
    assert [r['fresh_functions'] for r in plan['roles']]==[20,9,4] and [r['joint_functions'] for r in plan['roles']]==[24,25,14]
    assert plan['prior_actual_costs']['heads']==18538 and plan['prior_actual_costs']['all_complete_parameter_derivatives']==552
    assert plan['future_cumulative_caps']['heads']==18784<=82174 and plan['future_cumulative_caps']['all_complete_parameter_derivatives']==618<=2426
    assert plan['uniform_maximum_margin_functions']==25 and plan['max_joint_recoveries_per_role']==1 and plan['local_decision_floor']==0
    assert plan['only_changed_factor']=='add_all_observed_same_theta_V165_blocking_functions_to_joint_local_restoration'
    assert plan['new_fitting_permission'] is False and plan['quality_acceptance'] is False and plan['model_promotion'] is False
    assert plan['do_not_remove_any_original_protection_or_target'] and plan['all_original_class_and_retention_guards_unchanged'] and plan['finally_restore_original_complete_model_and_predictions']
    registry=read(ROOT/plan['failure_constraints']);assert plan['failure_case_actions']=={r['id']:r['required_action'] for r in registry['cases']}
    return dict(status='V166_coverage_finite_plan_reviewed',quality_acceptance=False)

def require_run_seal(path,entry):
    seal=read(path);assert seal['status']=='V166_sealed_before_actual_coverage_diagnostic' and (ROOT/seal['entry_path']).resolve()==Path(entry).resolve()
    assert sys.flags.optimize==0 and sys.version==seal['python_version'] and {n:importlib.metadata.version(n) for n in seal['package_versions']}==seal['package_versions']
    check_bindings(seal['source_sha256']);plan=read(ROOT/seal['plan_path']);assert sha(ROOT/seal['plan_path'])==seal['plan_sha256'];assert review_plan(plan)['status']=='V166_coverage_finite_plan_reviewed'
    return plan
