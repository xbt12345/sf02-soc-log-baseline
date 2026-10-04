"""Zero-fit V165 runtime gate, using the unchanged shared physical checker."""
import importlib.metadata,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

def review_plan(plan):
    expected=dict(heads=180,features=180,fixed_error_target_gradients=0,full_original_class_gradients=0,margin_gradients=0,finite_proposals=3,QP_solves=0,fits=0,permanent_updates=0)
    assert plan['protocol']=='V165-three-V164-endpoints-single-decision-floor-finite-diagnostic-v1'
    assert plan['new_caps']==expected and [r['head_cap'] for r in plan['roles']]==[66,48,66]
    assert [r['parameter_point'] for r in plan['roles']]==[1,1,3]
    assert plan['prior_actual_costs']['heads']==18358 and plan['prior_actual_costs']['complete_parameter_derivatives_all_types']==552
    assert plan['future_cumulative_actual_caps']['heads']==18538<=82174
    assert plan['future_cumulative_actual_caps']['complete_parameter_derivatives_all_types']==552<=2426
    assert plan['only_changed_factor']=='restore_correct_decision_floor_instead_of_original_confidence_margin'
    assert plan['all_original_M_S_counts_retention_16eps_Armijo_unchanged'] and plan['all_prior_and_cumulative_new_repairs_retained']
    assert plan['one_cached_treatment_per_role'] and plan['finally_restores_V164_last_accepted_checkpoint']
    assert plan['max_margin_functions_unchanged']==24 and plan['new_fitting_permission'] is False
    assert plan['quality_acceptance'] is False and plan['model_promotion'] is False
    constraints=read(ROOT/plan['failure_constraints'])
    assert constraints['actual_round']=='V164' and len(constraints['cases'])==5
    assert plan['failure_case_actions']=={r['id']:r['required_action'] for r in constraints['cases']}
    return dict(status='V165_zero_fit_finite_plan_passed',quality_acceptance=False)

def require_run_seal(seal_path,current_entry):
    seal=read(seal_path)
    assert seal['status']=='V165_sealed_before_actual_finite_diagnostic'
    assert (ROOT/seal['entry_path']).resolve()==Path(current_entry).resolve()
    assert sys.flags.optimize==0 and sys.version==seal['python_version']
    assert {n:importlib.metadata.version(n) for n in seal['package_versions']}==seal['package_versions']
    check_bindings(seal['source_sha256'])
    plan=read(ROOT/seal['plan_path']);assert sha(ROOT/seal['plan_path'])==seal['plan_sha256']
    assert review_plan(plan)['status']=='V165_zero_fit_finite_plan_passed'
    return plan
