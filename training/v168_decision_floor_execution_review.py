"""Single actual failed-role diagnostic, immutable source and cumulative caps."""
import importlib.metadata,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

def prospective_plan():
    plan=read(ROOT/'training/review_policy/v168_decision_aware_floor_draft.json');assert not plan['execution_authority'];check_bindings(plan['source_sha256']);plan['protocol']=plan['proposed_protocol'];plan['new_caps']=plan['proposed_caps'];old=read(ROOT/'training/review_policy/v167_trial_point_restoration_contract.json')['roles'][1]
    plan['role_spec']=dict(role=1,head_cap=98,margin_gradient_cap=50,QP_cap=1,finite_proposal_cap=1,OOF_chunks=4,complete_functions=25,parameter_point=old['parameter_point'],endpoint_parameter_sha256=old['endpoint_parameter_sha256'])
    registry=ROOT/'training/review_policy/v167_observed_runtime_boundaries.json';plan['failure_constraints']=registry.relative_to(ROOT).as_posix();plan['failure_case_actions']={r['id']:r['required_action'] for r in read(registry)['cases']};return plan

def review_plan(plan):
    assert plan['protocol']=='V168-single-decision-aware-floor-restoration-at-actual-V167-tie-point-v1' and plan['execute_only_actual_failed_role']==1 and plan['complete_functions']==25 and plan['margin_repetitions']==2
    assert plan['new_caps']==dict(heads=98,features=98,margin_gradients=50,fixed_error_target_gradients=0,full_original_class_gradients=0,QP_solves=1,finite_proposals=1,fits=0,permanent_updates=0)
    assert plan['role_spec']==dict(role=1,head_cap=98,margin_gradient_cap=50,QP_cap=1,finite_proposal_cap=1,OOF_chunks=4,complete_functions=25,parameter_point=1,endpoint_parameter_sha256='56d9600e98ff480de24147b1e5a632e9c85ff2fad2f3dac65f492008bf76c9fd')
    assert plan['origin_parameter_sha256']==plan['role_spec']['endpoint_parameter_sha256'] and plan['actual_failed_trial_parameter_sha256']=='8ae9b55cda7e4a710c4891dc8ab4e56f571ef90d8efb84f1c7a9e62aeb3fd431'
    assert plan['actual_failed_trial_path']=='artifacts/v167_trial_point_restoration_diagnostic_20261002/role1/probe1'
    assert plan['local_floor_rule']==dict(eps_dtype='float64',scale='max(1,abs(actual_trial_logq_truth),abs(actual_trial_logq_rival))',multiplier=16,positive_if='truth_index > rival_index',otherwise=0,not_an_actual_acceptance_tolerance=True,not_a_certified_CUDA_bound_or_transfer_safety_distance=True)
    assert plan['prior_actual_costs']==dict(heads=19080,all_complete_parameter_derivatives=718,fits_since_V159=9,permanent_updates_since_V159=170) and plan['future_cumulative_caps']==dict(heads=19178,all_complete_parameter_derivatives=768,fits_since_V159=9,permanent_updates_since_V159=170)
    assert plan['future_cumulative_caps']['heads']<=82174 and plan['future_cumulative_caps']['all_complete_parameter_derivatives']<=2426
    assert plan['prospective_full_class_error_caps']==dict(M=864,S=1156) and plan['additional_candidate_repair_goal']=='artifacts/v168_decision_floor_plan_20261002/four_observed_candidate_M_repairs.parquet'
    for key in ['new_Jacobians_at_actual_final_tie_parameter','old_class_gradients_remain_at_V164_origin','all_original_gold_rows_frequencies_denominators_targets_mixed_rows_and_cumulative_accepted_protection_unchanged','additional_goal_not_merged_into_origin_correct_mask','one_correction_only_no_automatic_second_or_floor_seed_step_change','qualification_and_physical_seal_still_required']:assert plan[key]
    assert not plan['new_fit_permission'] and not plan['quality_acceptance'] and not plan['model_promotion'];assert plan['raw_new_gradient_bytes']==424332800 and plan['largest_joint_matrix_bytes']==229139712
    registry=read(ROOT/plan['failure_constraints']);assert plan['failure_case_actions']=={r['id']:r['required_action'] for r in registry['cases']};return dict(status='V168_single_role_decision_floor_plan_reviewed',quality_acceptance=False)

def require_run_seal(path,entry):
    seal=read(path);assert seal['status']=='V168_sealed_before_actual_decision_floor_diagnostic' and (ROOT/seal['entry_path']).resolve()==Path(entry).resolve();assert sys.flags.optimize==0 and sys.version==seal['python_version'] and {n:importlib.metadata.version(n) for n in seal['package_versions']}==seal['package_versions'];check_bindings(seal['source_sha256']);plan=read(ROOT/seal['plan_path']);assert plan['execution_authority'] is True and sha(ROOT/seal['plan_path'])==seal['plan_sha256'];review_plan(plan);return plan
