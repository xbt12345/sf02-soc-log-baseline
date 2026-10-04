"""Dedicated V169 plan/seal upgrade; reuse project identity checks, no old budget scope."""
import importlib.metadata,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

def review_plan(plan):
    if plan['protocol']!='V169-three-role-paired-learnable-prior-bounded-training-v1':raise ValueError('V169 protocol mismatch')
    check_bindings(plan['source_sha256'])
    if plan['schedule']!=dict(accepted_updates=20,corrections=2,working_functions=64,backtracks=8):raise ValueError('New schedule qualification required')
    if plan['complete_parameter_widths']!=dict(A=1060832,B=1060833):raise ValueError('Full beta parameter not registered')
    if plan['objective_policy']!='strict_individual_M_S_actual_16eps_Armijo_and_full_original_row_protection':raise ValueError('Old mean objective forbidden for both arms')
    required=dict(roles=[0,1,2],arms=['A','B'],candidate='B',same_safe_initialization=True,bootstrap_in20=True,all_original_frequency_and_denominators_preserved=True,permanent_full_guard_ledger=True,full_guard_each_candidate=True,fresh_derivatives_at_actual_parameter_point=True,no_outer_selection=True,no_new_confirmations_or_model_promotion=True,unmatched_terminal_result='inconclusive',low_prior_slice_no_repair_cannot_claim_deep_error_solved=True)
    for key,value in required.items():
        if plan.get(key)!=value:raise ValueError('Missing applicable project constraint: '+key)
    if plan['prior_actual_costs']!=dict(heads=19178,features=19178,original_class_derivatives=368,fixed_target_derivatives=32,margin_derivatives=368,all_complete_derivatives=768,fits_since_V159=9,accepted_updates_since_V159=170):raise ValueError('Historical dimensions/costs changed')
    registry=read(ROOT/'training/review_policy/v168_observed_runtime_boundaries.json')
    if plan['inherited_observed_actions']!={c['id']:c['required_action'] for c in registry['cases']}:raise ValueError('Historical failure action missing')
    if plan['role_caps']!={str(r):dict(heads=9940 if r!=1 else 8284,features=9940 if r!=1 else 8284,fixed_target_derivatives=84,margin_derivatives=4864,QP=57,proposals=191,updates=20,fits=1) for r in [0,1,2]}:raise ValueError('Exact entry callgraph caps differ')
    if plan['new_caps']!=dict(heads=56328,features=56328,original_class_derivatives=0,fixed_target_derivatives=504,margin_derivatives=29184,all_complete_derivatives=29688,QP=342,proposals=1146,fits=6,updates=120):raise ValueError('Combined caps differ')
    if plan['full_task_quality_acceptance'] or plan['model_promoted']:raise ValueError('Preparation cannot claim complete quality')
    return dict(status='V169_dedicated_scope_identity_fixed_endpoint_and_historical_actions_reviewed',quality_acceptance=False)

def require_run_seal(path,current_trainer,phase='running'):
    path=Path(path)
    if not path.is_file():raise RuntimeError('V169 not sealed: zero official calls permitted')
    seal=read(path)
    if seal.get('status')!='V169_new_physical_seal_before_official_training' or (ROOT/seal['trainer_path']).resolve()!=Path(current_trainer).resolve():raise ValueError('New V169 trainer/physical seal required')
    check_bindings(seal['source_sha256']);plan_path=ROOT/seal['plan_path']
    if sha(plan_path)!=seal['plan_sha256']:raise ValueError('Execution contract changed')
    plan=read(plan_path);review_plan(plan)
    if not plan['execution_authority'] or not plan['new_fit_permission']:raise RuntimeError('Draft is not execution authority')
    root_review_path=ROOT/seal['root_review_path']
    if sha(root_review_path)!=seal['root_review_sha256']:raise ValueError('Root review changed')
    root=read(root_review_path);check_bindings(root['source_sha256'])
    if root.get('supports_physical_seal') is not True or root['reviewed_entry_sha256']!=sha(current_trainer):raise RuntimeError('Independent actual review does not support this entry')
    if seal.get('execution_dependency_closure_complete_reviewed') is not True or Path(current_trainer).resolve().relative_to(ROOT).as_posix() not in seal['source_sha256']:raise ValueError('Full source/dependency identity required')
    if sys.flags.optimize or sys.version!=seal['python_version'] or {n:importlib.metadata.version(n) for n in seal['package_versions']}!=seal['package_versions']:raise ValueError('Exact Python/package environment required')
    if phase not in ['initial','running']:raise ValueError('Known resource phase required')
    # One full-run prerequisite; later fits check the fixed external-state reserve.
    minimum=plan['resources']['minimum_free_disk_start_bytes'] if phase=='initial' else plan['resources']['fixed_free_disk_reserve_bytes']
    if shutil.disk_usage(ROOT).free<minimum:raise RuntimeError('Registered disk prerequisite/reserve not met')
    return plan
