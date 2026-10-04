"""Offline original-budget units, exact draft calls and conservative physical readiness."""
import ast,json,shutil
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v169_saved_budget_scope_and_callgraph_v2_20261002'

def main():
    assert not OUT.exists();entry=ROOT/'training/v169_prior_pair_training_entry_v3.py';text=entry.read_text(encoding='utf-8');ast.parse(text)
    assert 'for backtrack in range(8)' in text and "strict_class_step(" in text and 'class1_repeat0' in text and "require('initial')" in text
    lifecycle=ROOT/'training/v169_pair_lifecycle_v2.py';ltext=lifecycle.read_text(encoding='utf-8');assert 'range(1,schedule.accepted_updates+1)' in ltext and 'targets(observation,0)' in ltext and "backend.close_resources()" in ltext
    v160=read(ROOT/'training/review_policy/v160_fixed_endpoint_diagnostic_contract.json');actual=read(ROOT/'artifacts/v168_results_20261002/actual_result_summary.json')
    assert (v160['new_technical_class_gradient_cap'],v160['new_margin_gradient_cap'])==(2426,144)
    prior=dict(heads=actual['cumulative_heads'],features=actual['cumulative_features'],original_class_derivatives=actual['cumulative_full_original_class_gradients'],fixed_target_derivatives=actual['cumulative_fixed_error_target_gradients'],margin_derivatives=actual['cumulative_margin_gradients'],all_complete_derivatives=actual['cumulative_all_complete_derivatives'],fits_since_V159=actual['cumulative_fits_since_V159'],accepted_updates_since_V159=actual['cumulative_updates_since_V159'])
    assert prior==dict(heads=19178,features=19178,original_class_derivatives=368,fixed_target_derivatives=32,margin_derivatives=368,all_complete_derivatives=768,fits_since_V159=9,accepted_updates_since_V159=170)
    roles={};raw_scopes=0
    for role,k in enumerate([10,4,10]):
        d=k+12;target=(1+20)*2*2;margin=19*2*64*2;proposals=1+19*(8+2);qps=19*(1+2);heads=target*k+12+proposals*d+d+margin
        roles[str(role)]=dict(heads=heads,features=heads,fixed_target_derivatives=target,margin_derivatives=margin,QP=qps,proposals=proposals,updates=20,fits=1)
        assert heads==(8284 if role==1 else 9940)
        raw_scopes+=2*(proposals+20+1)
    caps={key:2*sum(r[key] for r in roles.values()) for key in ['heads','features','fixed_target_derivatives','margin_derivatives','QP','proposals','fits','updates']};caps['original_class_derivatives']=0;caps['all_complete_derivatives']=caps['fixed_target_derivatives']+caps['margin_derivatives']
    assert caps==dict(heads=56328,features=56328,fixed_target_derivatives=504,margin_derivatives=29184,QP=342,proposals=1146,fits=6,updates=120,original_class_derivatives=0,all_complete_derivatives=29688)
    x=load_npz(ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz');assert x.shape==(22546,66287) and x.nnz==4006092;maximum=int(np.diff(x.indptr).max());assert maximum==211
    # Current v1 lossless encoder stores both indices+uint64 values; no dense-mode
    # or structural sparsity guard exists in the executed v3 entry yet. Do not
    # pretend the small observed synthetic margin size bounds every saved vector.
    maximum_complete_vector_bytes=1060833*16+65536
    target_vector_bound=504*maximum_complete_vector_bytes
    margin_vector_bound=29184*maximum_complete_vector_bytes
    finite_vector_bound=1146*maximum_complete_vector_bytes
    correction_vector_bound=228*maximum_complete_vector_bytes
    checkpoint_bound=126*(1060833*8+65536)
    margin_outputs_bound=29184*(2048*3*8*2+65536)
    candidate_outputs_bound=1146*(22546*3*8*4+65536)
    target_outputs_bound=504*(22546*3*8*2+65536)
    # Parquet/string serialization and JSON logs remain separately unresolved.
    numeric_files_bound=sum([target_vector_bound,margin_vector_bound,finite_vector_bound,correction_vector_bound,checkpoint_bound,margin_outputs_bound,candidate_outputs_bound,target_outputs_bound]);free=shutil.disk_usage(ROOT).free
    doc=ROOT/'docs/V169_INCREMENTAL_RESEARCH_AND_TRAINING_REVIEW_20261002.md';assert sha(doc)=='42c566eb2c7c09b1999ed57c2fb001ac27db0b1ab2d03582c583ee1a86bc563c'
    files=[Path(__file__).resolve(),entry,lifecycle,ROOT/'training/v169_prior_pair_model.py',ROOT/'training/v169_pair_execution_review.py',ROOT/'training/v169_exact_gradient_storage.py',ROOT/'training/v169_saved_state_quality.py',doc,ROOT/'training/review_policy/v169_learnable_prior_pair_draft.json',ROOT/'training/review_policy/v160_fixed_endpoint_diagnostic_contract.json',ROOT/'training/review_policy/v164_short_trajectory_prospective_budget.json',ROOT/'artifacts/v168_results_20261002/actual_result_summary.json',ROOT/'artifacts/v169_saved_prospective_budget_review_20261002/review.json',ROOT/'artifacts/v169_budget_scope_direction_20261002/scope_review.json',ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz']
    report=dict(status='V169_v2_budget_units_and_prepared_v3_callgraph_reviewed_physical_storage_not_ready',
      original_V160_class_gradient_cap=2426,original_V160_margin_derivatives_separate_cap=144,
      inherited_1658_only_if_later_combined2426_cap_applies=True,
      not_original_V160_class_budget_remaining=True,not_new_V169_registered_budget=True,
      historical_categories=prior,new_prepared_entry_caps=caps,per_fit_role_caps=roles,
      combined_future_if_executed=dict(heads=19178+56328,original_class_derivatives=368,fixed_target_derivatives=32+504,margin_derivatives=368+29184,all_complete_derivatives=768+29688,fits_since_V159=9+6,updates_since_V159=170+120),
      callgraph=dict(target_points_per_fit=list(range(21)),class_target_repetitions=2,classes=[1,2],baseline_deployment_heads_per_fit=12,final_original_OOF_and_deployment_replay_per_fit=True,bootstrap_finite_candidates=1,new_direction_origins=19,line_search_steps=[2.**(-j) for j in range(8)],corrections_per_origin=2,working_functions64_current_not_history_cap=True,fresh_margin_pairs_at_each_actual_trial=True,QP_internal_iterations_not_accepted_updates=True),
      actual_input_max_row_nnz=maximum,conditional_model_local_margin_support_bound=211*16+241,
      conditional_sparse_support_bound_not_yet_runtime_enforced_or_officially_validated=True,
      current_lossless_encoder_worst_case_vector_bytes=maximum_complete_vector_bytes,
      conservative_numeric_files_only_upper_bytes=numeric_files_bound,
      current_free_disk_bytes=free,even_numeric_upper_bound_exceeds_current_free_disk=numeric_files_bound>free,
      complete_parquet_log_and_failure_storage_upper_still_unresolved=True,
      full_resource_budget_and_registration_not_ready=True,
      required_next=['Lossless dense-or-indexed storage and/or exact margin structural-support qualification without dropping signs/subnormals',
        'Remove redundant saved displacement copies through bit-verified deterministic recipes or another exact representation',
        'Compute all Parquet/JSON/log/failure storage upper bounds and actual RAM/GPU prerequisites',
        'One initial full-run free-space prerequisite, later fixed external-state reserve instead of restarting whole-run prerequisite',
        'Actual backend and dedicated scope/seal adversarial qualification, dependency closure and root independent review before any seal'],
      official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,new_fit_permission=False,no_beta_effect_or_infeasibility_claim=True,
      source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    OUT.mkdir();(OUT/'review.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status=report['status'],new_heads=56328,new_complete_derivatives=29688,numeric_files_upper_bytes=numeric_files_bound,current_free_disk_bytes=free,official_calls=0)))

if __name__=='__main__':main()
