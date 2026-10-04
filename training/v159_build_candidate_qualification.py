"""Freeze one concrete candidate/finite budget; never activate a formal fit."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v159_class_direction import class_direction,finite_armijo

OUT=ROOT/'artifacts/v159_candidate_qualification_20261001'
CONTRACT=ROOT/'training/review_policy/v159_candidate_qualification_contract.json'
BANK=ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists() and not CONTRACT.exists();OUT.mkdir()
    paths=[Path(__file__).resolve()]+[ROOT/p for p in [
        'training/v159_current_input_boundary_v2.py','training/v159_current_input_boundary.py','training/v159_class_direction.py',
        'training/v159_mgda_synthetic_qualification.py','training/experiment_review.py',
        'artifacts/v159_boundary_torch_synthetic_qualification_v3_20261001/qualification.json',
        'artifacts/v159_boundary_torch_synthetic_qualification_v2_20261001/qualification.json',
        'artifacts/v159_boundary_torch_synthetic_qualification_20261001/failure.json',
        'artifacts/v159_independent_OOF_capacity_and_input_review_v2_20261001/audit.json',
        'artifacts/v159_current_expert_capacity_review_20261001/audit.json',
        'artifacts/v159_mgda_synthetic_qualification_20261001/qualification.json',
        'artifacts/v159_saved_probability_numerical_audit_20261001/audit.json',
        'artifacts/v159_saved_logit_scale_and_risk_audit_20261001/audit.json',
        'artifacts/v159_zero_probability_gradient_qualification_20261001/qualification.json',
        'docs/V159_INDEPENDENT_NEXT_TRAINING_DECISION_AND_RESEARCH.md','docs/V159_NUMERICAL_INITIALIZATION_AND_GRADIENT_REVIEW.md',
        'training/v85_protection.py','training/v87_solver.py','training/v110_layer_probes.py','training/v128_train_r3.py','training/v127_model.py',
        'docs/V85_PROTECTION_TRAINING_RESULTS.md','docs/V89_READOUT_SUPPORT_TRAINING_RESULTS.md','docs/V95_FOUR_ARM_TRAINING_RESULTS_AND_FAILURE_ANALYSIS.md',
        'artifacts/v158_fusion_trial_20261001/final_delivery.json','artifacts/v158_fusion_trial_20261001/run_seal.json',
        'artifacts/v158_legal_fusion_bank_v2_20261001/pre_saved_array_bindings.json','artifacts/v158_legal_fusion_bank_v2_20261001/qualification.json',
        'artifacts/v124_header_trial_20260929/B_header_ASA.npz']]
    for f in range(3):paths.extend([BANK/f'fold{f}/{n}' for n in ['legal_FIT_reference.parquet','OOF_probabilities.npy','deployment_probabilities.npy']])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    save(OUT/'pre_qualification_bindings.json',dict(status='bound_before_role_decode_and_synthetic_direction_qualification',source_sha256=bindings))
    seal=read(ROOT/'artifacts/v158_fusion_trial_20261001/run_seal.json')
    for p in paths:
        name=p.relative_to(ROOT).as_posix()
        if name in seal['source_sha256']:assert bindings[name]==seal['source_sha256'][name],name
    toy=read(ROOT/'artifacts/v159_boundary_torch_synthetic_qualification_v3_20261001/qualification.json')
    assert toy['official_fits']==toy['official_classifier_calls']==toy['official_gradient_calls']==0 and toy['CUDA_toy']['toy_sparse_CUDA_origin_probability_tolerance_pass']
    budgets=[]
    for f in range(3):
        frame=pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet');k=math.ceil(frame.local.nunique()/2048)
        assert frame.truth.isin([1,2]).all() and (frame.fold!=f).all()
        budgets.append(dict(role=f,original_rows=len(frame),original_class_mass=[int(frame.truth.eq(c).sum()) for c in range(3)],locals=int(frame.local.nunique()),chunks=k,
            preflight_classifier_cap_A=6*k+24,preflight_classifier_cap_B=2*k+24,preflight_full_class_gradients_A=4,preflight_full_class_gradients_B=0,
            fit_classifier_cap_per_arm=1603*k+12,fit_full_class_gradient_cap_per_arm=400,fit_gradient_iterations_per_arm=200,
            proposals_per_arm=600,accepted_updates_per_arm=200,evaluation_classifier_cap_per_arm=72+k))
    rng=np.random.default_rng(15901);direction_calls=0
    for width in [1,2,7]:
        for _ in range(50):
            gm,gs=rng.normal(size=(2,width))
            for arm in ['A','B']:
                r=class_direction(gm,gs,9,2,arm);direction_calls+=1
                assert r['status']=='direction_qualified_for_finite_guarded_proposal' and np.abs(r['direction']).max()==1
                if arm=='A':assert np.allclose(r['direction'],-(9*gm+2*gs)/11/np.abs((9*gm+2*gs)/11).max(),rtol=1e-12,atol=1e-12)
                else:assert max(r['class_slopes'])<0
    zero=class_direction(np.array([1.]),np.array([-1.]),9,2,'B');direction_calls+=1
    assert zero['status']=='zero_direction_no_automatic_fallback'
    # Real finite-risk inequalities and classification retention are separate.
    assert finite_armijo([2,3],[1.9,2.9],[-1,-1],9,2,'B',1,True)
    assert not finite_armijo([2,3],[1.9,3.001],[-1,-1],9,2,'B',1,True)
    assert not finite_armijo([2,3],[1.9,2.9],[-1,-1],9,2,'B',1,False)
    assert finite_armijo([2,3],[1.9,3.001],[-1,-1],9,2,'A',1,True)
    total=dict(fits=6,new_base_fits=0,gradient_iterations=1200,fit_full_class_gradients=2400,preflight_full_class_gradients=12,
        full_class_gradients=2412,proposal_evaluations=3600,accepted_updates=1200,
        preflight_classifier_forward_chunks=sum(z['preflight_classifier_cap_A']+z['preflight_classifier_cap_B'] for z in budgets),
        fit_classifier_forward_chunks=2*sum(z['fit_classifier_cap_per_arm'] for z in budgets),evaluation_classifier_forward_chunks=2*sum(z['evaluation_classifier_cap_per_arm'] for z in budgets))
    total['classifier_forward_chunks']=sum(total[k] for k in ['preflight_classifier_forward_chunks','fit_classifier_forward_chunks','evaluation_classifier_forward_chunks'])
    assert total['classifier_forward_chunks']==77832
    contract=dict(version='V159-single-refined-current-input-probability-base-qualification-not-activation',latest_actual_training='V158',
        candidate_module='training/v159_current_input_boundary_v2.py',candidate_count=1,arms=['A','B'],candidate_arm='B',seed=15901,
        shape=dict(current_input=66287,text=65792,facts=495,members=16,opinion=11,hidden=16,outputs=3,parameters=1060832),
        input_scale=dict(source='artifacts/v124_header_trial_20260929/B_header_ASA.npz',values='unchanged existing float32 CSR values cast to float64',normalization_fit=False,text_sketch=False,legacy_N1=False,source_root_clock_fold_or_label_features=False),
        initialization=dict(observation_weight_std='1/sqrt(66287)',opinion_weight_std='1/sqrt(11)',hidden_bias='zero',output_weight='zero',identical_AB=True),
        probability_base=dict(current16_mean=True,floor=1e-12,origin_probability_tolerance=3e-12,origin_argmax_changes_allowed=0,softmax_mean_logits=False,conditional_zero_delta_branch=False,stable_training_log_softmax=True,probability_clipped_CE_floor=1e-300,clipped_CE_is_diagnostic_only=True),
        solver=dict(A='original frequency mean CE gradient reconstructed from two complete class gradients',B='exact two original class mean gradient minimum norm direction',direction_scale='one common infinity-norm normalization after A or B direction; no separate class scaling',initial_step=1.,max_step=1.,min_step=2**-40,shrink=.5,grow=2.,armijo=1e-4,max_backtracks=40,optimizer='finite backtracking only; no Adam',zero_direction='stop; no fallback or extra budget',B_finite_guard='both actual stable class risks meet their Armijo bounds and every initially correct OOF row retained',A_finite_guard='original-frequency mean stable CE meets its Armijo bound',both_arms_deployment_guard='every current-correct FIT original role row and all registered joint TRAIN scopes retained'),
        N_policy=dict(output='third free softmax output retained',OOF_truth_N=0,N_risk='absent/not applicable, never fabricated zero',N_denominator_gradient=True,non_ASA_route='frozen old predictions; all full N/M/S metrics still scored',ASA_predictions_of_N='count as M/S errors and full N false calls',claim_of_new_N_learning=False),
        role_call_budgets=budgets,total_future_caps=total,
        forward_count_derivation=dict(preflight_A='2K OOF repeat +24 full deploy repeat +4K repeated M/S gradients',preflight_B='2K OOF repeat +24 full deploy repeat',fit='2K initial +400K class gradients +1200K proposal OOF/deploy +K endpoint OOF +12 full endpoint deploy',evaluation='six fixed endpoint/window full deploy 72 +K endpoint OOF verification',setup_dummy='outside official caps, separately counted before sealing; currently0'),
        endpoint='last accepted state at registered budget/finite-guard stop; no label-selected checkpoint',checkpoints=[0,1,20,50,100,150,200],last_window=5,
        source_validation=dict(query_root_and_outer_excluded_from_current45_base_fits=True,reuse_current45=True,old9_N1_diagnostic_only=True,head_training_OOF_is_head_validation=False,existing_outer_folds='already inspected development evaluation, not blind',inner_head_selection=False,new_independent_source_population='requires correct deeper exclusion and new prospective cost registration'),
        budget_scale_counterexample=dict(old_member_logit_maximum_required_margin=3546.9754155491937,old_step_001_tanh16_200_update_upper_margin=64.,selected_probability_base_maximum_initial_class_pair_margin=float(-np.log(1e-12)),unit_infinity_direction_200_update_theoretical_upper_margin=6400.,upper_bound_is_not_learnability_proof=True),
        quality_acceptance=False,new_training_entry_registered=False,formal_runtime_ready=False,official_zero_step_replay=False,allowed_activation_entries=[],
        actual_this_qualification=dict(official_classifier_calls=0,official_features_calls=0,official_gradient_calls=0,official_fits=0,official_updates=0,synthetic_direction_evaluations=direction_calls,synthetic_finite_guard_evaluations=4),source_sha256=bindings)
    check_bindings(bindings);save(CONTRACT,contract)
    save(OUT/'qualification.json',dict(status='single_function_initialization_N_input_scale_and_corrected_counted_budget_bound_not_formal_activation',latest_actual_training='V158',
        actual_official_classifier_calls=0,actual_official_features_calls=0,actual_official_gradients=0,actual_official_fits=0,actual_official_updates=0,
        total_future_caps=total,contract_path=CONTRACT.relative_to(ROOT).as_posix(),contract_sha256=sha(CONTRACT),
        synthetic_full_parameter_head_gradient_qualified=True,new_training_entry_registered=False,formal_runtime_ready=False,official_zero_step_replay=False,quality_acceptance=False,
        remaining=['Implement and independently review counted official entry and full acceptance before activation.','Seal actual imported source/runtime/physical data then perform real-row origin and repeated complete class gradient preflight within declared caps.','Full row classification/source gates, not finite descent, determine completion; do not replenish spent budgets.']))
    print(json.dumps(dict(status='candidate_qualification_bound_not_activated',future_caps=total,synthetic_direction_evaluations=direction_calls,official_calls=0),ensure_ascii=False))

if __name__=='__main__':main()
