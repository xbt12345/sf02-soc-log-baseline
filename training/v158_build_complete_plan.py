"""One prospective registered design; data integrity only, no model functions."""
import math
from pathlib import Path
import numpy as np
from scipy import sparse
from v158_nested_runtime import ROOT,PLAN,QUAL,read,sha,save,expected_contract,validate
from v107_matched_training import ROWS,FOLDS,FID,ASA_IDS,X,PREP
from v131_common import INPUT,TRACE,OFFICIAL,REVIEW
from run_v75 import load_sparse

PREFLIGHT=ROOT/'artifacts/v158_formal_preflight_20261001'

def main():
    assert not PLAN.exists() and not PREFLIGHT.exists();PREFLIGHT.mkdir()
    matrix=load_sparse(X);ids=np.load(ASA_IDS);delta=sparse.load_npz(PREP/'N1_delta.npz')
    original=sparse.load_npz(PREP/'N1_ASA.npz');actual=matrix[ids]+delta[ids]
    difference=(actual.astype(np.float64)-original.astype(np.float64)).tocsr();difference.eliminate_zeros()
    gap=float(np.abs(difference.data).max()) if difference.nnz else 0.
    assert matrix.shape==(457566,66287) and actual.shape==(22546,66287) and gap<=2e-6
    source_paths=[ROWS,FOLDS,FID,ASA_IDS,OFFICIAL,INPUT,TRACE,REVIEW/'training_role_error_ledger.parquet',PREP/'N1_delta.npz',PREP/'N1_ASA.npz']
    source_paths.extend(Path(str(X)+suffix) for suffix in ['.json','.data','.indices','.indptr'])
    save(PREFLIGHT/'actual_N1_training_query_input_integrity.json',dict(status='existing_V107_N1_actual_fit_query_path_verified',
        full_numeric_shape=list(matrix.shape),ASA_query_shape=list(actual.shape),different_stored_coordinates=int(difference.nnz),
        maximum_fit_query_coordinate_difference=gap,registered_tolerance=2e-6,
        official_classifier_calls=0,new_fits=0,new_gradients=0,new_updates=0,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths}))
    p=expected_contract();roles=read(QUAL/'qualification.json')['roles']
    bank=ROOT/'artifacts/v158_legacy_expert_bank_qualification_20261001'
    floors=read(bank/'actual_inner_canonical_floors.json')['roles']
    p['role_budgets']=[]
    for r,floor in zip(roles,floors):
        n=r['fit_locals'];nb=math.ceil(n/256);k=math.ceil(n/2048)
        p['role_budgets'].append(dict(outer_fold=r['outer_fold'],excluded_inner=r['excluded_inner'],fit_locals=n,
            fit_rows=r['fit_rows'],fit_class_mass=[0,r['fit_M'],r['fit_S']],canonical_floor=floor['canonical_floor'],
            full_network_gradient_updates=100*nb,full_network_forward_cap=201*nb+89,
            LBFGS_stage_forward_cap=402*k+12,Armijo_stage_forward_cap=1402*k+12))
    p['legacy_role_budgets']=read(bank/'legacy_full_nested_roles_and_cost.json')['roles']
    p['legacy_ASA_forward_chunks_per_fit']=3
    p['task_quality']=read(ROOT/'training/review_policy/v137_single_issue_plan.json')['task_adoption_quality']
    p['risk_actions']={r['id']:r['required_action'] for r in read(ROOT/'training/review_policy/history_cases.json')['risks']}
    p['cost_scope']=dict(base_minibatch_gradients=35600,base_dense_global_gradient_cap=7200,
        legacy_full_population_gradient_cap=9000,reserved_fusion_global_gradient_cap=1200,
        gradient_evaluations_cap_of_different_population_types=53000,
        base_actual_classifier_call_cap=sum(r['full_network_forward_cap']+3*r['LBFGS_stage_forward_cap']+r['Armijo_stage_forward_cap'] for r in p['role_budgets']),
        legacy_full_risk_forward_cap=9000,legacy_ASA_query_forward_cap=27,
        setup_dummy_forwards=1,setup_dummy_gradients=1,setup_dummy_updates=0)
    p['scope']='Previously inspected developmental sources; new legal source-excluded supervision generation plus one A/B condition factor, not independent external acceptance.'
    p['bank_prior_source']=(bank/'initial_legacy_priors.json').relative_to(ROOT).as_posix()
    p['TRAIN_acceptance']=dict(pure_errors=0,M_errors=0,S_by_outer_fold=[22,6,28],correct_row_new_errors=0,
        last_distinct_states=5,guard_entry='training/v142_retention_check.py')
    p['matched_effect_acceptance']=dict(M_no_increase=True,S_no_increase=True,one_class_improves=True,two_folds_improve=True,outside_top3_S_no_increase=True)
    p['failure_policy']='Preserve sources, seal, actual attempts and incomplete state; no in-place repair, restart, scan, substituted OOF teacher or quality promotion.'
    own=['v158_nested_runtime.py','v158_nested_base_train.py','v158_legacy_nested_train.py','v158_conditions.py',
        'v158_build_complete_plan.py','v158_fusion_contract.py','v158_fusion_prototype_v2.py','v158_bank_initialization.py',
        'test_v158_complete_nested_trial.py','test_v158_fusion_bank_v2.py','experiment_review.py',
        'v131_common.py','v131_model.py','v135_model.py','v138_train.py','v138_readout.py','v141_second_model.py','v146_optimizer.py',
        'v107_matched_training.py','v85_protection.py','run_v75.py','v142_retention_check.py']
    source_paths.extend(ROOT/'training'/name for name in own)
    source_paths.extend([QUAL/'qualification.json',QUAL/'pre_execution_bindings.json',QUAL/'nested_source_roles.parquet',
        bank/'qualification.json',bank/'pre_saved_array_bindings.json',bank/'initial_legacy_priors.json',
        bank/'actual_inner_canonical_floors.json',bank/'legacy_full_nested_roles_and_cost.json',
        PREFLIGHT/'actual_N1_training_query_input_integrity.json',
        ROOT/'training/review_policy/history_cases.json',ROOT/'training/review_policy/v137_single_issue_plan.json',
        ROOT/'docs/EXPERIMENT_REVIEW_RULES.md',ROOT/'docs/V158_COMPLETE_NESTED_TRIAL_DESIGN_AND_EXECUTION.md'])
    p['source_sha256']={path.relative_to(ROOT).as_posix():sha(path) for path in sorted(set(source_paths))}
    validate(p);save(PLAN,p)
    save(PREFLIGHT/'plan_review.json',dict(status='eligible_for_bounded_base_supervision_generation',formal_fits_max=60,
        actual_new_fits=0,actual_official_classifier_calls=0,actual_gradients=0,actual_updates=0,
        actual_N1_data_integrity_passed=True,plan_sha256=sha(PLAN),quality_acceptance=False,model_promoted=False))
    print(dict(plan_created=True,formal_fits_max=60,N1_fit_query_coordinate_max_gap=gap,base_forward_cap=p['cost_scope']['base_actual_classifier_call_cap']))

if __name__=='__main__':main()
