"""Bind actual legal bank and implemented fusion/evaluation entry before activation."""
import math
from pathlib import Path
import pandas as pd
from v158_fusion_runtime import ROOT,BANK,CONTRACT,PLAN,read,sha,save

def main():
    assert not CONTRACT.exists();p=read(PLAN);bank=read(BANK/'qualification.json');assert bank['base_fits']==45 and bank['legacy_fits']==9
    c=dict(version='V158-six-fixed-probability-fusion',fits=6,folds=[0,1,2],arms=['A','B'],candidate='B',plan_sha256=sha(PLAN),
        expert_count=17,condition_width=527,full_gradients_per_fit=200,proposals_per_fit=600,accepted_updates_per_fit=200,
        preflight_gradients=0,activation_entries=['training/v158_fusion_train.py','training/v158_fusion_evaluate.py'],
        endpoint=p['fusion_endpoint'],solver=p['fusion_solver'],OOF_query_labels_used_for_base_fit=False,
        deployment_FIT_guard_labels_are_legal=True,outer_labels_used_for_selection=False,
        automatic_repeat=False,automatic_confirmation=False,quality_acceptance=False,role_call_budgets=[])
    for f in range(3):
        frame=pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet');k=math.ceil(frame.local.nunique()/2048)
        c['role_call_budgets'].append(dict(fold=f,fit_original_rows=len(frame),fit_locals=frame.local.nunique(),fit_chunks=k,
            preflight_classifier_forward_cap_per_arm=2*k+24,fit_classifier_forward_cap_per_arm=1404*k+12,
            evaluation_classifier_forward_cap_per_arm=6*12+k,global_gradient_cap_per_arm=200))
    paths=[ROOT/'training'/n for n in ['v158_fusion_runtime.py','v158_fusion_train.py','v158_fusion_evaluate.py','v158_assemble_legal_bank.py','v158_assemble_legal_bank_v2.py',
        'v158_build_fusion_contract.py','v158_fusion_prototype_v2.py','v158_conditions.py','test_v158_fusion_execution.py',
        'v142_retention_check.py','v140_retention_check.py','v138_retention_check.py','v137_issue_guard.py','experiment_review.py',
        'v131_evaluate.py','v135_evaluate.py','v138_closeout.py']]
    paths.extend([ROOT/'docs/V158_FUSION_EXECUTION_AND_FULL_ACCEPTANCE.md',BANK/'qualification.json',BANK/'pre_saved_array_bindings.json',
        ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001/base_phase_completion.json',ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001/legacy_phase_completion.json',
        ROOT/'artifacts/v146_guarded_pair_training_20261001/full_prediction_ledger.parquet',
        ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet',ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet',
        ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet',
        ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001/new_M_regressions_vs_V146_A.parquet',
        ROOT/'artifacts/v158_independent_design_counterchecks_20261001/audit.json'])
    paths.extend([ROOT/'artifacts/v158_zero_update_stage_identity_audit_20261001/audit.json',
        ROOT/'artifacts/v158_legal_fusion_bank_20261001/failure.json',ROOT/'training/v158_verify_zero_update_stages.py'])
    from v131_common import INPUT,TRACE,OFFICIAL,REVIEW
    from v107_matched_training import ROWS,FOLDS
    paths.extend([INPUT,TRACE,OFFICIAL,REVIEW/'training_role_error_ledger.parquet',ROWS,FOLDS])
    # Existing joint guards and checkpoints are execution dependencies too.
    for folder,registry in [('v138_single_issue_round1_20260930','scoped_training_capabilities.json'),
        ('v140_ensemble_training_round2_20261001','additional_verified_TRAIN_scopes.json'),
        ('v142_second_layer_training_20261001','verified_TRAIN_mastery_registry.json')]:
        root=ROOT/'artifacts'/folder
        if (root/registry).is_file():
            paths.append(root/registry);r=read(root/registry)
            for item in r.get('scopes',r.get('verified_training_scopes',[])):
                for key in ['guard','checkpoint','baseline_rows']:
                    if key in item:paths.append(ROOT/item[key])
                if folder=='v138_single_issue_round1_20260930':
                    import re
                    match=re.search(r'fold(\d+)',item['guard']);assert match
                    paths.append(root/f'fold{match.group(1)}_H_L/endpoint_original_rows.parquet')
    c['source_sha256']={q.relative_to(ROOT).as_posix():sha(q) for q in sorted(set(paths))}
    save(CONTRACT,c)
    from v158_fusion_runtime import review
    review();print(dict(fusion_contract_bound=True,teacher_fits_completed=54,fusion_fits_max=6,quality_acceptance=False))

if __name__=='__main__':main()
