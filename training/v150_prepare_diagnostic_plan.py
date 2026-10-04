"""Register a fixed factual readout diagnostic, no classifier trial."""
import json
from pathlib import Path
from experiment_review import sha,check_bindings

ROOT=Path(__file__).resolve().parents[1]
def main():
    target=ROOT/'training/review_policy/v150_field_readout_plan.json';assert not target.exists()
    paths=[ROOT/'training/review_policy/v148_observed_qualification_cases.json',
        ROOT/'artifacts/v149_independent_input_fidelity_20261001/audit.json',
        ROOT/'artifacts/v146_guarded_pair_training_20261001/run_seal.json',
        ROOT/'artifacts/v92_evidence_training_20260928/representation_contract.json',
        ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet']
    for p in paths[:3]:check_bindings(json.loads(p.read_text(encoding='utf-8'))['source_sha256'])
    plan=dict(version='V150-field-diagnostic',kind='frozen_behavior_field_readability_diagnostic',
        latest_actual_classifier='V146',decoder_fits=6,classifier_fits=0,layers=['H1','V146_B_H2'],folds=[0,1,2],ridge=1e-4,
        fit_solver='one_original_frequency_multivariate_ridge_normal_equation_per_layer_fold',
        split='legal_TRAIN_roots_SHA256_V150_field_readout_mod5_eq0_inner_held',
        targets='all_13_existing_fields_bit_and_enum_with_unknown_state_retained_no_threat_labels',
        feature='flatten_all_16_members_128_dims_no_mean_pool_canonical_min_local',
        baseline='fit_original_frequency_modal_field',selection='none_fixed_ridge_no_layer_or_field_selection',
        held_labels_used_for_fit=False,backbone_seen_inner_held_class_labels=True,new_classifier_parameters=0,
        new_classifier_updates=0,automatic_repeat=False,automatic_class_training=False,
        rationale='V149 exact input coordinates retain known behavior fields; V148 whole-hidden relation lacks task relevance. Diagnose readable field content before adding component supervision.',
        endpoint='single_fixed_ridge_normal_equation_solve_no_tuning',
        target_thresholds=dict(bits=.5,enum='argmax_if_max_ge_0.5_else_unknown'),
        scope='Probe reads frozen states. Inner-held roots unseen by decoder but seen by backbone classification; outer development previously inspected. Neither is a new blind test.',
        evaluation=['readout_FIT','inner_HELD','outer_HELD','all_original_known_and_unknown','correct_and_wrong_classifier_controls','canonical_seen_unseen','same_and_opposite_class_support'],
        causal_limits=['Linear decoder failure is not proof information absent or cannot be decoded nonlinearly.',
            'Field readability cannot create missing threat labels or turn port/role values into a threat rule.',
            'Canonical CSR-equivalent locals use fixed minimum-local feature representative; no float-cache variant as new input evidence.',
            'FACTS_DIRECT bypass stays in classifier, so hidden-field errors do not establish whole-classifier information loss.',
            'No target field, ridge, layer, sample, endpoint or classifier plan chosen using outer labels/errors.'],
        risk_actions=dict(original_frequency='count every occurrence in vector MSE, report all role rows including unknown/conflict/correct controls',
            old_training_budgets='no classifier fitting; old LP/V142/V146 budgets unchanged',
            no_false_generalization='backbone exposure explicit in every receipt',
            no_unearned_auxiliary='readout failure is bounded linear-readability evidence only; no automatic next classifier fit',
            actual_compute='count six learned multivariate factual decoders as six fits, not zero fits or six classifier trainings'),
        evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    target.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(registered_plan=True,max_factual_decoder_fits=6,classifier_fits=0),ensure_ascii=False))

if __name__=='__main__':main()
