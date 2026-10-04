"""Create the one current plan profile and evidence-backed historical risk registry."""
import json
from pathlib import Path
from experiment_review import ROOT, sha, review_plan

POLICY = ROOT / 'training/review_policy'
DEST = ROOT / 'artifacts/v119_review_constraints_20260929'


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    assert not POLICY.exists() and not DEST.exists(), 'Do not silently replace a sealed review.'
    POLICY.mkdir(); DEST.mkdir()
    cases = [
        ('EARLY_SELECTION', 'V116', 'Relative ties on weak K/U metrics selected epoch15 although all trajectories completed25; S regression2542.',
         'fixed_terminal_no_tie_early', ['training/test_v116.py', 'artifacts/v116_nested_selection_20260929/locked_selection.json']),
        ('OMITTED_POPULATION', 'V116', 'Missing parameters were absent from K/U selection; fold1 collateral6936 rows omitted from those scores.',
         'score_every_registered_row', ['artifacts/v116_nested_selection_20260929/split_policy.json', 'artifacts/v117_selection_failure_review_20260929/audit.json']),
        ('WEAK_SUPPORT', 'V117', 'Known ICMP3/13 parameter had180 M and0 S in inner fit but685 S in K; parameter presence is not class support.',
         'report_class_parameter_support_no_stop_authority', ['artifacts/v117_selection_failure_review_20260929/audit.json']),
        ('LOSS_NOT_TRANSFER', 'V81,V110', 'Internal fit or source validation can improve while broader held-out classification deteriorates.',
         'training_loss_diagnostic_only', ['docs/V81_ROOT_CAUSE_AND_TRAINING_RESET.md', 'docs/V110_LAYER_PROBE_TRAINING_RESULTS_AND_STOP_DECISION.md']),
        ('CLASS_TRADEOFF', 'V104,V113', 'More epochs or lower total error can increase malicious misses; V113 M318->370 despite total2412->2190.',
         'protect_each_class_from_raw_predictions', ['docs/V104_OFFICIAL_COVERAGE_AND_INDEPENDENT_MODEL_TRAINING.md', 'artifacts/v113_case_training_20260929/OOF_ASA_decisions.parquet']),
        ('SOURCE_CONCENTRATION', 'V113', 'S gains in one large group did not establish wider source improvements.',
         'multiple_folds_and_outside_largest_groups', ['docs/V113_CASE_TRAINING_RESULTS_AND_STOP.md']),
        ('NUMERIC_AND_REFERENCE', 'V116,V117', 'Historical versus fresh baseline changed364 predictions; determinism flags did not achieve exact repeats.',
         'fresh_paired_baseline_and_conditional_seed_confirmation', ['artifacts/v116_nested_selection_20260929/baseline_drift_audit.json', 'artifacts/v117_selection_failure_review_20260929/frozen_member_audit.json']),
        ('LOCAL_MECHANISM', 'V117,V118', 'Batch-gradient reduction at one state did not generalize: fold2 ratio1.379507.',
         'bounded_primary_trial_without_fold_override', ['artifacts/v118_retrospective_and_training_contract_20260929/probe_results.json']),
        ('POSTHOC_RULE_CHANGE', 'V116,V118', 'Changing a checkpoint rule after inspected outer errors would invalidate claimed unseen model selection.',
         'seal_before_fit_reject_changed_sources', ['artifacts/v118_retrospective_and_training_contract_20260929/training_contract.json']),
    ]
    risks = []
    for ident, rounds, observation, action, files in cases:
        risks.append({'id': ident, 'rounds': rounds, 'observed_failure': observation,
                      'required_action': action, 'status': 'historical_counterexample_retained',
                      'evidence_sha256': {p: sha(ROOT / p) for p in files}})
    registry = POLICY / 'history_cases.json'
    save(registry, {'schema_version': 1, 'scope': 'SF02 current N1 batch experiment; not universal model-training law.', 'risks': risks})
    sources = {k: v for risk in risks for k, v in risk['evidence_sha256'].items()}
    sources.update({registry.relative_to(ROOT).as_posix(): sha(registry),
                    'training/experiment_review.py': sha(ROOT / 'training/experiment_review.py'),
                    'docs/EXPERIMENT_REVIEW_RULES.md': sha(ROOT / 'docs/EXPERIMENT_REVIEW_RULES.md')})
    plan = {'schema_version': 1, 'experiment_id': 'next_N1_mass_stratified_batches',
        'status': 'reviewed_plan_not_training_registration',
        'risk_registry': registry.relative_to(ROOT).as_posix(), 'evidence_sha256': sources,
        'primary_endpoint': 'per_class_predictions_on_complete_population',
        'changed_factors': ['batch_organization'], 'paired_fresh_baseline': True,
        'preserve_input_class_mass': True, 'selection_labels': 'none_fixed_terminal',
        'evaluation_scope': 'previously_inspected_development',
        'confirmation_condition': 'all_primary_quality_gates_pass', 'best_seed_selection': False,
        'primary_fits': 6, 'confirmation_fits': 12, 'max_fits': 18,
        'primary_seed': 10201, 'confirmation_seeds': [10202, 10203], 'folds': [0, 1, 2],
        'fold_specific_method_override': False, 'complete_population_scoring': True,
        'training_epochs': 25, 'selector': {'kind': 'fixed_terminal', 'epoch': 25, 'tie_policy': 'retain_registered_reference'},
        'monitor': ['per_class_correct', 'per_class_checkpoint_CE', 'per_class_Brier', 'fit_nonconflict_errors',
                    'support_by_class_and_parameter', 'missing_parameter_rows', 'source_concentration',
                    'paired_repairs_regressions', 'online_CE_separate_from_checkpoint_CE'],
        'risk_actions': {r['id']: r['required_action'] for r in risks},
        'quality_profile': {'expected_full_rows': 2056871, 'asa_error_limits': {'M': 318, 'S': 2094, 'total': 2170},
                            'minimum_improved_folds': 2, 'protected_S_roots': [21702, 20849, 29],
                            'required_full_classes': [0, 1, 2]},
        'execution_limits': ['No new classifier fitted by this review.', 'A real trainer and all its data/code dependencies must be sealed before use.',
                             '25 epochs and error budgets apply to this plan only; revised policies need new evidence and replay before outcome inspection.']}
    save(POLICY / 'next_batch_plan.json', plan)
    review = review_plan(plan)
    assert review['plan_review_passed'], review
    save(DEST / 'plan_review.json', review)
    print(json.dumps({'risk_cases': len(risks), 'plan_status': review['status'], 'quality_acceptance': False, 'new_fits': 0}))


if __name__ == '__main__':
    main()
