"""Design consistency checks only; this is not a training entry or run seal."""
import copy
import json
from pathlib import Path
from experiment_review import ROOT, check_bindings, read

PLAN = ROOT / 'training/review_policy/v139_source_regression_plan.json'


def validate(plan):
    def need(condition, reason):
        if not condition:
            raise ValueError(reason)

    need(plan['latest_actual_training'] == 'V138' and
         (plan['new_fits'], plan['new_updates']) == (0, 0), 'Design cannot be actual training')
    decision = plan['decision']
    need(not decision['old_C_margin_round2_eligible'] and not decision['old_issue_closed'],
         'No fabricated feasibility or closed training issue')
    budget = plan['budget']
    need(budget['old_fits_consumed'] == 6 and budget['new_primary_fits_max'] == 6 and
         budget['cumulative_chain_fits_max'] == 12 and budget['cumulative_chain_rounds_max'] == 2,
         'No budget reset or extension')
    need(budget['folds'] == [0, 1, 2] and budget['arms'] == ['C', 'P'], 'No fold or arm cherry-picking')
    need(plan['arms']['P']['start'] == plan['arms']['C']['start'] and
         plan['arms']['P']['anchor'] == 'same fold V135 R_decay actual readout',
         'Matched starting point and registered anchor required')
    need(plan['arms']['P']['anchor_penalty'] and not plan['arms']['C']['anchor_penalty'],
         'Candidate and control must differ in anchor penalty')
    need(not plan['scope']['HELD_in_gradient_weights_or_lambda'] and
         not plan['objective']['lambda_grid_or_outer_selection'], 'No held-label training or outer tuning')
    need(plan['scope']['legal_TRAIN_original_frequencies_only'] and
         plan['scope']['backbone_inputs_and_other_formats_frozen'], 'No scope or weighting confound')
    need(plan['retention']['registered_new_errors_max'] == 0 and
         plan['retention']['guard_rows'] == 40879 and
         plan['retention']['all_H_L_original_correct_pure_TRAIN_rows_and_64_repairs_replayed'],
         'Do not discard the verified repair scope')
    old = read(ROOT / 'training/review_policy/v137_single_issue_plan.json')
    need(plan['task_adoption_quality'] == old['task_adoption_quality'], 'Task quality gate cannot be weakened')
    need(plan['honesty']['already_inspected_development_folds'] and
         not plan['honesty']['independent_blind_or_external_test_executed'], 'No invented blind test')
    need(not plan['runtime_readiness']['trainer_implemented'] and
         not plan['runtime_readiness']['run_seal_created'], 'Design review is not execution readiness')
    return True


def review():
    plan = read(PLAN)
    validate(plan)
    check_bindings(plan['evidence_sha256'])
    mutations = [
        ('held-label fitting', ('scope', 'HELD_in_gradient_weights_or_lambda'), True),
        ('outer lambda tuning', ('objective', 'lambda_grid_or_outer_selection'), True),
        ('budget reset', ('budget', 'old_fits_consumed'), 0),
        ('extra fits', ('budget', 'new_primary_fits_max'), 12),
        ('closed unresolved issue', ('decision', 'old_issue_closed'), True),
        ('fabricated certificate', ('decision', 'old_C_margin_round2_eligible'), True),
        ('discard repair protection', ('retention', 'registered_new_errors_max'), 1),
        ('weaker task M gate', ('task_adoption_quality', 'ASA_M_errors_max'), 1748),
        ('wrong anchor', ('arms', 'P', 'anchor'), 'V138 H_L'),
        ('different control start', ('arms', 'C', 'start'), 'V135 R_decay'),
        ('design reported as training', ('new_fits',), 6),
        ('inspected folds as blind', ('honesty', 'independent_blind_or_external_test_executed'), True),
    ]
    results = []
    for name, keys, value in mutations:
        bad = copy.deepcopy(plan)
        cursor = bad
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
        try:
            validate(bad)
        except ValueError as exc:
            results.append({'case': name, 'rejected': True, 'reason': str(exc)})
        else:
            raise RuntimeError('Historical risk not rejected: ' + name)
    return {'status': 'design_consistency_passed', 'evidence_bindings_checked': True,
            'negative_cases': results, 'new_fits': 0, 'new_updates': 0,
            'not_runtime_or_model_quality_acceptance': True}


if __name__ == '__main__':
    print(json.dumps(review(), ensure_ascii=False, indent=2))
