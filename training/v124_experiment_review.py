"""Versioned V124 input-only review gates. Preparation never proves quality.

This copy leaves the historical experiment_review.py and its V121 seal intact.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class ReviewError(ValueError):
    pass


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_bindings(bindings):
    if not bindings:
        raise ReviewError('Evidence/source bindings are missing.')
    for rel, expected in bindings.items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise ReviewError('Missing or changed bound evidence: ' + rel)


def review_plan(plan):
    """Reject an input experiment disguised as the old batch experiment."""
    check_bindings(plan['evidence_sha256'])
    registry = read(ROOT / plan['risk_registry'])
    if plan['risk_registry'] not in plan['evidence_sha256']:
        raise ReviewError('The risk registry must be bound to this plan.')
    preflight = read(ROOT / plan['input_contract']['preflight_path'])
    expected_input = {
        'A': 'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz',
        'B': 'artifacts/v124_header_trial_20260929/B_header_ASA.npz',
        'preflight_path': 'artifacts/v124_header_trial_20260929/input_preflight.json',
        'factor': 'observed_ASA_sanitized_clock_header_only',
        'old_local_sampling_units_preserved': True,
        'B_re_deduplication': False,
        'event_body_and_495_metadata_facts_unchanged': True,
        'unknown_headers': 'fail_closed_no_guess'
    }
    issues = []
    rules = {
        'experiment_identity': plan.get('experiment_id') == 'V124_N1_ASA_header_only',
        'actual_classification_is_primary': plan.get('primary_endpoint') == 'per_class_predictions_on_complete_population',
        'matched_single_factor': plan.get('changed_factors') == ['ASA_sanitized_clock_header_input'] and plan.get('paired_fresh_baseline') is True,
        'input_contract': plan.get('input_contract') == expected_input,
        'preflight_682_body_facts_isolation': (
            preflight.get('status') == 'prepared_no_training' and preflight.get('all_ASA_rows') == 112807
            and preflight.get('changed_rows') == 682 and preflight.get('M_changed') == 44
            and preflight.get('S_changed') == 638 and preflight.get('changed_old_local_units') == 289
            and preflight.get('event_body_bytes_changed') == 0 and preflight.get('fact_metadata_nnz_changed') == 0
            and preflight.get('old_local_splits') == 0 and preflight.get('cross_fold_input_keys') == 0
            and preflight.get('empirical_collision_floor') == 28
            and preflight.get('all_original_rows_and_labels_retained') is True),
        'no_label_or_frequency_changes': plan.get('preserve_input_class_mass') is True,
        'no_outer_answer_selection': plan.get('selection_labels') == 'none_fixed_terminal',
        'explicit_development_scope': plan.get('evaluation_scope') == 'previously_inspected_development',
        'conditional_confirmation': plan.get('confirmation_condition') == 'all_primary_quality_gates_pass',
        'no_best_seed': plan.get('best_seed_selection') is False,
        'full_population_scoring': plan.get('complete_population_scoring') is True,
        'bounded_primary_before_replication': plan.get('primary_fits') == 6 and plan.get('confirmation_fits') == 12 and plan.get('max_fits') == 18,
        'risk_is_not_fold_override': plan.get('fold_specific_method_override') is False,
        'source_of_direction': plan.get('source_of_direction') == 'training/review_policy/v123_targeted_plan.json',
        'registered_input_hashes': all(
            plan.get('evidence_sha256', {}).get(p) == sha(ROOT/p)
            for p in (expected_input['A'],expected_input['B'],expected_input['preflight_path'])),
        'frozen_non_ASA_population': (plan.get('evaluation_contract',{}).get('raw_ASA_rows') == 112807
            and plan.get('evaluation_contract',{}).get('full_task_rows') == 2056871
            and plan.get('evaluation_contract',{}).get('current_teacher_excluded_ASA_rows') == 0
            and plan.get('evaluation_contract',{}).get('frozen_non_ASA_errors') == 107),
        'same_objective_architecture': plan.get('architecture') == {
            'input':'old_N1_vs_header_canonical_N1', 'model':'SparseTabM','members':16,
            'dtype':'float32','loss':'original_row_frequency_member_mean_three_class_CE'},
        'same_optimizer': plan.get('optimizer') == {
            'name':'AdamW','lr':.002,'weight_decay':.0003,'same_initial_state_per_pair':True},
        'same_uniform_old_local_batches': plan.get('batch_contract') == {
            'A':'uniform_old_local','B':'uniform_old_local','batch_size':256,
            'updates_per_logical_batch':1,'drop_last':False,'rng':'seed + fold initialized once per arm'},
        'registered_schedule': [q['registered_optimizer_steps_per_arm'] for q in plan.get('expected_primary_schedule',[])] == [1825,800,1825],
        'registered_checkpoints': plan.get('diagnostic_checkpoints') == [1,2,5,10,15,20,25],
    }
    selector = plan.get('selector', {})
    budget = plan.get('training_epochs')
    rules['budget_and_candidate_consistent'] = (
        isinstance(budget, int) and not isinstance(budget, bool) and budget > 0
        and selector.get('kind') == 'fixed_terminal' and selector.get('epoch') == budget
        and selector.get('tie_policy') == 'retain_registered_reference')
    monitor = set(plan.get('monitor', []))
    rules['learning_not_inferred_from_counts_alone'] = {
        'per_class_correct', 'per_class_checkpoint_CE', 'per_class_Brier',
        'fit_nonconflict_errors', 'support_by_class_and_parameter',
        'missing_parameter_rows', 'source_concentration', 'paired_repairs_regressions',
        'online_CE_separate_from_checkpoint_CE'} <= monitor
    for key, passed in rules.items():
        if not passed:
            issues.append(key)
    actions = plan.get('risk_actions', {})
    for risk in registry['risks']:
        if actions.get(risk['id']) != risk['action']:
            issues.append('unhandled_risk:' + risk['id'])
    profile = plan.get('quality_profile', {})
    if (not {'expected_full_rows', 'asa_error_limits', 'minimum_improved_folds',
             'protected_S_roots', 'required_full_classes'} <= set(profile)
            or profile.get('required_full_classes') != [0, 1, 2]):
        issues.append('missing_quality_profile')
    return {'status': 'eligible_for_bounded_trial' if not issues else 'blocked_before_trial',
            'plan_review_passed': not issues, 'violations': issues, 'rules': rules,
            'quality_acceptance': False, 'model_promoted': False,
            'scope': 'Design/evidence preconditions only. Trainer integration and actual classification remain required.'}


def seal_run(plan_path, trainer_path, source_paths, destination):
    """New runners must verify this immutable seal before any optimizer step."""
    plan_path, trainer_path, destination = map(Path, (plan_path, trainer_path, destination))
    if destination.exists():
        raise ReviewError('Never overwrite a run seal.')
    plan = read(plan_path)
    review = review_plan(plan)
    if not review['plan_review_passed']:
        raise ReviewError(str(review['violations']))
    if not trainer_path.is_file():
        raise ReviewError('Actual trainer source must exist before sealing execution.')
    files = [plan_path, trainer_path, Path(__file__)] + list(map(Path, source_paths))
    bindings = {}
    for p in files:
        if not p.is_file():
            raise ReviewError('Missing execution source: ' + str(p))
        bindings[str(p.resolve().relative_to(ROOT)).replace('\\', '/')] = sha(p)
    value = {'status': 'sealed_before_fit', 'plan_path': plan_path.resolve().relative_to(ROOT).as_posix(),
             'trainer_path': trainer_path.resolve().relative_to(ROOT).as_posix(),
             'source_sha256': bindings, 'quality_acceptance': False}
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def require_run_seal(seal_path, current_trainer):
    seal = read(seal_path)
    if seal.get('status') != 'sealed_before_fit':
        raise ReviewError('A registered execution seal is required.')
    if (ROOT / seal['trainer_path']).resolve() != Path(current_trainer).resolve():
        raise ReviewError('Seal belongs to a different trainer.')
    check_bindings(seal['source_sha256'])
    plan = read(ROOT / seal['plan_path'])
    if not review_plan(plan)['plan_review_passed']:
        raise ReviewError('Current plan no longer passes review.')
    return plan


def require_checkpoint(plan, metadata):
    epoch = plan['selector']['epoch']
    if (metadata.get('status') != 'fit_executed' or metadata.get('completed_epochs') != plan['training_epochs']
            or metadata.get('prediction_epoch') != epoch):
        raise ReviewError('Incomplete fit or unregistered checkpoint. Budget completion is not convergence.')


def align_predictions(reference, predictions):
    # Labels and route/group metadata come from the separately bound reference.
    needed = {'row_position', 'fold', 'root', 'truth', 'route'}
    if not needed <= set(reference.columns) or not {'row_position', 'pred_A', 'pred_B'} <= set(predictions.columns):
        raise ReviewError('Incomplete reference or prediction ledger.')
    if reference.row_position.duplicated().any() or predictions.row_position.duplicated().any():
        raise ReviewError('Duplicate row identities.')
    if len(reference) != len(predictions) or set(reference.row_position) != set(predictions.row_position):
        raise ReviewError('Missing or extra scored rows, including unscored collateral.')
    if not reference.truth.isin([0, 1, 2]).all() or not predictions[['pred_A', 'pred_B']].isin([0, 1, 2]).all().all():
        raise ReviewError('Invalid or absent labels/predictions.')
    if reference[list(needed)].isna().any().any():
        raise ReviewError('Reference identities and class labels must be complete.')
    if 'truth' in predictions:
        joined = reference[['row_position', 'truth']].merge(predictions[['row_position', 'truth']], on='row_position', validate='one_to_one')
        if not np.array_equal(joined.truth_x, joined.truth_y):
            raise ReviewError('Prediction ledger changed the authoritative labels.')
    return reference.merge(predictions[['row_position', 'pred_A', 'pred_B']], on='row_position', validate='one_to_one')


def class_counts(d, field):
    result = {}
    for c in (0, 1, 2):
        actual, predicted = d.truth.eq(c), d[field].eq(c)
        n = int(actual.sum()); tp = int((actual & predicted).sum()); fp = int((~actual & predicted).sum())
        denom = n + int(predicted.sum())
        result[str(c)] = {'support': n, 'correct': tp, 'missed': n - tp, 'false_called': fp,
                          'recall': tp / n if n else None,
                          'precision': tp / int(predicted.sum()) if predicted.any() else None,
                          'f1': 2 * tp / denom if n and denom else None}
    return result


def evaluate_primary(reference, predictions, profile):
    d = align_predictions(reference, predictions)
    asa = d[d.route.eq('asa') & d.truth.isin([1, 2])]
    byclass = {name: class_counts(asa, field) for name, field in [('A', 'pred_A'), ('B', 'pred_B')]}
    full = {name: class_counts(d, field) for name, field in [('A', 'pred_A'), ('B', 'pred_B')]}
    changes = {}
    for c in (1, 2):
        q = asa[asa.truth.eq(c)]
        changes[str(c)] = {'repaired': int(((q.pred_A != c) & (q.pred_B == c)).sum()),
                           'regressed': int(((q.pred_A == c) & (q.pred_B != c)).sum())}
    folds = [{'fold': int(k), 'A_errors': int((q.pred_A != q.truth).sum()),
              'B_errors': int((q.pred_B != q.truth).sum())} for k, q in asa.groupby('fold')]
    limits = profile['asa_error_limits']
    outside = asa[asa.truth.eq(2) & ~asa.root.isin(profile['protected_S_roots'])]
    gates = {
        'complete_full_population': len(d) == profile['expected_full_rows'],
        'all_required_classes_present': all(full['A'][str(c)]['support'] > 0 for c in profile['required_full_classes']),
        'M_S_present_in_ASA': all(byclass['A'][str(c)]['support'] > 0 for c in (1, 2)),
        'ASA_M_absolute': byclass['B']['1']['missed'] <= limits['M'],
        'ASA_S_absolute': byclass['B']['2']['missed'] <= limits['S'],
        'ASA_total_absolute': int((asa.pred_B != asa.truth).sum()) <= limits['total'],
        'paired_M_S_protected': all(byclass['B'][str(c)]['missed'] <= byclass['A'][str(c)]['missed'] for c in (1, 2)),
        'multiple_folds_improve': sum(q['B_errors'] < q['A_errors'] for q in folds) >= profile['minimum_improved_folds'],
        'outside_largest_S_groups_protected': bool(len(outside) and (outside.pred_B == 2).sum() >= (outside.pred_A == 2).sum()),
        'full_class_recall_F1_protected': all(
            full['B'][str(c)][metric] is not None and full['A'][str(c)][metric] is not None
            and full['B'][str(c)][metric] >= full['A'][str(c)][metric]
            for c in profile['required_full_classes'] for metric in ('recall', 'f1')),
    }
    return {'stage': 'primary_quality_only', 'ASA': byclass, 'full_task': full, 'paired_changes': changes,
            'folds': folds, 'gates': {k: bool(v) for k, v in gates.items()},
            'primary_quality_passed': all(gates.values()), 'model_promoted': False,
            'limits': 'Predictions are recounted. Source/model replay and seed confirmation are separate required gates.'}


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('plan'); p.add_argument('--plan', required=True)
    p = sub.add_parser('seal'); p.add_argument('--plan', required=True); p.add_argument('--trainer', required=True)
    p.add_argument('--source', action='append', default=[]); p.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.stage == 'plan':
        result = review_plan(read(args.plan)); print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result['plan_review_passed']: raise SystemExit(2)
    else:
        seal_run(args.plan, args.trainer, args.source, args.out)


if __name__ == '__main__':
    main()
