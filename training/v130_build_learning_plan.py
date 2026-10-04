"""Publish the focused learning plan and a TRAIN-only 64-input probe. No fit."""
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v130_learning_review_20260930_r2'
PLAN = ROOT/'training/review_policy/v130_learning_qualification_plan.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    panel = pd.read_parquet(OUT/'fold1_train_only_panel.parquet')
    assert len(panel) == 871 and panel.single_class_canonical_fit_input.all()
    assert panel.truth.value_counts().to_dict() == {2: 527, 1: 344}
    assert panel.canonical_key.nunique() == 314
    # Select by input hash, including both labels from every paired body key.
    # Labels/old predictions are only those of legal outer TRAIN rows.
    by_input = panel.groupby(['canonical_key', 'truth'], as_index=False).agg(
        local=('local', 'min'), body_key=('body_key', 'first'), original_row_mass=('row_position', 'size'))
    assert not by_input.canonical_key.duplicated().any()
    paired = panel.groupby('body_key').truth.nunique()
    paired = set(paired.index[paired == 2])
    chosen = set()
    for body in sorted(paired):
        for cl in [1, 2]:
            chosen.add(by_input[(by_input.body_key == body) & (by_input.truth == cl)].canonical_key.min())
    for cl in [1, 2]:
        picked = int(by_input[(by_input.truth == cl) & by_input.canonical_key.isin(chosen)].shape[0])
        candidates = by_input[(by_input.truth == cl) & ~by_input.canonical_key.isin(chosen)].sort_values('canonical_key')
        chosen.update(candidates.head(32-picked).canonical_key)
    probe = by_input[by_input.canonical_key.isin(chosen)].sort_values(['truth', 'canonical_key']).copy()
    assert len(probe) == 64 and probe.truth.value_counts().to_dict() == {1: 32, 2: 32}
    probe.to_parquet(OUT/'tiny64_train_only_inputs.parquet', index=False)
    ledger = pd.read_parquet(OUT/'training_role_error_ledger.parquet')
    schedules = []
    loss_counterexamples = []
    for fold in range(3):
        fit = ledger[ledger.outer_fit_role == fold]
        schedules.append({'fold': fold, 'original_fit_rows': len(fit), 'M': int((fit.truth == 1).sum()),
            'S': int((fit.truth == 2).sum()), 'old_local_batch_units': int(fit.local.nunique()),
            'updates_per_100_epoch_fit': math.ceil(fit.local.nunique()/256)*100})
        mixed = fit[fit.canonical_key_mixed].groupby(['canonical_key', 'truth']).size().unstack(fill_value=0)
        nm, ns = int((fit.truth == 1).sum()), int((fit.truth == 2).sum())
        for key, c in mixed.iterrows():
            weighted = [c[1]/nm, c[2]/ns]
            loss_counterexamples.append({'fold': fold, 'canonical_key': key,
                'M_rows': int(c[1]), 'S_rows': int(c[2]),
                'unrestricted_row_CE_optimal_class': 1 if c[1] > c[2] else 2,
                'unrestricted_full_class_CE_optimal_class': 1 if weighted[0] > weighted[1] else 2,
                'class_CE_optimal_S_probability': weighted[1]/sum(weighted),
                'scope': 'Analytical independent-input optimum; not a fitted network or proof of the actual network optimum.'})
    counter = next(x for x in loss_counterexamples if x['fold'] == 1)
    assert (counter['M_rows'], counter['S_rows'], counter['unrestricted_full_class_CE_optimal_class']) == (72, 6, 2)
    (OUT/'loss_collision_counterexamples.json').write_text(
        json.dumps(loss_counterexamples, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    hard = panel[panel.truth == 2]
    plan = {
        'version': 'V130-learning-qualification', 'status': 'designed_not_trained',
        'scope': 'Solve the observed training-side classification gap first; no automatic promotion.',
        'latest_actual_delivery': 'artifacts/v128_nested_score_trial_20260929_r3/delivery.json',
        'supersedes_direction': 'V129 R/D/X transfer trial deferred, evidence retained.',
        'input': {'canonical_full_sparse': 'artifacts/v124_header_trial_20260929/B_header_ASA.npz',
                  'shape': [22546, 66287], 'retain_all_legal_features': True,
                  'ordered_body_reserved': 'No new body-only branch in this qualification trial.',
                  'forbidden_features': ['row_position', 'local', 'root', 'fold', 'truth', 'teacher_role', 'old_prediction']},
        'labels': {'source': 'data/official/train.parquet', 'output_classes': ['benign', 'malicious', 'suspicious'],
                   'M_S_training': 'All legal ASA outer TRAIN rows, unchanged official labels.',
                   'mixed_inputs': 'Retain in full fit/evaluation; separately report empirical collision error floor.'},
        'model': {'base': 'Custom SparseTabM adapter, not an exact reproduction of the paper.',
                  'members': 16, 'small_hidden': 128, 'wide_hidden': 256,
                  'full_end_to_end_supervision': True, 'teacher_logit_or_distillation': False,
                  'initialization': {'shared_seed': 10201, 'extra_seed': 13017,
                      'R_O': 'Identical state.', 'C_CO': 'Identical state.',
                      'small_to_wide': 'Copy common 128 blocks, factors and facts head; random new channels; zero new-to-common second-layer weights and extra head columns. Initial logits must match within 1e-6; do not initialize every extra weight to zero.',
                      'delayed_new_channel_gradient': 'Expected at update zero; inspect again after updates 1, 10 and 100.'}},
        'loss': {'row_CE': '(1/N) sum_i,c n_ic mean_member CE(z_i,c)',
                 'pure_class_CE': '0.5 * sum_c in {M,S} [sum_i in single-label canonical TRAIN inputs n_ic mean_member CE(z_i,c) / N_pure_c]',
                 'focused_CE': '0.5 * row_CE + 0.5 * pure_class_CE',
                 'accounting': 'Full-epoch original row/class mass conservation. Denominators fixed to whole legal fit population; unbiased batch scaling by number of logical batches. No per-batch class renormalization.',
                 'member_training': 'Mean of individual member losses, never CE of averaged predictions.',
                 'inference': 'Argmax of mean member probabilities; no threshold tuning.',
                 'mixed_safeguard': 'Mixed inputs appear in row_CE with unchanged relative original label masses, and receive no pure-class auxiliary loss. Do not add class weights to mixed labels.',
                 'weighting_claim': 'The auxiliary pure-class risk is a bounded diagnostic factor, not established debiasing or a deployable solution.',
                 'rejected_blind_class_balance': 'Fold1 one numerical input has 72M/6S. Unrestricted full class-normalized CE favors S and 72 M errors; this contradicts the zero-M-error TRAIN gate. Retain as an executable counterexample.'},
        'tiny_probe': {'inputs': 'artifacts/v130_learning_review_20260930_r2/tiny64_train_only_inputs.parquet',
            'population': '64 distinct single-label canonical TRAIN inputs: 32 M, 32 hard S, all six paired-body keys represented.',
            'models': ['R128', 'C256'], 'fits': 2, 'fixed_updates_per_fit': 2000,
            'optimizer': {'name': 'AdamW', 'lr': 0.002, 'weight_decay': 0.0},
            'batch': 'Same full 64-input batch every update, original selected row masses retained.',
            'checkpoint_updates': [0, 1, 10, 100, 500, 1000, 2000],
            'endpoint_gate': {'M_errors': 0, 'S_errors': 0, 'worst_ensemble_truth_probability_min': 0.9},
            'failure_action': 'If both fail, stop full trials and inspect exact dense-vs-sparse forward/gradient, target alignment, input resolution and activation/update paths. Failure does not prove labels or task are impossible.',
            'acceptance_boundary': 'Memorization/pipeline qualification only; discard probe model, reset before full fitting.'},
        'primary_trial': {'fold': 1, 'factorial': [
                {'arm': 'R', 'hidden': 128, 'loss': 'row_CE'},
                {'arm': 'O', 'hidden': 128, 'loss': 'focused_CE'},
                {'arm': 'C', 'hidden': 256, 'loss': 'row_CE'},
                {'arm': 'CO', 'hidden': 256, 'loss': 'focused_CE'}],
            'fit_order': ['R', 'O', 'C', 'CO'], 'fits': 4, 'fixed_epochs': 100,
            'optimizer': {'name': 'AdamW', 'lr': 0.002, 'weight_decay': 0.0003},
            'schedule': 'Constant, identical to the old direct-model learning rule; fixed longer budget isolated by R25/R100 trajectory. No new LR schedule in this factorial.',
            'unique_local_batch': 256, 'permutation_seed_base': 10201, 'permutation_seed_rule': '10201 + fold',
            'fit_original_rows': 40957, 'fit_M': 38886, 'fit_S': 2071,
            'checkpoints_epochs': [0, 1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100],
            'qualify_epoch': 100, 'early_stop': False,
            'explanation': '100 epochs is a predeclared diagnostic budget, not proven sufficiency. Earlier results show trajectories; they cannot replace the registered endpoint.'},
        'learning_gate': {'full_train_M_errors_max': 0, 'full_train_S_errors_max': 32,
            'canonical_nonconflict_S_errors_max': 26, 'hard_527_S_errors_max': 26,
            'matched_344_M_errors_max': 0,
            'new_errors_on_B124_correct_train_S_max': 0,
            'hard_S_source_groups_with_errors_max': math.floor(hard.root.nunique()*0.2),
            'old_hard_S_source_groups': int(hard.root.nunique()),
            'meaning': 'At least 95% of the 527 observed pure-input S mistakes repaired while every legal TRAIN M remains correct. The 6-row empirical collision floor is retained in scoring, not relabeled.',
            'claim': 'A new project target, not an achieved result or a guarantee from research.'},
        'selection': {'authority': 'TRAIN qualification only, before reading any newly generated held results.',
            'fixed_preference': ['R', 'C', 'O', 'CO'],
            'rule': 'First eligible arm at epoch100; no held-driven choice among arms/epochs/seeds.',
            'none_pass': 'Stop: training-side classification remains unsolved. Do not run a transfer confirmation.',
            'loss_only_improves': 'Record probability learning but mark classification learning failed.',
            'safety': 'New held-output files are sealed and remain unread until the TRAIN-only choice is recorded.'},
        'record': {'every_epoch': ['M/S original row mass seen', 'original-row and class-normalized CE', 'M/S errors', 'single-class-input errors', '527S/344M repair-regression', 'per-source training errors'],
            'checkpoints': ['model hash', 'per-row probabilities', 'per-member no-correct-member counts', 'truth-vs-other margins', 'activation sparsity', 'module gradient/update norms'],
            'gradient_diagnosis': 'At epochs0/25/50/100 compute whole-fit M and S parameter gradients, original-weight and pure-class-normalized module norms/cosines, without optimizer updates. Negative cosine is a local observation, not proof that gradient surgery will help.',
            'all_errors': 'Repaired, persistent, newly wrong and mixed-input rows remain in distinct ledgers.'},
        'confirmation': {'automatic_model_promotion': False,
            'on_learning_pass': 'Run predeclared same selected configuration in folds0/2 with matched R100 controls if selected arm differs from R. Do not alter method by fold.',
            'additional_fits': {'if_R_selected': 2, 'otherwise': 4},
            'quality_eligibility': 'Must pass preserved full original-row/per-class/source protections before later independent seed or official blind confirmation.',
            'learning_vs_transfer': 'Improved TRAIN classification with poor held classification solves only learning fit, not support coverage or stable transfer.'},
        'preserved_full_quality_gate': {'official_original_rows': 2056871, 'ASA_rows': 112807,
            'ASA_M_errors_max': 318, 'ASA_S_errors_max': 2074, 'ASA_total_errors_max': 2170,
            'all_classes_precision_recall_F1_at_least_A0': True,
            'S_source_mean_recall_min': 0.1151252713, 'S_zero_recall_roots_max': 198,
            'S_errors_outside_top3_max_exclusive': 1373, 'improved_outer_folds_min': 2,
            'root2868_M_errors_max': 48, 'header682_errors_max': 0,
            'unknown186_S_errors_max': 184, 'non_ASA_frozen_error_count': 107},
        'historical_constraints': [
            'V116: never qualify an early checkpoint instead of the registered endpoint.',
            'V118/V121: no automatic reuse of class-mass batch rearrangement; it worsened real classification.',
            'V104/V113/V128: an S gain bought with M damage is not acceptance.',
            'V124-V128: no-correct-member, full actual-input conflicts and body conflicts must be separate.',
            'V129: facts-only projection loses discriminative information; no deletion of legal full input.',
            'V62 and later weighting controls: balancing may just move the boundary; require simultaneous M/S classification gates.'],
        'fit_budget': {'this_review_fits': 0, 'this_review_updates': 0, 'primary_with_probes_fits_max': 6,
                      'primary_with_probes_updates_max': 16800, 'through_development_confirmation_fits_max': 10,
                      'through_development_confirmation_updates_max': 46000},
        'schedules': schedules,
        'runtime_readiness': {'trainer_implemented': False, 'run_seal_created': False,
            'required_before_updates': ['V130-specific checker and historical counterexample replay',
                'seal_run/check_bindings of actual entry, all imported sources, views, independent labels, splits and panels',
                'exact 128/wide initial output comparison', 'independent full-row evaluator',
                'audit late checkpoint identities and cumulative original class mass']},
        'evidence_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            OUT/'audit.json', OUT/'training_role_error_ledger.parquet', OUT/'fold1_train_only_panel.parquet',
            OUT/'tiny64_train_only_inputs.parquet', OUT/'learning_trajectory.json',
            OUT/'loss_collision_counterexamples.json', Path(__file__)]}
    }
    assert schedules[1]['updates_per_100_epoch_fit'] == 3200
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': plan['status'], 'hard_S_groups': hard.root.nunique(),
        'tiny64_original_row_mass': probe.groupby('truth').original_row_mass.sum().to_dict(),
        'plan_sha256': sha(PLAN), 'schedules': schedules}, ensure_ascii=False))


if __name__ == '__main__':
    main()
