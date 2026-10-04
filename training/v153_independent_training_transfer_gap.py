"""Bind actual TRAIN/outer probability ledgers and narrower factual controls.

No model imports, forwards, fits, gradients or new classification rules.
All probabilities are existing V146 receipts, not new unseen validation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v153_independent_training_transfer_gap_20261001'
TRAIN = ROOT / 'artifacts/v146_guarded_pair_training_20261001'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
KNOWN = ROOT / 'artifacts/v147_independent_conditional_support_20261001/all_known_port_conditional_support.parquet'
FACTS = ROOT / 'artifacts/v149_independent_input_fidelity_20261001/all_ASA_input_decoding_ledger.parquet'
LADDER = ROOT / 'artifacts/v123_targeted_plan_20260929/support_ladder.parquet'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def quantiles(values):
    return {str(k): float(v) for k, v in values.quantile([0, .1, .5, .9, 1]).items()}


def main():
    assert not OUT.exists(), 'Preserve executed evidence.'
    data = pd.read_parquet(TRACE).sort_values('row_position').reset_index(drop=True)
    frozen = pd.read_parquet(TRAIN / 'ASA_prediction_ledger.parquet').sort_values('row_position').reset_index(drop=True)
    for key in ['row_position', 'root', 'local', 'truth', 'fold']:
        assert np.array_equal(data[key], frozen[key])
    assert len(data) == 112807 and data.row_position.is_unique
    data['canonical_key'] = frozen.canonical_key
    known = pd.read_parquet(KNOWN).set_index('row_position')
    data['both_ports_observed'] = data.row_position.isin(known.index)
    data['known_578_cohort'] = data.row_position.map(known.known_578_cohort).eq(True)
    assert int(data.known_578_cohort.sum()) == 578
    f = data.facts_json.map(json.loads)
    family_fields = ['action', 'outcome', 'transport_protocol', 'src_role', 'dst_role',
                     'src_port_range', 'dst_port_range']
    data['known_port_family'] = f.map(lambda x: json.dumps([x.get(k) for k in family_fields], separators=(',', ':')))
    historical = pd.read_parquet(LADDER).set_index('row_position')
    assert set(data.row_position) == set(historical.index)
    for key in ['root', 'fold', 'truth']:
        assert np.array_equal(data[key], historical.loc[data.row_position, key])
    for key in ['parameter_observed', 'exact_M_rows', 'exact_M_roots', 'exact_S_rows', 'exact_S_roots',
                'destination_M_rows', 'destination_M_roots', 'destination_S_rows', 'destination_S_roots',
                'behavior_M_rows', 'behavior_M_roots', 'behavior_S_rows', 'behavior_S_roots']:
        data['historical_support_' + key] = historical.loc[data.row_position, key].to_numpy()
    evidence_paths = [Path(__file__), TRACE, KNOWN, FACTS, LADDER,
                      TRAIN / 'ASA_prediction_ledger.parquet', TRAIN / 'run_seal.json',
                      TRAIN / 'verification.json', TRAIN / 'final_delivery.json']
    role_panels, receipts, summaries = [], [], []
    for arm in ['A', 'B']:
        role_path = TRAIN / (arm + '_training_role_ledger.parquet')
        evidence_paths.append(role_path)
        train_rows = pd.read_parquet(role_path)
        assert len(train_rows) == 225614
        assert not train_rows.duplicated(['row_position', 'training_role']).any()
        assert train_rows.groupby('row_position').size().eq(2).all()
        refs = data.set_index('row_position')
        for key in ['root', 'local', 'truth', 'canonical_key']:
            assert np.array_equal(train_rows[key], refs.loc[train_rows.row_position, key])
        held_probs = np.empty((len(data), 3), dtype=np.float64)
        for fold in range(3):
            folder = TRAIN / f'fold{fold}_{arm}'
            fit_path = folder / 'fit.json'
            fit = read(fit_path)
            p_path = folder / 'sealed_all_prob.npy'
            model_path = folder / 'endpoint.pt'
            assert sha(p_path) == fit['probability_sha256']
            assert sha(model_path) == fit['model_sha256']
            assert fit['fold'] == fold and fit['arm'] == arm and not fit['selected_by_score']
            assert fit['seal_sha256'] == sha(TRAIN / 'run_seal.json')
            probs = np.load(p_path)
            assert probs.shape == (22546, 3) and np.isfinite(probs).all()
            assert np.allclose(probs.sum(1), 1, atol=2e-6, rtol=0)
            rows = train_rows.loc[train_rows.training_role.eq(fold)]
            assert refs.loc[rows.row_position, 'fold'].ne(fold).all()
            assert set(rows.row_position) == set(data.loc[data.fold.ne(fold), 'row_position'])
            assert not set(rows.root) & set(data.loc[data.fold.eq(fold), 'root'])
            assert np.array_equal(rows[['p0', 'p1', 'p2']].to_numpy(), probs[rows.local])
            assert np.array_equal(rows.pred, probs[rows.local].argmax(1))
            held = data.fold.eq(fold)
            held_probs[held] = probs[data.loc[held, 'local']]
            evidence_paths += [fit_path, p_path, model_path]
            receipts.append(dict(arm=arm, fold=fold, training_role_rows=len(rows),
                source_disjoint=True, actual_probability_file_and_checkpoint_sha_verified=True,
                saved_TRAIN_predictions_and_probabilities_equal=True))
        assert np.array_equal(held_probs.argmax(1), frozen['pred_' + arm])
        for cls in range(3):
            data[f'outer_{arm}_p{cls}'] = held_probs[:, cls]
        data[f'outer_{arm}_pred'] = held_probs.argmax(1)
        panel = train_rows.merge(data[['row_position', 'fold', 'known_578_cohort', 'both_ports_observed',
            'known_port_family', f'outer_{arm}_pred', f'outer_{arm}_p0', f'outer_{arm}_p1', f'outer_{arm}_p2']],
            on='row_position', how='left', validate='many_to_one')
        panel['actual_TRAIN_correct'] = panel.pred.eq(panel.truth)
        panel['historical_outer_correct'] = panel[f'outer_{arm}_pred'].eq(panel.truth)
        assert not panel.fold.eq(panel.training_role).any()
        for cohort, selected in [('all_original_ASA', np.ones(len(panel), bool)),
                                  ('known_578', panel.known_578_cohort)]:
            group = panel.loc[selected]
            summaries.append(dict(arm=arm, cohort=cohort, TRAIN_role_rows=len(group),
                unique_original_rows=int(group.row_position.nunique()),
                pure_TRAIN_role_rows=int(group.pure_TRAIN_input.sum()),
                pure_TRAIN_errors=int((group.pure_TRAIN_input & ~group.actual_TRAIN_correct).sum()),
                all_TRAIN_role_errors=int((~group.actual_TRAIN_correct).sum()),
                historical_outer_unique_errors=int(group.loc[~group.historical_outer_correct, 'row_position'].nunique()),
                TRAIN_S_probability=quantiles(group.loc[group.truth.eq(2), 'p2'])))
        role_panels.append(panel)
    hard_families = set(data.loc[data.known_578_cohort, 'known_port_family'])
    hard_family_folds = set(zip(data.loc[data.known_578_cohort, 'known_port_family'],
                                data.loc[data.known_578_cohort, 'fold']))
    data['correct_known_port_other_S'] = data.truth.eq(2) & data.both_ports_observed & ~data.known_578_cohort & data.outer_B_pred.eq(2)
    data['same_family_control_S'] = data.correct_known_port_other_S & data.known_port_family.isin(hard_families)
    data['same_family_and_outer_fold_control_S'] = data.correct_known_port_other_S & pd.Series(
        [(family, fold) in hard_family_folds for family, fold in zip(data.known_port_family, data.fold)])
    family_summary = data.loc[data.truth.eq(2) & data.both_ports_observed].groupby(
        ['known_port_family', 'fold', 'known_578_cohort']).agg(original_rows=('row_position', 'size'),
            roots=('root', 'nunique'), A_errors=('outer_A_pred', lambda s: int(s.ne(2).sum())),
            B_errors=('outer_B_pred', lambda s: int(s.ne(2).sum()))).reset_index()
    OUT.mkdir()
    data.to_parquet(OUT / 'all_original_classifier_gap_and_control_ledger.parquet', index=False)
    pd.concat(role_panels, ignore_index=True).to_parquet(OUT / 'both_arms_legal_TRAIN_outer_control.parquet', index=False)
    family_summary.to_parquet(OUT / 'same_known_port_family_profiles.parquet', index=False)
    result = dict(status='actual_training_mastery_transfer_gap_and_narrow_controls_bound',
        latest_actual_classifier='V146', own_model_forwards=0, own_gradients=0, own_fits=0, own_updates=0,
        new_classifier_predictions_generated=False, classifier_state_forward_replayed_by_parent=False,
        all_original_ASA_rows=len(data), six_probability_and_checkpoint_receipts=receipts,
        original_TRAIN_and_outer_probabilities_recounted=True, cohorts=summaries,
        correct_known_port_other_S=int(data.correct_known_port_other_S.sum()),
        same_protocol_direction_ranges_correct_control_S=int(data.same_family_control_S.sum()),
        same_family_and_outer_fold_correct_control_S=int(data.same_family_and_outer_fold_control_S.sum()),
        narrower_controls_selected_for_development_audit_only=True, quality_acceptance=False, issue_solved=False,
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in evidence_paths},
        limits=['Two legal TRAIN-role exposures are not two new independent records.',
                'Saved probability confidence is not calibrated deployment probability or proof of security understanding.',
                'Matched family includes ranges rather than exact port values; these are not counterfactual label-preserving pairs.',
                'Historical V123 support projections retain their historical keys and are not new model targets.',
                'Previously inspected outer results/cohort membership select descriptive controls only, not training or thresholds.',
                'Perfect pure TRAIN classifications do not imply source transfer; checkpoint hash receipts do not regenerate model outputs.'])
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'cohorts', 'correct_known_port_other_S',
        'same_protocol_direction_ranges_correct_control_S', 'same_family_and_outer_fold_correct_control_S']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
