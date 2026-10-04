"""Answer-aware descriptive threshold capacity; never select/deploy a rule.

This exhausts scalar score thresholds on inspected outer development scores.
It is not calibrated inference, official performance, or new model fitting.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v153_independent_training_transfer_gap_20261001'
INPUT = OUT / 'all_original_classifier_gap_and_control_ledger.parquet'
REF = ROOT / 'artifacts/v146_guarded_pair_training_20261001/ASA_prediction_ledger.parquet'
TARGET = OUT / 'scalar_score_order_capacity.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not TARGET.exists(), 'Preserve executed evidence.'
    d = pd.read_parquet(INPUT).sort_values('row_position').reset_index(drop=True)
    ref = pd.read_parquet(REF).sort_values('row_position').reset_index(drop=True)
    for name in ['row_position', 'truth', 'fold', 'local', 'canonical_key']:
        assert d[name].equals(ref[name])
    assert len(d) == 112807 and d.groupby('canonical_key').fold.nunique().eq(1).all()
    budget = int((ref.truth.eq(1) & ref.pred_A0.ne(1)).sum())
    reference_S_errors = int((ref.truth.eq(2) & ref.pred_A0.ne(2)).sum())
    assert budget == 318 and reference_S_errors == 2074
    summaries, curves = [], []
    representative = d.sort_values('local', kind='stable').drop_duplicates('canonical_key').set_index('canonical_key')
    for arm in ['A', 'B']:
        assert d['outer_' + arm + '_pred'].ne(0).all()
        for mode in ['actual_saved_scores', 'fixed_canonical_minimum_local']:
            if mode == 'actual_saved_scores':
                probs = d[['outer_' + arm + '_p1', 'outer_' + arm + '_p2']].to_numpy()
            else:
                probs = representative.loc[d.canonical_key, ['outer_' + arm + '_p1', 'outer_' + arm + '_p2']].to_numpy()
            scores = probs[:, 1] / probs.sum(1)
            assert np.isfinite(scores).all()
            table = pd.DataFrame(dict(score=scores, M=d.truth.eq(1).astype(np.int64),
                                      S=d.truth.eq(2).astype(np.int64))).groupby('score').sum().sort_index(ascending=False)
            curve = table.reset_index().rename(columns=dict(score='descriptive_score_boundary', M='block_M', S='block_S'))
            curve['M_errors'] = curve.block_M.cumsum()
            curve['S_errors'] = int(d.truth.eq(2).sum()) - curve.block_S.cumsum()
            curve = pd.concat([pd.DataFrame([dict(descriptive_score_boundary=np.inf, block_M=0, block_S=0,
                                                   M_errors=0, S_errors=int(d.truth.eq(2).sum()))]), curve], ignore_index=True)
            curve['total_errors'] = curve.M_errors + curve.S_errors
            allowed = curve.M_errors.le(budget)
            assert curve.iloc[-1].M_errors == 78748 and curve.iloc[-1].S_errors == 0
            assert curve.M_errors.is_monotonic_increasing and curve.S_errors.is_monotonic_decreasing
            summaries.append(dict(arm=arm, score_mode=mode, score_tie_blocks=len(table),
                fixed_reference_M_error_budget=budget, fixed_reference_S_errors=reference_S_errors,
                minimum_S_errors_under_M_budget=int(curve.loc[allowed, 'S_errors'].min()),
                minimum_total_errors_under_M_budget=int(curve.loc[allowed, 'total_errors'].min()),
                any_boundary_meets_project_ASA_2170_limit=bool(curve.loc[allowed, 'total_errors'].le(2170).any()),
                any_boundary_meets_reference_both_class_limits=bool(curve.loc[allowed, 'S_errors'].le(reference_S_errors).any()),
                selected_threshold=None, deployed_threshold=None))
            curve['arm'] = arm
            curve['score_mode'] = mode
            curves.append(curve)
    pd.concat(curves, ignore_index=True).to_parquet(OUT / 'answer_aware_descriptive_scalar_capacity.parquet', index=False)
    result = dict(status='scalar_ranking_capacity_exhausted_not_calibration_or_classifier_gain',
        official_ASA_original_rows=len(d), new_model_forwards=0, new_fits=0, new_gradients=0, new_updates=0,
        inspected_HELD_answers_used_only_for_descriptive_upper_capacity=True,
        new_inference_rule_or_prediction_file_created=False, selected_model=None, summaries=summaries,
        limitations=['This intentionally uses previously inspected outer answers, so it cannot select training/thresholds or establish blind quality.',
            'Bound applies to one global monotone scalar M/S score threshold on these fixed cross-fold score receipts only.',
            'It does not exclude a source-independent multivariate decision improvement or all calibration families.',
            'Strictly monotone scalar recalibration preserves order; tied/nonmonotone/multivariate changes require separate scope.',
            'Other formats, new models, unseen environments, actual model forwards and full-task promotion are not evaluated.',
            'Fixed canonical representative comparison excludes exploiting duplicate local numerical variants as new semantics.'],
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), INPUT, REF]})
    TARGET.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'summaries']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
