"""Read-only recount and counterexample checks; not a trainer or a run seal."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v132_training_mastery_review_20260930'
PLAN = ROOT / 'training/review_policy/v132_training_mastery_plan.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def recount(frame):
    if frame.row_position.duplicated().any():
        raise ValueError('Duplicate original rows in one fit role')
    counts = frame.groupby(['canonical_key', 'truth']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[1, 2], fill_value=0)
    mixed = counts[(counts > 0).sum(axis=1) > 1]
    pure = ~frame.canonical_key.isin(mixed.index)
    wrong = frame.pred.ne(frame.truth)
    m = int((frame.truth.eq(1) & wrong).sum())
    s = int((frame.truth.eq(2) & wrong).sum())
    pm = int((frame.truth.eq(1) & wrong & pure).sum())
    ps = int((frame.truth.eq(2) & wrong & pure).sum())
    floor = int((mixed.sum(axis=1) - mixed.max(axis=1)).sum())
    if frame.groupby('canonical_key').pred.nunique().max() != 1:
        raise ValueError('Identical numeric input has inconsistent deterministic labels')
    return dict(M_errors=m, S_errors=s, pure_M_errors=pm, pure_S_errors=ps,
                empirical_minimum_errors=floor, fit_rows=len(frame),
                all_mixed_keys_have_M_majority=bool((mixed[1] > mixed[2]).all()),
                endpoint_full_training_mastered=pm == ps == 0 and m == 0 and s == floor)


def fixed_window_pass(history, floor, m_support, s_support):
    if [h['epoch'] for h in history] != list(range(1, 101)):
        raise ValueError('Missing or duplicated full epoch history')
    for h in history:
        if h['class_mass_seen'] != [0, m_support, s_support]:
            raise ValueError('Original class mass not conserved')
        if h['per_class']['1']['support'] != m_support or h['per_class']['2']['support'] != s_support:
            raise ValueError('Scored population changed')
    return all(h['stats']['M_errors'] == 0 and h['stats']['pure_S_errors'] == 0
               and h['stats']['S_errors'] == floor for h in history[95:100])


def lr_at(epoch, decay):
    if not 1 <= epoch <= 100:
        raise ValueError('Wrong endpoint schedule')
    return 0.002 if not decay or epoch <= 60 else (
        0.00002 + 0.5 * (0.002 - 0.00002) * (1 + math.cos(math.pi * (epoch - 60) / 40)))


def counterexamples():
    """Actual historical failures plus input collision and accounting adversaries."""
    base = ROOT / 'artifacts/v131_learning_trial_20260930'
    hist_o = read(base / 'fold1_O/progress.json')
    hist_c = read(base / 'fold1_C/progress.json')
    assert hist_o[-1]['stats']['M_errors'] == 0
    assert not fixed_window_pass(hist_o, 6, 38886, 2071), 'Endpoint-only O incorrectly accepted'
    assert fixed_window_pass(hist_c, 6, 38886, 2071), 'Stable C should pass only its observed fold'
    toy = pd.DataFrame({'row_position': range(78), 'canonical_key': ['one'] * 78,
                        'truth': [1] * 72 + [2] * 6, 'pred': [1] * 78})
    assert recount(toy)['endpoint_full_training_mastered']
    flipped = toy.copy(); flipped['pred'] = 2
    assert not recount(flipped)['endpoint_full_training_mastered'], 'Mixed majority M cannot be sacrificed'
    benign = toy.copy(); benign['pred'] = 0
    assert not recount(benign)['endpoint_full_training_mastered'], 'All benign cannot pass'
    duplicate = pd.concat([toy, toy.iloc[:1]], ignore_index=True)
    try:
        recount(duplicate)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate original row silently accepted')
    broken = json.loads(json.dumps(hist_c)); broken.pop(96)
    try:
        fixed_window_pass(broken, 6, 38886, 2071)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing epoch silently accepted')
    broken = json.loads(json.dumps(hist_c)); broken[-1]['class_mass_seen'][2] -= 1
    try:
        fixed_window_pass(broken, 6, 38886, 2071)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing rare original row silently accepted')
    vote = read(OUT / 'member_counterfactuals.json')['results']
    assert sum(r['plurality_vote_errors']['2'] for r in vote) == 143
    assert sum(r['probability_average_errors']['2'] for r in vote) == 139
    return 8


def main():
    plan = read(PLAN)
    for path, expected in plan['evidence_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Plan evidence changed: ' + path)
    # Preserve the original receipts; later counterfactual is bound separately in the plan.
    for path, expected in read(OUT / 'source_receipt.json')['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Audit source changed: ' + path)
    for path, expected in read(OUT / 'output_receipt.json')['output_sha256'].items():
        if sha(OUT / path) != expected:
            raise ValueError('Audit output changed: ' + path)
    for path, expected in read(ROOT / 'artifacts/v131_learning_trial_20260930/run_seal.json')['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Sealed V131 source changed: ' + path)
    assert plan['status'] == 'designed_not_trained'
    assert plan['scope'] == 'training_mastery_diagnostic_not_automatic_replacement'
    assert plan['runtime_readiness']['trainer_implemented'] is False
    assert plan['runtime_readiness']['run_seal_created'] is False
    assert plan['primary']['arms'] == ['O_const', 'O_decay']
    assert plan['primary']['fixed_epochs'] == 100 and plan['mastery_gate']['fixed_epochs'] == [96,97,98,99,100]
    assert plan['primary']['seed'] == 10201 and plan['confirmation']['seed'] == 13701
    assert plan['primary']['fits'] == 6 and plan['primary']['updates'] == 35600
    assert plan['confirmation']['requires_training_and_quality'] is True
    assert plan['fit_budget']['maximum_fits'] == 9 and plan['fit_budget']['maximum_updates'] == 53400
    assert plan['selection']['authority'] == 'TRAIN_only_before_new_held_scores'
    assert plan['selection']['preference'] == ['O_const', 'O_decay']
    assert plan['confirmation']['automatic_promotion'] is False
    assert lr_at(60, True) == 0.002 and abs(lr_at(100, True) - 0.00002) < 1e-12
    assert all(lr_at(e, True) > lr_at(e+1, True) for e in range(60,100))
    audit = read(OUT / 'audit.json')
    frame = pd.read_parquet(OUT / 'training_mastery_ledger.parquet')
    truth = pq.read_table(ROOT / 'data/official/train.parquet', columns=['label_binary']).column(0).to_pandas()
    official = truth.map({'benign':0, 'malicious':1, 'suspicious':2}).to_numpy()
    assert len(official) == 2056871 and not pd.isna(official).any()
    assert np.array_equal(frame.truth, official[frame.row_position.to_numpy()])
    results = []
    for expected in audit['fold_arm_results']:
        fold, arm = expected['fold'], expected['arm']
        f = frame[frame.outer_fit_role.eq(fold) & frame.arm.eq(arm)]
        computed = recount(f)
        for key, value in computed.items():
            if expected[key] != value:
                raise ValueError(f'Recount mismatch {fold}/{arm}/{key}')
        hist = read(ROOT / f'artifacts/v131_learning_trial_20260930/fold{fold}_{arm}/progress.json')
        stable = fixed_window_pass(hist, computed['empirical_minimum_errors'], expected['M_support'], expected['S_support'])
        assert stable == expected['last5_input_minimum_reached_every_epoch']
        results.append(dict(fold=fold, arm=arm, endpoint_pass=computed['endpoint_full_training_mastered'], fixed_window_pass=stable))
    r = frame[frame.arm.eq('R')]
    counts = r.groupby(['outer_fit_role','canonical_key','truth']).size().unstack(fill_value=0)
    pure = counts.index[(counts > 0).sum(axis=1) == 1]
    mask = pd.MultiIndex.from_frame(r[['outer_fit_role','canonical_key']]).isin(pure)
    errors = r[mask & r.pred.ne(r.truth)]
    assert len(errors) == 89 and errors.row_position.nunique() == 63
    assert int(errors.truth.eq(1).sum()) == 6 and int(errors.truth.eq(2).sum()) == 83
    checks = counterexamples()
    result = dict(status='evidence_recount_and_plan_checks_passed_not_training_ready',
                  new_fits=0, new_updates=0, plan_sha256=sha(PLAN), checker_sha256=sha(__file__),
                  counterexample_checks=checks, audited_fit_roles=results,
                  fully_mastered_R=False, future_trainer_implemented=False, model_promoted=False,
                  limitations=['V131 per-epoch source CSV was overwritten; no complete per-epoch source-history acceptance.',
                               'Hash and stored prediction recount, not a new neural forward replay.',
                               'No new fits, no next-run seal, no new blind transfer validation.'])
    (OUT / 'plan_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
