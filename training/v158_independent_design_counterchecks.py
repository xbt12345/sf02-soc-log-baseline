"""Saved-array counterexamples to overclaims about the V158 candidate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v158_independent_design_counterchecks_20261001'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalized(scores):
    values = np.exp(scores - scores.max(-1, keepdims=True))
    return values / values.sum(-1, keepdims=True)


def main():
    assert not OUT.exists()
    ref_path = ROOT / 'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet'
    ref = pd.read_parquet(ref_path)
    assert len(ref) == 112807 and ref.row_position.is_unique
    ids_path = ROOT / 'artifacts/v79_execution_20260927/row_feature_id.npy'
    ids = np.load(ids_path, mmap_mode='r')
    root_profiles, old_score_profiles = [], []
    paths = [Path(__file__), ref_path, ids_path]
    for fold in range(3):
        legal = ref[ref.fold.ne(fold)].copy()
        def inner(root):
            key = f'V128|split=12801|outer={fold}|root={root}'
            return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % 3
        legal['inner'] = legal.root.map(inner)
        multiple = legal.groupby('local').inner.nunique()
        affected = legal.local.isin(multiple[multiple > 1].index)
        assert not affected.any()
        root_profiles.append(dict(outer=fold, multi_inner_locals=0, affected_original_rows=0,
                                  local_indexed_OOF_safe_for_current_population=True))
        current_path = ROOT / f'artifacts/v157_complete_function_logit_audit_v2_20261001/fold{fold}/logits.npy'
        legacy_path = ROOT / f'artifacts/v107_matched_training_20260928/fold{fold}_N1_teacher/scores_all_input_ids.npy'
        current = normalized(np.load(current_path))
        old_logits = np.load(legacy_path, mmap_mode='r')[ids[ref.row_position]]
        old = normalized(old_logits)
        assert np.array_equal(old.argmax(1), old_logits.argmax(1))
        held = ref.fold.eq(fold).to_numpy()
        hard = held & ref.known_578_cohort.to_numpy()
        no_s = (current[ref.local].argmax(2) == 1).all(1)
        target = hard & no_s
        bank = np.concatenate([current[ref.local][target], old[target, None, :]], axis=1)
        margin = bank[:, :, 2] - bank[:, :, 1]
        # With nonnegative normalized weights, every mixture margin is at most
        # the largest expert margin. No fitted oracle or new classifier call.
        impossible = margin.max(1) < 0
        old_score_profiles.append(dict(outer=fold, hard_original_rows=int(hard.sum()),
            all_current_members_predict_M=int(target.sum()),
            legacy_N_M_S_predictions=np.bincount(old[target].argmax(1), minlength=3).tolist(),
            all_17_have_strictly_negative_S_minus_M=int(impossible.sum()),
            largest_expert_S_minus_M=float(margin.max()) if len(margin) else None))
        paths += [current_path, legacy_path]
    hard_count = int(ref.known_578_cohort.sum())
    assert hard_count == 578
    result = dict(status='input_index_and_fixed_bank_counterexamples_checked',
        actual_original_ASA_rows=len(ref), actual_hard_original_rows=hard_count,
        local_source_profiles=root_profiles, fixed_deployment_bank_profiles=old_score_profiles,
        actual_fixed_bank_unrepairable_S_by_nonnegative_fusion=sum(
            p['all_17_have_strictly_negative_S_minus_M'] for p in old_score_profiles),
        scope='Fixed deployment bank only; new inner banks and learned fusion not evaluated.',
        own_official_classifier_calls=0, own_feature_calls=0, own_gradients=0, own_fits=0, own_updates=0,
        implications=['Current local-level OOF indexing requires an executable invariant, not a generic assumption.',
            'Shared conditions can change expert weights but cannot create a positive S-M margin if every expert has a negative one.',
            'A correct expert on another row is not a label-free deployable selection rule.',
            'The 32-coordinate text condition is lossy; the existing full text expert remains present.',
            'These inspected development cases are not an independent blind test.'],
        source_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in paths})
    OUT.mkdir()
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'local_source_profiles',
          'fixed_deployment_bank_profiles', 'actual_fixed_bank_unrepairable_S_by_nonnegative_fusion']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
