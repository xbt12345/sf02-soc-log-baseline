"""Distinguish observable-input collision floors from fixed-bank limitations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v158_independent_input_floor_scope_audit_20261001'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize(z):
    v = np.exp(z - z.max(-1, keepdims=True))
    return v / v.sum(-1, keepdims=True)


def floor(frame, columns):
    c = frame.groupby(columns + ['truth']).size().unstack(fill_value=0)
    return int((c.sum(1) - c.max(1)).sum()), c


def main():
    assert not OUT.exists()
    refpath = ROOT / 'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet'
    idpath = ROOT / 'artifacts/v79_execution_20260927/row_feature_id.npy'
    d = pd.read_parquet(refpath, columns=['row_position', 'local', 'canonical_key', 'truth', 'fold'])
    fid = np.load(idpath, mmap_mode='r')
    paths = [Path(__file__), refpath, idpath]
    reports = []
    for fold in range(3):
        t = d[d.fold.ne(fold)].reset_index(drop=True)
        oldpath = ROOT / f'artifacts/v107_matched_training_20260928/fold{fold}_N1_teacher/scores_all_input_ids.npy'
        currentpath = ROOT / f'artifacts/v157_complete_function_logit_audit_v2_20261001/fold{fold}/logits.npy'
        old = normalize(np.load(oldpath, mmap_mode='r')[fid[t.row_position]])
        current = normalize(np.load(currentpath))[t.local]
        t['normalized_legacy_key'] = [hashlib.sha256(p.tobytes()).hexdigest() for p in old]
        t['diagnostic_round12_key'] = [hashlib.sha256(np.round(p, 12).tobytes()).hexdigest() for p in old]
        old_floor, c = floor(t, ['canonical_key'])
        augmented_floor, _ = floor(t, ['canonical_key', 'normalized_legacy_key'])
        rounded_floor, _ = floor(t, ['canonical_key', 'diagnostic_round12_key'])
        assert [old_floor, augmented_floor, rounded_floor] == [[22,22,22], [6,4,4], [28,26,26]][fold]
        mixed = set(c.index[(c > 0).sum(1) > 1])
        split = []
        for key, group in t[t.canonical_key.isin(mixed)].groupby('canonical_key'):
            if group.normalized_legacy_key.nunique() <= 1:
                continue
            idx = group.index.to_numpy()
            bank = np.concatenate([current[idx], old[idx, None]], axis=1)
            forced = (bank[:, :, 2] - bank[:, :, 1]).max(1) < 0
            assert forced.all()
            split.append(dict(canonical_key=key, original_class_mass=np.bincount(group.truth, minlength=3).tolist(),
                legacy_probability_max_range=float(np.ptp(old[idx], axis=0).max()),
                all_fixed_bank_experts_S_below_M=True,
                forced_M_on_original_S_rows=int((group.truth.to_numpy() == 2).sum()),
                local_sublabel_mass=[dict(local=int(local),class_mass=np.bincount(
                    group.loc[group.local.eq(local), 'truth'], minlength=3).tolist()) for local in group.local.unique()]))
        reports.append(dict(outer=fold, current_numeric_input_floor=old_floor,
            current_key_plus_legacy_normalized_scores_floor=augmented_floor,
            diagnostic_floor_if_round12=rounded_floor, separated_groups=split))
        paths += [oldpath, currentpath]
    result = dict(status='input_collision_floor_and_fixed_expert_capability_distinguished', reports=reports,
        actual_model_calls=0, actual_feature_calls=0, new_fits=0, new_gradients=0, new_updates=0,
        limits=['Augmented key uses current canonical input plus saved normalized scores; not a newly deployed classifier.',
            'Rounding is a descriptive numerical check only, not a feature change or threshold search.',
            'Lower observable-input conflict does not show this convex classifier family can exploit it.',
            'No new probability calibration or raw-model replay was performed.'],
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in paths})
    OUT.mkdir()
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'reports':reports},ensure_ascii=False))


if __name__ == '__main__':
    main()
