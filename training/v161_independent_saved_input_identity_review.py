"""Recompute saved CSR/opinion byte identities and original-label purity only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v161_independent_saved_input_identity_review_20261002'
INPUT = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
PREP = ROOT / 'artifacts/v159_boundary_input_preparation_20261002'
BANK = ROOT / 'artifacts/v158_legal_fusion_bank_v2_20261001'
COHORT = ROOT / 'artifacts/v161_independent_frozen_error_cohort_review_20261002'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir()
    x = load_npz(INPUT).tocsr()
    assert x.shape == (22546, 66287) and x.dtype == np.float32
    assert x.has_canonical_format and np.isfinite(x.data).all()
    contents = [x.indices[x.indptr[i]:x.indptr[i + 1]].astype('<i8').tobytes() +
                x.data[x.indptr[i]:x.indptr[i + 1]].astype('<f4').tobytes()
                for i in range(len(x.indptr) - 1)]
    keys = {i: hashlib.sha256(raw).hexdigest() for i, raw in enumerate(contents)}
    bindings = {str(INPUT.relative_to(ROOT)): sha(INPUT)}
    roles = []
    for role in range(3):
        path = PREP / f'fold{role}/OOF_visible_input_rows.parquet'
        bank_path = BANK / f'fold{role}/OOF_probabilities.npy'
        cohort_path = COHORT / f'role{role}/fixed_pure_error_targets.parquet'
        frame = pd.read_parquet(path)
        ids = np.sort(frame.local.unique())
        p = np.load(bank_path, mmap_mode='r')[:, :16]
        assert p.shape == (22546, 16, 3) and np.isfinite(p[ids]).all()
        assert np.array_equal(frame.current_input_key, frame.local.map(keys))
        visible = {}
        for i in ids:
            order = np.lexsort((p[i, :, 2], p[i, :, 1], p[i, :, 0]))
            visible[int(i)] = hashlib.sha256(contents[i] + p[i][order].astype('<f8').tobytes()).hexdigest()
        assert np.array_equal(frame.visible_input_key, frame.local.map(visible))
        kinds = {}
        for key_name, flag in [('current_input_key', 'pure_current_input'),
                               ('visible_input_key', 'pure_visible_input')]:
            labels = frame.groupby(key_name).truth.nunique()
            computed = frame[key_name].map(labels).eq(1)
            assert np.array_equal(frame[flag], computed)
            counts = frame.groupby([key_name, 'truth']).size().unstack(fill_value=0)
            floor = int((counts.sum(axis=1) - counts.max(axis=1)).sum())
            kinds[key_name] = dict(input_groups=len(labels), pure_original_rows=int(computed.sum()),
                                   mixed_original_rows=int((~computed).sum()),
                                   original_frequency_majority_error_floor=floor)
        target = pd.read_parquet(cohort_path)
        target_frame = frame.set_index('row_position').loc[target.row_position]
        assert np.array_equal(target_frame.local, target.local)
        assert target_frame.pure_current_input.all() and target_frame.pure_visible_input.all()
        for source in [path, bank_path, cohort_path]:
            bindings[str(source.relative_to(ROOT))] = sha(source)
        roles.append(dict(role=role, original_rows=len(frame), target_rows=len(target), identity_scopes=kinds))
    report = dict(status='all_saved_current_and_complete_visible_input_keys_and_purity_independently_recomputed',
                  roles=roles, input_sha256=bindings, source_sha256=sha(Path(__file__)),
                  official_heads=0, official_features=0, official_gradients=0, official_fits=0,
                  permanent_updates=0, quality_acceptance=False,
                  scope='Saved CSR and first 16 role-legal opinions only. Pure target excludes current-CSR mixed labels conservatively; complete visible input can distinguish more rows. Neither byte conflicts nor majority floors prove conflicting official raw logs or feasibility of a parameterized protected classifier.')
    (OUT / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], roles=roles), ensure_ascii=False))


if __name__ == '__main__':
    main()
