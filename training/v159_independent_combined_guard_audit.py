"""Saved-array-only compatibility of OOF learning and deployment protection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'artifacts/v158_legal_fusion_bank_v2_20261001'
OUT = ROOT / 'artifacts/v159_independent_combined_guard_audit_20261002'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda: stream.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir()
    x_path = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    key_path = ROOT / 'artifacts/v159_independent_OOF_capacity_and_input_review_v2_20261001/all_sparse_input_identities.parquet'
    seal_path = ROOT / 'artifacts/v158_fusion_trial_20261001/run_seal.json'
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    sources = [Path(__file__).resolve(), x_path, key_path, seal_path]
    for f in range(3):
        sources += [BANK / f'fold{f}' / n for n in ('legal_FIT_reference.parquet', 'OOF_probabilities.npy', 'deployment_probabilities.npy')]
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    for k, h in bindings.items():
        if k in seal['source_sha256']:
            assert h == seal['source_sha256'][k], k
    (OUT / 'pre_bindings.json').write_text(json.dumps(bindings, indent=2), encoding='utf-8')
    x = load_npz(x_path).tocsr()
    structural = dict(has_sorted_indices=bool(x.has_sorted_indices), has_canonical_format=bool(x.has_canonical_format),
                      explicit_zero_entries=int((x.data == 0).sum()), dtype=str(x.dtype), shape=list(x.shape))
    x.sort_indices()
    canonical = x.copy()
    canonical.sum_duplicates()
    canonical.eliminate_zeros()
    canonical.sort_indices()
    assert np.array_equal(canonical.indptr, x.indptr) and np.array_equal(canonical.indices, x.indices) and np.array_equal(canonical.data, x.data)
    keys = pd.read_parquet(key_path).set_index('local').current_key
    results, ledgers = [], []
    for role in range(3):
        ref = pd.read_parquet(BANK / f'fold{role}/legal_FIT_reference.parquet')
        assert np.array_equal(ref.canonical_key, ref.local.map(keys))
        pure = ref.groupby('canonical_key').truth.nunique().eq(1)
        frames = []
        for scope in ('OOF', 'deployment'):
            p = np.load(BANK / f'fold{role}/{scope}_probabilities.npy', mmap_mode='r')[:, :16]
            joint = {}
            for i in ref.local.unique():
                v = np.asarray(p[i], dtype='<f8')
                order = np.lexsort((v[:, 2], v[:, 1], v[:, 0]))
                joint[i] = hashlib.sha256(keys.loc[i].encode('ascii') + v[order].tobytes()).hexdigest()
            part = ref[['row_position', 'local', 'truth', 'canonical_key']].copy()
            part['scope'] = scope
            part['joint_key'] = part.local.map(joint)
            part['initial_pred'] = p[part.local].mean(1).argmax(1)
            part['initial_correct'] = part.initial_pred.eq(part.truth)
            part['protected'] = part.initial_correct & (part.canonical_key.map(pure) if scope == 'OOF' else True)
            frames.append(part)
        both = pd.concat(frames, ignore_index=True)
        protected = both[both.protected].groupby('joint_key').truth.agg(lambda s: tuple(sorted(s.unique())))
        incompatible = protected.map(len).gt(1)
        assert not incompatible.any()
        oo = frames[0].copy()
        oo['forced_label'] = oo.joint_key.map(protected.map(lambda t: t[0]))
        counts = oo.groupby(['joint_key', 'truth']).size().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
        group_force = counts.index.to_series().map(protected.map(lambda t: t[0]))
        best = counts.max(1).to_numpy()
        for i, label in enumerate(group_force):
            if pd.notna(label):
                best[i] = counts.iloc[i, int(label)]
        minimum = int((counts.sum(1).to_numpy() - best).sum())
        oo['training_role'] = role
        ledgers.append(oo)
        results.append(dict(role=role, original_OOF_rows=len(oo),
                            original_deployment_rows=len(frames[1]),
                            identical_joint_keys_across_scopes=len(set(frames[0].joint_key) & set(frames[1].joint_key)),
                            multiple_required_labels_per_joint_input=int(incompatible.sum()),
                            combined_guard_constrained_minimum_OOF_errors=minimum,
                            registered_current_X_error_target=[22, 6, 28][role],
                            targets_compatible=minimum <= [22, 6, 28][role]))
    for k, h in bindings.items():
        assert sha(ROOT / k) == h, k
    pd.concat(ledgers, ignore_index=True).to_parquet(OUT / 'all_original_OOF_with_combined_guard_constraints.parquet', index=False)
    report = dict(status='combined_scope_exact_input_guard_compatibility_only', roles=results, sparse_structure=structural,
                  source_sha256=bindings, official_classifier_calls=0, official_feature_calls=0,
                  official_gradients=0, official_fits=0, official_updates=0,
                  scope='Finite saved-array equality only; feasible labels do not prove finite-network expressivity, optimizability or transfer.')
    (OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
