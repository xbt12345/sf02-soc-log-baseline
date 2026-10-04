"""Read-only exact input/retention lower bounds; no official head evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'artifacts/v158_legal_fusion_bank_v2_20261001'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def summarize(rows, key):
    counts = rows.groupby([key, 'truth']).size().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    totals = counts.sum(axis=1)
    protected = rows.loc[rows.initial_correct].groupby([key, 'truth']).size().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0).reindex(counts.index, fill_value=0)
    n_protected_labels = protected.gt(0).sum(axis=1)
    protected_class = protected.to_numpy().argmax(axis=1)
    majority = counts.to_numpy().max(axis=1)
    forced_correct = counts.to_numpy()[np.arange(len(counts)), protected_class]
    best_correct = np.where(n_protected_labels.to_numpy() == 0, majority, forced_correct)
    detail = counts.add_prefix('class_mass_')
    detail['total_original_rows'] = totals
    detail['unconstrained_minimum_errors'] = totals - majority
    detail['initially_correct_labels'] = n_protected_labels
    detail['retention_constrained_minimum_errors_if_feasible'] = totals - best_correct
    detail['retention_is_infeasible'] = n_protected_labels > 1
    mixed = counts.gt(0).sum(axis=1) > 1
    return dict(groups=len(counts), mixed_groups=int(mixed.sum()),
                unconstrained_minimum_original_errors=int((totals - majority).sum()),
                retention_constrained_minimum_original_errors=(int((totals - best_correct).sum()) if not (n_protected_labels > 1).any() else None),
                multiple_protected_labels_in_one_group=int((n_protected_labels > 1).sum()),
                pure_groups_original_rows=int(totals[~mixed].sum())), detail.reset_index()


def main():
    prior_path = ROOT / 'artifacts/v159_independent_OOF_capacity_and_input_review_v2_20261001/audit.json'
    keys_path = prior_path.parent / 'all_sparse_input_identities.parquet'
    contract_path = ROOT / 'training/review_policy/v159_candidate_qualification_contract.json'
    seal_path = ROOT / 'artifacts/v158_fusion_trial_20261001/run_seal.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    assert sha(seal_path) == prior['source_sha256'][seal_path.relative_to(ROOT).as_posix()]
    paths = [Path(__file__).resolve(), prior_path, keys_path, contract_path,
             ROOT / 'training/v159_current_input_boundary_v3.py', seal_path]
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    out = ROOT / 'artifacts/v159_joint_input_retention_feasibility_audit_20261002'
    assert not out.exists()
    out.mkdir()
    (out / 'pre_audit_bindings.json').write_text(json.dumps(dict(source_sha256=bindings), indent=2) + '\n', encoding='utf-8')
    keys = pd.read_parquet(keys_path).set_index('local').current_key
    results, original_rows, groups = [], [], []
    for role in range(3):
        ref_path = BANK / f'fold{role}/legal_FIT_reference.parquet'
        p_path = BANK / f'fold{role}/OOF_probabilities.npy'
        for path in (ref_path, p_path):
            assert sha(path) == seal['source_sha256'][path.relative_to(ROOT).as_posix()]
            bindings[path.relative_to(ROOT).as_posix()] = sha(path)
        rows = pd.read_parquet(ref_path)
        assert np.array_equal(rows.local.map(keys), rows.canonical_key)
        member = np.load(p_path, mmap_mode='r')[:, :16]
        ids = rows.local.unique()
        assert np.isfinite(member[ids]).all()
        joint, ordered = {}, {}
        for local in ids:
            p = np.asarray(member[local], dtype='<f8')
            # Shared opinion weights + mean expert aggregation has no expert
            # identity feature; use an unordered multiset, with no rounding.
            order = np.lexsort((p[:, 2], p[:, 1], p[:, 0]))
            prefix = keys.loc[local].encode('ascii')
            joint[local] = hashlib.sha256(prefix + p[order].tobytes()).hexdigest()
            ordered[local] = hashlib.sha256(prefix + p.tobytes()).hexdigest()
        rows['joint_symmetric_input_key'] = rows.local.map(joint)
        rows['joint_ordered_input_key'] = rows.local.map(ordered)
        rows['initial_pred'] = member[rows.local].mean(1).argmax(1)
        rows['initial_correct'] = rows.initial_pred == rows.truth
        rows['training_role'] = role
        raw_stats, _ = summarize(rows, 'canonical_key')
        sym_stats, detail = summarize(rows, 'joint_symmetric_input_key')
        ordered_stats, _ = summarize(rows, 'joint_ordered_input_key')
        detail['training_role'] = role
        inner_variation = rows.groupby('canonical_key').inner_fold.nunique()
        extra_variation = rows.groupby('canonical_key').joint_symmetric_input_key.nunique()
        results.append(dict(training_role=role, original_rows=len(rows),
                            initially_correct_original_rows=int(rows.initial_correct.sum()),
                            original_class_mass=[int(rows.truth.eq(c).sum()) for c in range(3)],
                            current_X_only=raw_stats, complete_X_and_unordered_current16_probabilities=sym_stats,
                            ordered_probability_control=ordered_stats,
                            raw_input_groups_split_by_probabilities=int(extra_variation.gt(1).sum()),
                            raw_input_groups_with_multiple_inner_teachers=int(inner_variation.gt(1).sum())))
        original_rows.append(rows)
        groups.append(detail)
    for path, h in bindings.items():
        assert sha(ROOT / path) == h, path
    pd.concat(original_rows, ignore_index=True).to_parquet(out / 'all_original_joint_input_and_initial_retention_rows.parquet', index=False)
    pd.concat(groups, ignore_index=True).to_parquet(out / 'all_symmetric_joint_input_groups.parquet', index=False)
    report = dict(status='exact_symmetric_joint_input_and_initial_retention_bounds_verified_not_head_execution',
                  latest_actual_training='V158', roles=results, source_sha256=bindings,
                  official_classifier_calls=0, official_feature_calls=0, official_gradient_calls=0,
                  official_fits=0, official_parameter_updates=0,
                  scope=['Exact finite saved numerical equality, no tolerances/rounding or label-dependent feature choice.',
                         'Unordered expert multiset reflects mathematical permutation invariance, not guaranteed bitwise invariance of floating reduction.',
                         'Numerically distinct inputs do not prove that the finite candidate or registered budget can classify them.',
                         'OOF score distinctions may reflect different source-excluded fitted teachers, not new independent behavior evidence.',
                         'This does not replace full sparse-input/gold verification inherited from the bound independent input audit.',
                         'No classifier is invoked on official data; initial predicates are saved-array arithmetic.'])
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
