"""Saved official derivative bytes only; no new official model/derivative calls."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from experiment_review import ROOT, sha, check_bindings

OUT = ROOT / 'artifacts/v169_root_saved_derivative_structural_support_review_20261002'


def support(columns):
    mask = np.zeros(1060832, bool)
    coordinates = (np.asarray(columns)[:,None]*16+np.arange(16)).ravel()
    mask[coordinates] = True
    mask[1060592:] = True
    return mask


def inspect(path, mask):
    g = np.load(path)
    assert g.dtype == np.float64 and g.shape == mask.shape and np.isfinite(g).all()
    bits = g.view(np.uint64)
    outside = int(np.count_nonzero(bits[~mask]))
    assert outside == 0, ('Nonzero bits outside proposed structural support', str(path), outside)
    return dict(path=path.relative_to(ROOT).as_posix(), full_width=len(g),
                structural_support_coordinates=int(mask.sum()),
                nonzero_bit_patterns=int(np.count_nonzero(bits)),
                negative_zero_bit_patterns=int(np.count_nonzero(bits == np.uint64(1 << 63))),
                nonzero_bits_outside_structural_support=outside)


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    xfile = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    x = load_npz(xfile).tocsr()
    assert x.shape == (22546,66287) and x.has_canonical_format
    paths = {Path(__file__).resolve(), xfile,
             ROOT / 'training/v159_current_input_boundary_v3.py',
             ROOT / 'training/v161_fixed_pure_error_risk.py',
             ROOT / 'training/v160_margin_normal.py'}
    target_records, normal_records = [], []
    for role, endpoint in enumerate([1,1,3]):
        tfile = ROOT / f'artifacts/v161_independent_frozen_error_cohort_review_20261002/role{role}/target_original_counts.npy'
        paths.add(tfile)
        targets = np.load(tfile)
        for cls in [1,2]:
            columns = np.unique(x[np.flatnonzero(targets[:,cls]>0)].indices)
            mask = support(columns)
            point = ROOT / f'artifacts/v164_short_supervised_trajectory_20261002/role{role}/parameter_point{endpoint}'
            for repetition in [0,1]:
                path = point / f'class{cls}_repeat{repetition}/complete_fixed_error_target_gradient.npy'
                paths.add(path)
                target_records.append(dict(role=role, class_id=cls, repetition=repetition,
                                           **inspect(path,mask)))
    refsfile = ROOT / 'artifacts/v168_decision_floor_diagnostic_20261002/role1/correction0/active_normal_references.json'
    paths.add(refsfile)
    refs = json.loads(refsfile.read_text(encoding='utf-8'))
    assert len(refs) == 25
    for identity, ref in refs.items():
        local = int(ref['metadata']['local'])
        mask = support(x.indices[x.indptr[local]:x.indptr[local+1]])
        parent = (ROOT / ref['gradient']).parent
        for repetition in [0,1]:
            path = parent / f'repeat{repetition}_gradient.npy'
            paths.add(path)
            normal_records.append(dict(input_identity=identity, local=local, repetition=repetition,
                                       **inspect(path,mask)))
    bindings = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    check_bindings(bindings)
    OUT.mkdir()
    report = dict(status='V164_endpoints_and_V168_actual_margin_saved_bytes_structural_support_verified',
        all_checks_passed=True, target_vectors=target_records, margin_vectors=normal_records,
        complete_saved_vectors_inspected=len(target_records)+len(normal_records),
        all_original_float64_bits_preserved_and_negative_zero_checked=True,
        bound_A_single_query=211*16+240, conditional_bound_B_single_query=211*16+241,
        future_beta_training_support_still_requires_new_runtime_qualification=True,
        structural_runtime_refusal_must_not_be_called_model_infeasibility=True,
        official_heads=0, official_features=0, official_derivatives=0, fits=0,
        permanent_updates=0, supports_physical_seal=False,
        source_sha256=bindings,
        scope='Checks previously saved complete derivatives against exact CSR coordinate support, including negative-zero bits. No derivative was recomputed; does not authorize truncation, dropping any bits, a new official call, or a training/quality claim.')
    (OUT/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','target_vectors','margin_vectors']}, ensure_ascii=False))


if __name__ == '__main__': main()
