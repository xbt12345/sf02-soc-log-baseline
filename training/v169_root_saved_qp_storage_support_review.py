"""Actual saved QP bytes; distinguish derivative support from solver support."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from experiment_review import ROOT, sha, check_bindings
from v169_structural_vector_storage import save_vector, load_vector

OUT = ROOT / 'artifacts/v169_root_saved_qp_storage_support_review_20261002'


def mask(columns):
    out = np.zeros(1060832, bool)
    out[(columns[:, None] * 16 + np.arange(16)).ravel()] = True
    out[1060592:] = True
    return out


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    xfile = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    x = load_npz(xfile)
    observed = np.unique(x.indices)
    extended = np.unique(np.r_[observed, np.arange(5)])
    assert len(observed) == 412
    pure, with_prefix = mask(observed), mask(extended)
    files = [Path(__file__).resolve(), xfile,
             ROOT / 'training/v169_structural_vector_storage.py',
             ROOT / 'training/v169_working_joint_restoration.py',
             ROOT / 'training/v169_prior_pair_training_entry_v7.py']
    OUT.mkdir()
    records = []
    parent = ROOT / 'artifacts/v168_decision_floor_diagnostic_20261002/role1/correction0'
    for name in ['correction', 'displacement']:
        path = parent / f'{name}.npy'
        v = np.load(path)
        assert v.dtype == np.float64 and v.shape == pure.shape and np.isfinite(v).all()
        bits = v.view(np.uint64)
        ids = np.flatnonzero((bits != 0) & ~pure)
        assert len(ids) == 26 and ids.tolist() == list(range(26))
        assert np.count_nonzero(bits[~with_prefix]) == 0
        pure_path = OUT / f'{name}_pure_data_support_refused.npz'
        try:
            save_vector(pure_path, v, observed)
        except RuntimeError as exc:
            assert 'Off-support bits not a uniform signed zero' in str(exc)
            refusal = str(exc)
        else:
            raise AssertionError('Real QP nonzero counterexample was not refused')
        assert load_vector(pure_path, len(v)).tobytes() == v.tobytes()
        actual_path = OUT / f'{name}_data_plus_prefix5.npz'
        encoded = save_vector(actual_path, v, extended)
        assert load_vector(actual_path, len(v)).tobytes() == v.tobytes()
        files.extend([path, pure_path, actual_path])
        records.append(dict(source=path.relative_to(ROOT).as_posix(),
            full_width=len(v), actual_nonzero_off_observation_support=len(ids),
            actual_nonzero_off_data_plus_prefix5_support=0,
            off_observation_coordinates=ids.tolist(),
            maximum_absolute_off_observation_value=float(np.max(np.abs(v[ids]))),
            pure_support_refusal=refusal,
            refused_vector_preserved_bit_exact=True,
            existing_prefix5_encoding_preserved_bit_exact=True,
            prefix5_encoded_bytes=encoded['bytes'],
            complete_raw_sha256=sha(path)))
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    check_bindings(bindings)
    report = dict(status='saved_real_QP_pure_data_support_counterexample_existing_prefix_qualified',
        all_checks_passed=True, vectors=records,
        applies_to_saved_vectors_only=True,
        derivative_support_does_not_automatically_bound_QP_outputs=True,
        prefix_coordinates=80, maximum_current_QP_matrix_rows=66,
        future_solver_outputs_require_full_raw_bit_runtime_checks=True,
        no_tiny_coordinate_was_zeroed_or_truncated=True,
        technical_support_refusal_is_not_model_infeasibility=True,
        root_previous_v5_failure_inference_retracted_because_prefix_already_present=True,
        official_heads=0, official_features=0, official_derivatives=0,
        fits=0, permanent_updates=0, supports_physical_seal=False,
        source_sha256=bindings,
        scope='Reads original saved QP correction/displacement and exercises lossless storage only. No official model, gradient, QP solve, training, or new accuracy evidence.')
    (OUT / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','vectors']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
