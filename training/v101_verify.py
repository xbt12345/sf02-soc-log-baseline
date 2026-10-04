"""Independent receipt, grouping, endpoint and metric verification for v10.0."""
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from run_v75 import ROOT, OUT, read, save, sha, load_sparse
from v75_views import matrix_hashes
from v89_common import LAST
from v92_train import Branch, csr
from v101_prepare import DEST, V92


def metric(y, p):
    cm = np.zeros((3, 3), np.int64)
    np.add.at(cm, (y, p), 1)
    terms = []
    for c in (1, 2):
        d = cm[c].sum()+cm[:, c].sum()
        terms.append(float(2*cm[c, c]/d) if d else 0.)
    return int(len(y)-np.trace(cm)), float(sum(terms)/2)


def main():
    target = DEST/'verification.json'
    if target.exists():
        raise FileExistsError(target)
    reg = read(DEST/'registration.json')
    checks = {}
    checks['registration_source'] = sha(ROOT/'training/v101_prepare.py') == reg['source_sha256']
    checks['registered_inputs'] = all(sha(ROOT/name) == digest for name, digest in reg['input_sha256'].items())
    checks['prepared_outputs'] = all(sha(DEST/name) == digest for name, digest in reg['output_sha256'].items())
    export = read(DEST/'source_export.json')
    checks['source_only_export'] = export['other_role_labels_exported'] == 0 and \
        export['source_rows'] == reg['source_rows'] and \
        export['ASA_distinct_text_input_pairs_checked_from_raw'] == reg['text_input_pair_count'] and \
        sha(DEST/'source_rows.parquet') == export['source_rows_sha256']
    src = pd.read_parquet(DEST/'source_rows.parquet')
    checks['folds_and_exposure'] = len(src) == 753709 and set(src.fold) == {0, 1, 2} and \
        src.groupby('component').fold.nunique().max() == 1 and \
        src.groupby('fid').fold.nunique().max() == 1
    groups = pd.read_parquet(DEST/'N1_ASA_group_map.parquet')
    paired = groups.merge(src[['row_position', 'fold']], on='row_position')
    checks['normalized_identical_input_isolation'] = paired.groupby('N1_group').fold.nunique().max() == 1
    old = sparse.load_npz(V92/'ASA_R0.npz')
    n1 = sparse.load_npz(DEST/'N1_ASA.npz')
    checks['facts_metadata_preserved'] = (n1[:, 65792:]-old[:, 65792:]).nnz == 0
    ids = np.load(V92/'ASA_input_ids.npy')
    delta = sparse.load_npz(DEST/'N1_delta.npz')
    input_difference = (old+delta[ids]-n1).tocsr()
    input_difference.eliminate_zeros()
    max_input_abs_difference = float(np.max(np.abs(input_difference.data))) if input_difference.nnz else 0.
    checks['N1_teacher_delta_matches_branch_input'] = max_input_abs_difference <= 1e-7
    source_ASA_fids = np.unique(src.loc[src.is_ASA, 'fid'])
    source_other_fids = np.unique(src.loc[~src.is_ASA, 'fid'])
    checks['ASA_and_other_R0_disjoint'] = np.intersect1d(source_ASA_fids, source_other_fids).size == 0
    # N1 may merge an ASA input with an unchanged non-ASA input even when
    # their old IDs differ. Check this cross-route possibility explicitly.
    base = load_sparse(LAST/'X')
    other_hashes = matrix_hashes(base[source_other_fids])
    other_by_hash = {}
    for f, digest in zip(source_other_fids, other_hashes):
        other_by_hash.setdefault(digest, []).append(int(f))
    fold_by_fid = src.groupby('fid').fold.first()
    n1_cross_route_equal = 0
    n1_cross_route_fold_violations = 0
    local_lookup = np.full(int(ids.max())+1, -1, np.int32)
    local_lookup[ids] = np.arange(len(ids))
    asa_digests = matrix_hashes(n1[local_lookup[source_ASA_fids]])
    for f, digest in zip(source_ASA_fids, asa_digests):
        local = local_lookup[int(f)]
        for other_fid in other_by_hash.get(digest, []):
            if (n1[local]-base[other_fid]).nnz == 0:
                n1_cross_route_equal += 1
                n1_cross_route_fold_violations += int(fold_by_fid[int(f)] != fold_by_fid[other_fid])
    checks['N1_cross_route_identical_input_isolation'] = n1_cross_route_fold_violations == 0
    selection = read(DEST/'selection.json')
    oof = pd.read_parquet(DEST/'oof_predictions.parquet')
    checks['source_OOF_identity'] = len(oof) == 31289 and \
        np.array_equal(oof.row_position.to_numpy(), src[src.is_ASA].row_position.to_numpy()) and \
        sha(DEST/'oof_predictions.parquet') == selection['oof_sha256']
    y = oof.truth_official.to_numpy()
    checks['OOF_metrics_recomputed'] = True
    for name in ['R0_teacher', 'N1_teacher'] + list(selection['arms']):
        p = oof[name].to_numpy()
        errors, f1 = metric(y, p)
        expected = selection['teacher_metrics'][name[:-8]] if name.endswith('_teacher') else selection['arms'][name]
        checks['OOF_metrics_recomputed'] &= errors == expected['errors'] and abs(f1-expected['M_S_equal_F1']) < 1e-12
    checks['all_18_valid_fits_receipted'] = True
    checks['checkpoint_scores_replayed'] = True
    max_replay_abs_difference = 0.
    chosen = np.array([0, 37, 101, 1000, 5000, 10000, 20000, len(ids)-1])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    for fold in range(3):
        for view in ('R0', 'N1'):
            folder = DEST/f'fold{fold}_{view}'
            teacher = read(folder/'teacher_fit.json')
            with np.load(folder/'counts.npz') as counts:
                assert int(counts['train'].sum()+counts['hold'].sum()) == len(src)
                assert int(counts['hold'].sum()) == int((src.fold == fold).sum())
            checks['all_18_valid_fits_receipted'] &= teacher['converged'] and \
                teacher['heldout_gradient_rows'] == 0 and sha(folder/'teacher.joblib') == teacher['model_sha256'] and \
                sha(folder/'teacher_ASA_scores.npy') == teacher['scores_sha256']
            asa = old if view == 'R0' else n1
            teacher_sample = np.load(folder/'teacher_ASA_scores.npy')[chosen]
            for arm in ('ERM', 'MAG'):
                fit = read(folder/f'{arm}_fit.json')
                progress = read(folder/f'{arm}_progress.json')
                steps = [item['step'] for item in progress]
                checks['all_18_valid_fits_receipted'] &= fit['steps_executed'] == steps[-1] and \
                    200 in steps and len(steps) == 81 and fit['heldout_gradient_rows'] == 0 and \
                    fit['source_rows_supervised'] == teacher['source_rows_supervised'] and \
                    fit['teacher_model_sha256'] == teacher['model_sha256']
                for tag in ('200', 'long'):
                    model_path = folder/f'{arm}_{tag}_model.pt'
                    scores_path = folder/f'{arm}_{tag}_ASA_scores.npy'
                    checks['all_18_valid_fits_receipted'] &= \
                        sha(model_path) == fit[f'endpoint_{tag}_model_sha256'] and \
                        sha(scores_path) == fit[f'endpoint_{tag}_scores_sha256']
                    state = torch.load(model_path, map_location='cpu', weights_only=True)
                    model = Branch(asa.shape[1]).to(device)
                    model.load_state_dict(state['state_dict'])
                    model.eval()
                    with torch.no_grad():
                        rerun = teacher_sample+model(csr(asa[chosen], device)).double().cpu().numpy()
                    stored = np.load(scores_path)[chosen]
                    max_replay_abs_difference = max(max_replay_abs_difference,
                        float(np.max(np.abs(rerun-stored))))
                    checks['checkpoint_scores_replayed'] &= bool(
                        np.allclose(rerun, stored, rtol=0, atol=1e-5) and
                        np.array_equal(rerun.argmax(1), stored.argmax(1)))
    # The earlier v10.0 run is preserved separately as an invalid pilot.
    variant = read(DEST/'source_variant_audit.json')
    checks['known_wrapper_invariance'] = variant['N1_prediction_flips_by_construction'] == 0 and \
        variant['N1_feature_equivalence_checks'] == 6693
    diagnosis = read(DEST/'source_diagnosis.json')
    checks['historic_error_and_focus_trace'] = diagnosis['historic_v97_error_rows_without_R0_label_conflict'] == 229 and \
        diagnosis['six_supported_S_rows'] == 6
    span_receipt = read(DEST/'span_map_receipt.json')
    checks['sanitized_raw_N1_span_chain'] = span_receipt['status'] == 'raw_to_stable_to_N1_span_chain_verified' and \
        span_receipt['representative_text_input_pairs'] == 22549 and \
        span_receipt['raw_training_rows_scanned'] == 2056871 and \
        sha(ROOT/'training/v101_span_map.py') == span_receipt['source_sha256'] and \
        sha(DEST/'N1_raw_span_map.parquet') == span_receipt['raw_span_map_sha256']
    prior = read(ROOT/'evidence/2026-09-28/v99_research_plan/delivery.json')
    checks['prior_v99_bound_artifacts'] = all(sha(ROOT/name) == digest for name, digest in prior['artifact_sha256'].items())
    checks['prior_receipts'] = all(sha(ROOT/name) == digest for name, digest in prior['verification']['receipt_sha256'].items())
    result = {'status': 'passed' if all(checks.values()) else 'failed',
        'checks': {k: bool(v) for k, v in checks.items()},
        'source_sha256': sha(__file__), 'valid_teacher_fits': 6,
        'N1_cross_route_equal_pairs': n1_cross_route_equal,
        'N1_cross_route_fold_violations': n1_cross_route_fold_violations,
        'max_N1_input_delta_replay_abs_difference': max_input_abs_difference,
        'max_checkpoint_score_replay_abs_difference': max_replay_abs_difference,
        'valid_branch_fits': 12, 'invalid_pilot_branch_fits_in_this_artifact': 0,
        'actual_classifier_fits_in_corrected_run': 18, 'actual_calibration_fits': 0,
        'selected': selection['selected'], 'quality_acceptance': False,
        'model_promoted': False,
        'scope': 'Receipt and deterministic source-replay checks. V and historic labels are not evaluated as candidates because source gate failed; no hidden or external transfer claim.'}
    save(target, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result['status'] != 'passed':
        raise RuntimeError('Verification failed')


if __name__ == '__main__':
    main()
