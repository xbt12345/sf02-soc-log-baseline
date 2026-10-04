"""Source-only legal wrapper perturbations with unchanged observable facts."""
import json

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from run_v75 import ROOT, OUT, save, sha
from v75_views import BYTE_FEATURES, byte_matrix
from v75_corrective import stable
from v92_train import Branch, csr
from v99_normalization_feasibility import normalize
from v101_prepare import DEST, V92


def main():
    output = DEST/'source_variant_audit.json'
    if output.exists():
        raise FileExistsError(output)
    source = pd.read_parquet(DEST/'source_rows.parquet', columns=['row_position', 'fid', 'fold', 'is_ASA'])
    source = source[source.is_ASA]
    rows = pd.read_parquet(OUT/'rows.parquet', columns=['row_position', 'new_text_id'])
    pairs = source[['row_position', 'fid', 'fold']].merge(rows, on='row_position').drop_duplicates(['fid', 'new_text_id'])
    ids = np.load(V92/'ASA_input_ids.npy')
    lookup = np.full(int(ids.max())+1, -1, np.int32)
    lookup[ids] = np.arange(len(ids))
    old = sparse.load_npz(V92/'ASA_R0.npz')
    dictionary = pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    first = pairs.drop_duplicates('fid').reset_index(drop=True)
    canonical = [stable(dictionary.loc[int(t)]) for t in first.new_text_id]
    variant = [t.replace('<IDENTITY>', 'CRED-<IDENTITY>', 1) for t in canonical]
    valid = np.array(['<IDENTITY>' in t and normalize(t, 'placeholder_cluster') ==
        normalize(v, 'placeholder_cluster') for t, v in zip(canonical, variant)])
    first = first[valid].reset_index(drop=True)
    canonical = [t for t, keep in zip(canonical, valid) if keep]
    variant = [t for t, keep in zip(variant, valid) if keep]
    assert len(first) > 0
    loc = lookup[first.fid.to_numpy()]
    original = old[loc]
    check = byte_matrix(canonical)-original[:, :BYTE_FEATURES]
    assert check.nnz == 0 or abs(check.data).max() < 1e-7
    altered = sparse.hstack([byte_matrix(variant), original[:, BYTE_FEATURES:]], format='csr')
    n1a = byte_matrix([normalize(t, 'placeholder_cluster') for t in canonical])
    n1b = byte_matrix([normalize(t, 'placeholder_cluster') for t in variant])
    assert (n1a-n1b).nnz == 0
    frequency = source.fid.value_counts()
    weights = first.fid.map(frequency).to_numpy()
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    for fold in range(3):
        take = first.fold.to_numpy() == fold
        xvar = altered[take]
        group_loc = loc[take]
        w = weights[take]
        folder = DEST/f'fold{fold}_R0'
        teacher = joblib.load(folder/'teacher.joblib')
        variant_teacher = np.asarray(xvar@teacher['coef'])+teacher['intercept']
        real_teacher = np.load(folder/'teacher_ASA_scores.npy')[group_loc]
        orig_label = real_teacher.argmax(1)
        var_label = variant_teacher.argmax(1)
        changed = orig_label != var_label
        results[f'fold{fold}_teacher'] = {'input_pairs': int(take.sum()),
            'decision_flips': int(changed.sum()), 'original_row_weighted_flips': int(w[changed].sum())}
        for arm in ('ERM', 'MAG'):
            for tag in ('200', 'long'):
                state = torch.load(folder/f'{arm}_{tag}_model.pt', map_location='cpu', weights_only=True)
                model = Branch(xvar.shape[1]).to(device)
                model.load_state_dict(state['state_dict'])
                model.eval()
                variant_scores = np.empty((xvar.shape[0], 3), np.float64)
                with torch.no_grad():
                    for start in range(0, xvar.shape[0], 4096):
                        stop = min(start+4096, xvar.shape[0])
                        variant_scores[start:stop] = variant_teacher[start:stop]+model(
                            csr(xvar[start:stop], device)).double().cpu().numpy()
                original_scores = np.load(folder/f'{arm}_{tag}_ASA_scores.npy')[group_loc]
                changed = original_scores.argmax(1) != variant_scores.argmax(1)
                results[f'fold{fold}_{arm}_{tag}'] = {'input_pairs': int(take.sum()),
                    'decision_flips': int(changed.sum()),
                    'original_row_weighted_flips': int(w[changed].sum())}
                del model
    result = {'status': 'source_only_legal_variant_audit',
        'source_sha256': sha(__file__), 'source_unique_R0_inputs': int(first.fid.nunique()),
        'source_original_row_weight': int(weights.sum()),
        'N1_feature_equivalence_checks': len(first),
        'N1_prediction_flips_by_construction': 0,
        'R0_matched_predictions': results,
        'variant': 'Add one known synthetic CRED- wrapper adjacent to existing <IDENTITY>; no real protocol, action, direction, number or fact field changed.',
        'unknown_formats_in_scope': False, 'independent_events_added': 0,
        'quality_claim': False}
    save(output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
