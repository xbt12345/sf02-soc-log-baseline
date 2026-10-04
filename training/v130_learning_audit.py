"""Read-only V130 training-side diagnosis. No optimizer, fitting or selection.

Historical held-out labels are not used to create the proposed training panel.
All errors below refer to original official rows in an outer TRAIN role.
Body-only collisions diagnose information available to the residual branch;
they are NOT an impossibility bound for a teacher-plus-body classifier.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v130_learning_review_20260930_r2'
V125 = ROOT / 'artifacts/v125_order_trial_20260929'
V128 = ROOT / 'artifacts/v128_nested_score_trial_20260929_r3'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
ROLES = ROOT / 'artifacts/v128_mechanism_review_20260929/nested_score_roles.parquet'
ORIGINAL = ROOT / 'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
CANON = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
OFFICIAL = ROOT / 'data/official/train.parquet'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csr_keys(path):
    x = sparse.load_npz(path).astype(np.float32).tocsr()
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    assert x.shape == (22546, 66287) and np.isfinite(x.data).all()
    return [hashlib.sha256(x.indices[x.indptr[i]:x.indptr[i+1]].astype('<i8').tobytes() +
                          x.data[x.indptr[i]:x.indptr[i+1]].astype('<f4').tobytes()).hexdigest()
            for i in range(x.shape[0])]


def conflicts(frame, key):
    c = frame.groupby([key, 'truth']).size().unstack(fill_value=0).reindex(columns=[1, 2], fill_value=0)
    mixed = (c[1] > 0) & (c[2] > 0)
    return set(c.index[mixed]), {'keys': len(c), 'mixed_keys': int(mixed.sum()),
        'rows_in_mixed_keys': int(c.loc[mixed].sum().sum()),
        'empirical_min_errors': int(c.min(axis=1).sum())}


def quantile(a):
    return np.quantile(a, [0, .1, .5, .9, 1]).tolist() if len(a) else []


def main():
    if OUT.exists():
        raise FileExistsError('Preserve published review: ' + str(OUT))
    paths = {Path(__file__), TRACE, ROLES, ORIGINAL, CANON, OFFICIAL,
             V125/'ordered_body_bytes.npy', V125/'body_lengths.npy'}
    d = pd.read_parquet(TRACE, columns=['row_position', 'local', 'root', 'fold', 'truth'])
    y = pd.read_parquet(OFFICIAL, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert len(y) == 2056871 and len(d) == 112807
    assert not d.row_position.duplicated().any()
    assert np.array_equal(y[d.row_position], d.truth)
    assert d.groupby('root').fold.nunique().max() == 1
    roles = pd.read_parquet(ROLES)
    original_keys, canonical_keys = csr_keys(ORIGINAL), csr_keys(CANON)
    body = np.load(V125/'ordered_body_bytes.npy', mmap_mode='r')
    lengths = np.load(V125/'body_lengths.npy')
    body_keys = [hashlib.sha256(bytes(body[i, :int(n)])).hexdigest() for i, n in enumerate(lengths)]
    for name, keys in [('original_key', original_keys), ('canonical_key', canonical_keys), ('body_key', body_keys)]:
        d[name] = d.local.map(dict(enumerate(keys)))
    reports, ledgers, trajectory, panel = [], [], [], []
    for fold in range(3):
        fit = d.loc[d.fold != fold].copy()
        role = roles.loc[roles.outer_fold == fold]
        assert set(role.row_position) == set(fit.row_position) and not role.row_position.duplicated().any()
        assert set(fit.root).isdisjoint(set(d.loc[d.fold == fold, 'root']))
        mixed = {}
        for key in ['original_key', 'canonical_key', 'body_key']:
            mixed[key], stats = conflicts(fit, key)
            reports.append({'fold': fold, 'kind': 'input_projection', 'view': key, **stats})
            fit[key+'_mixed'] = fit[key].isin(mixed[key])
        ix = fit.local.to_numpy(np.int64); truth = fit.truth.to_numpy()
        pp = V125/f'fold{fold}_A/epoch25_prob.npy'; paths.add(pp)
        p = np.load(pp)[ix]; pred = p.argmax(1)
        fit['A125_pred'] = pred
        fit['A125_p_truth'] = p[np.arange(len(fit)), truth]
        fit['A125_margin_probability'] = p[np.arange(len(fit)), truth]-np.max(
            np.where(np.arange(3)[None, :] == truth[:, None], -np.inf, p), axis=1)
        fit['A125_CE'] = -np.log(np.maximum(fit.A125_p_truth, 1e-12))
        pc = ROOT/f'artifacts/v124_header_trial_20260929/fold{fold}_B/checkpoints.json'; paths.add(pc)
        canonical = json.loads(pc.read_text(encoding='utf-8'))[-1]['fit_by_class']
        bprob = pc.with_name('epoch25_prob.npy'); paths.add(bprob)
        bp = np.load(bprob)[ix]
        fit['B124_pred'] = bp.argmax(1)
        for cl in (1, 2):
            mask = (truth == cl) & (pred != cl)
            pure = mask & ~fit.original_key_mixed.to_numpy()
            reports.append({'fold': fold, 'kind': 'direct_fit_endpoint', 'view': 'A125_original', 'class': cl,
                'support': int((truth == cl).sum()), 'errors': int(mask.sum()),
                'actual_input_nonconflict_errors': int(pure.sum()),
                'nonconflict_error_locals': int(fit.loc[pure, 'local'].nunique()),
                'nonconflict_error_roots': int(fit.loc[pure, 'root'].nunique()),
                'error_p_truth_quantiles': quantile(p[mask, cl]),
                'ensemble_CE': float(fit.loc[truth == cl, 'A125_CE'].mean()),
                'canonical_B124_errors': int(canonical[str(cl)]['support']-canonical[str(cl)]['correct']),
                'canonical_B124_actual_input_nonconflict_errors': int(((truth == cl) &
                    (fit.B124_pred.to_numpy() != cl) & ~fit.canonical_key_mixed.to_numpy()).sum()),
                'canonical_B124_no_correct_member_rows': int(canonical[str(cl)]['no_correct_member_rows'])})
            assert int(((truth == cl) & (fit.B124_pred.to_numpy() != cl)).sum()) == int(
                canonical[str(cl)]['support']-canonical[str(cl)]['correct'])
        for arm in ['P_IS', 'P_CF']:
            rf = role[['local', 'inner_fold']].drop_duplicates()
            assert not rf.local.duplicated().any()
            z = np.zeros((22546, 16, 3), np.float32)
            for inner in range(3):
                teacher = inner if arm == 'P_CF' else (inner+1) % 3
                zp = V128/f'teacher{fold}_{teacher}/canonical_member_logits.npy'; paths.add(zp)
                ids = rf.loc[rf.inner_fold == inner, 'local'].to_numpy(np.int64)
                z[ids] = np.load(zp, mmap_mode='r')[ids]
            residual_path = V128/f'fold{fold}_{arm}/epoch50_residual.npy'; paths.add(residual_path)
            residual = np.load(residual_path)
            q = softmax(z[ix]+residual[ix, None, :], axis=-1).mean(1)
            qp = q.argmax(1); fit[arm+'_pred'] = qp
            cp = V128/f'fold{fold}_{arm}/checkpoints.json'; paths.add(cp)
            checkpoints = json.loads(cp.read_text(encoding='utf-8'))
            for c in checkpoints:
                trajectory.append({'fold': fold, 'arm': arm, 'epoch': c['epoch'],
                                   'train_role': c['train_role']})
            for cl in (1, 2):
                mask = (truth == cl) & (qp != cl)
                assert int(mask.sum()) == checkpoints[-1]['train_role'][str(cl)]['errors']
                reports.append({'fold': fold, 'kind': 'residual_fit_endpoint', 'view': arm, 'class': cl,
                    'support': int((truth == cl).sum()), 'errors': int(mask.sum()),
                    'canonical_input_nonconflict_errors': int((mask & ~fit.canonical_key_mixed.to_numpy()).sum()),
                    'errors_on_body_pure_rows': int((mask & ~fit.body_key_mixed.to_numpy()).sum()),
                    'errors_on_body_mixed_rows': int((mask & fit.body_key_mixed.to_numpy()).sum()),
                    'error_p_truth_quantiles': quantile(q[mask, cl])})
        cp = V125/f'fold{fold}_A/checkpoints.json'; paths.add(cp)
        for c in json.loads(cp.read_text(encoding='utf-8')):
            trajectory.append({'fold': fold, 'arm': 'A125', 'epoch': c['epoch'], 'train_role': c['fit_by_class']})
        fit['outer_fit_role'] = fold
        fit['direct_and_both_residual_wrong'] = ((fit.A125_pred != fit.truth) &
            (fit.P_IS_pred != fit.truth) & (fit.P_CF_pred != fit.truth))
        for cl in (1, 2):
            subset = fit[(fit.truth == cl) & fit.direct_and_both_residual_wrong & ~fit.original_key_mixed]
            reports.append({'fold': fold, 'kind': 'persistent_training_error', 'class': cl,
                            'rows': len(subset), 'locals': int(subset.local.nunique()),
                            'roots': int(subset.root.nunique())})
        # A diagnostic proposal panel, TRAIN-only: all single-class input S
        # mistakes plus all TRAIN M with the same ordered body, not held errors.
        if fold == 1:
            hard = fit[(fit.truth == 2) & (fit.B124_pred != 2) & ~fit.canonical_key_mixed]
            chosen = set(hard.body_key)
            pair = fit[fit.body_key.isin(chosen)].copy()
            pair['panel_role'] = np.where(pair.truth == 2, 'S', 'matched_body_M')
            pair['single_class_canonical_fit_input'] = ~pair.canonical_key_mixed
            panel.append(pair)
            reports.append({'fold': fold, 'kind': 'train_only_panel', 'rows': len(pair),
                'hard_S_rows': len(hard), 'hard_S_locals': int(hard.local.nunique()),
                'S_rows': int((pair.truth == 2).sum()), 'M_rows': int((pair.truth == 1).sum()),
                'roots': int(pair.root.nunique()), 'body_keys': int(pair.body_key.nunique()),
                'M_S_same_body_keys': int(pair.groupby('body_key').truth.nunique().eq(2).sum())})
        ledgers.append(fit)
    OUT.mkdir()
    ledger = pd.concat(ledgers, ignore_index=True)
    ledger.to_parquet(OUT/'training_role_error_ledger.parquet', index=False)
    pd.concat(panel, ignore_index=True).to_parquet(OUT/'fold1_train_only_panel.parquet', index=False)
    save(OUT/'learning_trajectory.json', trajectory)
    summary = {'status': 'training_side_read_only_review', 'classifier_fits': 0,
        'optimizer_steps': 0, 'official_rows': len(y), 'ASA_unique_original_rows': len(d),
        'train_role_rows': len(ledger), 'note': 'Every ASA row has two outer TRAIN roles; do not sum as independent data.',
        'results': reports, 'limitations': [
            'Body projection conflicts are not an impossibility bound for the score-plus-body classifier.',
            'Hash equality is numerical input equality, not a semantic label guarantee.',
            'Ensemble-no-correct-member counts replay historical fit metadata, not newly replayed member models.',
            'No parameter gradient, optimization trial, new training or unseen transfer test executed.'],
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}}
    save(OUT/'audit.json', summary)
    save(OUT/'output_receipt.json', {'status': 'audit_outputs_recorded', 'classifier_fits': 0,
        'sha256': {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}})
    print(json.dumps({'status': summary['status'], 'train_role_rows': len(ledger),
        'results': reports}, ensure_ascii=False))


if __name__ == '__main__':
    main()
