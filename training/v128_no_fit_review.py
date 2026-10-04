"""V128 post-hoc evidence audit. No optimizer, fitting, or candidate selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/v127_frozen_branch_trial_20260929'
OUT = ROOT / 'artifacts/v128_mechanism_review_20260929'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
BODY = ROOT / 'artifacts/v125_order_trial_20260929/ordered_body_bytes.npy'
LENGTHS = BODY.with_name('body_lengths.npy')
OFFICIAL = ROOT / 'data/official/train.parquet'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def counts(groups, y, n):
    return np.bincount(groups * 3 + y, minlength=n * 3).reshape(n, 3)


def floor_report(cc):
    mixed = (cc[:, 1] > 0) & (cc[:, 2] > 0)
    return {'original_rows': int(cc.sum()), 'unique_inputs': int((cc.sum(1) > 0).sum()),
            'mixed_input_groups': int(mixed.sum()), 'rows_in_mixed_groups': int(cc[mixed].sum()),
            'empirical_single_input_classification_error_floor': int((cc.sum(1) - cc.max(1)).sum())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    delivery = read(PARENT / 'delivery.json')
    bound = delivery['artifact_sha256']
    for rel, h in bound.items():
        if sha(ROOT / rel) != h:
            raise ValueError('V127 identity changed: ' + rel)
    d = pd.read_parquet(TRACE)
    pred = pd.read_parquet(PARENT / 'expert_ASA_predictions.parquet')
    assert len(d) == len(pred) == 112807 and d.row_position.equals(pred.row_position)
    y = d.truth.to_numpy(dtype=np.int64)
    oy = pd.read_parquet(OFFICIAL, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert len(oy) == 2056871 and np.array_equal(y, oy[d.row_position])
    body = np.load(BODY); lengths = np.load(LENGTHS)
    strings = [bytes(b[:n]) for b, n in zip(body, lengths)]
    body_id, unique = pd.factorize(strings, sort=False)
    local = d.local.to_numpy(dtype=np.int64); fold = d.fold.to_numpy()
    gid = body_id[local]; n = len(unique)
    full_counts = counts(gid, y, n)
    floor = {'body_only': floor_report(full_counts), 'old_local_full_input': floor_report(counts(local, y, len(body))),
             'scope': 'Observed deterministic input collisions only, not Bayes error or performance ceiling of base+residual. All labels already inspected.'}
    d['body_id'] = gid
    d['body_sha256'] = [hashlib.sha256(strings[k]).hexdigest() for k in local]
    body_ledger = d[['row_position', 'local', 'root', 'fold', 'truth', 'body_id', 'body_sha256']].copy()
    for arm in ('A0', 'AH', 'K', 'W', 'P'):
        body_ledger['pred_' + arm] = pred['pred_' + arm]
    corrections = []; support = []; gradients = []; curves = []
    for f in range(3):
        fit = fold != f; held = ~fit
        fc = counts(gid[fit], y[fit], n)
        fit_local_counts = counts(local[fit], y[fit], len(body))
        held_local_counts = counts(local[held], y[held], len(body))
        z = np.load(PARENT / f'fold{f}_base_member_logits.npy').astype(np.float64)
        fmass = fit_local_counts.sum(1)
        hmass = held_local_counts.sum(1)
        status = np.full(held.sum(), 'unseen_body', dtype=object)
        s = fc[gid[held]]
        status[(s[:, 1] > 0) & (s[:, 2] == 0)] = 'fit_M_only'
        status[(s[:, 1] == 0) & (s[:, 2] > 0)] = 'fit_S_only'
        status[(s[:, 1] > 0) & (s[:, 2] > 0)] = 'fit_M_and_S'
        body_ledger.loc[held, 'fit_body_support'] = status
        for c in (1, 2):
            for name in ('unseen_body', 'fit_M_only', 'fit_S_only', 'fit_M_and_S'):
                m = (status == name) & (y[held] == c)
                support.append({'fold': f, 'truth': c, 'fit_body_support': name, 'rows': int(m.sum()),
                    **{a + '_errors': int((pred.loc[held, 'pred_' + a].to_numpy()[m] != c).sum()) for a in ('A0', 'AH', 'K', 'W', 'P')}})
        for arm in ('W', 'P'):
            checks = read(PARENT / f'fold{f}_{arm}/checkpoints.json')
            for cp in checks:
                curves.append({'fold': f, 'arm': arm, 'epoch': cp['epoch'],
                    'fit_M_errors': cp['fit_and_held']['fit']['1']['errors'],
                    'fit_S_errors': cp['fit_and_held']['fit']['2']['errors'],
                    'fit_M_CE': cp['fit_and_held']['fit']['1']['member_CE'],
                    'fit_S_CE': cp['fit_and_held']['fit']['2']['member_CE'],
                    'held_M_errors': cp['fit_and_held']['held']['1']['errors'],
                    'held_S_errors': cp['fit_and_held']['held']['2']['errors']})
            extra = np.load(PARENT / f'fold{f}_{arm}/epoch50_residual.npy').astype(np.float64)
            mean = (extra * fmass[:, None]).sum(0) / fmass.sum()
            real = softmax(z + extra[:, None, :], axis=-1).mean(1).argmax(1)
            constant = softmax(z + mean, axis=-1).mean(1).argmax(1)
            diff = real != constant
            assert int((diff & (hmass > 0)).sum()) == checks[-1]['real_vs_train_mean_held_prediction_flips']
            corrections.append({'fold': f, 'arm': arm,
                'changed_unique_old_local_inputs': int((diff & (hmass > 0)).sum()),
                'changed_original_rows': int((diff * hmass).sum()),
                'M_changed_original_rows': int((diff * held_local_counts[:, 1]).sum()),
                'S_changed_original_rows': int((diff * held_local_counts[:, 2]).sum()),
                'real_total_errors': int((held_local_counts * (real[:, None] != np.arange(3))).sum()),
                'constant_total_errors': int((held_local_counts * (constant[:, None] != np.arange(3))).sum())})
        # Differentiate the existing member-CE and ensemble-CE objectives analytically.
        # This computes gradients only; no parameter update, fit, or new predictions.
        for state in ('AH', 'P'):
            extra = np.zeros((len(body), 3)) if state == 'AH' else np.load(PARENT / f'fold{f}_P/epoch50_residual.npy')
            q = softmax(z + extra[:, None, :], axis=-1)
            p = q.mean(1)
            for c in (1, 2):
                member = p.copy(); member[:, c] -= 1
                ensemble = (q * q[:, :, c, None]).mean(1) / p[:, c, None]
                ensemble[:, c] -= 1
                # Directional derivative for residual shift (0, -t/2, t/2).
                gm = (member[:, 2] - member[:, 1]) / 2
                ge = (ensemble[:, 2] - ensemble[:, 1]) / 2
                mass = fit_local_counts[:, c]
                member_ce = -np.log(q[:, :, c].clip(1e-300)).mean(1)
                ensemble_ce = -np.log(p[:, c].clip(1e-300))
                gradients.append({'fold': f, 'state': state, 'fit_truth': c, 'fit_rows': int(mass.sum()),
                    'member_mean_CE': float(np.dot(mass, member_ce)/mass.sum()),
                    'ensemble_CE': float(np.dot(mass, ensemble_ce)/mass.sum()),
                    'same_row_MS_direction_disagreements': int(mass[(gm * ge) < 0].sum()),
                    'member_MS_absolute_gradient_mass': float(np.dot(mass, np.abs(gm))),
                    'ensemble_MS_absolute_gradient_mass': float(np.dot(mass, np.abs(ge)))})
    # Original-row provenance is retained, rather than copying labels into an inference route.
    OUT.mkdir()
    body_ledger.to_parquet(OUT / 'body_support_and_errors.parquet', index=False)
    pd.DataFrame(support).to_csv(OUT / 'body_support_by_fold_class.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT / 'class_checkpoint_trajectory.csv', index=False)
    result = {'status': 'posthoc_no_fit_mechanism_review', 'latest_actual_training': 'V127',
        'classifier_fits': 0, 'optimizer_steps': 0, 'model_promoted': False,
        'verified_parent_artifact_count': len(bound), 'population': {'full': len(oy), 'ASA': len(d)},
        'input_collision': floor, 'constant_replacement_units_correction': corrections,
        'gradient_objective_comparison': gradients,
        'inputs_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in (TRACE, BODY, LENGTHS, OFFICIAL, PARENT / 'delivery.json')},
        'source_sha256': sha(__file__),
        'scope': 'Post-hoc development audit, not independent validation. Body collision is not full-model collision. No label rewrite, training, tuning or external data.'}
    write(OUT / 'audit.json', result)
    print(json.dumps({'floor': floor, 'corrections': corrections, 'support': support, 'gradient': gradients}, ensure_ascii=False))


if __name__ == '__main__':
    main()
