"""CPU-only actual-state forward and analytic logit derivatives; no fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch
from v131_common import ROOT, OFFICIAL

OLD = ROOT / 'artifacts/v138_single_issue_round1_20260930'
NEW = ROOT / 'artifacts/v140_ensemble_training_round2_20261001'
OUT = ROOT / 'artifacts/v139_training_decision_review_20261001/v140_saturation_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Preserve the independent diagnosis')
    y = pd.read_parquet(OFFICIAL, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy(np.int8)
    facts = np.load(OLD / 'facts.npy', mmap_mode='r')
    results = []
    bindings = {}
    for fold in [0, 2]:
        erows = pd.read_parquet(NEW / f'fold{fold}_E/endpoint_original_rows.parquet')
        assert np.array_equal(y[erows.row_position], erows.truth)
        mask = erows.pure_TRAIN_input & (
            erows.pred_start.ne(erows.truth) | erows.pred.ne(erows.truth))
        cases = erows.loc[mask].groupby('local').agg(
            truth=('truth', 'first'), original_rows=('row_position', 'size'),
            original_sources=('root', 'nunique'), start_pred=('pred_start', 'first'),
            E_pred=('pred', 'first')).reset_index()
        ids = cases.local.to_numpy()
        hidden_path = OLD / f'fold{fold}_hidden.npy'
        h = np.load(hidden_path, mmap_mode='r')[ids].astype(np.float64)
        f = np.asarray(facts[ids], dtype=np.float64)
        for arm in ['V138', 'C', 'E']:
            folder = OLD / f'fold{fold}_H_L' if arm == 'V138' else NEW / f'fold{fold}_{arm}'
            model_path = folder / 'endpoint_readout.pt'
            state = torch.load(model_path, map_location='cpu', weights_only=True)['readout']
            state = {k: v.detach().numpy() for k, v in state.items()}
            z = np.einsum('nkh,khc->nkc', h, state['weight']) + state['bias']
            z += (f @ state['facts'].T)[:, None, :]
            lp = z - logsumexp(z, axis=-1, keepdims=True)
            p = np.exp(lp)
            q = p.mean(axis=1)
            saved = np.load(folder / 'sealed_all_prob.npy', mmap_mode='r')[ids]
            if not np.array_equal(q.argmax(1), saved.argmax(1)) or np.max(np.abs(q-saved)) > 2e-6:
                raise ValueError('CPU forward does not preserve actual decisions/probabilities')
            truth = cases.truth.to_numpy()
            onehot = np.eye(3)[truth, None, :]
            pt = p[np.arange(len(ids))[:, None], np.arange(p.shape[1])[None, :], truth[:, None]]
            responsibility = pt / pt.sum(1, keepdims=True)
            g_member = (p-onehot)/p.shape[1]
            g_mix = responsibility[:, :, None]*(p-onehot)
            for i, row in cases.iterrows():
                gn = float(np.linalg.norm(g_member[i]))
                gm = float(np.linalg.norm(g_mix[i]))
                results.append({'fold': fold, 'arm': arm, 'local': int(row.local),
                    'truth': int(row.truth), 'original_rows': int(row.original_rows),
                    'original_sources': int(row.original_sources), 'start_pred': int(row.start_pred),
                    'E_pred': int(row.E_pred), 'actual_pred': int(saved[i].argmax()),
                    'p_true': float(q[i, int(row.truth)]),
                    'member_CE': float(-lp[i, :, int(row.truth)].mean()),
                    'ensemble_CE': float(-np.log(q[i, int(row.truth)])),
                    'true_class_probability_below_1e6_members': int((pt[i] < 1e-6).sum()),
                    'true_class_probability_above_one_minus_1e6_members': int((pt[i] > 1-1e-6).sum()),
                    'member_logit_gradient_norm': gn, 'ensemble_logit_gradient_norm': gm,
                    'gradient_norm_ratio': gm/gn if gn else None})
            bindings[model_path.relative_to(ROOT).as_posix()] = sha(model_path)
        bindings[hidden_path.relative_to(ROOT).as_posix()] = sha(hidden_path)
    frame = pd.DataFrame(results)
    OUT.mkdir()
    frame.to_parquet(OUT / 'same_input_three_states.parquet', index=False)
    summary = []
    for arm in ['V138', 'C', 'E']:
        r = frame[(frame.arm == arm) & frame.E_pred.ne(frame.truth)]
        w = r.original_rows.to_numpy()
        summary.append({'arm': arm, 'population': 'same actually incorrect E pure TRAIN inputs',
            'input_keys': len(r), 'original_role_rows': int(w.sum()),
            'weighted_p_true': float(np.average(r.p_true, weights=w)),
            'weighted_member_CE': float(np.average(r.member_CE, weights=w)),
            'weighted_ensemble_CE': float(np.average(r.ensemble_CE, weights=w)),
            'weighted_logit_gradient_ratio': float(np.average(r.gradient_norm_ratio, weights=w)),
            'max_logit_gradient_ratio': float(r.gradient_norm_ratio.max()),
            'min_logit_gradient_ratio': float(r.gradient_norm_ratio.min())})
    result = {'status': 'readonly_actual_CPU_forward_and_analytic_logit_derivatives',
        'new_classifier_fits': 0, 'parameter_updates': 0, 'optimizer_or_LP_QP_calls': 0,
        'new_full_role_gradient_evaluations': 0,
        'summary': summary, 'evidence_sha256': bindings,
        'source_sha256': sha(__file__), 'rows_sha256': sha(OUT / 'same_input_three_states.parquet'),
        'limitations': [
            'Conditional diagnosis after observing errors; not unseen validation.',
            'Analytic derivatives are with respect to member logits, not a full-parameter stationarity proof.',
            'Saturation does not prove representation impossibility or justify changing the registered endpoint.',
            'All labels are independently verified legal TRAIN original rows; no held-label fitting or selection.']}
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
