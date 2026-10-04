"""Independent saved-logit arithmetic; no new classifier/features/gradients."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'artifacts/v158_legal_fusion_bank_v2_20261001'
BASE = ROOT / 'artifacts/v158_current_pipeline_OOF_trial_20261001'
CURRENT = ROOT / 'artifacts/v157_complete_function_logit_audit_v2_20261001'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    source_manifest = BANK / 'pre_saved_array_bindings.json'
    manifest = json.loads(source_manifest.read_text(encoding='utf-8'))['source_sha256']
    seal = json.loads((ROOT / 'artifacts/v158_fusion_trial_20261001/run_seal.json').read_text(encoding='utf-8'))
    bindings = {str(source_manifest.relative_to(ROOT)): sha(source_manifest)}

    def load_bound(path, source):
        h = sha(path)
        assert h == source[path.relative_to(ROOT).as_posix()]
        bindings[str(path.relative_to(ROOT))] = h
        return np.load(path, mmap_mode='r')

    assert sha(source_manifest) == seal['source_sha256'][source_manifest.relative_to(ROOT).as_posix()]
    summaries, ledgers = [], []
    for role in range(3):
        ref_path = BANK / f'fold{role}/legal_FIT_reference.parquet'
        assert sha(ref_path) == seal['source_sha256'][ref_path.relative_to(ROOT).as_posix()]
        bindings[str(ref_path.relative_to(ROOT))] = sha(ref_path)
        frame = pd.read_parquet(ref_path)
        oof = np.full((22546, 16, 3), np.nan, dtype=np.float64)
        for inner in range(3):
            path = BASE / f'outer{role}_inner{inner}/V146_A_second/member_logits.npy'
            values = load_bound(path, manifest)
            ids = frame.loc[frame.inner_fold.eq(inner), 'local'].unique()
            assert values.shape == oof.shape and np.isfinite(values[ids]).all()
            oof[ids] = values[ids]
        deployment = load_bound(CURRENT / f'fold{role}/logits.npy', manifest)
        for scope, logits in (('OOF', oof), ('deployment', deployment)):
            p_bank = load_bound(BANK / f'fold{role}/{scope}_probabilities.npy', seal['source_sha256'])
            ids = frame.local.to_numpy()
            z = logits[ids]
            p = softmax(z, axis=-1).mean(1)
            actual = p_bank[ids, :16].mean(1)
            gap = float(np.abs(p - actual).max())
            assert gap <= 1e-12
            lp = z - logsumexp(z, axis=-1, keepdims=True)
            logq = logsumexp(lp, axis=1) - np.log(16.)
            assert np.isfinite(logq).all()
            truth = frame.truth.to_numpy()
            true_z = np.take_along_axis(z, np.broadcast_to(truth[:, None, None], (len(frame), 16, 1)), 2)[..., 0]
            other_z = np.take_along_axis(z, np.broadcast_to((3 - truth)[:, None, None], (len(frame), 16, 1)), 2)[..., 0]
            # Necessary: a shared class correction must cross at least one
            # member's true-vs-other margin before the ensemble can be correct.
            required = (other_z - true_z).min(1)
            true_logq = logq[np.arange(len(frame)), truth]
            q_true = p[np.arange(len(frame)), truth]
            for cl in (1, 2):
                mask = truth == cl
                blocked = mask & (required > 0)
                required_blocked = required[blocked]
                summaries.append(dict(training_role=role, score_scope=scope, class_id=cl,
                                      original_rows=int(mask.sum()),
                                      mean_probability_reconstruction_gap=gap,
                                      original_true_probability_zero=int((mask & (q_true == 0)).sum()),
                                      stable_mean_log_CE=float(-true_logq[mask].mean()),
                                      saved_probability_tiny_clipped_mean_CE=float(-np.log(np.maximum(q_true[mask], np.finfo(float).tiny)).mean()),
                                      all16_strictly_wrong_MS_logit_margin_original_rows=int(blocked.sum()),
                                      necessary_margin_quantiles=(np.quantile(required_blocked, [0, .5, .9, 1]).tolist() if len(required_blocked) else None),
                                      necessary_margin_above_64=int((mask & (required > 64)).sum()),
                                      necessary_margin_above_745=int((mask & (required > 745)).sum())))
            rows = frame[['row_position', 'local', 'root', 'truth', 'inner_fold']].copy()
            rows['training_role'] = role
            rows['score_scope'] = scope
            rows['minimum_shared_class_correction_needed'] = required
            rows['stable_true_class_log_probability'] = true_logq
            rows['saved_true_class_probability'] = q_true
            ledgers.append(rows)
    out = ROOT / 'artifacts/v159_saved_logit_scale_and_risk_audit_20261001'
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(ledgers, ignore_index=True).to_parquet(out / 'all_legal_OOF_and_deployment_logit_scale_rows.parquet', index=False)
    report = dict(status='saved_current_logit_identity_scale_and_stable_risk_verified', latest_actual_training='V158',
                  roles=summaries, source_sha256=bindings, script_sha256=sha(Path(__file__)),
                  conditional_budget_counterexample=dict(hidden=16, output_initialization='zero', updates=200,
                                                        maximum_step=0.01, output_component_gradient_bound=1,
                                                        maximum_shared_class_pair_correction=64,
                                                        assumptions=['tanh hidden bounded by 1; no output bias',
                                                                     'plain mean gradient or convex common-class gradient without preconditioning/rescaling',
                                                                     'actual finite step at most .01 for all 200 updates']),
                  official_classifier_calls=0, official_feature_calls=0, official_gradient_calls=0,
                  official_fits=0, official_parameter_updates=0,
                  limits=['The 64 bound applies only under the listed conditional budget; it does not describe a registered V159 training run.',
                          'Crossing a member margin is necessary but not sufficient for a correct ensemble argmax.',
                          'Stable saved-logit risk and clipped probability CE are distinct; no official derivative is evaluated.',
                          'Legal role/bank identity is checked here; independent full-gold checks are in the earlier V159 input audit.',
                          'No new probability floors, class weights, temperatures or fitted parameters are selected.'])
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
