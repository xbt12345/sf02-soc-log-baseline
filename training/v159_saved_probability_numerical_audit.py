"""Saved probability arithmetic, no classifier/feature/gradient calls."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'artifacts/v158_legal_fusion_bank_v2_20261001'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    seal_path = ROOT / 'artifacts/v158_fusion_trial_20261001/run_seal.json'
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    bindings = {str(seal_path.relative_to(ROOT)): sha(seal_path)}
    report = []
    for role in range(3):
        ref_path = BANK / f'fold{role}/legal_FIT_reference.parquet'
        ref = pd.read_parquet(ref_path)
        for scope in ('OOF', 'deployment'):
            path = BANK / f'fold{role}/{scope}_probabilities.npy'
            for p in (ref_path, path):
                assert sha(p) == seal['source_sha256'][p.relative_to(ROOT).as_posix()]
                bindings[str(p.relative_to(ROOT))] = sha(p)
            q = np.load(path, mmap_mode='r')[ref.local, :16].mean(1)
            assert np.isfinite(q).all() and (q >= 0).all()
            y = ref.truth.to_numpy()
            qt = q[np.arange(len(q)), y]
            qm = q[:, 1:]
            corrected = np.maximum(q, 1e-12)
            corrected /= corrected.sum(1, keepdims=True)
            row = dict(training_role=role, score_scope=scope,
                       population='all_legal_FIT_original_role_rows',
                       original_rows=len(ref),
                       mean_min=float(q.min()),
                       normalization_max_error=float(np.abs(q.sum(1) - 1).max()),
                       original_rows_with_any_zero_MS=int((qm == 0).any(1).sum()),
                       original_rows_with_zero_true_probability=int((qt == 0).sum()),
                       original_rows_with_true_probability_below_1e12=int((qt < 1e-12).sum()),
                       diagnostic_floor_1e12_max_probability_change=float(np.abs(corrected - q).max()),
                       diagnostic_floor_1e12_changed_argmax=int((corrected.argmax(1) != q.argmax(1)).sum()))
            report.append(row)
    out = ROOT / 'artifacts/v159_saved_probability_numerical_audit_20261001'
    out.mkdir(parents=True, exist_ok=True)
    data = dict(status='saved_current16_probability_domain_verified', latest_actual_training='V158',
                source_sha256=bindings, script_sha256=sha(Path(__file__)), roles=report,
                new_classifier_calls=0, new_feature_calls=0, new_gradients=0, new_fits=0, new_updates=0,
                scope=['Legal FIT identities and saved arrays rehashed; gold proof remains in the independent full input audit.',
                       'OOF unused NaN rows excluded through registered legal roles, not by class or difficulty.',
                       'Floor arithmetic is diagnostic; it is not an approved classifier initialization, probability calibration or new fit.',
                       'An exact-zero-output branch may mask parameter derivatives; no official gradient is computed here.'])
    (out / 'audit.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in data.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
