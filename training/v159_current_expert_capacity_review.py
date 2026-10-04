"""Read-only saved-array audit of the 16 current source-excluded experts.

No classifier/feature functions, official gradients, fits or parameter updates.
Gold and complete-input identity evidence is inherited from the separately
executed v159 independent OOF/input audit; bank identities are checked here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'artifacts/v158_legal_fusion_bank_v2_20261001'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    out = ROOT / 'artifacts/v159_current_expert_capacity_review_20261001'
    out.mkdir(parents=True, exist_ok=True)
    seal_path = ROOT / 'artifacts/v158_fusion_trial_20261001/run_seal.json'
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    prior_audit_path = ROOT / 'artifacts/v159_independent_OOF_capacity_and_input_review_v2_20261001/audit.json'
    prior = json.loads(prior_audit_path.read_text(encoding='utf-8'))
    assert prior['latest_actual_training'] == 'V158'
    assert sha(seal_path) == prior['source_sha256'][seal_path.relative_to(ROOT).as_posix()]
    identities = {str(prior_audit_path.relative_to(ROOT)): sha(prior_audit_path),
                  str(seal_path.relative_to(ROOT)): sha(seal_path)}
    results = []
    for fold in range(3):
        reference = BANK / f'fold{fold}/legal_FIT_reference.parquet'
        assert sha(reference) == seal['source_sha256'][reference.relative_to(ROOT).as_posix()]
        identities[str(reference.relative_to(ROOT))] = sha(reference)
        frame = pd.read_parquet(reference)
        bank_path = BANK / f'fold{fold}/OOF_probabilities.npy'
        assert sha(bank_path) == seal['source_sha256'][bank_path.relative_to(ROOT).as_posix()]
        identities[str(bank_path.relative_to(ROOT))] = sha(bank_path)
        scores = np.load(bank_path, mmap_mode='r')[frame.local]
        y = frame.truth.to_numpy()
        assert scores.shape == (len(frame), 17, 3) and np.isfinite(scores).all()
        true = np.take_along_axis(scores, np.broadcast_to(y[:, None, None], (len(frame), 17, 1)), 2)[..., 0]
        other = np.take_along_axis(scores, np.broadcast_to((3 - y)[:, None, None], (len(frame), 17, 1)), 2)[..., 0]
        margins = true - other
        blocked16 = margins[:, :16].max(axis=1) < 0
        blocked17 = margins.max(axis=1) < 0
        current_mean = scores[:, :16].mean(axis=1).argmax(axis=1)
        for cl in (1, 2):
            mask = y == cl
            oldwrong = mask & (current_mean != y)
            results.append(dict(training_role=fold, class_id=cl,
                                original_rows=int(mask.sum()),
                                current16_mean_wrong=int(oldwrong.sum()),
                                strict_current16_convex_blocked=int((mask & blocked16).sum()),
                                strict_all17_convex_blocked=int((mask & blocked17).sum()),
                                old_N1_unblocks_negative_current16_margin=int((mask & blocked16 & ~blocked17).sum()),
                                current16_mean_wrong_with_nonnegative_expert_margin=int((oldwrong & ~blocked16).sum())))
    report = dict(status='current16_saved_OOF_capacity_verified_not_new_training',
                  latest_actual_training='V158', rows=results,
                  source_sha256=identities,
                  script_sha256=sha(Path(__file__)),
                  official_classifier_calls=0, official_feature_calls=0,
                  official_gradient_calls=0, official_fits=0, official_parameter_updates=0,
                  limits=['Negative M/S margin is a convex-family limitation, not raw-data impossibility.',
                          'A nonnegative expert margin need not mean a correct three-class argmax.',
                          'Old N1 contribution is diagnostic only; historical clock-header input has not been approved for the new head.',
                          'This sidecar rechecks saved bank/role identities and inherits independent full-gold checks from the bound prior audit.'])
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
