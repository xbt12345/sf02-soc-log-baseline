"""Audit saved legacy roles/initial margins, without any model function call."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v158_legacy_expert_bank_qualification_20261001'
OUT = ROOT / 'artifacts/v158_independent_legacy_role_and_prior_audit_20261001'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(bindings):
    for name, identity in bindings.items():
        assert sha(ROOT / name) == identity, name


def main():
    assert not OUT.exists()
    qual = read(RUN / 'qualification.json')
    assert qual['official_classifier_calls'] == qual['official_features_calls'] == 0
    assert qual['new_fits'] == qual['new_gradients'] == qual['new_updates'] == 0
    assert qual['total_planned_fits'] == 60 and not qual['formal_training_registered']
    verify(qual['source_sha256'])
    bindings = read(RUN / 'pre_saved_array_bindings.json')
    verify(bindings['source_sha256'])
    roles = read(RUN / 'legacy_full_nested_roles_and_cost.json')['roles']
    priors = read(RUN / 'initial_legacy_priors.json')['folds']
    rrpath = ROOT / 'artifacts/v75_four_arm_20260921_r2/rows.parquet'
    foldpath = ROOT / 'artifacts/v106_frozen_audit_20260928/proposed_body_closed_folds.parquet'
    goldpath = ROOT / 'data/official/train.parquet'
    fidpath = ROOT / 'artifacts/v79_execution_20260927/row_feature_id.npy'
    tracepath = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    trace = pd.read_parquet(tracepath, columns=['row_position', 'local', 'root', 'fold', 'truth'])
    rr = pd.read_parquet(rrpath, columns=['row_position', 'route', 'label_index'])
    folds = pd.read_parquet(foldpath, columns=['row_position', 'root', 'proposed_fold'])
    gold = pd.read_parquet(goldpath, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy(np.int8)
    assert len(rr) == len(folds) == len(gold) == 2056871
    assert np.array_equal(rr.row_position, np.arange(len(gold)))
    assert np.array_equal(folds.row_position, rr.row_position)
    assert np.array_equal(rr.label_index, gold)
    assert np.array_equal(gold[trace.row_position], trace.truth)
    assert np.array_equal(folds.iloc[trace.row_position].root, trace.root)
    assert np.array_equal(folds.iloc[trace.row_position].proposed_fold, trace.fold)
    assert folds.groupby('root').proposed_fold.nunique().max() == 1
    fid = np.load(fidpath, mmap_mode='r')
    assert len(fid) == len(rr) and len(trace) == 112807
    paths = [Path(__file__), RUN / 'qualification.json', RUN / 'pre_saved_array_bindings.json',
             RUN / 'legacy_full_nested_roles_and_cost.json', RUN / 'initial_legacy_priors.json',
             rrpath, foldpath, goldpath, fidpath, tracepath]
    role_reports, prior_reports = [], []
    roots = folds.root.unique()
    for fold in range(3):
        lookup = {}
        for root in roots:
            key = f'V128|split=12801|outer={fold}|root={root}'
            lookup[root] = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % 3
        inner = folds.root.map(lookup).to_numpy()
        legal = folds.proposed_fold.ne(fold).to_numpy()
        held_roots = set(folds.loc[~legal, 'root'])
        for excluded in range(3):
            expected = roles[fold * 3 + excluded]
            assert (expected['outer_fold'], expected['excluded_inner']) == (fold, excluded)
            fit = legal & (inner != excluded)
            query = legal & (inner == excluded)
            fit_roots = set(folds.loc[fit, 'root'])
            assert not fit_roots & held_roots
            assert not fit_roots & set(folds.loc[query, 'root'])
            assert expected['fit_rows'] == int(fit.sum())
            assert expected['fit_class_mass'] == np.bincount(gold[fit], minlength=3).tolist()
            assert expected['fit_roots'] == len(fit_roots)
            assert expected['fit_actual_feature_ids'] == np.unique(fid[fit]).size
            assert expected['OOF_full_rows'] == int(query.sum())
            assert expected['OOF_class_mass'] == np.bincount(gold[query], minlength=3).tolist()
            assert expected['OOF_ASA_rows'] == int((query & rr.route.eq('asa').to_numpy()).sum())
            role_reports.append(dict(outer=fold, excluded_inner=excluded,
                original_fit_rows=int(fit.sum()), fit_class_mass=expected['fit_class_mass'],
                source_and_outer_exclusion_verified=True))
        folder = ROOT / f'artifacts/v107_matched_training_20260928/fold{fold}_N1_teacher'
        receipt = read(folder / 'fit.json')
        scores_path = folder / 'scores_all_input_ids.npy'
        assert sha(scores_path) == receipt['scores_sha256']
        assert sha(folder / 'teacher.joblib') == receipt['model_sha256']
        scores = np.load(scores_path, mmap_mode='r')[fid[trace.row_position]]
        ex = np.exp(scores - scores.max(1, keepdims=True))
        legacy = ex / ex.sum(1, keepdims=True)
        assert np.array_equal(legacy.argmax(1), scores.argmax(1))
        qpath = ROOT / f'artifacts/v155_guarded_full_gradient_sam_20261001/fold{fold}_zero_probability.npy'
        current = np.load(qpath)[trace.local]
        fit_mask = trace.fold.ne(fold).to_numpy()
        truth = trace.truth.to_numpy()
        ids = np.flatnonzero(fit_mask & (current.argmax(1) == truth))
        margins = current[ids, truth[ids], None] - current[ids]
        old_margins = legacy[ids, truth[ids], None] - legacy[ids]
        slope = old_margins - margins
        harm = slope < 0
        bound = float(np.min(margins[harm] / -slope[harm], initial=1.))
        alpha = .25 * min(1., bound)
        expected = priors[fold]
        assert expected['outer_fold'] == fold and expected['safety_fraction'] == .25
        assert alpha == expected['legacy_prior'] and bound == expected['linear_class_preserving_bound']
        assert expected['protected_legal_FIT_rows'] == len(ids)
        assert alpha > 0 and np.all(margins + alpha * slope >= .75 * margins - 1e-15)
        prior_reports.append(dict(outer=fold, protected_legal_FIT_rows=len(ids),
            alpha=alpha, preserving_bound=bound, saved_score_argmax_preserved=True,
            minimum_algebraic_retention_slack=float(np.min(margins + alpha * slope - .75 * margins))))
        paths.extend([folder / 'fit.json', folder / 'teacher.joblib', scores_path, qpath])
    result = dict(status='full_population_source_roles_and_legal_FIT_prior_independently_verified',
        original_gold_rows=len(gold), original_gold_class_mass=np.bincount(gold, minlength=3).tolist(),
        original_ASA_rows=len(trace), nested_legacy_roles=9, role_reports=role_reports, prior_reports=prior_reports,
        own_official_classifier_calls=0, own_official_feature_calls=0, own_gradients=0, own_fits=0, own_updates=0,
        training_activated=False, quality_acceptance=False,
        limits=['Saved-array arithmetic does not regenerate old classifier scores.',
                'Normalized OVA logits preserve argmax, but are not proven calibrated probabilities.',
                'Positive algebraic prior protects initialization only, not learned fusion or unseen sources.',
                'Existing inspected development folds are not a new blind test.'],
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    OUT.mkdir()
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'original_gold_rows', 'original_gold_class_mass', 'prior_reports']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
