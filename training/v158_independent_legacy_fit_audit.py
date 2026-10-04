"""Verify nine full-population N1 fits from gold, saved outputs and logs only."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v158_current_pipeline_OOF_trial_20261001'
OUT = ROOT / 'artifacts/v158_independent_legacy_fit_audit_20261001'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    assert not OUT.exists()
    planpath = ROOT / 'training/review_policy/v158_complete_nested_trial_v2_plan.json'
    sealpath = RUN / 'run_seal_v2.json'
    completionpath = RUN / 'legacy_phase_completion.json'
    plan, seal, completion = read(planpath), read(sealpath), read(completionpath)
    assert seal['plan_sha256'] == sha(planpath)
    paths = [Path(__file__), planpath, sealpath, completionpath]
    goldpath = ROOT / 'data/official/train.parquet'
    foldpath = ROOT / 'artifacts/v106_frozen_audit_20260928/proposed_body_closed_folds.parquet'
    rowpath = ROOT / 'artifacts/v75_four_arm_20260921_r2/rows.parquet'
    tracepath = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    gold = pd.read_parquet(goldpath, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy(np.int8)
    folds = pd.read_parquet(foldpath, columns=['row_position', 'root', 'proposed_fold'])
    rows = pd.read_parquet(rowpath, columns=['row_position', 'label_index'])
    trace = pd.read_parquet(tracepath, columns=['row_position', 'local', 'root', 'fold', 'truth'])
    assert len(gold) == len(folds) == len(rows) == 2056871
    assert np.array_equal(rows.row_position, np.arange(len(gold)))
    assert np.array_equal(folds.row_position, rows.row_position)
    assert np.array_equal(rows.label_index, gold)
    assert np.array_equal(gold[trace.row_position], trace.truth)
    paths += [goldpath, foldpath, rowpath, tracepath]
    reports = []
    for outer in range(3):
        def inner(root):
            key = f'V128|split=12801|outer={outer}|root={root}'
            return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % 3
        lookup = {root: inner(root) for root in folds.root.unique()}
        inner_all = folds.root.map(lookup).to_numpy()
        legal = folds.proposed_fold.ne(outer).to_numpy()
        trace_inner = trace.root.map(lookup).to_numpy()
        for excluded in range(3):
            folder = RUN / f'legacy_outer{outer}_inner{excluded}'
            receiptpath = folder / 'fit.json'
            startedpath = folder / 'started.json'
            receipt, started = read(receiptpath), read(startedpath)
            fit = legal & (inner_all != excluded)
            query = legal & (inner_all == excluded)
            fit_roots = set(folds.loc[fit, 'root'])
            assert not fit_roots & set(folds.loc[query, 'root'])
            assert not fit_roots & set(folds.loc[~legal, 'root'])
            mass = np.bincount(gold[fit], minlength=3).tolist()
            budget = plan['legacy_role_budgets'][outer * 3 + excluded]
            assert receipt['outer'] == started['outer'] == outer
            assert receipt['excluded_inner'] == started['excluded_inner'] == excluded
            assert receipt['class_mass'] == receipt['gradients_per_call_original_class_mass'] == mass == budget['fit_class_mass']
            assert receipt['fit_rows'] == int(fit.sum()) == budget['fit_rows']
            assert started['fit_roots'] == sorted(fit_roots)
            assert not started['prior_supervised_model_weights_loaded']
            assert receipt['seal_sha256'] == started['seal_sha256'] == sha(sealpath)
            assert not receipt['selected_by_score'] and receipt['query_or_outer_labels_used'] == 0
            assert receipt['new_fits'] == 1 and not receipt['quality_acceptance']
            modelpath, scorepath = folder / 'endpoint.joblib', folder / 'ASA_logits.npy'
            assert sha(modelpath) == receipt['model_sha256']
            assert sha(scorepath) == receipt['scores_sha256']
            model = joblib.load(modelpath)
            assert set(model) == {'coef', 'intercept'}
            assert model['coef'].shape == (66287, 3) and model['intercept'].shape == (3,)
            assert np.isfinite(model['coef']).all() and np.isfinite(model['intercept']).all()
            scores = np.load(scorepath)
            assert scores.shape == (22546, 3) and np.isfinite(scores).all()
            gradients = lines(folder / 'gradient_calls.jsonl')
            attempts = [item for item in gradients if item['event'] == 'gradient_attempt']
            completed = [item for item in gradients if item['event'] == 'gradient_completed']
            assert len(attempts) == len(completed) == receipt['full_gradient_evaluations'] <= 1000
            assert all(item['original_class_mass'] == mass for item in attempts)
            assert [item['call'] for item in attempts] == list(range(1, len(attempts)+1))
            assert [item['call'] for item in completed] == list(range(1, len(completed)+1))
            assert all(np.isfinite(item['objective']) and np.isfinite(item['gradient_inf']) for item in completed)
            progress = read(folder / 'progress.json')
            assert len(progress) == receipt['accepted_iterations'] <= 1000
            assert [item['iteration'] for item in progress] == list(range(1, len(progress)+1))
            forward = lines(folder / 'ASA_forward_calls.jsonl')
            assert sum(item['event'] == 'attempt' for item in forward) == sum(item['event'] == 'completed' for item in forward) == receipt['ASA_classifier_forward_chunks'] == 3
            original_query = trace.fold.ne(outer).to_numpy() & (trace_inner == excluded)
            query_truth = trace.loc[original_query, 'truth'].to_numpy()
            query_pred = scores[trace.loc[original_query, 'local']].argmax(1)
            reports.append(dict(outer=outer, excluded_inner=excluded, actual_fit_rows=int(fit.sum()),
                original_class_mass=mass, actual_full_gradients=len(completed),
                actual_accepted_iterations=len(progress), saved_original_ASA_query_rows=int(original_query.sum()),
                saved_query_class_mass=np.bincount(query_truth, minlength=3).tolist(),
                saved_query_class_errors=np.bincount(query_truth[query_pred != query_truth], minlength=3).tolist(),
                source_exclusion_verified=True, model_and_score_identity_verified=True))
            paths += [receiptpath, startedpath, modelpath, scorepath, folder/'gradient_calls.jsonl',
                      folder/'progress.json', folder/'ASA_forward_calls.jsonl']
    gradients = sum(item['actual_full_gradients'] for item in reports)
    iterations = sum(item['actual_accepted_iterations'] for item in reports)
    assert completion['new_fits'] == len(reports) == 9
    assert completion['full_gradients'] == gradients <= plan['legacy_full_gradient_cap']
    assert completion['accepted_iterations'] == iterations
    assert completion['ASA_classifier_forward_chunks'] == 27
    result = dict(status='nine_full_population_legacy_fits_independently_verified', original_gold_rows=len(gold),
        actual_fits=9, actual_full_gradients=gradients, actual_accepted_iterations=iterations,
        actual_saved_ASA_forward_chunks=27, reports=reports,
        own_official_classifier_calls=0, own_feature_calls=0, own_gradients=0, own_fits=0, own_updates=0,
        task_quality_acceptance=False,
        limits=['Saved score/gold counts are diagnostics, not a new classifier evaluation or fusion quality.',
            'Legacy logs do not include per-accepted-state class/source errors, Brier, or parameter identities.',
            'The legacy input and model function were not independently replayed.',
            'The currently inspected source folds are not a new blind external test.'],
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in paths})
    OUT.mkdir()
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ['status','actual_fits','actual_full_gradients','actual_accepted_iterations','reports']},ensure_ascii=False))


if __name__ == '__main__':
    main()
