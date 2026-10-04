"""Read-only audit of completed V158 fits and logging-only continuation.

No classifier/feature function, optimizer, gradient or official training call.
Run once per immutable snapshot; newer completed fits require a new snapshot.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v158_current_pipeline_OOF_trial_20261001'
STAGES = ['V135_R_decay_full_network', 'V138_H_L_readout',
          'V140_C_readout', 'V142_S2_second', 'V146_A_second']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_identity(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def statements(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text(encoding='utf-8')).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def events(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def recompute(frame, probability, pure, before, floor):
    y = frame.truth.to_numpy()
    pred = probability[frame.local].argmax(1)
    bad = pred != y
    was_correct = before == y
    mask = pure[frame.local.to_numpy()]
    result = {name: int(value.sum()) for name, value in {
        'M_errors': bad & (y == 1), 'S_errors': bad & (y == 2),
        'pure_M_errors': bad & mask & (y == 1),
        'pure_S_errors': bad & mask & (y == 2),
        'old_correct_pure_regressions': bad & mask & was_correct,
        'repaired_vs_start': ~bad & ~was_correct,
        'new_errors_vs_start': bad & was_correct}.items()}
    result['empirical_minimum_errors'] = floor
    result['mastered'] = (result['M_errors'] == 0 and
                          result['S_errors'] == floor and
                          result['pure_M_errors'] == result['pure_S_errors'] == 0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    assert args.snapshot.replace('_', '').isalnum()
    out = ROOT / ('artifacts/v158_independent_completed_fit_audit_' + args.snapshot)
    assert not out.exists()
    # Snapshot only completed receipts. Active fits are neither discarded nor scored.
    receipt_paths = sorted(RUN.glob('outer*_inner*/*/fit.json'))
    assert receipt_paths
    old_path = ROOT / 'training/review_policy/v158_complete_nested_trial_plan.json'
    new_path = ROOT / 'training/review_policy/v158_complete_nested_trial_v2_plan.json'
    old, new = read(old_path), read(new_path)
    keys = set(old) | set(new)
    changed = sorted(key for key in keys if old.get(key) != new.get(key))
    assert changed == ['execution_entries', 'source_sha256', 'technical_retry', 'version']
    retry = new['technical_retry']
    assert retry['inherited_completed_fits'] == 1
    assert retry['inherited_batch_gradients'] == retry['inherited_updates'] == 6300
    assert retry['additional_total_fits_max'] == 59
    assert retry['cumulative_fits_max'] == new['total_new_fits_max'] == 60
    assert retry['no_completed_fit_repeated'] and retry['no_budget_reset']
    prior_source = ROOT / 'training/v158_nested_base_train.py'
    current_source = ROOT / 'training/v158_nested_base_train_v2.py'
    a, b = statements(prior_source), statements(current_source)
    invariant_functions = ['Counted', 'load_role', 'full_network', 'dense',
                           'log_rows', 'source_stats', 'expected_next', 'folder_for']
    assert all(a[name] == b[name] for name in invariant_functions)
    original_legacy = (ROOT / 'training/v158_legacy_nested_train.py').read_text(encoding='utf-8')
    expected_legacy = original_legacy.replace('from v158_nested_runtime import ',
                                             'from v158_nested_runtime_v2 import ')
    expected_legacy = expected_legacy.replace("OUT/'run_seal.json'", "OUT/'run_seal_v2.json'")
    assert expected_legacy == (ROOT / 'training/v158_legacy_nested_train_v2.py').read_text(encoding='utf-8')
    failure = read(RUN / 'execution_failure.json')
    assert failure['error_type'] == 'TypeError'
    assert 'multiple values' in failure['error'] and "'stage'" in failure['error']
    assert "save(folder/'fit.json',result);emit(stage=" in prior_source.read_text(encoding='utf-8')
    seals = [read(RUN / 'run_seal.json'), read(RUN / 'run_seal_v2.json')]
    assert seals[0]['plan_sha256'] == sha(old_path)
    assert seals[1]['plan_sha256'] == sha(new_path)
    union = {}
    for seal in seals:
        for name, identity in seal['source_sha256'].items():
            assert name not in union or union[name] == identity, name
            union[name] = identity
    for name, identity in union.items():
        assert sha(ROOT / name) == identity, name
    registration = read(RUN / 'registration_v2.json')
    assert registration['inherited_completed_fits'] == 1
    assert registration['total_new_fits_max'] == 60
    assert registration['seal_sha256'] == sha(RUN / 'run_seal_v2.json')
    assert registration['cumulative_setup_dummy_forwards'] == 2
    assert registration['cumulative_setup_dummy_gradients'] == 2
    assert registration['setup_dummy_optimizer_updates'] == 0
    trace_path = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    key_path = ROOT / 'artifacts/v130_learning_review_20260930_r2/training_role_error_ledger.parquet'
    official_path = ROOT / 'data/official/train.parquet'
    trace = pd.read_parquet(trace_path, columns=['row_position', 'local', 'root', 'fold', 'truth'])
    gold = pd.read_parquet(official_path, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy(np.int8)
    assert len(gold) == 2056871 and len(trace) == 112807
    assert np.array_equal(gold[trace.row_position], trace.truth)
    keys = pd.read_parquet(key_path, columns=['local', 'canonical_key'])
    assert keys.groupby('local').canonical_key.nunique().max() == 1
    lookup = keys.drop_duplicates('local').set_index('local').canonical_key
    trace['canonical_key'] = trace.local.map(lookup)
    assert trace.canonical_key.notna().all()
    paths = [Path(__file__), old_path, new_path, prior_source, current_source,
             RUN / 'run_seal.json', RUN / 'run_seal_v2.json', RUN / 'execution_failure.json',
             RUN / 'registration_v2.json', trace_path, key_path, official_path]
    reports = []
    for receipt_path in receipt_paths:
        receipt = read(receipt_path)
        folder = receipt_path.parent
        fold, excluded, stage = receipt['outer_fold'], receipt['excluded_inner'], receipt['stage']
        assert stage in STAGES and not receipt['selected_by_score']
        def inner(root):
            key = f'V128|split=12801|outer={fold}|root={root}'
            return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % 3
        legal = trace.fold.ne(fold)
        inner_roles = trace.root.map(inner)
        frame = trace[legal & inner_roles.ne(excluded)].copy()
        query_roots = set(trace.loc[legal & inner_roles.eq(excluded), 'root'])
        assert not set(frame.root) & query_roots
        assert not set(frame.root) & set(trace.loc[~legal, 'root'])
        reference = pd.read_parquet(folder / 'FIT_reference.parquet')
        for column in ['row_position', 'local', 'root', 'fold', 'truth', 'canonical_key']:
            assert np.array_equal(reference[column], frame[column]), (str(folder), column)
        started = read(folder / 'started.json')
        assert started['fit_roots'] == sorted(set(frame.root))
        assert started['query_or_outer_labels_used'] == 0
        mass = np.bincount(frame.truth, minlength=3).tolist()
        assert receipt['fit_rows'] == len(frame) and receipt['class_mass'] == mass
        canonical = frame.groupby(['canonical_key', 'truth']).size().unstack(fill_value=0)
        floor = int((canonical.sum(1) - canonical.max(1)).sum())
        assert floor == receipt['canonical_label_collision_floor']
        pure_keys = set(canonical.index[(canonical > 0).sum(1) == 1])
        pure = np.zeros(22546, bool)
        for loc, key in frame[['local', 'canonical_key']].drop_duplicates().itertuples(index=False):
            pure[loc] = key in pure_keys
        endpoint = folder / 'endpoint.pt'
        logits_path = folder / 'member_logits.npy'
        probability_path = folder / 'mean_probability.npy'
        assert sha(endpoint) == receipt['source_model_sha256']
        assert sha(logits_path) == receipt['logits_sha256']
        state_file = torch.load(endpoint, map_location='cpu', weights_only=True)
        state = state_file['model' if stage == STAGES[0] else 'state']
        identity = tensor_identity(state)
        logits = np.load(logits_path)
        q = np.load(probability_path)
        assert logits.shape == (22546, 16, 3) and q.shape == (22546, 3)
        assert np.isfinite(logits).all() and np.isfinite(q).all()
        exponent = np.exp(logits.astype(np.float64) - logits.max(-1, keepdims=True))
        recomposed = (exponent / exponent.sum(-1, keepdims=True)).mean(1)
        tolerance = 2e-7 if logits.dtype == np.float32 else 2e-14
        gap = float(np.abs(q - recomposed).max())
        assert gap <= tolerance
        assert np.array_equal(q.argmax(1), recomposed.argmax(1))
        initial_path = folder / 'update0_FIT_rows.parquet'
        initial = pd.read_parquet(initial_path)
        assert np.array_equal(initial.row_position, frame.row_position)
        before = initial.pred.to_numpy()
        stats = recompute(frame, q, pure, before, floor)
        progress_path = folder / 'progress.json'
        if progress_path.exists():
            history = read(progress_path)
            assert history and stats == history[-1]['stats']
            key = 'parameter_sha256' if stage == STAGES[0] else 'state_sha256'
            assert identity == history[-1][key]
        else:
            assert stage != STAGES[0] and receipt['accepted_updates'] == 0
            assert receipt['termination'] in ['numerical_no_change', 'gradient_budget_trial_rolled_back', 'zero_gradient', 'no_feasible_step']
            initial_state_path = folder / 'accepted0.pt'
            initial_state = torch.load(initial_state_path, map_location='cpu', weights_only=True)['state']
            assert identity == tensor_identity(initial_state)
            saved_initial = initial[['p0', 'p1', 'p2']].to_numpy()
            assert np.max(np.abs(q[frame.local] - saved_initial)) <= 2e-12
            assert np.array_equal(q[frame.local].argmax(1), before)
            history = []
            paths.append(initial_state_path)
        assert receipt['last5_FIT_stats'] == [row['stats'] for row in history[-5:]]
        call_events = events(folder / 'classifier_calls.jsonl')
        attempts = [v for v in call_events if v['event'] == 'attempt']
        completed = [v for v in call_events if v['event'] == 'completed']
        assert len(attempts) == len(completed) == receipt['classifier_forward_calls']
        assert [v['attempt'] for v in attempts] == list(range(1, len(attempts) + 1))
        assert [v['completed'] for v in completed] == list(range(1, len(completed) + 1))
        budget = new['role_budgets'][fold * 3 + excluded]
        if stage == STAGES[0]:
            assert len(history) == 101 and [v['epoch'] for v in history] == list(range(101))
            assert receipt['termination'] == 'fixed_100_epochs'
            assert receipt['accepted_updates'] == receipt['batch_gradients'] == budget['full_network_gradient_updates']
            assert len(attempts) == budget['full_network_forward_cap']
            logs = events(folder / 'steps.jsonl')
            updates = [v for v in logs if v['event'] == 'batch_gradient_and_update_completed']
            gradient_attempts = [v for v in logs if v['event'] == 'batch_gradient_attempt']
            assert len(updates) == len(gradient_attempts) == receipt['batch_gradients']
            for epoch in range(1, 101):
                rows = [v for v in updates if v['epoch'] == epoch]
                assert np.sum([v['original_class_mass'] for v in rows], axis=0).tolist() == mass
        else:
            assert stats == receipt['endpoint_FIT_stats']
            assert len(history) == receipt['accepted_updates'] <= 200
            gradient_events = events(folder / 'gradients.jsonl')
            gradients = [v for v in gradient_events if v['event'] == 'full_gradient_completed']
            assert len(gradients) == receipt['full_gradients'] <= 200
            assert all(v['class_mass'] == mass for v in gradients)
            bound = budget['Armijo_stage_forward_cap' if stage == STAGES[-1] else 'LBFGS_stage_forward_cap']
            assert len(attempts) <= bound
            if stage == STAGES[-1]:
                assert stats['mastered'] and receipt['baseline_correct_FIT_regressions'] == 0
        reports.append(dict(outer=fold, excluded_inner=excluded, stage=stage,
            actual_fit_rows=len(frame), original_class_mass=mass, canonical_floor=floor,
            independently_recomputed_stats=stats, batch_gradients=receipt['batch_gradients'],
            full_gradients=receipt['full_gradients'], updates=receipt['accepted_updates'],
            classifier_forward_calls=len(attempts), saved_probability_recomposition_gap=gap,
            source_and_query_roots_excluded=True, endpoint_state_matches_last_accepted=True))
        paths += [receipt_path, folder / 'started.json', folder / 'FIT_reference.parquet',
                  endpoint, logits_path, probability_path, initial_path]
        if progress_path.exists():
            paths.append(progress_path)
    result = dict(status='completed_fit_snapshot_and_logging_continuation_independently_verified',
        snapshot=args.snapshot, actual_current_chain_fits_verified=len(reports),
        changed_plan_fields=changed, unchanged_learning_functions=invariant_functions,
        original_logging_failure_preserved=True, completed_first_fit_not_retrained=True,
        cumulative_fit_cap=60, physical_seal_union_verified=len(union),
        original_gold_rows=len(gold), reports=reports,
        own_official_classifier_calls=0, own_feature_calls=0, own_gradients=0, own_updates=0, own_fits=0,
        full_task_quality_acceptance=False,
        limits=['Explicit zero-accepted endpoint verified against initial tensors and probabilities; no missing histories silently skipped.',
                'Immutable completed-fit snapshot; other fits may still be running.',
                'CPU saved-array arithmetic does not independently replay model functions.',
                'FIT mastery is not OOF classification or cross-source quality.',
                'Canonical collision floor is for this numerical input, not raw-log impossibility.'],
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    out.mkdir()
    (out / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'actual_current_chain_fits_verified',
          'physical_seal_union_verified', 'reports']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
