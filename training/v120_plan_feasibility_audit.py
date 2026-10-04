"""Audit the next plan using saved data and batch schedules; never fit/update a model.

This is a retrospective feasibility audit, not a preregistered efficacy experiment.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_review import ROOT, check_bindings, read, sha, class_counts
from v117_gradient_batch_audit import stratified
from v107_matched_training import FOLDS, ROWS, FID, DEST as TEACHERS

DEST = ROOT / 'artifacts/v120_plan_refinement_20260929'
PREVIOUS = ROOT / 'artifacts/v116_nested_selection_20260929'


def save(path, value):
    import json
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    if DEST.exists():
        raise FileExistsError('Keep the prior audit immutable: ' + str(DEST))
    prior = ROOT / 'artifacts/v119_review_constraints_20260929/review_receipt.json'
    check_bindings(read(prior)['artifact_sha256'])
    manifest_path = PREVIOUS / 'inner_split_manifest.parquet'
    decisions_path = PREVIOUS / 'OOF_ASA_decisions.parquet'
    manifest = pd.read_parquet(manifest_path)
    decisions = pd.read_parquet(decisions_path)
    assert not decisions.row_position.duplicated().any()
    n = int(manifest.local.max()) + 1
    results, support = [], []
    for fold in range(3):
        train = manifest[manifest.outer_fold.eq(fold) & manifest.fold.ne(fold)]
        test = manifest[manifest.outer_fold.eq(fold) & manifest.fold.eq(fold)].copy()
        assert not set(train.root) & set(test.root)
        assert not train.row_position.duplicated().any()
        counts = np.bincount(train.local.to_numpy() * 3 + train.truth.to_numpy(),
                             minlength=n * 3).reshape(-1, 3)
        assert counts[:, 0].sum() == 0, 'Existing ASA planner supports M/S mass only.'
        used = np.flatnonzero(counts.sum(1))
        batches = math.ceil(len(used) / 256)
        rng_a = np.random.default_rng(10201 + fold)
        rng_b = np.random.default_rng(10201 + fold)
        forward_a = forward_b = oversized = micro_steps_if_naive = 0
        minimum, maximum, max_error = n, 0, 0.0
        schedule_hashes = []
        import hashlib
        for epoch in range(1, 26):
            order = used[rng_a.permutation(len(used))]
            a = [(order[i:i+256], counts[order[i:i+256]].astype(float))
                 for i in range(0, len(used), 256)]
            b = stratified(counts, batches, rng_b)
            assert len(a) == len(b) == batches
            for name, plan in [('A', a), ('B', b)]:
                restored = np.zeros_like(counts, dtype=float)
                for ids, mass in plan:
                    assert len(ids) and len(ids) == len(set(ids))
                    assert np.isfinite(mass).all() and (mass >= 0).all()
                    restored[ids] += mass
                error = float(np.max(np.abs(restored - counts)))
                assert error <= 1e-8, (fold, epoch, name, error)
                max_error = max(max_error, error)
            h = hashlib.sha256()
            for ids, mass in b:
                h.update(ids.astype('<i8').tobytes()); h.update(mass.astype('<f8').tobytes())
                assert np.allclose(mass.sum(0), counts.sum(0) / batches, atol=1e-8, rtol=0)
                maximum = max(maximum, len(ids)); minimum = min(minimum, len(ids))
                oversized += int(len(ids) > 256)
                micro_steps_if_naive += math.ceil(len(ids) / 256)
            schedule_hashes.append(h.hexdigest())
            forward_a += sum(len(ids) for ids, _ in a)
            forward_b += sum(len(ids) for ids, _ in b)
        icmp = train[train.parameter.eq('["icmp",3,13]')]
        results.append({'fold': fold, 'train_rows': len(train), 'class_mass': counts.sum(0).tolist(),
            'unique_inputs': len(used), 'logical_batches_per_epoch': batches,
            'registered_optimizer_steps_per_arm': 25 * batches,
            'B_unique_inputs_per_batch_min': minimum, 'B_unique_inputs_per_batch_max': maximum,
            'B_batches_exceeding_256': oversized, 'B_steps_if_each_microbatch_updates': micro_steps_if_naive,
            'forward_input_evaluations_A': forward_a, 'forward_input_evaluations_B': forward_b,
            'maximum_input_label_mass_reconstruction_error': max_error,
            'unique_epoch_schedule_hashes': len(set(schedule_hashes)), 'epoch_B_schedule_sha256': schedule_hashes,
            'icmp_3_13_fit_M': int(icmp.truth.eq(1).sum()), 'icmp_3_13_fit_S': int(icmp.truth.eq(2).sum())})
        known = set(train.parameter.dropna())
        roots = train[train.parameter.notna()].groupby(['parameter', 'truth']).root.nunique().to_dict()
        def bucket(row):
            if pd.isna(row.parameter): return 'parameter_missing'
            if row.parameter not in known: return 'parameter_unseen'
            count = roots.get((row.parameter, row.truth), 0)
            return ('parameter_seen_class_absent' if count == 0 else
                    'same_class_one_root' if count == 1 else 'same_class_multiple_roots')
        test['support_bin'] = test.apply(bucket, axis=1)
        test = test.merge(decisions[['row_position', 'A_epoch25', 'B_selected']],
                          on='row_position', validate='one_to_one')
        assert test[['A_epoch25', 'B_selected']].notna().all().all()
        for (label, name), q in test.groupby(['truth', 'support_bin']):
            support.append({'fold': fold, 'truth': int(label), 'support_bin': name, 'rows': len(q),
                'roots': int(q.root.nunique()), 'A25_errors': int(q.A_epoch25.ne(q.truth).sum()),
                'selected_errors': int(q.B_selected.ne(q.truth).sum())})
    # Reconstruct the historical deployed composition from stored, hash-verified OOF scores.
    r = pd.read_parquet(ROWS, columns=['row_position', 'route', 'label_index'])
    f = pd.read_parquet(FOLDS, columns=['row_position', 'proposed_fold'])
    assert np.array_equal(r.row_position, np.arange(len(r)))
    assert np.array_equal(f.row_position, r.row_position)
    feature_ids = np.load(FID, mmap_mode='r')
    teacher = np.empty(len(r), dtype=np.int8)
    sources = [prior, manifest_path, decisions_path, ROWS, FOLDS, FID,
               ROOT / 'training/v117_gradient_batch_audit.py', Path(__file__)]
    for fold in range(3):
        path = TEACHERS / f'fold{fold}_N1_teacher/scores_all_input_ids.npy'
        receipt = path.parent / 'fit.json'
        assert sha(path) == read(receipt)['scores_sha256']
        score = np.load(path, mmap_mode='r')
        ids = np.flatnonzero(f.proposed_fold.eq(fold))
        teacher[ids] = score[np.asarray(feature_ids[ids], dtype=np.int64)].argmax(1)
        sources += [path, receipt]
    pos = np.flatnonzero(r.route.eq('asa'))
    assert np.array_equal(pos, decisions.row_position)
    assert np.array_equal(r.label_index.to_numpy()[pos], decisions.truth)
    gate = teacher[pos] != 0
    output = {'A': teacher.copy(), 'B': teacher.copy()}
    historical = read(PREVIOUS / 'evaluation.json')
    routes = {}
    for arm, field in [('A', 'A_epoch25'), ('B', 'B_selected')]:
        raw = decisions[field].to_numpy()
        output[arm][pos[gate]] = raw[gate]
        frame = pd.DataFrame({'truth': r.label_index, 'prediction': output[arm]})
        classes = class_counts(frame, 'prediction')
        errors = int(frame.truth.ne(frame.prediction).sum())
        assert errors == historical['full_task'][arm]['errors']
        raw_error = raw != decisions.truth.to_numpy()
        routed_error = output[arm][pos] != decisions.truth.to_numpy()
        routes[arm] = {'raw_ASA_errors': int(raw_error.sum()), 'routed_ASA_errors': int(routed_error.sum()),
            'full_task_errors': errors, 'full_task_class_metrics': classes,
            'gate_excluded_ASA_rows': int((~gate).sum()),
            'raw_correct_but_gate_to_benign': int(((~raw_error) & (~gate)).sum()),
            'raw_wrong_also_gate_to_benign': int((raw_error & (~gate)).sum()),
            'non_ASA_errors': int((frame.truth.ne(frame.prediction) & r.route.ne('asa')).sum())}
    assert np.array_equal(output['A'][~r.route.eq('asa')], output['B'][~r.route.eq('asa')])
    sources.append(PREVIOUS / 'evaluation.json')
    DEST.mkdir()
    save(DEST / 'feasibility_audit.json', {'status': 'retrospective_zero_fit_plan_audit',
        'classifier_fits': 0, 'optimizer_steps': 0, 'model_promoted': False,
        'batch_schedule': results, 'support_slices': support, 'routing_replay': routes,
        'limits': ['One primary-seed schedule simulation, not training or classification efficacy.',
                   'Support labels used only for retrospective diagnosis, never a model input or sampling weights.',
                   'Counts preserve the inherited official-data mapping; original parquet not independently rehashed here.',
                   'Shared outer folds are repeatedly inspected development data, not blind validation.'],
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}})
    print({'batch_over_256': sum(q['B_batches_exceeding_256'] for q in results),
           'logical_steps': sum(q['registered_optimizer_steps_per_arm'] for q in results),
           'routing_errors': {a: {k: v for k, v in q.items() if k != 'full_task_class_metrics'} for a, q in routes.items()},
           'new_fits': 0})


if __name__ == '__main__':
    main()
