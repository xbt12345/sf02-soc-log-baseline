"""Read-only model replay, input perturbation and artifact identity verification."""
import argparse
import collections
import gc
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(a):
    started = time.perf_counter()
    root, run, out = Path(a.root).resolve(), Path(a.run).resolve(), Path(a.out).resolve()
    assert not out.exists(), 'Preserve verification receipts'
    runtime = run / 'frozen_training_runtime'
    sys.path.insert(0, str(runtime))
    import v40_core as core
    from audit_v37_prepared import variants
    from run_v39_prepare import sha, save
    from run_v40_train import metrics

    stage = run / 'prepared'
    receipt = read(stage / 'complete.json')
    prep = root / receipt['v39_prepared']
    assert sha(prep / 'complete.json') == receipt['v39_prepared_receipt_sha256']
    old_receipt = read(prep / 'complete.json')
    verified = {}

    def check(path, expected):
        actual = sha(path)
        assert actual == expected, str(path)
        verified[path.relative_to(root).as_posix()] = actual

    check(root / 'data/official/train.parquet', receipt['official_sha256'])
    assert receipt['official_sha256'] == old_receipt['official_sha256']
    for base, values in [(stage, receipt['files']), (runtime, receipt['runtime_sources']),
                         (prep, old_receipt['files'])]:
        for name, expected in values.items():
            check(base / name, expected)
    input_checks = read(run / 'input_checks.json')
    assert input_checks['all_checks_passed']
    check(stage / 'complete.json', input_checks['prepared_sha256'])
    check(stage / 'near_copy_review.json', input_checks['near_copy_review_sha256'])
    check(run / 'raw_inference_cases.json', input_checks['raw_inference_cases_sha256'])
    check(root / 'training/audit_v40_inputs.py', input_checks['script_sha256'])
    selection = read(run / 'primary_review/selection.json')
    check(run / 'primary_review/primary_review.json', selection['primary_review_sha256'])
    check(stage / 'configuration.json', selection['configuration_sha256'])
    check(root / 'training/review_v40_primary.py', selection['script_sha256'])
    assert selection['selected_view'] is None and not selection['proceed_to_stress']
    assert not (run / 'old_protocol_stress').exists()
    assert len(list((run / 'primary').glob('fold_*/*/model.joblib'))) == 9

    pr = pq.read_table(prep / 'projections.parquet', columns=['text', 'facts']).to_pandas()
    cases = read(run / 'raw_inference_cases.json')
    records, expected_ids, counts = [], [], collections.Counter()
    for item in cases:
        raw, pid = item['raw'], item['projection_id']
        records.append({'message_sanitized': raw}); expected_ids.append(pid)
        counts['original'] += 1
        records.append({'message_sanitized': raw, 'timestamp': '2099-01-01T00:00:00Z',
                        'product_name': None, 'vendor_name': 'unseen_vendor',
                        'src_ip': '203.0.113.255', 'dst_ip': '192.0.2.1',
                        'username': 'unseen_identity', 'event_id': 'replacement',
                        'pipeline': 'replacement', 'src_port': 65535,
                        'src_host': 'new_host', 'dst_host': 'new_host',
                        'label_binary': 'suspicious'})
        expected_ids.append(pid); counts['outside_metadata_replaced'] += 1
        if item['route'] == 'clock_probe':
            changes = []
            for clock in ['2099-01-01T00:00:00Z', '2011-12-31T23:59:59Z']:
                changed = core.prior.ISO_LITERAL.sub(clock, raw)
                changed = core.prior.TASK_BOUNDARY.sub(lambda m: m[1] + clock + m[3], changed)
                changes.append(('absolute_clock', changed))
        else:
            changes = variants(raw, item['route'])
            if item['route'] == 'asa':
                changes.append(('address_interface', re.sub(r'dmz[-_]\d+', 'dmz-999',
                    re.sub(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)', '203.0.113.99', raw))))
        for name, changed in changes:
            records.append({'message_sanitized': changed}); expected_ids.append(pid)
            counts[name] += 1
    unique_ids, lookup = np.unique(expected_ids, return_inverse=True)
    original_text = pr.text.iloc[unique_ids].tolist()
    original_facts = [json.loads(v) for v in pr.facts.iloc[unique_ids]]
    model_results = []
    for fold in range(3):
        for view in ['R', 'N', 'I']:
            folder = run / ('primary/fold_%s/%s' % (fold, view))
            bound = read(folder / 'complete.json')
            for name, key in [('model.joblib', 'model_sha256'), ('evaluation.parquet', 'predictions_sha256'),
                              ('report.json', 'report_sha256'), ('binding.json', 'binding_sha256')]:
                check(folder / name, bound[key])
            binding = read(folder / 'binding.json')
            assert binding['prepared_sha256'] == sha(stage / 'complete.json')
            assert binding['runtime_sources'] == receipt['runtime_sources']
            baseline = prep.parent / ('primary/fold_%s/SEMANTIC' % fold)
            check(baseline / 'model.joblib', binding['reference_model_sha256'])
            model = joblib.load(folder / 'model.joblib')
            expected = model['model'].predict_proba(core.matrix(model, original_text, original_facts))
            raw_difference, flips = 0.0, 0
            for start in range(0, len(records), 512):
                p = core.predict_records(model, records[start:start+512])
                ref = expected[lookup[start:start+512]]
                raw_difference = max(raw_difference, float(np.abs(p-ref).max()))
                flips += int((p.argmax(1) != ref.argmax(1)).sum())
            assert raw_difference <= 1e-10 and flips == 0
            evaluation = pq.read_table(folder / 'evaluation.parquet').to_pandas()
            pids, inverse = np.unique(evaluation.projection_id.to_numpy(), return_inverse=True)
            p = model['model'].predict_proba(core.matrix(model, pr.text.iloc[pids].tolist(),
                [json.loads(v) for v in pr.facts.iloc[pids]]))[inverse]
            saved = evaluation[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy()
            full_difference = float(np.abs(p-saved).max())
            assert full_difference <= 1e-10
            recounted = metrics(evaluation.label_index.to_numpy(dtype=int), p)
            reported = read(folder / 'report.json')['evaluation']
            assert recounted == reported
            item = {'fold': fold, 'view': view, 'evaluation_rows': len(evaluation),
                    'evaluation_probability_max_difference': full_difference,
                    'actual_raw_inference_calls': len(records), 'raw_probability_max_difference': raw_difference,
                    'raw_prediction_flips': flips, 'all_metrics_recomputed_match': True,
                    'model_sha256': bound['model_sha256']}
            model_results.append(item)
            print(json.dumps(item), flush=True)
            del model, evaluation, p, saved
            gc.collect()
    out.mkdir(parents=True)
    save(out / 'verification.json', {'all_checks_passed': True, 'model_checks': model_results,
         'verified_files': verified, 'raw_inference_cases': len(cases), 'calls_per_model': len(records),
         'inference_case_counts': dict(counts), 'new_primary_fits': 9, 'new_stress_fits': 0,
         'stress_gate_obeyed': True, 'quality_accepted': False, 'script_sha256': sha(__file__),
         'seconds': time.perf_counter()-started,
         'scope': 'Actual saved-model replay, specified raw-input invariance and byte identity; not blind or external quality validation',
         'semantic_correction': 'Missing maps to an identifiable zero vector. No active dedicated missing-code term does not mean missingness information is erased.'})
    print(json.dumps({'all_checks_passed': True, 'quality_accepted': False, 'models': len(model_results)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
