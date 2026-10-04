"""Final raw-message inference, source identity and completion evidence."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
import v39_core as core
from run_v39_prepare import sha, save
from audit_v37_prepared import variants


def main(a):
    root = Path(a.root); out = Path(a.output)
    if out.exists():
        raise FileExistsError('Preserve verification')
    run = root / 'artifacts/v39_local_r2_20260913'
    prepared = run / 'prepared'
    receipt = json.loads((prepared / 'complete.json').read_text(encoding='utf-8'))
    checks = {'official_data_unchanged': sha(root / 'data/official/train.parquet') == receipt['official_sha256'],
        'prepared_files_unchanged': all(sha(prepared / n) == h for n, h in receipt['files'].items())}
    review = json.loads((root / 'evidence/2026-09-13/v39_execution/primary_review/primary_review.json').read_text(encoding='utf-8'))
    for model in review['model_bindings']:
        checks['model_identity_' + model['view'] + '_' + str(model['fold'])] = sha(root / model['path'] / 'model.joblib') == model['model_sha256']
    first = root / 'artifacts/v39_local_r1_20260913'
    for name, binding, runtime in [
        ('primary_v10', first / 'primary/binding.json', first / 'frozen_training_runtime'),
        ('primary_v11', run / 'primary/binding.json', run / 'frozen_training_runtime'),
        ('stress_v11', run / 'old_protocol_stress/binding.json', run / 'frozen_training_runtime')]:
        bound = json.loads(binding.read_text(encoding='utf-8'))
        checks[name + '_source_identity'] = all(sha(runtime / n) == h for n, h in bound['sources'].items())
    cases = json.loads((prepared / 'audit_cases.json').read_text(encoding='utf-8'))
    projections = pq.read_table(prepared / 'projections.parquet', columns=['text', 'facts']).to_pandas()
    texts, facts, originals, changed = [], [], [], []
    for case in cases:
        original = len(texts)
        p = core.prepare_message(case['raw'])
        texts.append(p['text']); facts.append(p['facts'])
        for kind, raw in variants(case['raw'], case['route']):
            p = core.prepare_message(raw)
            originals.append(original); changed.append(len(texts))
            texts.append(p['text']); facts.append(p['facts'])
    results = []
    folders = [run / 'primary' / ('fold_' + str(k)) / 'SEMANTIC' for k in range(3)] + [run / 'old_protocol_stress']
    for folder in folders:
        bound = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
        assert sha(folder / 'model.joblib') == bound['model_sha256']
        bundle = joblib.load(folder / 'model.joblib')
        predictions = []
        for begin in range(0, len(texts), 256):
            x = sparse.hstack([bundle['text_encoder'].transform(texts[begin:begin+256]),
                bundle['fact_encoder'].transform(facts[begin:begin+256])], format='csr')
            predictions.append(bundle['model'].predict_proba(x))
        p = np.concatenate(predictions)
        diff = float(np.abs(p[originals] - p[changed]).max()) if originals else 0.
        flips = int((p[originals].argmax(axis=1) != p[changed].argmax(axis=1)).sum()) if originals else 0
        assert diff <= 1e-10 and flips == 0
        # Full saved evaluation replay through the serialized encoders.
        d = pq.read_table(folder / 'evaluation.parquet').to_pandas()
        keys, ids = np.unique(d.projection_id.to_numpy(), return_inverse=True)
        selected = projections.iloc[keys]
        x = sparse.hstack([bundle['text_encoder'].transform(selected.text.tolist()),
            bundle['fact_encoder'].transform([json.loads(v) for v in selected.facts])], format='csr')
        z = x.dot(bundle['model'].coef_.T) + bundle['model'].intercept_
        z -= z.max(axis=1, keepdims=True); e = np.exp(z); manual = (e/e.sum(axis=1, keepdims=True))[ids]
        expected = d[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy()
        replay_diff = float(np.abs(manual - expected).max())
        assert replay_diff <= 1e-10
        assert np.array_equal(manual.argmax(axis=1), expected.argmax(axis=1))
        results.append({'model': folder.relative_to(root).as_posix(), 'model_sha256': bound['model_sha256'],
            'saved_rows_replayed': len(d), 'probability_max_difference': replay_diff,
            'wrapper_pairs': len(originals), 'wrapper_max_difference': diff, 'wrapper_flips': flips})
    checks['four_semantic_models_full_replay_and_raw_wrappers'] = True
    old = root / 'artifacts/v38_local_r1_20260913/equal_classifiers_attempt2'
    for view in ('A_SEMANTIC', 'B_DESTINATION', 'C_BOTH'):
        r = json.loads((old / view / 'complete.json').read_text(encoding='utf-8'))
        checks['old_model_preserved_' + view] = sha(old / view / 'model.joblib') == r['model_sha256']
    final = {'checks': checks, 'all_execution_checks_passed': all(checks.values()),
        'raw_real_cases': len(cases), 'raw_inference_inputs_per_model': len(texts), 'models': results,
        'actual_fits_this_execution': 13, 'primary_comparison_fits': 9, 'superseded_semantic_fits': 3, 'stress_fits': 1,
        'unit_tests_passed': 13, 'optional_pair_fits': 0,
        'quality_accepted': False, 'platform_used': False, 'competition_submission_created': False,
        'scope': 'Runtime, input, protocol and specified inference checks; not external quality acceptance',
        'script_sha256': sha(Path(__file__))}
    save(out, final)
    print(json.dumps(final, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    main(p.parse_args())
