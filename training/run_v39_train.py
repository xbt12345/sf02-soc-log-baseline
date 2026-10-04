"""Nine bounded, equal-row-weight fits on frozen v39 development folds."""
import argparse
import collections
import gc
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.metrics import roc_auc_score
import v39_core as core
from run_v39_prepare import sha, save
from run_v38_train import text_encoder, actual_keys
from audit_v37_prepared import variants


def metric(y, p):
    r = core.learning.cm_metrics(y, p.argmax(axis=1))
    wrong = p.argmax(axis=1) != y
    r['errors'] = int(wrong.sum())
    r['normal_errors'] = int((wrong & (y == 0)).sum())
    r['false_alerts_per_10000_normal'] = float(r['normal_errors'] * 10000 / (y == 0).sum()) if (y == 0).any() else None
    return r


def real_input_checks(prepared, projections):
    cases = json.loads((prepared / 'audit_cases.json').read_text(encoding='utf-8'))
    checks = collections.Counter(); failures = []; predictions_cases = []
    for case in cases:
        raw, route = case['raw'], case['route']
        direct = core.prepare_message(raw)
        cached = projections.iloc[case['projection_id']]
        if direct['text'] != cached.text or core.canonical(direct['facts']) != cached.facts:
            failures.append(['cached_direct', case['row_position']])
        previous = core.previous.prior.prepare_message(raw)
        key = core.body_identity(raw, route, previous['evidence'], case['row_position'])[0]
        changes = variants(raw, route)
        # Preserve existing actor-independent behavior while stress-testing a
        # globally changed address/identity and transport wrapper.
        if route == 'asa':
            import re
            changes.append(('address_and_interface_identity', re.sub(r'dmz[-_]\d+', 'dmz-999',
                re.sub(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)', '203.0.113.99', raw))))
        for name, changed in changes:
            other = core.prepare_message(changed)
            if (direct['text'], direct['facts']) != (other['text'], other['facts']):
                failures.append(['feature_invariance', case['row_position'], name])
            checks[name] += 1
        changed = core.prepare_record({'message_sanitized': raw, 'timestamp': '2099', 'product_name': None, 'label_binary': 'do_not_read', 'src_ip': '203.0.113.1'})
        if (direct['text'], direct['facts']) != (changed['text'], changed['facts']):
            failures.append(['outer_field', case['row_position']])
        checks['outer_field'] += 1
        predictions_cases.append({'projection_id': case['projection_id'], 'text': direct['text'], 'facts': direct['facts']})
    return {'cases': len(cases), 'checks': dict(checks), 'failures': failures, 'passed': not failures}, predictions_cases


def pair_support(rows, facts, texts, ids, feature_keys, fit, configuration):
    counts = np.bincount(feature_keys[ids[fit]] * 3 + rows.label_index.to_numpy()[fit], minlength=(feature_keys.max() + 1) * 3).reshape(-1, 3)
    conflicted = (counts > 0).sum(axis=1) > 1
    unique = rows.loc[fit, ['projection_id', 'body_group', 'label_index']].drop_duplicates()
    # Map original projection IDs to compact positions in this invocation.
    lookup = dict(zip(rows.projection_id.to_numpy(), ids))
    cells = collections.defaultdict(list)
    rules = configuration['pair_gate']
    for item in unique.itertuples(index=False):
        k = lookup[item.projection_id]
        if conflicted[feature_keys[k]]:
            continue
        f = facts[k]
        for target in rules['target_fields']:
            if target not in f:
                continue
            if target == 'icmp_code' and not (f.get('transport_protocol') == 'icmp' and f.get('icmp_type') == 3):
                continue
            others = dict(f); value = others.pop(target)
            if target == 'icmp_code':
                others.pop('icmp_unreachable', None)
            if target == 'outcome':
                others.pop('auth_result', None)
            # Exact remaining prose prevents unrelated narratives being paired
            # merely because their schema and original labels differ.
            context = core.canonical([target, texts[k], others])
            cells[context].append((item.body_group, item.label_index, str(value), k))
    candidates = []
    for context, items in cells.items():
        support = {c: len({r[0] for r in items if r[1] == c}) for c in range(3)}
        classes = [c for c, n in support.items() if n >= rules['min_distinct_bodies_per_class_per_fit_cell']]
        if len(classes) < 2:
            continue
        if len({r[2] for r in items if r[1] in classes}) < 2:
            continue
        candidates.append({'context': context, 'class_body_support': support, 'distinct_fact_values': sorted({r[2] for r in items})})
    return {'fit_only': True, 'fit_conflicting_keys_excluded': int(conflicted.sum()),
            'fit_rows_in_conflicted_keys': int(conflicted[feature_keys[ids[fit]]].sum()),
            'candidate_cells': candidates, 'eligible': bool(candidates),
            'reason': 'No cross-body mixed-label cell with documented varying fact and identical remaining context' if not candidates else 'Candidate support found; auxiliary optimizer still requires separate invocation'}


def conditional(rows, y, p, ids, masks, fit, ev):
    result = []
    mm = masks[ids]; routes = rows.route.to_numpy(); body = rows.body_group.to_numpy()
    for route in sorted(set(routes[ev])):
        for mask in np.unique(mm[ev & (routes == route)]):
            m = ev & (routes == route) & (mm == mask)
            for a, b in [(0, 1), (0, 2), (1, 2)]:
                take = m & np.isin(y, [a, b])
                ns = [int((take & (y == c)).sum()) for c in [a, b]]
                if min(ns) == 0:
                    continue
                fs = [len(np.unique(body[fit & (routes == route) & (mm == mask) & (y == c)])) for c in [a, b]]
                es = [len(np.unique(body[take & (y == c)])) for c in [a, b]]
                score = p[ids[take], b] / np.maximum(p[ids[take], a] + p[ids[take], b], 1e-300)
                result.append({'route': str(route), 'mask': int(mask), 'classes': [a, b], 'rows': ns,
                    'fit_body_keys': fs, 'evaluation_body_keys': es, 'supported_across_bodies': min(fs + es) >= 3,
                    'conditional_AUC': float(roc_auc_score(y[take] == b, score)),
                    'three_class_errors': int((p[ids[take]].argmax(axis=1) != y[take]).sum())})
    return result


def run(a):
    start = time.perf_counter(); prepared = Path(a.prepared); out = Path(a.output)
    if out.exists():
        raise FileExistsError('New training output required')
    receipt = json.loads((prepared / 'complete.json').read_text(encoding='utf-8'))
    for name, digest in receipt['files'].items():
        assert sha(prepared / name) == digest, name
    configuration = json.loads((prepared / 'configuration.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True)
    save(out / 'binding.json', {'prepared_receipt_sha256': sha(prepared / 'complete.json'),
        'python': platform.python_version(), 'packages': {n: __import__(n).__version__ for n in ['numpy', 'scipy', 'sklearn', 'pyarrow']},
        'sources': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}})
    full_projections = pq.read_table(prepared / 'projections.parquet').to_pandas()
    check, cases = real_input_checks(prepared, full_projections)
    save(out / 'real_input_checks.json', check)
    if not check['passed']:
        raise RuntimeError('Real input invariance/cache audit failed before training')
    rows = pq.read_table(prepared / 'rows.parquet').to_pandas()
    rows = rows[rows.fold >= 0].copy().reset_index(drop=True)
    active, ids = np.unique(rows.projection_id.to_numpy(), return_inverse=True)
    projections = full_projections.iloc[active].reset_index(drop=True)
    del full_projections; gc.collect()
    y = rows.label_index.to_numpy(dtype=np.int64); folds = rows.fold.to_numpy()
    texts = {'BASELINE_C': projections.baseline_text.tolist(), 'SEMANTIC': projections.text.tolist()}
    facts = {'BASELINE_C': [json.loads(v) for v in projections.baseline_facts], 'SEMANTIC': [json.loads(v) for v in projections.facts]}
    ax, availability_names = core.availability(facts['SEMANTIC'])
    masks = actual_keys(ax)
    summaries = []
    invoked_views = a.views.split(',') if a.views else configuration['models']
    if not set(invoked_views).issubset(configuration['models']):
        raise ValueError('Unknown model view')
    for fold in range(3):
        fit = folds != fold; ev = folds == fold
        for view in invoked_views:
            folder = out / ('fold_' + str(fold)) / view; folder.mkdir(parents=True)
            print(json.dumps({'stage': 'fit_start', 'fold': fold, 'view': view, 'fit_rows': int(fit.sum()), 'evaluation_rows': int(ev.sum())}), flush=True)
            t = time.perf_counter()
            if view == 'AVAILABILITY':
                x = ax; te = fe = None
            else:
                te = text_encoder(texts[view], ids, fit)
                tx = te.transform(texts[view])
                fe = core.learning.FixedFacts('C_BOTH') if view == 'BASELINE_C' else core.SemanticFacts()
                fe.fit([facts[view][k] for k in np.unique(ids[fit])])
                x = sparse.hstack([tx, fe.transform(facts[view])], format='csr')
                del tx
            keys = actual_keys(x)
            if view == 'SEMANTIC':
                support = pair_support(rows, facts[view], texts[view], ids, keys, fit, configuration)
                save(folder / 'pair_support.json', support)
            model, optimizer = core.learning.fit_aggregated(x, ids[fit], y[fit], configuration['C'])
            assert optimizer['sum_weights'] == int(fit.sum())
            unique_p = model.predict_proba(x)
            p = unique_p[ids[ev]]
            result = metric(y[ev], p)
            bundle = {'version': core.VERSION, 'view': view, 'fold': fold, 'text_encoder': te, 'fact_encoder': fe,
                      'model': model, 'availability_names': availability_names, 'decision': 'three_class_argmax'}
            joblib.dump(bundle, folder / 'model.joblib', compress=3)
            table = rows.loc[ev, ['row_position', 'label_index', 'product', 'route', 'body_group', 'union_group', 'projection_id', 'previous_projection_id']].copy()
            table['fold'] = fold
            for j, name in enumerate(['p_benign', 'p_malicious', 'p_suspicious']):
                table[name] = p[:, j]
            table['pred_label'] = np.asarray(['benign', 'malicious', 'suspicious'])[p.argmax(axis=1)]
            count = np.bincount(keys[ids[fit]] * 3 + y[fit], minlength=(keys.max()+1)*3).reshape(-1, 3)
            table['actual_key_seen_in_fit'] = count.sum(axis=1)[keys[ids[ev]]] > 0
            table['actual_key_conflicted_in_fit'] = ((count > 0).sum(axis=1) > 1)[keys[ids[ev]]]
            table['availability_mask'] = masks[ids[ev]]
            pq.write_table(pa.Table.from_pandas(table, preserve_index=False), folder / 'evaluation.parquet', compression='zstd')
            route_metrics = {str(route): metric(y[ev][table.route.to_numpy() == route], p[table.route.to_numpy() == route]) for route in sorted(set(table.route))}
            source_metrics = {str(source): metric(y[ev][table['product'].to_numpy() == source], p[table['product'].to_numpy() == source]) for source in sorted(set(table['product']))}
            ec = np.bincount(keys[ids[ev]] * 3 + y[ev], minlength=(keys.max()+1)*3).reshape(-1, 3)
            collisions = {'mixed_keys': int(((ec > 0).sum(axis=1) > 1).sum()), 'empirical_minimum_errors': int((ec.sum(axis=1)-ec.max(axis=1)).sum())}
            # Independent manual softmax from reloaded parameters for every
            # scored row. No re-fitting or accuracy claims from serialization.
            loaded = joblib.load(folder / 'model.joblib')['model']
            z = x.dot(loaded.coef_.T) + loaded.intercept_
            z -= z.max(axis=1, keepdims=True); e = np.exp(z); manual = e / e.sum(axis=1, keepdims=True)
            replay_difference = float(np.abs(manual - unique_p).max())
            assert replay_difference <= configuration['probability_tolerance']
            assert np.array_equal(manual.argmax(axis=1), unique_p.argmax(axis=1))
            report = {'fold': fold, 'view': view, 'evaluation': result, 'routes': route_metrics, 'sources': source_metrics,
                'apparent_fit': metric(y[fit], unique_p[ids[fit]]), 'optimizer': optimizer,
                'encoded_collision': collisions, 'actual_fit_seen_rows': int(table.actual_key_seen_in_fit.sum()),
                'conditional_comparisons': conditional(rows, y, unique_p, ids, masks, fit, ev),
                'manual_softmax_replay_max_difference': replay_difference,
                'seconds': time.perf_counter()-t, 'fresh_blind_test': False, 'quality_accepted': False}
            save(folder / 'report.json', report)
            save(folder / 'complete.json', {'model_sha256': sha(folder / 'model.joblib'), 'predictions_sha256': sha(folder / 'evaluation.parquet'),
                 'report_sha256': sha(folder / 'report.json'), 'fit_rows': int(fit.sum()), 'evaluation_rows': int(ev.sum())})
            summaries.append({'fold': fold, 'view': view, **result, 'seconds': report['seconds']})
            print(json.dumps({'stage': 'fit_complete', 'fold': fold, 'view': view, 'macro_f1': result['macro_f1'], 'errors': result['errors'], 'recall': result['class_recall'], 'seconds': round(report['seconds'])}), flush=True)
            del x, model, loaded, z, e, manual, unique_p, p, bundle, table, te, fe
            gc.collect()
    save(out / 'complete.json', {'primary_fits_completed': len(summaries), 'invoked_views': invoked_views, 'results': summaries, 'seconds': time.perf_counter()-start,
        'optional_pair_training_executed': False, 'quality_accepted': False, 'fresh_blind_test': False})
    print(json.dumps({'stage': 'primary_training_complete', 'fits': len(summaries), 'seconds': round(time.perf_counter()-start)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--prepared', required=True); p.add_argument('--output', required=True); p.add_argument('--views')
    run(p.parse_args())
