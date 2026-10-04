"""One conditional v39 fit on the exact inherited template/source stress roles."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
import v39_core as core
from run_v39_prepare import sha, save
from run_v38_train import text_encoder, actual_keys
from run_v39_train import metric


def main(a):
    root = Path(a.root); out = Path(a.output); prepared = Path(a.prepared)
    if out.exists():
        raise FileExistsError('Preserve prior stress result')
    review_path = root / 'evidence/2026-09-13/v39_execution/primary_review/primary_review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    assert review['proceed_to_old_protocol_stress']
    receipt = json.loads((prepared / 'complete.json').read_text(encoding='utf-8'))
    for name, digest in receipt['files'].items():
        assert sha(prepared / name) == digest
    out.mkdir(parents=True); start = time.perf_counter()
    rows = pq.read_table(prepared / 'rows.parquet').to_pandas()
    rows = rows[rows.inner_role >= 0].reset_index(drop=True)
    fit = rows.inner_role.to_numpy() != 2; ev = ~fit
    assert int(fit.sum()) == 1199575 and int(ev.sum()) == 179075
    assert not rows.loc[fit, 'product'].isin(['Duo', 'Barracuda WAF']).any()
    assert not set(rows.loc[fit, 'union_group']) & set(rows.loc[ev, 'union_group'])
    assert not set(rows.loc[fit, 'body_group']) & set(rows.loc[ev, 'body_group'])
    save(out / 'binding.json', {'protocol': 'exact_v38_inner_roles_template_plus_allowed_Duo_WAF_holdout',
        'fit_rows': int(fit.sum()), 'evaluation_rows': int(ev.sum()), 'C': 0.1,
        'weights': 'original rows equal', 'decision': 'three_class_argmax', 'fresh_blind_test': False,
        'primary_gate_sha256': sha(review_path), 'prepared_sha256': sha(prepared / 'complete.json'),
        'sources': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}})
    allp = pq.read_table(prepared / 'projections.parquet').to_pandas()
    active, ids = np.unique(rows.projection_id.to_numpy(), return_inverse=True)
    pr = allp.iloc[active]; texts = pr.text.tolist(); facts = [json.loads(v) for v in pr.facts]
    y = rows.label_index.to_numpy()
    te = text_encoder(texts, ids, fit)
    fe = core.SemanticFacts().fit([facts[k] for k in np.unique(ids[fit])])
    x = sparse.hstack([te.transform(texts), fe.transform(facts)], format='csr')
    print(json.dumps({'stage': 'old_protocol_stress_fit', 'fit_rows': int(fit.sum())}), flush=True)
    model, optimizer = core.learning.fit_aggregated(x, ids[fit], y[fit], .1)
    bundle = {'version': core.VERSION, 'view': 'SEMANTIC', 'protocol': 'old_template_source', 'text_encoder': te,
              'fact_encoder': fe, 'model': model, 'decision': 'three_class_argmax'}
    joblib.dump(bundle, out / 'model.joblib', compress=3)
    allprob = model.predict_proba(x); p = allprob[ids[ev]]
    d = rows.loc[ev, ['row_position', 'label_index', 'body_group', 'union_group', 'projection_id', 'route', 'product']].copy()
    for i, name in enumerate(['p_benign', 'p_malicious', 'p_suspicious']):
        d[name] = p[:, i]
    d['pred_label'] = np.asarray(['benign', 'malicious', 'suspicious'])[p.argmax(axis=1)]
    pq.write_table(pa.Table.from_pandas(d, preserve_index=False), out / 'evaluation.parquet', compression='zstd')
    baseline_dir = root / 'artifacts/v38_local_r1_20260913/equal_classifiers_attempt2/C_BOTH'
    br = json.loads((baseline_dir / 'complete.json').read_text(encoding='utf-8'))
    assert sha(baseline_dir / 'evaluation.parquet') == br['predictions_sha256']
    b = pq.read_table(baseline_dir / 'evaluation.parquet').to_pandas()
    assert np.array_equal(b.row_position.to_numpy(), d.row_position.to_numpy())
    assert np.array_equal(b.label_index.to_numpy(), d.label_index.to_numpy())
    bp = b[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy()
    report = {'evaluation': metric(y[ev], p), 'baseline_same_roles': metric(y[ev], bp),
        'apparent_fit': metric(y[fit], allprob[ids[fit]]), 'optimizer': optimizer,
        'routes': {str(r): metric(y[ev][d.route.to_numpy() == r], p[d.route.to_numpy() == r]) for r in sorted(set(d.route))},
        'sources': {str(r): metric(y[ev][d['product'].to_numpy() == r], p[d['product'].to_numpy() == r]) for r in sorted(set(d['product']))},
        'decision_changes': {'fixed': int(((bp.argmax(axis=1) != y[ev]) & (p.argmax(axis=1) == y[ev])).sum()),
                             'regressed': int(((bp.argmax(axis=1) == y[ev]) & (p.argmax(axis=1) != y[ev])).sum())},
        'seconds': time.perf_counter() - start, 'fit_count': 1, 'quality_accepted': False,
        'scope': 'Already reviewed old development stress; not independent or external validation'}
    save(out / 'report.json', report)
    save(out / 'complete.json', {'model_sha256': sha(out / 'model.joblib'), 'predictions_sha256': sha(out / 'evaluation.parquet'), 'report_sha256': sha(out / 'report.json')})
    print(json.dumps({k: report[k] for k in ['evaluation', 'baseline_same_roles', 'decision_changes', 'seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'prepared', 'output']:
        p.add_argument('--' + name, required=True)
    main(p.parse_args())
