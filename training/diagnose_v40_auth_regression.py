"""Inspect frozen class margins for new authentication regressions; never fit."""
import argparse
import gc
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pyarrow.parquet as pq


def main(a):
    root, run = Path(a.root).resolve(), Path(a.run).resolve()
    sys.path.insert(0, str(run / 'frozen_training_runtime'))
    import v40_core as core
    from run_v39_prepare import sha, save
    out = run / 'primary_review/auth_margin_diagnosis.json'
    assert not out.exists()
    prep = root / 'artifacts/v39_local_r2_20260913/prepared'
    rows = pq.read_table(prep / 'rows.parquet', columns=['row_position', 'fold', 'product', 'projection_id', 'label_index']).to_pandas()
    rows = rows[(rows.fold >= 0) & (rows['product'] == 'Duo') & (rows.label_index == 2)].copy()
    pr = pq.read_table(prep / 'projections.parquet', columns=['text', 'facts']).to_pandas().iloc[rows.projection_id.unique()]
    records = []
    for fold in range(3):
        selected = rows[rows.fold == fold]
        for view in ['B', 'N', 'I']:
            folder = prep.parent / ('primary/fold_%s/SEMANTIC' % fold) if view == 'B' else run / ('primary/fold_%s/%s' % (fold, view))
            model = joblib.load(folder / 'model.joblib')
            texts = pr.loc[selected.projection_id, 'text'].tolist()
            facts = [json.loads(v) for v in pr.loc[selected.projection_id, 'facts']]
            x = core.matrix(model, texts, facts)
            tn = model['text_encoder'].names(); fn = model['fact_encoder'].names()
            pn = model['parameter_encoder'].names() if model.get('parameter_encoder') is not None else np.array([])
            names = list(tn) + list(fn) + list(pn)
            w = model['model'].coef_[2] - model['model'].coef_[0]
            intercept = float(model['model'].intercept_[2] - model['model'].intercept_[0])
            p = model['model'].predict_proba(x)
            for i, row in enumerate(selected.itertuples(index=False)):
                xr = x.getrow(i)
                contribution = xr.data * w[xr.indices]
                margin = float(contribution.sum() + intercept)
                assert abs(margin - np.log(p[i, 2]/p[i, 0])) < 1e-10
                buckets = {'text': 0.0, 'fact': 0.0, 'parameter_effect': 0.0}
                terms = []
                for idx, value, gain in zip(xr.indices, xr.data, contribution):
                    block = 'text' if idx < len(tn) else ('fact' if idx < len(tn) + len(fn) else 'parameter_effect')
                    buckets[block] += float(gain)
                    terms.append({'block': block, 'feature': str(names[idx]), 'value': float(value), 'S_minus_B_contribution': float(gain)})
                records.append({'row_position': int(row.row_position), 'fold': fold, 'view': view,
                    'original_facts': facts[i], 'model_facts': core.canonicalize(facts[i])[0] if view != 'B' else facts[i],
                    'p_B_M_S': p[i].tolist(), 'S_minus_B_margin': margin, 'intercept_S_minus_B': intercept,
                    'block_contributions_S_minus_B': buckets, 'active_terms': terms,
                    'model_sha256': sha(folder / 'model.joblib')})
            del model, x
            gc.collect()
    save(out, {'records': records, 'rows': len(rows), 'fitted_models': 0, 'script_sha256': sha(__file__),
        'scope': 'Exact frozen linear score decomposition. N jointly changes finite coding and semantic aliases; this is not a causal ablation separating those interventions.'})
    for view in ['B', 'N', 'I']:
        values = [v for v in records if v['view'] == view]
        print(json.dumps({'view': view, 'suspicious_correct': int(sum(np.argmax(v['p_B_M_S']) == 2 for v in values)),
            'margin_range': [min(v['S_minus_B_margin'] for v in values), max(v['S_minus_B_margin'] for v in values)],
            'parameter_contribution': sorted(set(v['block_contributions_S_minus_B']['parameter_effect'] for v in values))}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); parser.add_argument('--run', required=True)
    main(parser.parse_args())
