"""Natural vs simulated missingness controls on fit/calibration only; no fit."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import run_v54 as v
import run_v55 as n
import run_v56 as t
import run_v57_stage2 as run

OUT = t.ROOT / 'evidence/2026-09-14/v57_coverage'

def describe(x, y, probability, bodies):
    pred = probability.argmax(axis=1)
    tab = pd.crosstab(pd.Series(v.frame_keys(x), name='observation'), pd.Series(y, name='label')).reindex(columns=[0, 1, 2], fill_value=0)
    return {'rows': len(y), 'body_groups': int(pd.Series(bodies).nunique()), 'classes_B_M_S': np.bincount(y, minlength=3).tolist(), 'base_errors_B_M_S': [int(((y == k) & (pred != k)).sum()) for k in range(3)], 'class_NLL_B_M_S': [float(-np.log(np.maximum(probability[y == k, k], 1e-30)).mean()) if (y == k).any() else None for k in range(3)], 'unique_observations': len(tab), 'empirical_min_total_errors': int((tab.sum(axis=1) - tab.max(axis=1)).sum()) if len(tab) else 0, 'risk_available_M_S': bool((y == 1).any() and (y == 2).any())}

def main():
    assert not OUT.exists()
    _, r, x = run.load()
    OUT.mkdir(parents=True)
    records = []
    for protocol in t.PROTOCOLS:
        folder = run.RUN / protocol
        complete = v.read(folder / 'complete.json')
        for p, h in complete['bindings'].items():
            assert v.sha(folder / p) == h
        roles = pd.read_parquet(run.RUN / (protocol + '_split.parquet')).role.to_numpy()
        base = joblib.load(folder / 'base.joblib')
        context = (r.route.eq('asa') & x.transport_protocol.eq('tcp') & x.src_role.eq('outside') & x.dst_role.eq('dmz')).to_numpy()
        both = x.src_port_fixed.ne(v.MISSING).to_numpy() & x.dst_port_fixed.ne(v.MISSING).to_numpy()
        for missing in ['src', 'dst']:
            other = 'dst' if missing == 'src' else 'src'
            natural = context & x[missing + '_port_fixed'].eq(v.MISSING).to_numpy() & x[other + '_port_fixed'].ne(v.MISSING).to_numpy()
            fitted_projection = None
            for role in ['fit', 'calibration']:
                for kind, subset in [('natural', natural), ('synthetic', context & both)]:
                    mask = (roles == role) & subset
                    xx = x.loc[mask].reset_index(drop=True).copy()
                    if kind == 'synthetic':
                        xx[[missing + '_port_fixed', missing + '_port_range']] = v.MISSING
                    y = r.loc[mask, 'label_index'].to_numpy()
                    p = n.predict(base, xx)[1] if len(xx) else np.zeros((0, 3))
                    info = describe(xx, y, p, r.loc[mask, 'body_group'].to_numpy())
                    if role == 'fit' and kind == 'synthetic':
                        tmp = pd.DataFrame({'key': v.frame_keys(xx), 'y': y})
                        fitted_projection = pd.crosstab(tmp.key, tmp.y).reindex(columns=[0, 1, 2], fill_value=0)
                    if role == 'calibration' and kind == 'natural':
                        support = fitted_projection.reindex(v.frame_keys(xx), fill_value=0)
                        strata = np.where(support.sum(axis=1).to_numpy() == 0, 'unseen_projection', np.where(support[2].to_numpy() == 0, 'seen_without_S', 'seen_with_S'))
                        info['synthetic_fit_support'] = {name: np.bincount(y[strata == name], minlength=3).tolist() for name in sorted(set(strata))}
                    records.append({'protocol': protocol, 'missing_port': missing, 'role': role, 'kind': kind, **info})
    output = {'scope': 'No new model fit or evaluation prediction. Same semantic TCP outside-to-dmz context; fit/calibration natural missingness vs predeclared deletion of visible source/destination port fields. Per-row labels unchanged. Groups differ between natural and synthetic, so score/rate differences describe distributions, not a causal treatment effect. Missing class risk is unavailable rather than zero.', 'records': records}
    v.save(OUT / 'controls.json', output)
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    v.save(OUT / 'receipt.json', {'files': {p.name: v.sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps([z for z in records if z['missing_port'] == 'src' and (z['role'] == 'calibration' or z['kind'] == 'synthetic')]), flush=True)

if __name__ == '__main__':
    main()
