"""Four fixed paired fits; calibration guard precedes any new evaluation."""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import joblib
import run_v54 as v
import run_v56 as t
import v56_shared_control as prev
import v57_applicability as m

ROOT = t.ROOT
RUN = ROOT / 'artifacts/v57_applicability_20260914'

def prepare():
    assert not RUN.exists()
    _, r, x = prev.load()
    sources = {}
    for receipt in ['evidence/2026-09-14/v56_delivery/delivery.json', 'evidence/2026-09-14/v56_followup_delivery.json']:
        for p, h in v.read(ROOT / receipt)['bindings'].items():
            assert v.sha(ROOT / p) == h, p
        sources[receipt] = v.sha(ROOT / receipt)
    strengths = {}
    for protocol in t.PROTOCOLS:
        selected = v.read(prev.RUN / protocol / 'selection.json')['selected']
        strengths[protocol] = float(selected.split('_')[1])
        for p in [t.RUN / (protocol + '_split.parquet'), t.RUN / protocol / 'base.joblib', t.RUN / protocol / 'base_calibration.parquet', prev.RUN / protocol / (selected + '_calibration.parquet')]:
            sources[p.relative_to(ROOT).as_posix()] = v.sha(p)
    for name in ['v57_applicability.py', 'run_v57_stage1.py', 'test_v57.py', 'v56_pooling.py', 'run_v54.py', 'run_v55.py', 'run_v56.py']:
        sources['training/' + name] = v.sha(ROOT / 'training' / name)
    RUN.mkdir(parents=True)
    cfg = {'version': 'v57-stage1-1', 'protocols': t.PROTOCOLS, 'strengths': strengths, 'optimizer': v.read(t.RUN / 'configuration.json')['optimizer'], 'delta': 'New single-field branch only: nonmissing observed values; ICMP fields apply only to ICMP, port ranges only to TCP/UDP. Same full 13-field base and 3 combination layers, original record mean CE and support penalty. No new threshold, augmentation or base fit.', 'selection': 'Exactly one previously selected lambda per protocol held fixed for paired structural comparison, plus zero correction. Calibration must have nonincreasing M, S, ASA-to-normal and normal-control errors separately in each of all 6 views, with at least one strict M/S improvement. Numerical success AND scaled gradient <=5e-6. Failed candidate never evaluated; select zero and stop its expansion.', 'scope': 'Adaptive official fit-side development only, previous pressure evaluation physically absent. Old body-only calibration retained for this controlled structural comparison. A passing candidate still requires behavior-isolated inner validation before expansion or promotion.', 'new_fits_planned': 4, 'source_bindings': sources}
    v.save(RUN / 'configuration.json', cfg)
    v.save(RUN / 'preregistered.json', {'configuration_sha256': v.sha(RUN / 'configuration.json'), 'new_fits': 0})
    print(json.dumps({'prepared': True, 'fixed_strengths': strengths}), flush=True)

def load():
    c = v.read(RUN / 'configuration.json')
    assert v.sha(RUN / 'configuration.json') == v.read(RUN / 'preregistered.json')['configuration_sha256']
    for p, h in c['source_bindings'].items():
        assert v.sha(ROOT / p) == h, p
    _, r, x = prev.load()
    return c, r, x

def run(protocol):
    c, r, x = load()
    folder = RUN / protocol
    assert not folder.exists()
    folder.mkdir()
    roles = pd.read_parquet(t.RUN / (protocol + '_split.parquet')).role
    fit, cal, ev = (roles.eq(k).to_numpy() for k in ['fit', 'calibration', 'evaluation'])
    base = joblib.load(t.RUN / protocol / 'base.joblib')
    audit = m.design_audit(m.fit_design(x.loc[fit], r.loc[fit, 'body_group']), x.loc[fit])
    assert audit['shared_missing_parameter_count'] == 0
    v.save(folder / 'before_fit.json', audit)
    start = time.monotonic()
    bundle = m.fit(base, x.loc[fit], r.loc[fit, 'label_index'], r.loc[fit, 'body_group'], c['strengths'][protocol], c['optimizer'])
    bundle.update(base_source=(t.RUN / protocol / 'base.joblib').relative_to(ROOT).as_posix(), base_sha256=v.sha(t.RUN / protocol / 'base.joblib'))
    joblib.dump(bundle, folder / 'model.joblib')
    a = m.predictions(bundle, protocol, 'calibration', r, x, cal)
    a.to_parquet(folder / 'calibration.parquet', index=False)
    b = pd.read_parquet(t.RUN / protocol / 'base_calibration.parquet')
    gate = m.compare(b, a)
    old = v.read(prev.RUN / protocol / 'selection.json')['selected']
    oldcal = pd.read_parquet(prev.RUN / protocol / (old + '_calibration.parquet'))
    selected = 'applicable' if bundle['optimizer']['eligible'] and gate['quality_passed'] else 'zero'
    result = {'protocol': protocol, 'optimizer': bundle['optimizer'], 'selected': selected, 'against_base': gate, 'against_old_shared': m.compare(oldcal, a)}
    v.save(folder / 'selection.json', result)
    v.save(folder / 'before_evaluation.json', {'selected': selected, 'bindings': {p.name: v.sha(p) for p in folder.iterdir() if p.is_file()}, 'evaluation_generated': False})
    if selected == 'applicable':
        m.predictions(bundle, protocol, 'evaluation', r, x, ev).to_parquet(folder / 'evaluation.parquet', index=False)
    v.save(folder / 'complete.json', {'bindings': {p.name: v.sha(p) for p in folder.iterdir() if p.is_file()}, 'new_fits': 1, 'selected': selected, 'evaluation_executed': selected == 'applicable', 'seconds': time.monotonic() - start})
    print(json.dumps({'protocol': protocol, 'selected': selected, 'optimizer': bundle['optimizer'], 'calibration_failures': gate['failures'], 'seconds': time.monotonic() - start}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--protocol', choices=t.PROTOCOLS)
    a = parser.parse_args()
    prepare() if a.mode == 'prepare' else run(a.protocol)
