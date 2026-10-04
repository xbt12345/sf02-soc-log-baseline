"""Coverage-qualified natural/synthetic source-port transfer, fixed alternatives."""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from scipy.special import softmax
import run_v54 as v
import run_v55 as n
import run_v56 as t
import run_v57_stage1 as first
import run_v57_stage2 as second
import v57_applicability as m

ROOT = t.ROOT
RUN = ROOT / 'artifacts/v57_port_transfer_20260914'

def active(x):
    return (x.transport_protocol.isin(['tcp', 'udp']) & x.src_port_fixed.eq(v.MISSING) & x.dst_port_fixed.ne(v.MISSING)).to_numpy()

def transformed(x):
    out = x.copy()
    out[['src_port_fixed', 'src_port_range']] = v.MISSING
    return out

def prepare():
    assert not RUN.exists()
    _, r, x = second.load()
    fine = n.behavior(x, True)
    asa = r.route.eq('asa').to_numpy()
    tcp = (x.transport_protocol.eq('tcp') & x.src_role.eq('outside') & x.dst_role.eq('dmz')).to_numpy()
    natural = tcp & active(x)
    donors = tcp & x.src_port_fixed.ne(v.MISSING).to_numpy() & x.dst_port_fixed.ne(v.MISSING).to_numpy()
    splits, audits = {}, {}
    for protocol in t.PROTOCOLS:
        old = pd.read_parquet(t.RUN / (protocol + '_split.parquet'))
        idx = np.flatnonzero(old.role.ne('evaluation').to_numpy() & asa)
        records, selected = [], None
        splitter = StratifiedGroupKFold(3, shuffle=True, random_state=20260916)
        for fold, (train, cal) in enumerate(splitter.split(idx, r.label_index.to_numpy()[idx], fine[idx])):
            roles = old.role.to_numpy(copy=True)
            roles[idx[train]], roles[idx[cal]] = 'fit', 'calibration'
            donor_rows = int(((roles == 'fit') & donors).sum())
            natural_rows = int(((roles == 'calibration') & natural).sum())
            ok = donor_rows > 0 and natural_rows > 0
            records.append({'fold': fold, 'fit_TCP_donors': donor_rows, 'calibration_natural_TCP_source_missing': natural_rows, 'coverage_qualified': ok})
            if ok and selected is None:
                selected = fold
                split = old.copy(); split['role'] = roles
                splits[protocol] = split
        audits[protocol] = {'selected_fold': selected, 'folds': records, 'selection_inputs': 'Observed protocol, roles and port availability counts only; no prediction, loss or classification score. Labels used only by the fixed stratified splitter.'}
    RUN.mkdir(parents=True)
    for p, split in splits.items():
        split.to_parquet(RUN / (p + '_split.parquet'), index=False)
    sources = [ROOT / 'training/run_v57_port_transfer.py', ROOT / 'training/v57_applicability.py', ROOT / 'training/run_v57_stage2.py', ROOT / 'evidence/2026-09-14/v57_coverage/controls.json', second.RUN / 'configuration.json']
    cfg = {'version': 'v57-port-transfer-1', 'protocols': list(splits), 'coverage_audits': audits, 'strength': 1., 'optimizer': v.read(first.RUN / 'configuration.json')['optimizer'], 'network': v.read(t.RUN / 'configuration.json')['base_network'], 'epochs': 10, 'seed': 20260914, 'variants': ['natural_only', 'natural_plus_synthetic'], 'model': 'Frozen newly fitted same-spec base plus observed-applicable facts/combination correction activating only when TCP/UDP source port unknown and destination port known. Preserve actual destination port; never guess source port. natural_only learns from active original fit rows; natural_plus_synthetic additionally masks source port on eligible original rows with visible destination port. Each participating parent appears once. No relabeling, class balancing or duplicate row generation.', 'selection': 'Each fixed variant vs zero must pass same all-6-view per-class calibration guard and numeric condition. Among passing variants choose lower mean ASA calibration log loss, natural_only on tie. If none pass, zero; rejected candidates not evaluated. Original evaluation membership never changed.', 'scope': 'Separate adaptive follow-up after discovering stage2 lacked TCP donor/target coverage. Select first coverage-qualified of the pre-existing 3 inner folds, not the best score. Infeasible protocols are recorded, not silently replaced. No OOF residual learning, original pressure, external data or deployment.', 'source_bindings': {p.relative_to(ROOT).as_posix(): v.sha(p) for p in sources}, 'local_bindings': {p.name: v.sha(p) for p in RUN.iterdir() if p.is_file()}}
    v.save(RUN / 'configuration.json', cfg)
    v.save(RUN / 'preregistered.json', {'configuration_sha256': v.sha(RUN / 'configuration.json'), 'new_fits': 0})
    print(json.dumps({'prepared': True, 'protocols': list(splits), 'coverage': audits}), flush=True)

def load():
    c = v.read(RUN / 'configuration.json')
    assert v.sha(RUN / 'configuration.json') == v.read(RUN / 'preregistered.json')['configuration_sha256']
    for root, name in [(ROOT, 'source_bindings'), (RUN, 'local_bindings')]:
        for p, h in c[name].items():
            assert v.sha(root / p) == h
    _, r, x = second.load()
    return c, r, x

def predictions(baseline, bundle, original):
    parts = []
    for view, fields in v.VIEWS.items():
        xx = original.copy(); xx[fields] = v.MISSING
        mask = active(xx)
        b = baseline[baseline.scenario == view].reset_index(drop=True).copy()
        if mask.any():
            z = np.log(np.maximum(b.loc[mask, ['p_0', 'p_1', 'p_2']].to_numpy().astype(float), 1e-30))
            p = softmax(z + m.matrix(bundle['design'], xx.loc[mask]) @ bundle['weights'], axis=1)
            b.loc[mask, 'pred'] = p.argmax(axis=1)
            for k in range(3):
                b['p_' + str(k)] = b['p_' + str(k)].astype(float)
                b.loc[mask, 'p_' + str(k)] = p[:, k]
        parts.append(b)
    return pd.concat(parts, ignore_index=True)

def run(protocol):
    c, r, x = load()
    assert protocol in c['protocols']
    folder = RUN / protocol
    assert not folder.exists(); folder.mkdir()
    roles = pd.read_parquet(RUN / (protocol + '_split.parquet')).role.to_numpy()
    fit, cal, ev = (roles == role for role in ['fit', 'calibration', 'evaluation'])
    start = time.monotonic()
    enc = v.fit_encoder(x.loc[fit])
    A = v.encode(enc, x.loc[fit]).astype(np.float32)
    cold = np.flatnonzero(np.asarray(A.sum(axis=0)).ravel() == 0)
    net = MLPClassifier(**c['network'], alpha=1., random_state=c['seed'])
    for _ in range(c['epochs']):
        net.partial_fit(A, r.loc[fit, 'label_index'].to_numpy(), classes=[0, 1, 2]); net.coefs_[0][cold] = 0
    base = {'input': enc, 'model': net}
    joblib.dump(base, folder / 'base.joblib')
    b = second.base_predictions(base, r, x, cal)
    b.to_parquet(folder / 'base_calibration.parquet', index=False)
    choices = {}
    for variant in c['variants']:
        parent = active(x) if variant == 'natural_only' else (x.transport_protocol.isin(['tcp', 'udp']) & x.dst_port_fixed.ne(v.MISSING)).to_numpy()
        participating = fit & parent
        xx = transformed(x.loc[participating])
        bundle = m.fit(base, xx, r.loc[participating, 'label_index'], r.loc[participating, 'body_group'], c['strength'], c['optimizer'])
        bundle.update(base_sha256=v.sha(folder / 'base.joblib'), variant=variant, participating_row_positions=r.loc[participating, 'row_position'].to_numpy())
        joblib.dump(bundle, folder / (variant + '.joblib'))
        pred = predictions(b, bundle, x.loc[cal].reset_index(drop=True))
        pred.to_parquet(folder / (variant + '_calibration.parquet'), index=False)
        gate = m.compare(b, pred)
        z = pred[pred.route == 'asa']; y = z.label_index.to_numpy(); p = z[['p_0', 'p_1', 'p_2']].to_numpy()
        loss = float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)).mean())
        choices[variant] = {'gate': gate, 'optimizer': bundle['optimizer'], 'calibration_log_loss': loss, 'accepted': bool(gate['quality_passed'] and bundle['optimizer']['eligible']), 'participating_rows': int(participating.sum()), 'natural_rows': int((participating & active(x)).sum()), 'synthetic_rows': int((participating & ~active(x)).sum())}
        print(json.dumps({'protocol': protocol, 'variant': variant, 'accepted': choices[variant]['accepted'], 'failures': gate['failures'], 'optimizer': bundle['optimizer']}), flush=True)
    eligible = [name for name, a in choices.items() if a['accepted']]
    selected = min(eligible, key=lambda name: (choices[name]['calibration_log_loss'], c['variants'].index(name))) if eligible else 'zero'
    v.save(folder / 'selection.json', {'candidates': choices, 'selected': selected})
    v.save(folder / 'before_evaluation.json', {'bindings': {p.name: v.sha(p) for p in folder.iterdir() if p.is_file()}, 'evaluation_generated': False})
    if selected != 'zero':
        b = second.base_predictions(base, r, x, ev)
        a = predictions(b, joblib.load(folder / (selected + '.joblib')), x.loc[ev].reset_index(drop=True))
        b.to_parquet(folder / 'base_evaluation.parquet', index=False)
        a.to_parquet(folder / 'evaluation.parquet', index=False)
        v.save(folder / 'evaluation_comparison.json', m.compare(b, a))
    v.save(folder / 'complete.json', {'bindings': {p.name: v.sha(p) for p in folder.iterdir() if p.is_file()}, 'new_base_fits': 1, 'new_adapter_fits': 2, 'selected': selected, 'evaluation_executed': selected != 'zero', 'seconds': time.monotonic() - start})
    print(json.dumps({'protocol': protocol, 'complete': True, 'selected': selected}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'run']); parser.add_argument('--protocol', choices=t.PROTOCOLS); args = parser.parse_args()
    prepare() if args.mode == 'prepare' else run(args.protocol)
