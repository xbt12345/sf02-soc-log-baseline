"""Isolated role-missing adapters with behavior-disjoint inner calibration."""
import argparse
import json
import time
from pathlib import Path
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
import v57_applicability as m

ROOT = t.ROOT
RUN = ROOT / 'artifacts/v57_isolated_missing_20260914'
HEADS = {'no_src_role': 'src_role', 'no_dst_role': 'dst_role'}

def active(x, head):
    own = HEADS[head]
    other = 'dst_role' if own == 'src_role' else 'src_role'
    return x[own].eq(v.MISSING).to_numpy() & x[other].ne(v.MISSING).to_numpy()

def prepare():
    assert not RUN.exists()
    c, r, x = first.load()
    assert all((first.RUN / p / 'complete.json').exists() for p in t.PROTOCOLS)
    keys = n.behavior(x, True)
    asa = r.route.eq('asa').to_numpy()
    roles_all, stats = {}, {}
    for protocol in t.PROTOCOLS:
        previous = pd.read_parquet(t.RUN / (protocol + '_split.parquet'))
        roles = previous.role.to_numpy(copy=True)
        idx = np.flatnonzero((roles != 'evaluation') & asa)
        train, cal = next(StratifiedGroupKFold(3, shuffle=True, random_state=20260916).split(idx, r.label_index.to_numpy()[idx], keys[idx]))
        roles[idx[train]] = 'fit'
        roles[idx[cal]] = 'calibration'
        assert set(keys[asa & (roles == 'fit')]).isdisjoint(keys[asa & (roles == 'calibration')])
        for aa, bb in [('fit', 'calibration'), ('fit', 'evaluation'), ('calibration', 'evaluation')]:
            assert set(r.loc[roles == aa, 'body_group']).isdisjoint(r.loc[roles == bb, 'body_group'])
        for role in ['fit', 'calibration', 'evaluation']:
            assert set(r.loc[roles == role, 'label_index']) == {0, 1, 2}
        previous['role'] = roles
        roles_all[protocol] = previous
        stats[protocol] = {role: {'labels_B_M_S': np.bincount(r.loc[roles == role, 'label_index'], minlength=3).tolist(), 'ASA_behavior_keys': len(set(keys[asa & (roles == role)]))} for role in ['fit', 'calibration', 'evaluation']}
    RUN.mkdir(parents=True)
    for protocol, split in roles_all.items():
        split.to_parquet(RUN / (protocol + '_split.parquet'), index=False)
    sources = {p.relative_to(ROOT).as_posix(): v.sha(p) for p in [first.RUN / 'configuration.json', ROOT / 'evidence/2026-09-14/v57_stage1_verification/verification.json', ROOT / 'training/run_v57_stage2.py', ROOT / 'training/v57_applicability.py', ROOT / 'training/test_v57_stage2.py']}
    cfg = {'version': 'v57-stage2-1', 'protocols': t.PROTOCOLS, 'split_seed': 20260916, 'splitter': 'First of 3 StratifiedGroupKFold folds over original fine behavior keys for ASA on previous outer-fit+calibration union. Normal ACL keeps previous body split. Original evaluation unchanged. Each parent and transformed rows stay together.', 'splits': stats, 'network': v.read(t.RUN / 'configuration.json')['base_network'], 'alpha': 1., 'epochs': 10, 'seed': 20260914, 'strength': 1., 'optimizer': c['optimizer'], 'heads': HEADS, 'model': 'New same-spec base fitted only on new fit rows. Two independent role-missing applicable-value+combination residuals. A head activates only from exactly its own missing role and other role observed, never scenario/route/label/product/time. If both roles missing, base fallback. Base and other head are frozen. New base is needed because inner calibration is now behavior-disjoint; compare adapters to this same base, not attribute old/new split differences to adapters.', 'objective': 'Original labels retained. Conceptual per-parent loss mass: full 0.5, source-role missing 0.25, destination-role missing 0.25. Full loss constant with frozen base; independent adapter objectives/penalties scaled equally by 0.25 have exactly the same optimum as each stored row-mean CE+ridge. Counts are computational aggregation, not additional real observations.', 'selection': c['selection'].replace('Exactly one previously selected lambda per protocol held fixed for paired structural comparison', 'Exactly lambda=1 for each of two independent heads per protocol; each one evaluated against zero in all 6 calibration views. Combine only individually passing heads, then recheck combined calibration. No full-data refit'), 'evaluation_gate': 'Selected combined adapter must not increase observed M/S cell errors, ASA-to-normal, or normal errors; absent class cells marked unsupported, never assigned a zero risk. At least one strict supported M/S improvement required. This gate permits further development only. Full ASA predictions must be bitwise identical to this stage base.', 'scope': 'Adaptive official fit-side experiment after stage1 failed. No original pressure test or external data. Behavior-disjoint before masking; masked input overlap reported separately. Natural missing-role normal records can activate a head even in full view; protect them explicitly. OOF base prediction training remains a later hypothesis, not implemented here.', 'source_bindings': sources, 'local_bindings': {p.name: v.sha(p) for p in RUN.iterdir() if p.is_file()}}
    v.save(RUN / 'configuration.json', cfg)
    v.save(RUN / 'preregistered.json', {'configuration_sha256': v.sha(RUN / 'configuration.json'), 'new_fits': 0})
    print(json.dumps({'prepared': True, 'splits': stats}), flush=True)

def load():
    c = v.read(RUN / 'configuration.json')
    assert v.sha(RUN / 'configuration.json') == v.read(RUN / 'preregistered.json')['configuration_sha256']
    for root, field in [(ROOT, 'source_bindings'), (RUN, 'local_bindings')]:
        for p, h in c[field].items():
            assert v.sha(root / p) == h, p
    _, r, x = first.load()
    return c, r, x

def base_predictions(base, r, x, mask):
    parts = []
    for view, fields in v.VIEWS.items():
        xx = x.loc[mask].copy()
        xx[fields] = v.MISSING
        pred, prob = n.predict(base, xx)
        z = r.loc[mask].copy()
        z['scenario'] = view
        z['pred'] = pred
        z['eligible_stress'] = True if not fields else x.loc[mask, fields].ne(v.MISSING).any(axis=1).to_numpy()
        for k in range(3):
            z['p_' + str(k)] = prob[:, k]
        parts.append(z)
    return pd.concat(parts, ignore_index=True)

def apply_heads(baseline, heads, xxfull):
    parts = []
    for view, fields in v.VIEWS.items():
        xx = xxfull.copy()
        xx[fields] = v.MISSING
        b = baseline[baseline.scenario == view].reset_index(drop=True).copy()
        used = np.zeros(len(b), bool)
        for name, bundle in heads.items():
            mask = active(xx, name)
            assert not np.any(used & mask)
            used |= mask
            if not mask.any():
                continue
            offset = np.log(np.maximum(b.loc[mask, ['p_0', 'p_1', 'p_2']].to_numpy().astype(float), 1e-30))
            p = softmax(offset + m.matrix(bundle['design'], xx.loc[mask]) @ bundle['weights'], axis=1)
            b.loc[mask, 'pred'] = p.argmax(axis=1)
            for k in range(3):
                # Cast column first: avoid silently truncating corrected values to float32.
                b['p_' + str(k)] = b['p_' + str(k)].astype(float)
                b.loc[mask, 'p_' + str(k)] = p[:, k]
        parts.append(b)
    return pd.concat(parts, ignore_index=True)

def run(protocol):
    c, r, x = load()
    folder = RUN / protocol
    assert not folder.exists()
    folder.mkdir()
    roles = pd.read_parquet(RUN / (protocol + '_split.parquet')).role
    fit, cal, ev = (roles.eq(role).to_numpy() for role in ['fit', 'calibration', 'evaluation'])
    start = time.monotonic()
    encoding = v.fit_encoder(x.loc[fit])
    A = v.encode(encoding, x.loc[fit]).astype(np.float32)
    cold = np.flatnonzero(np.asarray(A.sum(axis=0)).ravel() == 0)
    model = MLPClassifier(**c['network'], alpha=c['alpha'], random_state=c['seed'])
    for _ in range(c['epochs']):
        model.partial_fit(A, r.loc[fit, 'label_index'].to_numpy(), classes=[0, 1, 2])
        model.coefs_[0][cold] = 0
    base = {'input': encoding, 'model': model}
    joblib.dump(base, folder / 'base.joblib')
    baseline = base_predictions(base, r, x, cal)
    baseline.to_parquet(folder / 'base_calibration.parquet', index=False)
    decisions, accepted, audits = {}, {}, {}
    for head, field in HEADS.items():
        xx = x.loc[fit].copy()
        xx[field] = v.MISSING
        participating = active(xx, head)
        # Inactive parents retain a constant frozen-base loss. They cannot train
        # a branch that would never act on their corresponding observation.
        bundle = m.fit(base, xx.loc[participating], r.loc[fit, 'label_index'].to_numpy()[participating], r.loc[fit, 'body_group'].to_numpy()[participating], c['strength'], c['optimizer'])
        bundle['head'] = head
        bundle['base_sha256'] = v.sha(folder / 'base.joblib')
        joblib.dump(bundle, folder / (head + '.joblib'))
        pred = apply_heads(baseline, {head: bundle}, x.loc[cal].reset_index(drop=True))
        pred.to_parquet(folder / (head + '_calibration.parquet'), index=False)
        gate = m.compare(baseline, pred)
        passed = bool(bundle['optimizer']['eligible'] and gate['quality_passed'])
        decisions[head] = {'accepted': passed, 'optimizer': bundle['optimizer'], 'gate': gate}
        if passed:
            accepted[head] = bundle
        trainkeys = set(v.frame_keys(xx.loc[participating]))
        calx = x.loc[cal].copy()
        calx[field] = v.MISSING
        audits[head] = {'fit_original_rows': int(fit.sum()), 'fit_active_rows': int(participating.sum()), 'fit_inactive_constant_loss_rows': int((~participating).sum()), 'fit_active_body_groups': int(r.loc[fit, 'body_group'].iloc[np.flatnonzero(participating)].nunique()), 'calibration_rows_with_seen_masked_input': int(sum(k in trainkeys for k in v.frame_keys(calx))), 'calibration_rows': int(cal.sum()), 'design': m.design_audit(bundle['design'], xx.loc[participating])}
        print(json.dumps({'protocol': protocol, 'head': head, 'accepted': passed, 'optimizer': bundle['optimizer'], 'failures': gate['failures']}), flush=True)
    combined = apply_heads(baseline, accepted, x.loc[cal].reset_index(drop=True))
    combined.to_parquet(folder / 'selected_calibration.parquet', index=False)
    combined_gate = m.compare(baseline, combined)
    eligible = bool(accepted and combined_gate['quality_passed'])
    if not eligible:
        assert not accepted, 'Disjoint individually passing heads should compose without new errors'
    v.save(folder / 'selection.json', {'heads': decisions, 'accepted': list(accepted), 'combined_gate': combined_gate, 'eligible_for_evaluation': eligible})
    v.save(folder / 'support_audit.json', audits)
    v.save(folder / 'before_evaluation.json', {'bindings': {p.name: v.sha(p) for p in folder.iterdir() if p.is_file()}, 'evaluation_generated': False})
    if eligible:
        baseev = base_predictions(base, r, x, ev)
        pred = apply_heads(baseev, accepted, x.loc[ev].reset_index(drop=True))
        baseev.to_parquet(folder / 'base_evaluation.parquet', index=False)
        pred.to_parquet(folder / 'evaluation.parquet', index=False)
        v.save(folder / 'evaluation_comparison.json', m.compare(baseev, pred))
    v.save(folder / 'complete.json', {'bindings': {p.name: v.sha(p) for p in folder.iterdir() if p.is_file()}, 'new_base_fits': 1, 'new_head_fits': 2, 'evaluation_executed': eligible, 'seconds': time.monotonic() - start})
    print(json.dumps({'protocol': protocol, 'complete': True, 'accepted': list(accepted), 'seconds': time.monotonic() - start}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--protocol', choices=t.PROTOCOLS)
    a = parser.parse_args()
    prepare() if a.mode == 'prepare' else run(a.protocol)
