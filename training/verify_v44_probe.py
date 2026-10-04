"""Replay saved classifiers, audit real nuisance changes, and diagnose thresholds.

Thresholds below deliberately use inspected evaluation labels: descriptive
capacity checks only. They are NEVER written back to model decision rules.
"""
import argparse
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier


def main(root, run):
    assert not (run / 'verification.json').exists()
    spec = importlib.util.spec_from_file_location('frozen_probe', run / 'probe_v44_ready_models.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    c, report, pre = [read(run / s) for s in ['configuration.json', 'report.json', 'preregistered.json']]
    assert probe.sha(run / 'configuration.json') == pre['configuration_sha256']
    assert probe.sha(run / 'probe_v44_ready_models.py') == pre['source_sha256']
    verified = {}
    for s, h in c['source_bindings'].items():
        assert probe.sha(root / s) == h
        verified[s] = h
    official = root / 'data/official/train.parquet'
    assert probe.sha(official) == '6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
    verified[official.relative_to(root).as_posix()] = probe.sha(official)
    frozen = root / 'artifacts/v42_local_r1_20260914/frozen_training_runtime'
    frec = read(frozen.parent / 'prepared/complete.json')
    for s, h in frec['runtime_sources'].items():
        assert probe.sha(frozen / s) == h
        verified[(frozen / s).relative_to(root).as_posix()] = h
    sys.path.insert(0, str(frozen))
    import v39_core as core
    from audit_v37_prepared import variants
    r, x, full_b, ab = probe.data(root)
    e = pd.read_parquet(run / 'evaluation.parquet')
    assert np.array_equal(e.row_position, r.row_position)
    assert np.array_equal(e.label_index, r.label_index)
    y = e.label_index.to_numpy()
    bpred = ab[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy().argmax(1)
    assert np.array_equal(e.B_pred, bpred)
    models, replay = {}, []
    for fold in range(3):
        ix = e.fold.to_numpy() == fold
        for n in ['LR', 'CAT']:
            p = run / ('fold_%s/%s.%s' % (fold, n, 'joblib' if n == 'LR' else 'cbm'))
            assert probe.sha(p) == report['model_sha256'][p.relative_to(run).as_posix()]
            model = joblib.load(p) if n == 'LR' else CatBoostClassifier().load_model(p)
            q = model.predict_proba(x.loc[ix])[:, 1]
            assert np.array_equal(q, e.loc[ix, n + '_qM'].to_numpy())
            pred = np.where(q >= .5, 1, 2)
            assert np.array_equal(pred, e.loc[ix, n + '_head_pred'].to_numpy())
            s = report['slices']['fold_%s' % fold][n]
            assert int((pred != y[ix]).sum()) == s['errors']
            for k in [1, 2]:
                assert int(((y[ix] == k) & (pred != k)).sum()) == s['confusion_M_S'][k-1][2-k]
            models[(fold, n)] = model
            replay.append({'fold': fold, 'model': n, 'rows': int(ix.sum()), 'max_probability_difference': 0.0})
    # Resolve original official messages for one representative of EVERY ASA
    # projection; no cached text or report projection is accepted as raw truth.
    reps = r.drop_duplicates('projection_id')
    wanted = dict(zip(reps.row_position.astype(int), reps.projection_id.astype(int)))
    positions = np.array(sorted(wanted), dtype=np.int64)
    raws = {}
    off = 0
    for batch in pq.ParquetFile(official).iter_batches(batch_size=8192, columns=['message_sanitized'], use_threads=False):
        ps = positions[np.searchsorted(positions, off):np.searchsorted(positions, off+len(batch))]
        for pos in ps:
            raws[wanted[int(pos)]] = batch.column(0)[int(pos)-off].as_py() or ''
        off += len(batch)
    assert len(raws) == 16874
    xp = x.copy()
    xp['pid'] = r.projection_id.to_numpy()
    expected_x = {int(k): list(v) for k, v in xp.drop_duplicates('pid').set_index('pid').iterrows()}
    originals, variant_values, variant_parents = [], [], []
    counts = collections.Counter()
    # All changed projections plus deterministic projection sampling. This is
    # implementation verification, not a separately selected performance test.
    changed = (e.LR_head_pred != e.B_pred) | (e.CAT_head_pred != e.B_pred)
    audit_ids = set(e.loc[changed, 'projection_id'].astype(int)) | set(sorted(raws)[::16])
    for index, (pid, raw) in enumerate(sorted(raws.items())):
        p = core.prepare_record({'message_sanitized': raw})
        xx = probe.encode(p['facts'])
        assert core.previous.prior.prepare_message(raw)['route'] == 'asa' and p['text'] == ''
        assert xx == expected_x[pid], pid
        originals.append(xx)
        if pid not in audit_ids:
            continue
        changes = [('outside_metadata', {'message_sanitized': raw, 'timestamp': '2099-01-01T00:00:00Z',
                    'product_name': None, 'vendor_name': 'new', 'src_ip': '192.0.2.99', 'dst_ip': '203.0.113.77',
                    'username': 'new', 'event_id': 'new', 'pipeline': 'new', 'label_binary': 'benign'})]
        changes.extend((n, {'message_sanitized': text}) for n, text in variants(raw, 'asa'))
        text = re.sub(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)', '203.0.113.99', raw)
        text = re.sub(r'dmz[-_]\d+', 'dmz-999', text)
        changes.append(('address_interface', {'message_sanitized': text}))
        text = re.sub(r'(?:CRED|USER|ORG|HOST|IP|EMAIL)-\d+', 'CRED-999999', raw)
        if text != raw:
            changes.append(('redaction_identity', {'message_sanitized': text}))
        for name, record in changes:
            vv = core.prepare_record(record)
            v = probe.encode(vv['facts'])
            assert core.previous.prior.prepare_message(record['message_sanitized'])['route'] == 'asa'
            assert vv['text'] == '' and v == xx, (pid, name)
            variant_values.append(v)
            variant_parents.append(index)
            counts[name] += 1
    original_df = pd.DataFrame(originals, columns=probe.FIELDS)
    variant_df = pd.DataFrame(variant_values, columns=probe.FIELDS)
    raw_checks = []
    for (fold, n), m in models.items():
        base = m.predict_proba(original_df)[:, 1]
        vp = m.predict_proba(variant_df)[:, 1]
        diff = float(np.max(np.abs(vp-base[np.array(variant_parents)])))
        assert diff == 0.0
        raw_checks.append({'fold': fold, 'model': n, 'original_projections': len(raws),
                           'variant_rows': len(variant_df), 'max_probability_difference': diff})
    # Missing sentinels are not real port/code values; equivalent absence must
    # canonicalize identically. Nuisance fields cannot enter the whitelist.
    for f in [json.loads(v) for v in pq.read_table(root / 'artifacts/v39_local_r2_20260913/prepared/projections.parquet', columns=['facts']).column(0).to_pylist()[:100]]:
        clean = probe.encode(f)
        ff = dict(f, timestamp='new', product_name=None, label_binary='malicious', src_ip='new')
        assert probe.encode(ff) == clean
        for k, sentinel in probe.FINITE.items():
            candidates = []
            for v in [None, sentinel, '', 'CRED-888', -1]:
                z = dict(f); z[k] = v
                candidates.append(probe.encode(z))
            assert all(v == candidates[0] for v in candidates)
    threshold = []
    for n in ['LR', 'CAT']:
        for fold in ['all', 0, 1, 2]:
            ix = np.ones(len(y), dtype=bool) if fold == 'all' else e.fold.to_numpy() == fold
            q, yy, bb, unseen = (e.loc[ix, n+'_qM'].to_numpy(), y[ix], bpred[ix], ~e.loc[ix, 'seen_fit_key'].to_numpy())
            order = np.argsort(-q, kind='stable');q=q[order];yy=yy[order];bb=bb[order];unseen=unseen[order]
            ends = np.r_[np.flatnonzero(q[:-1] != q[1:]), len(q)-1]
            m, s = yy == 1, yy == 2
            m_err = m.sum()-np.cumsum(m)[ends];s_err=np.cumsum(s)[ends]
            u_err=(m&unseen).sum()-np.cumsum(m&unseen)[ends]+np.cumsum(s&unseen)[ends]
            mb=int((m&(bb!=yy)).sum());sb=int((s&(bb!=yy)).sum());ub=int((unseen&(bb!=yy)).sum())
            ok=(m_err<=mb)&(s_err<=sb)&(u_err<=ub)
            ids=np.flatnonzero(ok)
            record={'model':n,'fold':fold,'feasible':bool(len(ids)), 'B_M_errors':mb,'B_S_errors':sb,'B_unseen_errors':ub}
            if len(ids):
                z=ids[np.argmin((m_err+s_err)[ids])]
                record.update(threshold=float(q[ends[z]]),M_errors=int(m_err[z]),S_errors=int(s_err[z]),unseen_errors=int(u_err[z]))
            threshold.append(record)
    probe.save(run / 'threshold_capacity_diagnostic.json', {'status':'EVALUATION_LABEL_ORACLE_DIAGNOSTIC_ONLY_DO_NOT_DEPLOY',
        'rule':'Search observed evaluation score thresholds with M errors, S errors and unseen-input total errors each no greater than B on the same rows; then minimize total errors.',
        'limits':'No learned threshold, no independent validation, no calibrated-probability claim. Thresholds were not applied to saved classifiers or report.',
        'results':threshold})
    # Trace regressions and gains into fit-only support; evaluation conflicts
    # are used for diagnosis only, never classifier routing.
    supports = pd.read_parquet(root / 'artifacts/v43_information_audit_20260914/fit_support_by_row.parquet')
    z = e.merge(supports.drop(columns='fold'), on='row_position', validate='one_to_one')
    support_summary = {}
    for s in ['both_labels_seen', 'three_bodies_per_label', 'pure_fit_opposite', 'unseen']:
        mask=z[s].to_numpy()
        support_summary[s]={'rows':int(mask.sum())}
        for n in ['B', 'LR', 'CAT']:
            pred=z.B_pred.to_numpy() if n=='B' else z[n+'_head_pred'].to_numpy()
            support_summary[s][n+'_errors']=int((mask&(pred!=z.label_index.to_numpy())).sum())
    probe.save(run / 'support_diagnosis.json', support_summary)
    probe.save(run / 'verification.json', {'all_implemented_checks_passed':True,'model_quality_passed':False,
        'reason':'Default-threshold heads increase malicious-to-suspicious errors; CAT also increases unseen-input errors.',
        'saved_prediction_replay':replay,'raw_inference':raw_checks,'raw_variant_types':dict(counts),
        'raw_sources_verified':len(raws),'source_bindings_after_training':verified,
        'class_and_row_alignment_passed':True,'metadata_whitelist_and_missing_equivalence_passed':True,
        'model_fits_in_verification':0,'scope':'Conditional ASA heads and offline protected overlay. No production API installation or external transfer test.'})
    print(json.dumps({'replay_rows':sum(v['rows'] for v in replay),'raw_sources':len(raws),'variants_per_model':len(variant_df),
                      'raw_probability_differences':0,'model_quality_passed':False,'support':support_summary}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();main(Path(a.root).resolve(),Path(a.run).resolve())
