"""Read-only priority/semantic audit. No classifier fit or source modification.

The Meraki pattern interpretation is a bounded audit hypothesis backed by the
vendor's syslog documentation, not a maliciousness rule or production parser.
"""
import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.metrics import confusion_matrix


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')


def pattern_action(raw):
    # Called only after the frozen native_flow grammar succeeds. No other field,
    # model prediction, label or device name is used to determine this action.
    m = re.search(r'\bpattern:\s+(1|0|allow|deny)\s+(all|dst\s+\S+)\s*$', raw, re.I)
    if not m: return None
    return 'deny' if m[1].lower() in ('1', 'deny') else 'allow'


def summary(d):
    q = d[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy()
    d = d.copy(); d['prediction'] = q.argmax(1); d['wrong'] = d.prediction != d.label_index
    result = []
    for route, g in d.groupby('route'):
        result.append({'route': route, 'rows': len(g), 'body_groups': int(g.body_group.nunique()),
            'errors': int(g.wrong.sum()), 'wrong_body_groups': int(g.loc[g.wrong, 'body_group'].nunique()),
            'M_to_B': int(((g.label_index == 1)&(g.prediction == 0)).sum()),
            'S_to_B': int(((g.label_index == 2)&(g.prediction == 0)).sum()),
            'class_counts_B_M_S': [int((g.label_index == k).sum()) for k in range(3)]})
    return sorted(result, key=lambda v: -v['errors'])


def main(root, out):
    assert not out.exists()
    base = root/'artifacts/v39_local_r2_20260913'; stress = base/'old_protocol_stress'
    receipt = read(stress/'complete.json')
    for name, key in [('model.joblib', 'model_sha256'), ('evaluation.parquet', 'predictions_sha256'), ('report.json', 'report_sha256')]:
        assert sha(stress/name) == receipt[key]
    old_report = read(stress/'report.json')
    c46 = read(root/'artifacts/v46_interaction_20260914/configuration.json')
    for name in ['rows.parquet', 'projections.parquet']:
        path = base/'prepared'/name
        assert sha(path) == c46['source_bindings'][path.relative_to(root).as_posix()]
    official = root/'data/official/train.parquet'
    old_audit = read(root/'evidence/2026-09-12/v35_direction_review/task_assumptions_audit.json')
    assert sha(official) == old_audit['source_file_sha256']
    doc = next((root/'docs/official').glob('*.docx'))
    assert sha(doc) == old_audit['official_doc_sha256']
    with zipfile.ZipFile(doc) as z: tree = ET.fromstring(z.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    pars = [''.join(n.text or '' for n in p.findall('.//w:t', ns)) for p in tree.findall('.//w:p', ns)]
    a = next(i for i, s in enumerate(pars) if 'SF-2026-02' in s)-1
    b = next(i for i, s in enumerate(pars) if 'SF-2026-03' in s)-1
    task = '\n'.join(pars[a:b])+'\n'
    assert task == (root/'evidence/2026-09-12/v35_direction_review/official_task2_verified.txt').read_text(encoding='utf-8')
    # Recount the product/label confounding; do not use these fields for prediction.
    source = pq.read_table(official, columns=['product_name', 'label_binary']).to_pandas().fillna('')
    sc = pd.crosstab(source.product_name, source.label_binary).reindex(columns=['benign', 'malicious', 'suspicious'], fill_value=0)
    assert int(sc.malicious.sum()) == 111728
    source_summary = {'rows': len(source), 'all_malicious_product_blank': bool(sc.loc['', 'malicious'] == sc.malicious.sum()),
        'named_products': int((sc.index != '').sum()),
        'named_products_with_one_observed_class': int(((sc.loc[sc.index != ''] > 0).sum(axis=1) == 1).sum()),
        'counts': [{'product': k, 'B_M_S': row.astype(int).tolist()} for k, row in sc.iterrows()]}
    rows = pd.read_parquet(base/'prepared/rows.parquet'); allowed = rows[rows.inner_role >= 0].copy()
    assert len(allowed) == 1378650
    ev = pd.read_parquet(stress/'evaluation.parquet').sort_values('row_position').reset_index(drop=True)
    fit = allowed[allowed.inner_role != 2]; expected_ev = allowed[allowed.inner_role == 2].sort_values('row_position')
    assert np.array_equal(ev.row_position, expected_ev.row_position)
    assert not set(fit.body_group)&set(expected_ev.body_group)
    assert not set(fit.union_group)&set(expected_ev.union_group)
    current = pd.concat([pd.read_parquet(base/('primary/fold_%s/SEMANTIC/evaluation.parquet'%f)) for f in range(3)])
    primary_s = summary(current); stress_s = summary(ev)
    total_errors = sum(z['errors'] for z in stress_s)
    assert total_errors == old_report['evaluation']['errors'] == 3988
    flow = allowed[allowed.route == 'native_flow'].copy().sort_values('row_position')
    assert len(flow) == 4026
    projections = pd.read_parquet(base/'prepared/projections.parquet')
    frozen = root/'artifacts/v42_local_r1_20260914/frozen_training_runtime'
    closure = read(frozen.parent/'prepared/complete.json')
    for name, digest in closure['runtime_sources'].items(): assert sha(frozen/name) == digest
    sys.path.insert(0, str(frozen)); import v39_core as core
    poses = flow.row_position.to_numpy(); raw = {}; offset = 0
    for batch in pq.ParquetFile(official).iter_batches(batch_size=16384, columns=['message_sanitized'], use_threads=False):
        wanted = poses[np.searchsorted(poses, offset):np.searchsorted(poses, offset+len(batch))]
        for pos in wanted: raw[int(pos)] = batch.column(0)[int(pos)-offset].as_py() or ''
        offset += len(batch)
    records = []; examples = []; orig_facts = []; repaired_facts = []; texts = []
    for row in flow.itertuples():
        message = raw[int(row.row_position)]
        parsed = core.prepare_record({'message_sanitized': message})
        old = projections.iloc[int(row.projection_id)]
        assert parsed['text'] == old.text and core.canonical(parsed['facts']) == core.canonical(json.loads(old.facts))
        assert core.previous.prior.prepare_message(message)['route'] == 'native_flow'
        facts = copy.deepcopy(parsed['facts']); action = pattern_action(message)
        existing = facts.get('action'); conflict = action is not None and existing is not None and action != existing
        assert not conflict, 'Conflicting observed actions require review, never silent priority.'
        changed = action is not None and existing is None
        if changed:
            facts['action'] = action; facts['outcome'] = 'blocked' if action == 'deny' else 'allowed'
        tail = re.search(r'\bpattern:\s+(.+)$', message)
        records.append({'row_position': int(row.row_position), 'body_group': int(row.body_group), 'projection_id': int(row.projection_id),
            'role': 'pressure' if row.inner_role == 2 else 'fit', 'label_index': int(row.label_index),
            'protocol': parsed['facts'].get('transport_protocol', '<missing>'),
            'action_before': existing or '<missing>', 'documented_action': action or '<unresolved>',
            'action_after': facts.get('action', '<missing>'), 'changed': changed,
            'pattern_token': tail[1].split()[0] if tail else '<none>'})
        signature = (records[-1]['role'], records[-1]['protocol'], records[-1]['pattern_token'])
        if sum(z['signature'] == list(signature) for z in examples) < 2:
            examples.append({'signature': list(signature), 'row_position': int(row.row_position), 'raw': message,
                'before_facts': parsed['facts'], 'audit_proposed_facts': facts})
        orig_facts.append(parsed['facts']); repaired_facts.append(facts); texts.append(parsed['text'])
    details = pd.DataFrame(records)
    grouped = details.groupby(['role', 'protocol', 'action_before', 'action_after', 'pattern_token']).agg(
        rows=('row_position', 'size'), bodies=('body_group', 'nunique')).reset_index().to_dict(orient='records')
    # Inference-only sensitivity with the SAME frozen stress model. Changes are
    # not called a retrained score and not promoted; fit-side semantics also need repair.
    model = joblib.load(stress/'model.joblib')
    prediction = lambda ff: model['model'].predict_proba(sparse.hstack([
        model['text_encoder'].transform(texts), model['fact_encoder'].transform(ff)], format='csr'))
    before = prediction(orig_facts); after = prediction(repaired_facts)
    pressure = details.role.to_numpy() == 'pressure'; ef = ev[ev.route == 'native_flow'].sort_values('row_position')
    assert np.array_equal(details.loc[pressure, 'row_position'], ef.row_position)
    difference = float(np.max(np.abs(before[pressure]-ef[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy())))
    assert difference < 1e-12
    yy = details.loc[pressure, 'label_index'].to_numpy(); old_pred = before[pressure].argmax(1); new_pred = after[pressure].argmax(1)
    probe = {'scope': 'Frozen-model inference sensitivity only. No training or promotion; before/after are paired on the same pressure rows.',
        'rows': int(pressure.sum()), 'original_errors': int((old_pred != yy).sum()), 'after_audit_repair_errors': int((new_pred != yy).sum()),
        'original_confusion_B_M_S': confusion_matrix(yy, old_pred, labels=[0, 1, 2]).tolist(),
        'after_confusion_B_M_S': confusion_matrix(yy, new_pred, labels=[0, 1, 2]).tolist(),
        'baseline_replay_max_abs_diff': difference,
        'warning': 'All eligible native_flow rows are malicious. This cannot measure normal false positives or establish deny implies attack.'}
    # Syntax-only counterexamples, no synthetic security labels or training rows.
    assert pattern_action('flows src=x dst=y protocol=udp pattern: 1 all') == 'deny'
    assert pattern_action('flows src=x dst=y protocol=udp pattern: 0 all') == 'allow'
    assert pattern_action('flows src=x dst=y protocol=udp pattern: allow all') == 'allow'
    assert pattern_action('flows src=x dst=y protocol=udp pattern: 7 all') is None
    flow_benign = int(((allowed.route == 'native_flow')&(allowed.label_index == 0)).sum())
    out.mkdir(parents=True)
    details.to_parquet(out/'flow_semantics.parquet', index=False)
    pred_frame = details.loc[pressure, ['row_position', 'body_group', 'label_index']].copy()
    for i, name in enumerate(['benign', 'malicious', 'suspicious']):
        pred_frame['before_'+name] = before[pressure, i]; pred_frame['audit_after_'+name] = after[pressure, i]
    pred_frame.to_parquet(out/'inference_sensitivity.parquet', index=False)
    (out/'official_task_verified.txt').write_text(task, encoding='utf-8')
    save(out/'raw_semantic_examples.json', examples)
    save(out/'audit.json', {'status': 'direction_audit_no_training', 'new_fits': 0, 'production_modified': False,
        'official_rows': len(source), 'source_support': source_summary,
        'body_development': primary_s, 'old_template_pressure': stress_s,
        'old_pressure_errors': total_errors,
        'old_pressure_native_flow_error_fraction': next(z['errors'] for z in stress_s if z['route'] == 'native_flow')/total_errors,
        'old_pressure_errors_outside_ASA': sum(z['errors'] for z in stress_s if z['route'] != 'asa'),
        'flow_semantics': grouped, 'flow_rows_with_missing_documented_action': int(details.changed.sum()),
        'flow_pressure_rows_with_missing_documented_action': int((details.changed&pressure).sum()),
        'flow_normal_support_rows': flow_benign, 'inference_only_probe': probe,
        'limits': ['Existing pressure set was inspected before this review. Not a blind or external benchmark.',
            'Old stress and recent body-fold scores have different train/evaluation roles; not a same-model cross-test score comparison.',
            'Numeric pattern action is recognized only inside the audited Meraki native grammar; never map arbitrary number 1 to deny.',
            'Vendor denial semantics is not an independent malicious label or a blanket detection rule.'],
        'source_bindings': {str(p.relative_to(root).as_posix()): sha(p) for p in [official, doc,
            base/'prepared/rows.parquet', base/'prepared/projections.parquet', stress/'evaluation.parquet', stress/'model.joblib',
            frozen/'v37_representation.py', frozen/'v38_representation.py', frozen/'v39_core.py']}})
    shutil.copyfile(__file__, out/Path(__file__).name)
    save(out/'verification.json', {'all_audit_checks_passed': True, 'official_doc_and_data_hashes_verified': True,
        'official_extract_reproduced': True, 'old_pressure_counts_reproduced': True,
        'native_raw_rows_reparsed': len(details), 'frozen_stress_probability_rows_replayed': int(pressure.sum()),
        'baseline_max_abs_difference': difference, 'syntax_counterexamples_passed': 4,
        'model_training_executed': False, 'evaluation_labels_changed': False, 'source_modified': False,
        'audit_code_sha256': sha(__file__)})
    print(json.dumps({'flow_action_gaps': int(details.changed.sum()), 'flow_grouped': grouped, 'probe': probe, 'verified': True}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); main(Path(a.root).resolve(), Path(a.out).resolve())
