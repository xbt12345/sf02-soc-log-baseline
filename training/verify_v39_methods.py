"""Recompute method-review evidence without fitting or changing old artifacts."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/2026-09-13/v39_methods'
RUN = ROOT / 'artifacts/v38_local_r1_20260913'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(cm):
    tp = cm.diagonal()
    f1 = 2 * tp / (cm.sum(axis=0) + cm.sum(axis=1))
    return {'errors': int(cm.sum() - tp.sum()), 'macro_f1': float(f1.mean()),
            'class_f1': f1.tolist(), 'S_precision': float(tp[2] / cm.sum(axis=0)[2])}


def main():
    target = OUT / 'verification.json'
    if target.exists():
        raise FileExistsError('Preserve prior verification')
    checks = {}
    assumptions = read(OUT / 'assumption_probes.json')
    tradeoff = read(OUT / 'decision_tradeoff.json')
    checks['old_plan_snapshot_identity'] = sha(OUT / 'plan_before_method_review.md') == assumptions['old_plan_sha256'] == 'f643672568e8cdc09522d65d00ff55ac1c70c1bb32ac9784551e53cd836e0d60'
    checks['assumption_script_identity'] = sha(ROOT / 'training/audit_v39_method_assumptions.py') == assumptions['script_sha256']
    checks['decision_script_identity'] = sha(ROOT / 'training/audit_v39_decision_tradeoff.py') == tradeoff['script_sha256']
    prepared_receipt = read(RUN / 'prepared/complete.json')
    checks['official_data_identity'] = sha(ROOT / 'data/official/train.parquet') == prepared_receipt['official_sha256']
    for name in ['rows.parquet', 'projections.parquet']:
        checks['prepared_identity_' + name] = sha(RUN / 'prepared' / name) == prepared_receipt['files'][name]
    binding = read(RUN / 'equal_classifiers_attempt2/binding.json')
    checks['frozen_training_sources_unchanged'] = all(sha(RUN / 'frozen_training_runtime' / n) == h for n, h in binding['sources'].items())
    for name in ['A_SEMANTIC', 'B_DESTINATION', 'C_BOTH']:
        directory = RUN / 'equal_classifiers_attempt2' / name
        receipt = read(directory / 'complete.json')
        checks['model_unchanged_' + name] = sha(directory / 'model.joblib') == receipt['model_sha256']
    directory = RUN / 'equal_classifiers_attempt2/C_BOTH'
    checks['decision_input_identity'] = sha(directory / 'evaluation.parquet') == assumptions['input_predictions_sha256']
    checks['assumption_model_identity'] = sha(directory / 'model.joblib') == assumptions['input_model_sha256']

    d = pq.read_table(directory / 'evaluation.parquet').to_pandas()
    y = d.label_index.to_numpy(dtype='i8')
    logp = np.log(d[['p_benign', 'p_malicious', 'p_suspicious']].to_numpy())
    before = logp.argmax(axis=1)
    checks['saved_decisions_match_argmax'] = bool(np.array_equal(before, d.pred_label.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()))
    cm = np.bincount(y * 3 + before, minlength=9).reshape(3, 3)
    comparisons = [{'target': 'frozen_C', **metrics(cm)}]
    checks['baseline_4008_errors'] = comparisons[0]['errors'] == 4008
    for trial in tradeoff['trials']:
        scores = logp.copy()
        scores[:, 2] += trial['offset']
        after = scores.argmax(axis=1)
        cm = np.bincount(y * 3 + after, minlength=9).reshape(3, 3)
        name = trial['oracle_target']
        checks['decision_recount_' + name] = cm.tolist() == trial['confusion_matrix']
        checks['M_to_S_recount_' + name] = int(((y == 1) & (before == 1) & (after == 2)).sum()) == trial['previously_correct_M_now_S']
        checks['zero_new_normal_errors_' + name] = int(((y == 0) & (after != 0)).sum()) == trial['new_benign_errors'] == 0
        comparisons.append({'target': name, **metrics(cm)})
    checks['both_offsets_lower_macro_f1'] = all(q['macro_f1'] < comparisons[0]['macro_f1'] for q in comparisons[1:])

    audit = pq.read_table(OUT / 'asa_code13_audit.parquet').to_pandas().sort_values('row_position')
    checks['865_distinct_row_positions'] = len(audit) == audit.row_position.nunique() == 865
    wanted = set(audit.row_position.tolist())
    original = {}
    offset = 0
    for batch in pq.ParquetFile(ROOT / 'data/official/train.parquet').iter_batches(batch_size=16384, columns=['message_sanitized', 'label_binary'], use_threads=False):
        selected = sorted(wanted.intersection(range(offset, offset + len(batch))))
        for pos in selected:
            original[pos] = (batch.column(0)[pos - offset].as_py(), batch.column(1)[pos - offset].as_py())
        offset += len(batch)
    checks['official_row_count'] = offset == 2056871
    rebuilt = []
    label_map = {'benign': 0, 'malicious': 1, 'suspicious': 2}
    for row in audit.itertuples(index=False):
        raw, label = original[row.row_position]
        body = raw[raw.lower().index('deny icmp '):].strip()
        h = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()
        normalized = re.sub(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)', '<IP>', body)
        normalized = re.sub(r'(?:USER|HOST|ORG|CRED)(?:-\d+)+', '<ENTITY>', normalized)
        normalized = re.sub(r'dmz-?\d+', 'dmz', normalized, flags=re.I)
        assert h(raw) == row.raw_hash and h(body) == row.body_hash
        assert h(normalized) == row.without_identity_hash and label_map[label] == row.label_index
        interfaces = re.search(r'\bsrc\s+([^:\s]+):\S+\s+dst\s+([^:\s]+):', body, re.I)
        assert interfaces and interfaces[1] == row.src_interface and interfaces[2] == row.dst_interface
        rebuilt.append((h(raw), h(body), h(normalized), row.label_index))
    checks['original_messages_labels_and_interfaces_match'] = len(rebuilt) == 865
    f = pd.DataFrame(rebuilt, columns=['raw', 'body', 'normalized', 'label'])
    counts = pd.crosstab(f.normalized, f.label)
    checks['386_raw_2_bodies_1_normalized'] = [f[c].nunique() for c in ['raw', 'body', 'normalized']] == [386, 2, 1]
    checks['code13_minimum_180'] = int((counts.sum(axis=1) - counts.max(axis=1)).sum()) == 180
    checks['code13_label_support'] = f.label.value_counts().to_dict() == {2: 685, 1: 180}

    source_receipts = read(OUT / 'source_receipts_git.json')
    source_files = [f for repo in source_receipts['results'] for f in repo.get('files', [])]
    checks['13_source_snapshots_unchanged'] = len(source_files) == 13 and all(sha(ROOT / f['local']) == f['sha256'] and (ROOT / f['local']).stat().st_size == f['bytes'] for f in source_files)
    documents = ['docs/V39_METHOD_RESEARCH.md', 'docs/V39_NEXT_PLAN.md', 'docs/TRAINING_PLAN.md', 'README.md']
    missing = []
    for name in documents:
        path = ROOT / name
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):
                continue
            dest = (path.parent / link.split('#')[0]).resolve()
            if not dest.exists() and dest != target.resolve():
                missing.append([name, link])
    checks['local_links_exist_or_this_receipt'] = not missing
    paths = [p for p in OUT.rglob('*') if p.is_file()]
    paths += [ROOT / n for n in documents]
    paths += [Path(__file__), ROOT / 'training/audit_v39_method_assumptions.py', ROOT / 'training/audit_v39_decision_tradeoff.py', ROOT / 'training/collect_v39_method_sources.py']
    result = {'scope': 'Input identity, original-record recount, fixed-prediction diagnostic and source/document verification; not new model acceptance.',
              'all_evidence_checks_passed': all(checks.values()), 'checks': checks,
              'diagnostic_comparison': comparisons, 'missing_links': missing,
              'files_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              'new_model_trained': False, 'platform_used': False, 'external_training_data_used': False}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    assert target.exists()
    print(json.dumps({k: v for k, v in result.items() if k != 'files_sha256'}, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
