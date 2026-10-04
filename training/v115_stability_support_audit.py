"""Recount observed information and compositional support; no classifier fitting.

Labels of held-out rows are used only for retrospective error stratification.
Support dictionaries are always constructed from the other two frozen folds.
The fixed evidence decomposition is diagnostic, not a proposed true M/S rule.
"""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/v115_stability_support_20260929'
PREVIOUS = ROOT / 'artifacts/v114_mechanism_review_20260929_r2'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def encode(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def atoms(f):
    proto = f.get('transport_protocol')
    if proto not in ('tcp', 'udp', 'icmp'):
        return None, 'unparsed'
    keys = ('action', 'outcome', 'src_role', 'dst_role')
    if any(f.get(k) in (None, '', 'unknown') for k in keys):
        return None, 'incomplete_role_or_action'
    if proto == 'icmp':
        if any(k not in f or not 0 <= f[k] <= 255 for k in ('icmp_type', 'icmp_code')):
            return None, 'incomplete_parameter'
        # ICMP code is meaningful conditional on type: never split this atom.
        parameter = ('icmp', f['icmp_type'], f['icmp_code'])
    else:
        port = f.get('dst_port_fixed', 65536)
        if not 0 <= port <= 65535:
            return None, 'incomplete_parameter'
        parameter = (proto, port)
    result = {'protocol': proto, 'parameter': parameter}
    result.update({k: f[k] for k in keys})
    return tuple(sorted((k, encode(v)) for k, v in result.items())), None


def empirical_floor(keys, y):
    table = pd.crosstab(pd.Series(keys, name='key'), pd.Series(y, name='truth'))
    total = table.sum(axis=1)
    floor = int((total-table.max(axis=1)).sum())
    # Independent Counter arithmetic protects grouping/count mistakes.
    counts = Counter(zip(keys, y))
    best = {}
    for (key, _label), n in counts.items():
        best[key] = max(best.get(key, 0), n)
    assert floor == len(y)-sum(best.values())
    return {'groups': len(table), 'mixed_label_groups': int((table.gt(0).sum(axis=1) > 1).sum()),
            'empirical_minimum_errors': floor}


def main():
    assert not DEST.exists(), 'Refusing to overwrite an existing audit.'
    receipt = json.loads((PREVIOUS/'verification.json').read_text(encoding='utf-8'))
    assert receipt['all_checks_passed']
    for rel, digest in receipt['artifact_sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    diagnosis = json.loads((PREVIOUS/'diagnosis.json').read_text(encoding='utf-8'))
    dp = ROOT/'artifacts/v113_case_training_20260929/OOF_ASA_decisions.parquet'
    tp = ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    assert sha(dp) == diagnosis['decision_sha256']
    assert sha(tp) == diagnosis['trace_sha256']
    d = pd.read_parquet(dp)
    raw = pd.read_parquet(tp)
    for col in ('row_position', 'local', 'root', 'fold', 'truth'):
        assert np.array_equal(d[col], raw[col]), col
    assert len(d) == 112807 and d.groupby('root').fold.nunique().max() == 1
    d['facts_json'] = raw.facts_json
    facts = [json.loads(s) for s in d.facts_json]
    y = d.truth.to_numpy()
    projections = {
        'full_N1_model_input_id': d.local.tolist(),
        'parsed_message_facts_only': [encode(f) for f in facts],
        'parsed_message_facts_without_source_port': [encode({k: v for k, v in f.items() if not k.startswith('src_port')}) for f in facts],
        'parsed_message_facts_plus_lowercase_literal_interface_pair': [encode([f, p.lower() if isinstance(p, str) else None]) for f, p in zip(facts, d.interface_literal_pair)],
    }
    floors = {k: empirical_floor(v, y) for k, v in projections.items()}
    data = [atoms(f) for f in facts]
    d['atom_tuple'] = [a for a, _ in data]
    d['incomplete_reason'] = [r for _, r in data]
    d['joint_key'] = [encode(a) if a is not None else None for a, _ in data]
    d['support_bin'] = ''
    d['unseen_atom_names'] = ''
    d['all_pairs_seen'] = False
    d['exact_same_class_roots'] = 0
    d['exact_opposite_class_roots'] = 0
    role_counts = []
    for fold in range(3):
        train = d[d.fold != fold]
        valid = train[train.joint_key.notna()]
        all_atoms = {a for tup in valid.atom_tuple for a in tup}
        all_pairs = {pair for tup in valid.atom_tuple for pair in itertools.combinations(tup, 2)}
        all_joint = set(valid.joint_key)
        counts = valid.groupby(['joint_key', 'truth']).root.nunique().to_dict()
        # Cache per joint; rows with the same observed key must get the same support.
        cache = {}
        for idx, row in d[d.fold == fold].iterrows():
            if row.joint_key is None:
                d.at[idx, 'support_bin'] = row.incomplete_reason
                continue
            if row.joint_key not in cache:
                unseen = [k for k, v in row.atom_tuple if (k, v) not in all_atoms]
                support = ('exact_joint_seen' if row.joint_key in all_joint else
                           'unseen_joint_all_atoms_seen' if not unseen else 'unseen_atomic_value')
                pairs_seen = all(p in all_pairs for p in itertools.combinations(row.atom_tuple, 2))
                cache[row.joint_key] = support, ','.join(unseen), pairs_seen
            support, unseen, pairs_seen = cache[row.joint_key]
            d.at[idx, 'support_bin'] = support
            d.at[idx, 'unseen_atom_names'] = unseen
            d.at[idx, 'all_pairs_seen'] = pairs_seen
            d.at[idx, 'exact_same_class_roots'] = counts.get((row.joint_key, row.truth), 0)
            d.at[idx, 'exact_opposite_class_roots'] = counts.get((row.joint_key, 3-row.truth), 0)
        for role, z in [('fit', train), ('held_out', d[d.fold == fold])]:
            for cls in (1, 2):
                q = z[z.truth == cls]
                role_counts.append({'fold': fold, 'role': role, 'truth': cls,
                                    'rows': len(q), 'roots': int(q.root.nunique())})
    assert d.support_bin.ne('').all()
    d['A_wrong'] = d.A_N1_prediction != y
    d['B_wrong'] = d.B_case_prediction != y
    d['protocol'] = [f.get('transport_protocol', 'unparsed') for f in facts]
    d['parameter'] = [encode((f.get('transport_protocol'), f.get('icmp_type'), f.get('icmp_code')))
                      if f.get('transport_protocol') == 'icmp' else encode((f.get('transport_protocol'), f.get('dst_port_fixed'))) for f in facts]
    aggregates = {}
    for name, keys in {
        'support_bins': ['support_bin', 'truth'],
        'support_by_fold_protocol': ['fold', 'protocol', 'support_bin', 'truth'],
        'unseen_atoms': ['fold', 'unseen_atom_names', 'truth'],
        'icmp_parameters': ['fold', 'protocol', 'parameter', 'support_bin', 'truth'],
    }.items():
        z = d[d.protocol == 'icmp'] if name == 'icmp_parameters' else d
        table = z.groupby(keys, dropna=False).agg(rows=('local', 'size'), roots=('root', 'nunique'),
                     unique_inputs=('local', 'nunique'), A_errors=('A_wrong', 'sum'), B_errors=('B_wrong', 'sum')).reset_index()
        aggregates[name] = table
    root_classes = d.groupby('root').truth.nunique()
    no_same = d.joint_key.notna() & d.exact_same_class_roots.eq(0)
    zero_support_recount = []
    for cls in (1, 2):
        z = d[no_same & d.truth.eq(cls)]
        zero_support_recount.append({'truth': cls, 'rows': len(z), 'A_correct': int((~z.A_wrong).sum()),
                                     'B_correct': int((~z.B_wrong).sum())})
    report = {
        'status': 'retrospective_support_and_information_audit_no_fit',
        'classifier_fits': 0, 'calibration_fits': 0, 'rows': len(d),
        'class_rows': {str(c): int((y == c).sum()) for c in (1, 2)},
        'source_roots': {'total': len(root_classes), 'single_label': int(root_classes.eq(1).sum()), 'mixed': int(root_classes.gt(1).sum())},
        'projection_empirical_lower_bounds': floors,
        'support_bins': aggregates['support_bins'].to_dict('records'),
        'complete_joint_zero_same_class_support': zero_support_recount,
        'role_counts': role_counts,
        'all_pairs_seen_but_unseen_joint': {'rows': int((d.support_bin.eq('unseen_joint_all_atoms_seen') & d.all_pairs_seen).sum()),
            'A_errors': int((d.support_bin.eq('unseen_joint_all_atoms_seen') & d.all_pairs_seen & d.A_wrong).sum()),
            'B_errors': int((d.support_bin.eq('unseen_joint_all_atoms_seen') & d.all_pairs_seen & d.B_wrong).sum())},
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in (dp, tp, PREVIOUS/'verification.json', Path(__file__))},
        'limits': [
            'All frozen outer folds have been repeatedly inspected; these are developmental diagnostics.',
            'Support atoms are chosen observed protocol/roles/parameter; source ports, interface suffixes and residual text are outside this diagnostic key.',
            'Seen components do not establish their causal meaning, sufficiency, correct composition rule or transferability.',
            'Single-label isolation roots are not independently verified organizations or policy domains.',
            'Empirical collision floors use all labels and apply only to the stated projection on these rows; not Bayes error or a model evaluation.',
            'Parsed message facts do not include the full model input, residual text, or all record-level source-port features.',
            'No labels are changed, no new classifier is trained, no prediction rule is selected from held-out labels.'
        ]}
    DEST.mkdir()
    for name, table in aggregates.items():
        table.to_csv(DEST/(name+'.csv'), index=False)
    pd.DataFrame(role_counts).to_csv(DEST/'role_counts.csv', index=False)
    d.drop(columns=['facts_json', 'atom_tuple']).to_parquet(DEST/'support_decisions.parquet', index=False)
    save(DEST/'diagnosis.json', report)
    verification = {'all_checks_passed': True, 'classifier_fits': 0, 'calibration_fits': 0,
        'checks': {'V114_bound_evidence_hashes': True, 'V113_decision_and_official_checked_trace_hashes': True,
                   'all_row_label_fold_root_alignment': True, 'roots_do_not_cross_folds': True,
                   'projection_floors_independently_counted': True,
                   'support_population_and_errors_conserved': bool(aggregates['support_bins'].rows.sum() == 112807
                        and aggregates['support_bins'].A_errors.sum() == 2412 and aggregates['support_bins'].B_errors.sum() == 2190)},
        'scope': 'Artifact identity and arithmetic; not new training or transfer acceptance.',
        'artifact_sha256': {p.name: sha(p) for p in DEST.iterdir() if p.is_file()}}
    assert all(verification['checks'].values())
    save(DEST/'verification.json', verification)
    print(json.dumps({k: report[k] for k in ('source_roots', 'projection_empirical_lower_bounds', 'support_bins', 'complete_joint_zero_same_class_support', 'all_pairs_seen_but_unseen_joint')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
