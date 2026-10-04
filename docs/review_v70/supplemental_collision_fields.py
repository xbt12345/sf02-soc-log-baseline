"""Read-only field attribution for B's 72 additional target/bin collisions.

All matching fit-M rows are compared, retaining both row-pair and target counts.
No fitting, no inference-time feature changes, no alteration of old evidence.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
from run_v69_support_control import load
from run_v67_targeted import design

RUN = ROOT / 'artifacts/v69_support_control_20260921'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(v):
    return None if np.isnan(v) else float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.out:
        assert not args.out.exists()
    audit_path = Path(__file__).with_name('binning_audit.json')
    audit_hash = sha(audit_path)
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    f, context, manifest = load()
    positions = pd.Index(f.row_position)
    per_target = []
    examples = {}
    model_hashes = {}
    totals = Counter()
    with threadpool_limits(limits=4):
        for fold in range(3):
            prior = next(e for e in audit['models'] if e['fold'] == fold and e['arm'] == 'B_small_sources')
            targets = prior['target']['new_same_fit_M_row_positions']
            roles = pd.read_parquet(RUN / f'fold{fold}_roles.parquet')
            assert np.array_equal(roles.row_position, f.row_position)
            fit = roles.B_small_sources.to_numpy()
            model_path = RUN / f'fold{fold}/B_small_sources/model.joblib'
            model_hashes[str(model_path.relative_to(ROOT))] = sha(model_path)
            assert model_hashes[str(model_path.relative_to(ROOT))] == prior['model_sha256']
            bundle = joblib.load(model_path)
            model = bundle['model']
            x, _, names, _ = design(f, context, 'context_numeric', np.flatnonzero(fit), encoder=bundle['encoder'])
            xp = model._preprocess_X(x, reset=False)
            xb = model._bin_mapper.transform(xp)
            _, keys = np.unique(xb, axis=0, return_inverse=True)
            src = names.index('src_port_fixed'); dst = names.index('dst_port_fixed')
            fit_m = fit & f.label.eq(1).to_numpy()
            table = pd.DataFrame({'key': keys[fit_m], 'index': np.flatnonzero(fit_m)}).groupby('key')['index'].agg(list)
            for p in targets:
                i = positions.get_loc(p)
                assert roles.evaluation.iloc[i] and f.label.iloc[i] == 2
                matched = np.array(table.loc[keys[i]], dtype=int)
                same = (x[matched] == x[i]) | (np.isnan(x[matched]) & np.isnan(x[i]))
                diff = ~same
                assert diff.any(axis=1).all()
                other = np.ones(x.shape[1], bool); other[[src, dst]] = False
                assert not diff[:, other].any(), 'A field other than numeric ports differs'
                kind = np.where(diff[:, src] & diff[:, dst], 'both_ports',
                                np.where(diff[:, src], 'src_port_only', 'dst_port_only'))
                counts = Counter(kind)
                totals.update(counts)
                protocol = f.transport_protocol.iloc[i]
                empty = bool(context['stats'][i, 0] == 0)
                matrix = x[matched].copy(); matrix[np.isnan(matrix)] = np.inf
                details = {}
                for label in sorted(counts):
                    use = kind == label
                    details[label] = {
                        'M_row_pairs': int(use.sum()),
                        'M_source_symbols': int(f.group.iloc[matched[use]].nunique()),
                        'distinct_exact_M_designs': int(np.unique(matrix[use], axis=0).shape[0]),
                    }
                    if label not in examples:
                        j = matched[np.flatnonzero(use)[0]]
                        examples[label] = {
                            'difference_type': label, 'fold': fold,
                            'S_target_row_position': int(p), 'M_fit_row_position': int(f.row_position.iloc[j]),
                            'protocol': protocol, 'empty_context': empty,
                            'S_src_port': numeric(x[i, src]), 'M_src_port': numeric(x[j, src]),
                            'S_dst_port': numeric(x[i, dst]), 'M_dst_port': numeric(x[j, dst]),
                            'all_other_design_fields_exactly_equal': True,
                            'binned_inputs_exactly_equal': bool(np.array_equal(xb[i], xb[j])),
                            'S_normalized_text': f.text.iloc[i], 'M_normalized_text': f.text.iloc[j],
                        }
                per_target.append({
                    'fold': fold, 'row_position': int(p), 'group': int(f.group.iloc[i]),
                    'protocol': protocol, 'empty_context': empty,
                    'difference_types': sorted(counts), 'pair_counts': details,
                    'M_row_pairs': int(len(matched)),
                    'M_source_symbols': int(f.group.iloc[matched].nunique()),
                    'distinct_exact_M_designs': int(np.unique(matrix, axis=0).shape[0]),
                    'context_difference_pairs': 0,
                })
    assert len(per_target) == 72
    t = pd.DataFrame(per_target)
    by_type = {}
    slices = []
    for label in ['src_port_only', 'dst_port_only', 'both_ports']:
        use = t.difference_types.map(lambda v: label in v)
        by_type[label] = {'targets_with_at_least_one_pair': int(use.sum()),
                          'target_source_symbols': int(t.loc[use, 'group'].nunique()),
                          'total_M_row_pairs': int(totals[label])}
        for (protocol, empty), d in t[use].groupby(['protocol', 'empty_context']):
            slices.append({'difference_type': label, 'protocol': protocol, 'empty_context': bool(empty),
                           'target_rows': len(d), 'target_source_symbols': int(d.group.nunique())})
    combination = t.assign(combination=t.difference_types.map('|'.join)).groupby('combination').agg(
        target_rows=('row_position', 'size'), target_sources=('group', 'nunique')).reset_index().to_dict('records')
    by_context = t.groupby(['protocol', 'empty_context']).agg(target_rows=('row_position', 'size'),
                                                             target_sources=('group', 'nunique')).reset_index().to_dict('records')
    result = {
        'scope': 'B_small_sources only, 72 predefined target rows with new fit-M collisions introduced by HGB binning',
        'new_fits': 0, 'target_rows': len(t), 'target_source_symbols': int(t.group.nunique()),
        'all_matching_M_row_pairs': int(sum(totals.values())),
        'counting_units': {'pair': 'One target row paired with one matching training-M row; official duplicate rows retained',
                           'target_by_type': 'Target counted once per difference type; a target may have multiple types',
                           'combination': 'Exclusive target partition according to all observed pair difference types'},
        'by_difference_type': by_type, 'exclusive_target_type_combinations': combination,
        'by_protocol_context': by_context, 'by_type_protocol_context': slices,
        'context_continuous_field_difference_pairs': 0,
        'examples': list(examples.values()), 'targets': per_target,
        'limitations': [
            'A numeric port distinction is not automatically a semantically useful distinction.',
            'Source ports may be transient or environment-specific; precision restoration can restore a fingerprint.',
            'Destination ports may identify services but no service-label causality is established by a pair.',
            'This is observed representational loss, not proof these targets would be repaired after a model change.',
            'All other design fields equal does not certify equal hidden context or equal attack intent.',
            'Row-pair counts are highly correlated and not independent evidence units.',
        ],
        'binning_audit_sha256': audit_hash, 'model_sha256': model_hashes,
        'script_sha256': sha(__file__),
    }
    assert sha(audit_path) == audit_hash
    for path, digest in model_hashes.items(): assert sha(ROOT / path) == digest
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out: args.out.write_text(output + '\n', encoding='utf-8')
    else: print(output)


if __name__ == '__main__':
    main()
