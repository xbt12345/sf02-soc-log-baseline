"""Read-only model audit: observed-field relations are not threat labels.

No model import, model forward, gradient, fitting or parameter mutation.
All graph label diagnostics are retrospective and cannot construct a new target.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / 'artifacts/v148_factual_relation_qualification_20261001'
OUT = ROOT / 'artifacts/v148_independent_relation_relevance_20261001'
FIELDS = ('action', 'outcome', 'transport_protocol', 'src_role', 'dst_role',
          'src_port_fixed', 'src_port_range', 'dst_port_fixed', 'dst_port_range',
          'icmp_type', 'icmp_code', 'icmp_message', 'icmp_unreachable')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_observed(f, field):
    value = f.get(field)
    protocol = f.get('transport_protocol')
    if field in ('src_port_fixed', 'dst_port_fixed'):
        return protocol in ('tcp', 'udp') and isinstance(value, int) and 0 <= value <= 65535
    if field in ('src_port_range', 'dst_port_range'):
        return is_observed(f, field.replace('_range', '_fixed')) and value in ('system', 'user', 'dynamic')
    if field in ('icmp_type', 'icmp_code'):
        return protocol == 'icmp' and isinstance(value, int) and 0 <= value <= 255
    if field.startswith('icmp_'):
        return protocol == 'icmp' and isinstance(value, str) and value not in ('', 'unknown')
    if field.endswith('_role'):
        return value in ('inside', 'outside', 'dmz')
    return isinstance(value, str) and value not in ('', 'unknown')


def main():
    assert not OUT.exists(), 'Preserve executed evidence; use a new version for a correction.'
    identity_path = ROOT / 'artifacts/v147_independent_observability_20261001/observable_identity_ledger.parquet'
    trace_path = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    official_path = ROOT / 'data/official/train.parquet'
    d = pd.read_parquet(identity_path)
    trace = pd.read_parquet(trace_path, columns=['row_position', 'facts_json'])
    assert np.array_equal(d.row_position, trace.row_position)
    official = pd.read_parquet(official_path, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy(np.int8)
    assert len(official) == 2056871
    assert np.array_equal(d.truth.to_numpy(), official[d.row_position.to_numpy()])
    facts = trace.facts_json.map(json.loads)
    capacity_path = GRAPH / 'capacity.json'
    cap = json.loads(capacity_path.read_text(encoding='utf-8'))
    sources = [Path(__file__), identity_path, trace_path, official_path, capacity_path,
               ROOT / 'training/v148_observed_relation.py',
               ROOT / 'training/v148_fact_relation_capacity_audit.py']
    reports = []
    for fold in range(3):
        node_path, edge_path = GRAPH / f'fold{fold}_nodes.parquet', GRAPH / f'fold{fold}_edges.parquet'
        sources.extend([node_path, edge_path])
        node, edge = pd.read_parquet(node_path), pd.read_parquet(edge_path)
        receipt = cap['folds'][fold]
        assert sha(node_path) == receipt['node_sha256'] and sha(edge_path) == receipt['edge_sha256']
        train = d.loc[~d.fold.eq(fold)]
        expected = train.groupby(['local', 'root', 'canonical_key'], sort=True).size().reset_index(name='mass')
        assert node.equals(expected)
        assert not (set(node.root) & set(d.loc[d.fold.eq(fold), 'root']))
        assert int(node.mass.sum()) == len(train) == receipt['original_classification_rows']
        fact_lookup = {}
        for local, inds in train.groupby('local', sort=True).groups.items():
            texts = facts.loc[inds].map(lambda f: json.dumps(f, sort_keys=True, separators=(',', ':')))
            assert texts.nunique() == 1
            fact_lookup[int(local)] = json.loads(texts.iloc[0])
        fs = [fact_lookup[int(i)] for i in node.local]
        vals = np.array([[json.dumps(f.get(k), sort_keys=True) for k in FIELDS] for f in fs], dtype=object)
        masks = np.array([[is_observed(f, k) for k in FIELDS] for f in fs], dtype=bool)
        active = np.array([len(set(vals[masks[:, j], j])) > 1 for j in range(len(FIELDS))])
        masks &= active[None, :]
        assert [k for k, a in zip(FIELDS, active) if a] == receipt['active_fields']
        li, ri = edge.left_node.to_numpy(), edge.right_node.to_numpy()
        assert np.array_equal(node.local.to_numpy()[li], edge.left_local)
        assert np.array_equal(node.local.to_numpy()[ri], edge.right_local)
        assert np.array_equal(node.root.to_numpy()[li], edge.left_root)
        assert np.array_equal(node.root.to_numpy()[ri], edge.right_root)
        assert not edge.left_root.eq(edge.right_root).any()
        assert not np.any(node.canonical_key.to_numpy()[li] == node.canonical_key.to_numpy()[ri])
        common = masks[li] & masks[ri]
        n_common = common.sum(axis=1)
        assert np.all(n_common > 0)
        distances = (((vals[li] != vals[ri]) & common).sum(axis=1) / n_common)
        assert np.array_equal(n_common, edge.observed_fields)
        assert np.array_equal(distances, edge.target_distance)
        degree = edge.groupby('left_node').size()
        assert np.array_equal(edge.anchor_mass, node.mass.to_numpy()[li])
        expected_weights = edge.anchor_mass / edge.left_node.map(degree)
        assert np.array_equal(expected_weights, edge.weight)
        labels = train.groupby(['local', 'root']).truth.agg(lambda a: tuple(sorted(set(a))))
        node_labels = [labels.loc[(int(n.local), int(n.root))] for n in node.itertuples()]
        edge['left_label'] = [ls[0] if len(ls) == 1 else -1 for ls in (node_labels[i] for i in li)]
        edge['right_label'] = [ls[0] if len(ls) == 1 else -1 for ls in (node_labels[i] for i in ri)]
        pure = edge.left_label.ge(0) & edge.right_label.ge(0)
        edge['different_threat_class'] = edge.left_label.ne(edge.right_label)
        zero = edge.target_distance.eq(0)
        zero_cross = pure & zero & edge.different_threat_class
        profiles = []
        for (kind, slot, target), block in edge.loc[pure].groupby(['kind', 'slot', 'target_distance'], sort=True):
            different = block.different_threat_class.to_numpy()
            profiles.append(dict(kind=kind, slot=slot, target_distance=float(target), edges=len(block),
                same_class_edges=int((~different).sum()), different_class_edges=int(different.sum()),
                nominal_anchor_weight=float(block.weight.sum()),
                different_class_nominal_weight=float(block.loc[block.different_threat_class, 'weight'].sum())))
        # An ambiguity diagnostic over the target alone; this is NOT a fitted model or OOD score.
        contradictory_targets = sum(p['same_class_edges'] > 0 and p['different_class_edges'] > 0 for p in profiles)
        floor = sum(min(p['same_class_edges'], p['different_class_edges']) for p in profiles)
        report = dict(fold=fold, legal_train_original_rows=len(train), nodes=len(node), edges=len(edge),
            official_truth_and_role_identity_exact=True, targets_and_weights_independently_recomputed=True,
            graph_original_classification_mass=int(node.mass.sum()),
            zero_distance_edges=int(zero.sum()), zero_distance_different_threat_class_edges=int(zero_cross.sum()),
            zero_distance_cross_class_original_anchor_mass=int(node.loc[np.unique(li[zero_cross]), 'mass'].sum()),
            positive_distance_same_class_edges=int((pure & ~zero & ~edge.different_threat_class).sum()),
            mixed_label_edges=int((~pure).sum()),
            target_kind_slot_profiles_with_both_threat_relations=int(contradictory_targets),
            target_kind_slot_only_empirical_edge_relation_error_floor=int(floor),
            target_kind_slot_profiles=profiles,
            conclusion='Observed-field geometry can be lawful auxiliary information, but is not same/different threat supervision.')
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != 'target_kind_slot_profiles'}, ensure_ascii=False), flush=True)
    OUT.mkdir()
    result = dict(status='lawful_graph_recomputed_threat_relation_claim_not_established',
        latest_actual_training='V146', new_fits=0, new_model_forwards=0, new_gradients=0, new_updates=0,
        official_rows=len(official), ASA_original_rows=len(d), roles=reports,
        registration_or_training_permission_granted=False,
        limitations=[
            'Zero factual distance with different labels is not raw or numerical input identity conflict.',
            'Contradictory graph targets refute interpreting geometry as threat-class equality; they do not prove an auxiliary objective cannot help.',
            'Anchor weights repeated across directed edges are graph objective mass, not extra independent labeled records.',
            'No held labels are used to create relations, tune weights, select a candidate or fit a classifier.',
            'No learned teacher or factual reconstruction quality is demonstrated by graph capacity.',
            'The edge relation floor is a descriptive target-only count, not full-task error floor or acceptance metric.'
        ], source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources})
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
