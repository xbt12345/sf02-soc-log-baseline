"""Label-free cross-root relations between explicitly observed body facts.

The relation is equality/difference of typed observed fields, not equality of
threat class or whole original behavior. Original model inputs remain intact.
"""
import json
import numpy as np
import pandas as pd

FIELDS = ('action', 'outcome', 'transport_protocol', 'src_role', 'dst_role',
    'src_port_fixed', 'src_port_range', 'dst_port_fixed', 'dst_port_range',
    'icmp_type', 'icmp_code', 'icmp_message', 'icmp_unreachable')
CONTEXT = ('action', 'outcome', 'transport_protocol', 'src_role', 'dst_role')
MAX_PER_SLOT = 4


def observed(f, name):
    protocol = f.get('transport_protocol')
    value = f.get(name)
    if name in ['src_port_fixed', 'dst_port_fixed']:
        return protocol in ['tcp', 'udp'] and isinstance(value, int) and 0 <= value <= 65535
    if name in ['src_port_range', 'dst_port_range']:
        port = name.replace('_range', '_fixed')
        return observed(f, port) and value in ['system', 'user', 'dynamic']
    if name in ['icmp_type', 'icmp_code']:
        return protocol == 'icmp' and isinstance(value, int) and 0 <= value <= 255
    if name.startswith('icmp_'):
        return protocol == 'icmp' and isinstance(value, str) and value not in ['', 'unknown']
    if name.endswith('_role'):
        return value in ['inside', 'outside', 'dmz']
    return isinstance(value, str) and value not in ['', 'unknown']


def slots(f):
    result = []
    for name in ['src_port_fixed', 'dst_port_fixed']:
        if observed(f, name):
            result.append((name, str(f[name])))
    if observed(f, 'icmp_type') and observed(f, 'icmp_code'):
        result.append(('icmp_type_code', f"{f['icmp_type']}/{f['icmp_code']}"))
    return result


def make_graph(frame, facts):
    # The topology receives no truth, predictions, error IDs, metadata or clocks.
    nodes = frame[['local', 'root', 'canonical_key']].groupby(['local', 'root', 'canonical_key'], sort=True).size().reset_index(name='mass')
    facts_by_local = {}
    for local, indices in frame.groupby('local', sort=True).groups.items():
        originals = facts.loc[indices].map(lambda v: json.dumps(v, sort_keys=True, separators=(',', ':')))
        if originals.nunique() != 1:
            raise ValueError('One registered numeric entry has inconsistent body facts')
        facts_by_local[int(local)] = json.loads(originals.iloc[0])
    fs = [facts_by_local[int(i)] for i in nodes.local]
    values = np.array([[json.dumps(f.get(k), sort_keys=True) for k in FIELDS] for f in fs], dtype=object)
    masks = np.array([[observed(f, k) for k in FIELDS] for f in fs], dtype=bool)
    active = np.array([len(set(values[masks[:, j], j])) > 1 for j in range(len(FIELDS))])
    masks &= active[None, :]
    pools, alternatives, contexts = {}, {}, []
    for i, f in enumerate(fs):
        context = tuple(json.dumps(f.get(k), sort_keys=True) for k in CONTEXT) if all(observed(f, k) for k in CONTEXT) else None
        contexts.append(context)
        if context is None:
            continue
        for name, value in slots(f):
            pools.setdefault((context, name, value), []).append(i)
            alternatives.setdefault((context, name), set()).add(value)
    # One representative per root, ordered by numeric identity, not threat label.
    representatives = {}
    for key, members in pools.items():
        ids = sorted(members, key=lambda i: (nodes.canonical_key.iloc[i], int(nodes.root.iloc[i]), int(nodes.local.iloc[i])))
        seen = set(); chosen = []
        for i in ids:
            root = int(nodes.root.iloc[i])
            if root not in seen:
                seen.add(root); chosen.append(i)
        representatives[key] = chosen
    edges = []
    for i, f in enumerate(fs):
        context = contexts[i]
        if context is None:
            continue
        selected = {}
        def add(candidates, kind, slot):
            taken = 0
            for j in candidates:
                if nodes.root.iloc[j] == nodes.root.iloc[i] or nodes.canonical_key.iloc[j] == nodes.canonical_key.iloc[i]:
                    continue
                if j not in selected:
                    selected[j] = (kind, slot)
                taken += 1
                if taken >= MAX_PER_SLOT:
                    break
        for name, value in slots(f):
            add(representatives[(context, name, value)], 'shared_observed_component', name)
            # Different concrete values are retained as witnesses; never relabeled.
            different = sorted(alternatives[(context, name)] - {value})
            witness = []
            for other in different:
                witness += representatives[(context, name, other)][:MAX_PER_SLOT]
                if len(witness) >= 2 * MAX_PER_SLOT:
                    break
            add(witness, 'different_observed_component', name)
        for j, (kind, slot) in selected.items():
            common = masks[i] & masks[j]
            if not common.any():
                raise ValueError('Relation without common observed fields')
            target = float((values[i, common] != values[j, common]).mean())
            edges.append(dict(left_node=i, right_node=j, left_local=int(nodes.local.iloc[i]), right_local=int(nodes.local.iloc[j]),
                left_root=int(nodes.root.iloc[i]), right_root=int(nodes.root.iloc[j]), kind=kind, slot=slot,
                target_distance=target, observed_fields=int(common.sum()), anchor_mass=int(nodes.mass.iloc[i])))
    edges = pd.DataFrame(edges)
    if edges.empty:
        raise ValueError('No actual cross-root observed relations')
    degree = edges.groupby('left_node').size()
    edges['weight'] = edges.anchor_mass / edges.left_node.map(degree)
    assert not edges.left_root.eq(edges.right_root).any()
    assert len(nodes) == int(frame[['local', 'root', 'canonical_key']].drop_duplicates().shape[0])
    return nodes, edges, dict(active_fields=[k for k, a in zip(FIELDS, active) if a],
        inactive_constant_fields=[k for k, a in zip(FIELDS, active) if not a],
        nodes=len(nodes), edges=len(edges), supervised_anchor_nodes=int(degree.size),
        supervised_original_anchor_mass=int(nodes.loc[degree.index, 'mass'].sum()),
        all_original_classification_mass=int(nodes.mass.sum()), labels_used_for_topology=0,
        target='Masked nominal field difference / common observed nonconstant fields. It is not a threat-class target.',
        component_match_is_not_whole_behavior_equivalence=True)
