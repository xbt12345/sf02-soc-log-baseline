"""Independent recount of V115 support classifications and observed-label floors."""
import json
from collections import Counter, defaultdict

import pandas as pd

from v115_stability_support_audit import ROOT, DEST, encode, save, sha


def main():
    receipt = json.loads((DEST/'verification.json').read_text(encoding='utf-8'))
    for name, digest in receipt['artifact_sha256'].items():
        assert sha(DEST/name) == digest, name
    d = pd.read_parquet(DEST/'support_decisions.parquet')
    trace = pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet')
    assert d.row_position.equals(trace.row_position)
    # Reconstruct keys independently, from the raw-derived fact trace.
    keys = []
    for s in trace.facts_json:
        f = json.loads(s)
        proto = f.get('transport_protocol')
        names = ('action', 'outcome', 'src_role', 'dst_role')
        ok = proto in ('tcp', 'udp', 'icmp') and all(f.get(k) not in (None, '', 'unknown') for k in names)
        if proto == 'icmp':
            ok &= all(k in f and 0 <= f[k] <= 255 for k in ('icmp_type', 'icmp_code'))
            param = (proto, f.get('icmp_type'), f.get('icmp_code'))
        else:
            ok &= 0 <= f.get('dst_port_fixed', 65536) <= 65535
            param = (proto, f.get('dst_port_fixed'))
        if not ok:
            keys.append(None)
        else:
            a = dict(protocol=proto, parameter=param, **{k: f[k] for k in names})
            keys.append(tuple(sorted((k, encode(v)) for k, v in a.items())))
    assert all((None if k is None else encode(k)) == v for k, v in zip(keys, d.joint_key))
    for fold in range(3):
        train_keys = {key for key, k in zip(keys, d.fold) if k != fold and key is not None}
        atom_set = {a for key in train_keys for a in key}
        root_sets = defaultdict(set)
        for key, k, cls, root in zip(keys, d.fold, d.truth, d.root):
            if key is not None and k != fold:
                root_sets[key, cls].add(root)
        for i in d.index[d.fold.eq(fold)]:
            key = keys[i]
            if key is None:
                assert d.at[i, 'support_bin'].startswith('incomplete_')
                continue
            expected = ('exact_joint_seen' if key in train_keys else 'unseen_joint_all_atoms_seen'
                        if set(key) <= atom_set else 'unseen_atomic_value')
            assert d.at[i, 'support_bin'] == expected
            cls = d.at[i, 'truth']
            assert d.at[i, 'exact_same_class_roots'] == len(root_sets[key, cls])
            assert d.at[i, 'exact_opposite_class_roots'] == len(root_sets[key, 3-cls])
    z = d[d.support_bin.eq('exact_joint_seen')].copy()
    z['label_support'] = ['both' if s > 0 and o > 0 else 'same_only' if s > 0 else 'opposite_only'
                          for s, o in zip(z.exact_same_class_roots, z.exact_opposite_class_roots)]
    t = z.groupby(['label_support', 'truth']).agg(rows=('local', 'size'), roots=('root', 'nunique'),
                         A_errors=('A_wrong', 'sum'), B_errors=('B_wrong', 'sum')).reset_index()
    assert not (DEST/'supplement.json').exists()
    t.to_csv(DEST/'known_joint_label_support.csv', index=False)
    facts = [json.loads(s) for s in trace.facts_json]
    code13 = [f.get('transport_protocol') == 'icmp' and f.get('icmp_type') == 3 and f.get('icmp_code') == 13 for f in facts]
    c = d[code13]
    assert len(c) == 865 and c.fold.nunique() == 1 and c.fold.iloc[0] == 2
    assert len({encode(f) for f, flag in zip(facts, code13) if flag}) == 1
    assert all(f.get('icmp_unreachable') == 'administratively_prohibited' for f, flag in zip(facts, code13) if flag)
    out = {'status': 'independent_support_recount_no_fit', 'all_checks_passed': True,
           'classifier_fits': 0, 'calibration_fits': 0,
           'checks': {'bound_evidence_hashes': True, 'keys_rebuilt_from_fact_trace': True,
                      'all_support_bins_independently_recounted': True, 'training_root_counts_independently_recounted': True},
           'exact_joint_label_support': t.to_dict('records'),
           'ICMP_3_13': {'rows': 865, 'M': int(c.truth.eq(1).sum()), 'S': int(c.truth.eq(2).sum()),
                'all_in_fold': 2, 'parsed_message_fact_profiles': 1,
                'existing_semantic_name_present': True,
                'A_errors': int(c.A_wrong.sum()), 'B_errors': int(c.B_wrong.sum()),
                'note': 'Identical current parsed facts do not mean identical full raw inputs or incorrect labels.'},
           'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in (DEST/'diagnosis.json', DEST/'verification.json', ROOT/'training/v115_stability_support_audit.py', ROOT/'training/v115_verify_support_audit.py')},
           'scope': 'All support rows recounted without fitting; diagnosis is not a deployable decision rule.'}
    save(DEST/'supplement.json', out)
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
