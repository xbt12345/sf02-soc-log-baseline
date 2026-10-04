"""Read-only input audit. No label repair, fitting, or candidate scoring."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import sys
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'training'))
import v331_prepare as prior
import v351_safeguards as guard


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).digest()


def summarize(groups):
    mixed = [v for v in groups.values() if sum(x > 0 for x in v) > 1]
    return {'rows': sum(map(sum, groups.values())), 'unique_groups': len(groups),
            'mixed_groups': len(mixed), 'mixed_rows': sum(map(sum, mixed)),
            'minimum_empirical_label_errors': sum(sum(v)-max(v) for v in groups.values())}


def main():
    prepared = ROOT/'artifacts/v36_prepared_r13_20260912/prepared.parquet'
    protocol = pq.read_table(ROOT/'artifacts/v36_cloud_20260912T191112Z/work/prepared_attempt001/protocol.parquet', columns=['asa_hard'])['asa_hard'].to_numpy()
    groups = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    names = ['raw_message_exact', 'asa_body_exact_includes_identities', 'visible_exact_interfaces_no_acl_identity', 'visible_role_and_interface_equality', 'b0', 'b2']
    raw_iter = pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192, columns=['message_sanitized', 'label_binary'], use_threads=False)
    prep_iter = pq.ParquetFile(prepared).iter_batches(batch_size=8192, columns=['row_position','route','label_index','b0','b1','raw_hash'], use_threads=False)
    offset = 0
    identities_verified = 0
    for rb, pb in zip(raw_iter, prep_iter):
        assert len(rb) == len(pb)
        for raw, p in zip(rb.to_pylist(), pb.to_pylist()):
            assert p['row_position'] == offset
            role = int(protocol[offset]); offset += 1
            if p['route'] != 'asa':
                continue
            msg = raw['message_sanitized']
            assert digest(msg) == p['raw_hash']
            assert raw['label_binary'] == ['benign','malicious','suspicious'][p['label_index']]
            identities_verified += 1
            parts = prior.asa_parts(msg)
            f = guard.asa_visible_facts(msg)
            v = {k:f[k] for k in ['action','protocol','src','dst','icmp']}
            relation = json.loads(json.dumps(v))
            relation['same_interface'] = v['src']['interface'] == v['dst']['interface']
            relation['src']['interface'] = prior.role(parts['facts']['src'])
            relation['dst']['interface'] = prior.role(parts['facts']['dst'])
            keys = [msg, parts['body'], json.dumps(v,sort_keys=True), json.dumps(relation,sort_keys=True),p['b0'],p['b1']]
            scopes = ['all_asa'] + (['asa_hard_evaluation'] if role == 3 else []) + (['asa_hard_fit_and_selection'] if role in [0,1] else [])
            for scope in scopes:
                for name, key in zip(names, keys):
                    groups[(scope,name)][digest(key)][p['label_index']] += 1
    assert offset == len(protocol) == 2056871
    out = {'scope': 'Exploratory observability audit of inspected data; raw/body identity keys are diagnostic only, not predictor recommendations.',
           'all_rows_aligned': offset, 'asa_message_hash_and_label_checks': identities_verified,
           'warning': 'Minimum errors describe deterministic rules on fixed observed keys only, not Bayes risk, transferable performance, or a license to use identities.',
           'results': {scope:{name:summarize(groups[(scope,name)]) for name in names} for scope in ['all_asa','asa_hard_fit_and_selection','asa_hard_evaluation']}}
    (Path(__file__).parent/'asa_observability.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
