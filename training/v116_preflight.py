"""Register one deterministic K/U split; no fitting, no search over splits."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'artifacts/v116_nested_selection_20260929'
TRACE = ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
VIEW = ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
PLAN = ROOT/'docs/V115_STABLE_MS_CRITERIA_REVIEW_AND_PLAN.md'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def parameter(s):
    f = json.loads(s)
    p = f.get('transport_protocol')
    if p == 'icmp' and all(k in f and 0 <= f[k] <= 255 for k in ('icmp_type','icmp_code')):
        return json.dumps([p, f['icmp_type'], f['icmp_code']], separators=(',', ':'))
    if p in ('tcp','udp') and 0 <= f.get('dst_port_fixed',65536) <= 65535:
        return json.dumps([p, f['dst_port_fixed']], separators=(',', ':'))
    return None


def bucket(kind, fold, value, modulo):
    key = f'v116-fixed-20260929|{kind}|{fold}|{value}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % modulo


def main():
    assert not DEST.exists(), 'One registered split only; no overwrite or split search.'
    previous = ROOT/'artifacts/v115_stability_support_20260929/review_receipt.json'
    receipt = json.loads(previous.read_text(encoding='utf-8'))
    for rel, digest in receipt['artifact_sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    # Policy is persisted BEFORE constructing/counting any proposed split.
    DEST.mkdir()
    policy = {'status':'split_policy_registered_before_split_counts', 'created_unix':time.time(),
        'classifier_fits':0, 'calibration_fits':0, 'outer_folds':[0,1,2],
        'parameter_definition':'protocol+known destination port; ICMP type/code together; unknown excluded',
        'U_selection':'parameter SHA256 bucket modulo 10 == 0; remove every connected root in its entirety',
        'K_selection':'remaining root SHA256 bucket modulo 5 == 0; K evaluates only parameters present in fit',
        'collateral_rows':'retain in manifest, exclude from fitting and K/U selection; diagnostic only',
        'minimum_distinct_roots_per_class_per_role':2,
        'minimum_note':'Minimal replication in fit/K/U, not a power calculation or guarantee of precise risk.',
        'no_search_or_retry_with_other_seed':True,
        'gate':'All three outer folds must have >=2 independent roots for each M/S in fit, K, U; otherwise stop before any fit.',
        'epochs':[1,2,5,10,15,20,25], 'model_seed':10201, 'max_new_fits':6,
        'source_sha256':sha(__file__), 'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (TRACE,VIEW,PLAN,previous)}}
    save(DEST/'split_policy.json',policy)
    # No prediction/probability columns are read for split construction.
    d = pd.read_parquet(TRACE, columns=['row_position','local','root','fold','truth','facts_json'])
    assert len(d)==112807 and d.groupby('root').fold.nunique().max()==1
    assert d.groupby('local').root.nunique().max()==1
    d['parameter'] = d.facts_json.map(parameter)
    d = d.drop(columns='facts_json')
    pieces=[]; stats=[]; checks=[]
    for fold in range(3):
        pool = d[d.fold != fold].copy()
        # Only outer-train facts participate in choosing parameter/root holdouts.
        params = set(pool.parameter.dropna())
        reserved = {p for p in params if bucket('U_parameter',fold,p,10)==0}
        uroots = set(pool.loc[pool.parameter.isin(reserved),'root'])
        kroots = {int(r) for r in set(pool.root)-uroots if bucket('K_root',fold,int(r),5)==0}
        fitmask = ~pool.root.isin(uroots|kroots)
        seen = set(pool.loc[fitmask,'parameter'].dropna())
        q = d.copy()
        q['outer_fold']=fold
        q['role']='outer_test'
        q.loc[q.fold.ne(fold),'role']='validation_collateral'
        q.loc[q.fold.ne(fold)&~q.root.isin(uroots|kroots),'role']='fit'
        q.loc[q.fold.ne(fold)&q.root.isin(uroots)&q.parameter.isin(reserved),'role']='U'
        q.loc[q.fold.ne(fold)&q.root.isin(kroots)&q.parameter.isin(seen),'role']='K'
        assert reserved.isdisjoint(seen)
        for role in ('fit','K','U'):
            for cls in (1,2):
                z=q[q.role.eq(role)&q.truth.eq(cls)]
                stat={'outer_fold':fold,'role':role,'truth':cls,'rows':len(z),
                      'roots':int(z.root.nunique()),'unique_inputs':int(z.local.nunique()),
                      'known_parameters':int(z.parameter.nunique())}
                stats.append(stat)
                checks.append({'fold':fold,'role':role,'truth':cls,'passed':stat['roots']>=2})
        # Including collateral rows, all groups and equal inputs remain isolated.
        group_role=q.role.replace({'K':'validation','U':'validation','validation_collateral':'validation'})
        assert pd.DataFrame({'root':q.root,'role':group_role}).groupby('root').role.nunique().max()==1
        assert pd.DataFrame({'local':q.local,'role':group_role}).groupby('local').role.nunique().max()==1
        assert set(q.loc[q.role.eq('K'),'parameter']) <= seen
        assert not set(q.loc[q.role.eq('U'),'parameter']) & seen
        pieces.append(q)
    manifest=pd.concat(pieces,ignore_index=True)
    manifest.to_parquet(DEST/'inner_split_manifest.parquet',index=False)
    pd.DataFrame(stats).to_csv(DEST/'inner_population.csv',index=False)
    result={'status':'preflight_passed' if all(c['passed'] for c in checks) else 'stopped_insufficient_K_U_source_coverage',
        'all_checks_passed':all(c['passed'] for c in checks),'classifier_fits':0,'calibration_fits':0,
        'split_attempts':1,'checks':checks,'populations':stats,
        'issues':[] if all(c['passed'] for c in checks) else ['Registered K/U validation lacks replicated class support; V115 explicitly requires stopping rather than reselecting a convenient split.'],
        'policy_sha256':sha(DEST/'split_policy.json'),'manifest_sha256':sha(DEST/'inner_split_manifest.parquet'),
        'role_population':manifest.groupby(['outer_fold','role']).size().rename('rows').reset_index().to_dict('records'),
        'verified':{'outer_test_excluded_from_split_dictionaries':True,'root_and_equal_input_isolation':True,
                    'K_parameter_seen_in_fit':True,'U_parameter_absent_from_fit':True,'all_original_rows_accounted':len(manifest)==3*112807}}
    save(DEST/'preflight.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
