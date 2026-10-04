"""Prepare a frozen new-source-entity experiment and corpus-only context audit."""
import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold
from v61_common import FIELDS, MISSING, save, sha

ASA = re.compile(r'^(?P<header>.*?)(?P<body>Deny\s+(?P<proto>tcp|udp|icmp6?)\s+src\s+(?P<src>\S+)\s+dst\s+(?P<dst>\S+)(?:\s+\(type\s+\S+,\s*code\s+\S+\))?\s+by\s+[A-Za-z0-9_-]+[-_]group\s+"(?P<acl>[^"]*)"\s+\[(?P<hashes>0x[0-9a-f]+,\s*0x[0-9a-f]+)\])\s*$',re.I|re.S)
ACL = re.compile(r'^(?P<header>.*?)TCP\s+\S+\s+denied by ACL from (?P<src>\S+) to (?P<dst>\S+)\s*$',re.I|re.S)
CTX_NAMES = ['log_other_fact_states','log_other_destinations','log_other_dst_port_symbols',
             'log_other_src_port_symbols','same_destination_fraction','same_dst_port_fraction',
             'tcp_fraction','udp_fraction','icmp_fraction','src_port_missing_fraction',
             'dst_port_missing_fraction','outside_destination_fraction','dmz_destination_fraction',
             'inside_destination_fraction','protocol_diversity','destination_role_diversity']


def endpoint(value):
    left, port = value.rsplit('/',1) if '/' in value else (value,'')
    zone, address = left.split(':',1) if ':' in left else ('',left)
    return zone,address,port


def normalize(raw, facts):
    m = ASA.fullmatch(raw.strip())
    if m:
        body = m['body']; src=endpoint(m['src']); dst=endpoint(m['dst'])
        replacement=[]
        for side, ep in [('src',src),('dst',dst)]:
            role=facts[side+'_role']; value=facts[side+'_port_fixed']
            role='unknown_role' if role==MISSING else role
            port='opaque_port' if value==MISSING else value
            replacement.append((m[side], role+':[address]/'+port))
        for before, after in replacement: body=body.replace(before,after,1)
        body=body.replace('"'+m['acl']+'"','"[policy]"').replace(m['hashes'],'[rule_hashes]')
        # An organization alias can occur in the policy-keyword slot itself.
        body=re.sub(r'\bORG-\d+', '[organization]', body)
    else:
        m=ACL.fullmatch(raw.strip()); assert m, 'Unknown grammar; do not silently discard text.'
        src=endpoint(m['src']);dst=endpoint(m['dst'])
        body='TCP [rule] denied by ACL from [address]/'+('opaque_port' if facts['src_port_fixed']==MISSING else facts['src_port_fixed'])
        body+=' to '+facts['dst_role']+':[address]/'+facts['dst_port_fixed']
    device=re.search(r'(USER-\S+)\s*$',m['header'])
    assert device
    bad=re.findall(r'(?:\d{1,3}\.){3}\d{1,3}|USER-\d|CRED-\d|ORG-\d',body)
    assert not bad, repr(body)
    return body.lower(), {'source':src[1],'destination':dst[1], 'source_port':src[2],
        'destination_port':dst[2],'namespace':device.group(1), 'header':m['header'],
        'raw_hash':hashlib.sha256(raw.encode()).hexdigest()}


def contexts(facts, metadata, roles, cap=32):
    """Only equal-symbol corpus cooccurrence; no time, labels or cross-role records."""
    fact_tuples=list(map(tuple,facts)); n=len(facts)
    buckets=defaultdict(list)
    for i,m in enumerate(metadata): buckets[(roles[i],m['namespace'],m['source'])].append(i)
    stats=np.zeros((n,len(CTX_NAMES)),dtype=np.float32)
    neighbors=np.full((n,cap),-1,dtype=np.int32); relation=np.zeros((n,cap,2),dtype=np.float32)
    coverage=defaultdict(lambda:{'rows':0,'with_other_states':0,'buckets':0})
    for (role,namespace,source),indices in buckets.items():
        coverage[role]['buckets']+=1
        # Equal facts can still concern DIFFERENT targets/port symbols. Preserve that relation.
        # Multiplicity of an identical symbolic event state is never a packet rate.
        by_state=defaultdict(list)
        for i in indices:
            m=metadata[i]
            by_state[(fact_tuples[i],m['destination'],m['source_port'],m['destination_port'])].append(i)
        for key,targets in by_state.items():
            other_keys=[k for k in by_state if k!=key]
            coverage[role]['rows']+=len(targets)
            if not other_keys: continue
            coverage[role]['with_other_states']+=len(targets)
            other=[by_state[k][0] for k in other_keys]
            # Counts are sets of symbols, not counts of packets or repeated rows.
            destinations={metadata[j]['destination'] for j in other}
            dports={metadata[j]['destination_port'] for j in other if metadata[j]['destination_port']}
            sports={metadata[j]['source_port'] for j in other if metadata[j]['source_port']}
            reps=other
            ff=facts[reps]; protocols=ff[:,2]
            base=[np.log1p(len(other_keys)),np.log1p(len(destinations)),np.log1p(len(dports)),np.log1p(len(sports)),0,0,
                  np.mean(protocols=='tcp'),np.mean(protocols=='udp'),np.mean(np.isin(protocols,['icmp','icmp6'])),
                  np.mean(ff[:,5]==MISSING),np.mean(ff[:,6]==MISSING),np.mean(ff[:,4]=='outside'),
                  np.mean(ff[:,4]=='dmz'),np.mean(ff[:,4]=='inside'),len(set(protocols)),len(set(ff[:,4]))]
            # Cache by identity-equality relation, not literal address text or label.
            target_sets=defaultdict(list)
            for i in targets: target_sets[(metadata[i]['destination'],metadata[i]['destination_port'])].append(i)
            for (dest,pt),tids in target_sets.items():
                entries={}
                all_relations=[]
                for k in other_keys:
                    rel=(int(k[1]==dest),int(bool(pt) and k[3]==pt))
                    all_relations.append(rel)
                    # The neural neighbor set collapses equal facts/equality flags;
                    # C and D both receive complete state-count statistics as well.
                    entries[(k[0],rel)]=by_state[k][0]
                ordered=sorted(entries,key=lambda z:hashlib.sha256(repr(z).encode()).hexdigest())
                vals=np.array(base,dtype=np.float32)
                vals[4]=np.mean([r[0] for r in all_relations]);vals[5]=np.mean([r[1] for r in all_relations])
                chosen=ordered[:cap]
                for i in tids:
                    stats[i]=vals
                    for j,entry in enumerate(chosen): neighbors[i,j]=entries[entry];relation[i,j]=entry[1]
    return stats,neighbors,relation,dict(coverage)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--split-reference',type=Path);a=ap.parse_args()
    root=a.root.resolve();out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
    prior=root/'artifacts/v55_risk_validation_20260914'
    rows=pd.read_parquet(prior/'rows.parquet');obs=pd.read_parquet(prior/'observations.parquet').astype(str)
    assert len(rows)==99398 and rows.inner_role.ne(2).all() and obs.columns.tolist()==FIELDS
    positions={int(p):i for i,p in enumerate(rows.row_position)};raw=[None]*len(rows);offset=0
    for b in pq.ParquetFile(root/'data/official/train.parquet').iter_batches(batch_size=65536,columns=['event_id','message_sanitized']):
        for pos in positions.keys() & range(offset,offset+b.num_rows):
            i=positions[pos];assert b.column(0)[pos-offset].as_py()==rows.iloc[i].event_id
            raw[i]=b.column(1)[pos-offset].as_py()
        offset+=b.num_rows
    assert all(v is not None for v in raw)
    normalized=[normalize(t,f) for t,f in zip(raw,obs.to_dict('records'))]
    texts=[v[0] for v in normalized];meta=[v[1] for v in normalized]
    source=np.array([m['source'] for m in meta]);group_ids=pd.factorize(source,sort=True)[0]
    # Source equality is a conservative boundary across collector aliases.
    assert pd.DataFrame({'b':rows.body_group,'s':source}).groupby('b').s.nunique().max()==1
    y=rows.label_index.to_numpy(); fold=np.full(len(y),-1)
    if a.split_reference:
        ref=pd.read_parquet(a.split_reference)
        assert np.array_equal(ref.row_position,rows.row_position) and np.array_equal(ref.label,y)
        assert np.array_equal(ref.group,group_ids)
        roles=ref.role.to_numpy()
    else:
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260915)
        for k,(_,va) in enumerate(sg.split(np.zeros(len(y)),y,group_ids)):fold[va]=k
        roles=np.where(fold==0,'evaluation',np.where(fold==1,'selection','fit'))
    assert pd.DataFrame({'g':group_ids,'role':roles}).groupby('g').role.nunique().max()==1
    coverage={}
    for role in ['fit','selection','evaluation']:
        mask=roles==role; counts=np.bincount(y[mask],minlength=3)
        coverage[role]={'rows':int(mask.sum()),'labels':counts.tolist(),'sources':int(len(set(source[mask]))),
                       'source_groups_per_class':[int(len(set(source[mask&(y==j)]))) for j in range(3)]}
        assert (counts>0).all(), 'Fixed split lacks a class: diagnose before fitting, do not pick by scores.'
    f=obs.to_numpy(dtype=str)
    print('Input and fixed source split prepared; constructing corpus-only context.',flush=True)
    st,ne,rel,ctx_coverage=contexts(f,meta,roles)
    # Baseline one-hot vocab and all normalization are fitted later, on fit only.
    payload=pd.DataFrame({'row_position':rows.row_position,'event_id':rows.event_id,
        'label':y,'role':roles,'group':group_ids,'body_group':rows.body_group,'text':texts})
    payload[FIELDS]=obs
    payload.to_parquet(out/'records.parquet',index=False)
    np.savez_compressed(out/'context.npz',stats=st,neighbors=ne,relation=rel)
    pd.DataFrame(meta).to_parquet(out/'private_join_audit.parquet',index=False)
    # Only opaque corpus grouping is qualified. Physical hosts/timing/rates are NOT established.
    audit={'fixed_split':coverage,'context_coverage':ctx_coverage,'scope':'Offline unlabelled corpus cooccurrence, per role and collector/source; not verified physical sessions, rates or online detection',
        'online_context_qualified':False,'offline_symbolic_context_qualified':True,
        'normal_controls':int((y==0).sum()),'normal_source_groups':int(len(set(source[y==0]))),
        'normal_sources_shared_with_threat_rows':len(set(source[y==0])&set(source[y>0])),
        'context_names':CTX_NAMES,'neighbor_cap':32,'literal_id_in_model':False,'absolute_time_in_model':False,
        'current_exact_symbolic_state_all_copies_excluded_from_neighbors':True,
        'same_facts_different_targets_preserved':True,'repeat_count_used_as_rate':False}
    save(out/'data_audit.json',audit)
    config={'version':'v61-source-corpus-factorial-2','input_rows':len(rows),'scope':'ASA plus 16 normal ACL controls; not whole-SOC integration',
        'protocol':'New source entity; fixed SGKF5 seed 20260915, fold0 evaluation, fold1 selection, rest fit. Existing body groups and literal source addresses disjoint. Old pressure rows absent.',
        'context':'Corpus-only offline secondary estimand; model statistics and vocabulary fit only; each role gets its own unlabelled context. No timestamp or labels in context.',
        'arms':['A','B','C','D'],'fields':FIELDS,'context_names':CTX_NAMES,
        'seed':20260915,'neural_epochs':3,'source_plan_sha256':sha(root/'training/v60_research_plan.json'),
        'source_bindings':{p.relative_to(root).as_posix():sha(p) for p in [prior/'rows.parquet',prior/'observations.parquet',root/'training/prepare_v61.py',root/'training/v61_common.py']},
        'data_bindings':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
        'evaluation_policy':'Do not open outer predictions until all first-seed arm selection results are fixed; no outer score tuning.'}
    if a.split_reference:
        config['split_reference']={'path':str(a.split_reference),'sha256':sha(a.split_reference),
            'reason':'Reuse pre-model frozen membership; SGKF behavior differs across sklearn versions.'}
    save(out/'configuration.json',config)
    print(audit,flush=True)


if __name__=='__main__':main()
