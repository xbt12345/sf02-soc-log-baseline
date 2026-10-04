"""Official context, lawful class conditions and ordered residual scope, zero fits.

No model imports or classification rules. Projection floors diagnose observed
collisions only, not threat truth, inference policy or achievable quality.
"""
import hashlib,json,re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from experiment_review import sha,read,check_bindings
from v75_views import view

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v151_context_coherence_20261001'
TRACE=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
OFFICIAL=ROOT/'data/official/train.parquet'
BODY=re.compile(r'\bDeny\s+(tcp|udp|icmp)\s+src\s+([^:\s]+):([^\s]+)\s+dst\s+([^:\s]+):([^\s]+)',re.I)
IP4=re.compile(r'^([0-9]{1,3}\.){3}[0-9]{1,3}$')
PRI=re.compile(r'^<([0-9]{1,3})>')
FIELDS=['event_id','timestamp','pipeline','src_ip','dst_ip','src_port','src_host','dst_host','username','message_sanitized','product_name','vendor_name','label_binary']

def digest(v):return hashlib.sha256(v.encode('utf-8')).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def state(v):return 'null' if v is None else 'empty' if isinstance(v,str) and not v.strip() else 'observed'
def port(v):
    if not isinstance(v,str) or not re.fullmatch(r'[0-9]+',v.strip()):return None
    n=int(v.strip());return n if 0<=n<=65535 else None
def ip(v):
    if not isinstance(v,str) or not IP4.fullmatch(v) or any(int(x)>255 for x in v.split('.')):return None
    return v
def coherence(a,b):return 'not_comparable' if a is None or b is None else 'equal' if a==b else 'different'

def projection_stats(rows,key):
    c=rows.groupby([key,'truth'],sort=False).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
    mixed=c[(c>0).sum(1)>1];masses=c.sum(1)
    return dict(original_rows=len(rows),class_mass=[0,int(c[1].sum()),int(c[2].sum())],
        unique_keys=len(c),mixed_keys=len(mixed),mixed_original_rows=int(mixed.sum().sum()),
        deterministic_observed_population_error_floor=int((masses-c.max(1)).sum()),
        floor_scope='This fixed projection and observed population only; not full-input or official-data unavoidable errors.')

def legal_support(rows,query,key):
    group=rows.groupby([key,'truth'],sort=False).agg(original_rows=('row_position','size'),roots=('root','nunique'))
    result=[]
    for target in [1,2]:
        for cl in [1,2]:
            table=group.xs(cl,level='truth').reindex(query[key]).fillna(0)
            result.append((target,cl,table.original_rows.to_numpy(np.int64),table.roots.to_numpy(np.int64)))
    return result

def main():
    assert not OUT.exists(),'Never overwrite observed audit'
    worker=read(ROOT/'artifacts/v150_field_readout_diagnostic_20261001/result_audit.json')
    check_bindings(worker['source_sha256'])
    independent=read(ROOT/'artifacts/v150_independent_result_review_20261001/audit.json')
    trace=pd.read_parquet(TRACE);assert len(trace)==112807 and not trace.row_position.duplicated().any()
    indices=set(trace.row_position);parts=[];offset=0
    file=pq.ParquetFile(OFFICIAL)
    for batch in file.iter_batches(batch_size=32768,columns=FIELDS):
        frame=batch.to_pandas();pos=np.array([i for i in range(offset,offset+len(frame)) if i in indices],dtype=np.int64)
        if len(pos):
            take=frame.iloc[pos-offset].copy();take['row_position']=pos;parts.append(take)
        offset+=len(frame)
    assert offset==2056871
    original=pd.concat(parts,ignore_index=True).sort_values('row_position').reset_index(drop=True)
    trace=trace.sort_values('row_position').reset_index(drop=True)
    assert np.array_equal(original.row_position,trace.row_position) and np.array_equal(original.message_sanitized,trace.raw_message)
    assert np.array_equal(original.label_binary.map({'benign':0,'malicious':1,'suspicious':2}),trace.truth)
    assert original.event_id.nunique()==112807
    facts=trace.facts_json.map(json.loads).tolist();parsed=[BODY.search(s) for s in trace.raw_message]
    assert all(parsed),'Unmatched grammar must be retained and diagnosed before semantic claims'
    ordered=[];fac=[];sev=[];source_ip=[];dest_ip=[];body_ports=[];trailing=[]
    for raw,m in zip(trace.raw_message,parsed):
        text,ledger=view(raw)
        assert ''.join(raw[a:b] for a,b,_ in ledger['spans'])==raw
        ordered.append(digest(text))
        pri=PRI.match(raw);p=int(pri[1]) if pri else None
        fac.append(None if p is None else p//8);sev.append(None if p is None else p%8)
        source_ip.append(ip(m[3].rsplit('/',1)[0]));dest_ip.append(ip(m[5].rsplit('/',1)[0]))
        body_ports.append(port(m[3].rsplit('/',1)[1]) if m[1].lower() in ('tcp','udp') and '/' in m[3] else None)
        tail=raw[m.end():].strip()
        trailing.append(dict(acl_clause=bool(re.search(r'\bby\s+\S+-group\s+',tail,re.I)),
            quoted_acl_name=bool(re.search(r'"[^"]*"',tail)),hash_pair=bool(re.search(r'\[0x[0-9a-f]+,\s*0x[0-9a-f]+\]',tail,re.I))))
    ledger=trace[['row_position','local','root','fold','truth','facts_json']].copy()
    ledger['facts_key']=trace.facts_json.map(lambda s:digest(json.dumps(json.loads(s),sort_keys=True,separators=(',',':'))))
    ledger['ordered_identity_clock_removed_key']=ordered
    ledger['facts_plus_ordered_residual_key']=[digest(a+b) for a,b in zip(ledger.facts_key,ordered)]
    ledger['syslog_facility']=fac;ledger['syslog_severity']=sev
    ledger['packet_plus_logging_level_key']=[digest(a+str(b)+':'+str(c)) for a,b,c in zip(ledger.facts_key,fac,sev)]
    ledger['src_ip_record_body_relation']=[coherence(ip(a),b) for a,b in zip(original.src_ip,source_ip)]
    ledger['dst_ip_record_body_relation']=[coherence(ip(a),b) for a,b in zip(original.dst_ip,dest_ip)]
    ledger['src_port_record_body_relation']=[coherence(port(a),b) for a,b in zip(original.src_port,body_ports)]
    ledger['body_src_dst_ip_relation']=[coherence(a,b) for a,b in zip(source_ip,dest_ip)]
    for field in ['pipeline','src_ip','dst_ip','src_port','src_host','dst_host','username','product_name','vendor_name']:
        ledger[field+'_state']=original[field].map(state)
    for key in ['acl_clause','quoted_acl_name','hash_pair']:ledger[key]=[v[key] for v in trailing]
    # Context values retain their original provenance in official rows. Never
    # include identity/clock/product values in an accepted threat target here.
    context_names=['src_ip_record_body_relation','dst_ip_record_body_relation','src_port_record_body_relation','body_src_dst_ip_relation']
    context_names += [x+'_state' for x in ['pipeline','src_ip','dst_ip','src_port','src_host','dst_host','username','product_name','vendor_name']]
    ledger['unqualified_context_state_key']=ledger[context_names].apply(lambda r:digest(json.dumps(r.to_dict(),sort_keys=True)),axis=1)
    ledger['facts_plus_unqualified_context_key']=[digest(a+b) for a,b in zip(ledger.facts_key,ledger.unqualified_context_state_key)]
    keys=['facts_key','packet_plus_logging_level_key','ordered_identity_clock_removed_key','facts_plus_ordered_residual_key','facts_plus_unqualified_context_key']
    reports=[dict(role='whole_observed_population',projection=key,**projection_stats(ledger,key)) for key in keys]
    assert reports[0]['deterministic_observed_population_error_floor']==2510 and reports[0]['mixed_keys']==215
    support=[]
    for fold in range(3):
        legal=ledger[ledger.fold.ne(fold)];held=ledger[ledger.fold.eq(fold)]
        assert not set(legal.root)&set(held.root)
        for key in keys:
            reports.append(dict(role='legal_TRAIN',fold=fold,projection=key,**projection_stats(legal,key)))
        q=held[['row_position','truth','root','facts_key','facts_plus_ordered_residual_key']].copy()
        for key in ['facts_key','facts_plus_ordered_residual_key']:
            counts=legal.groupby([key,'truth']).agg(rows=('row_position','size'),roots=('root','nunique'))
            for cl in [1,2]:
                tab=counts.xs(cl,level='truth').reindex(q[key]).fillna(0)
                q[key+'_class'+str(cl)+'_rows']=tab.rows.to_numpy(np.int64)
                q[key+'_class'+str(cl)+'_roots']=tab.roots.to_numpy(np.int64)
        support.append(q)
    query=pd.concat(support,ignore_index=True);assert len(query)==112807 and not query.row_position.duplicated().any()
    known=pd.read_parquet(ROOT/'artifacts/v147_independent_conditional_support_20261001/all_known_port_conditional_support.parquet',columns=['row_position','known_578_cohort']).set_index('row_position')
    ledger=ledger.join(known,on='row_position');ledger['known_578_cohort']=ledger.known_578_cohort.eq(True)
    h=ledger.known_578_cohort;assert int(h.sum())==578
    # Timestamp/duplicate evidence never treats repeated records as distinct
    # packet observations or discards their required classification frequency.
    attrs=[f for f in FIELDS if f not in ['event_id','label_binary']]
    attr_key=pd.util.hash_pandas_object(original[attrs],index=False).to_numpy()
    ledger['available_event_attribute_hash']=attr_key
    duplicates=original.assign(attribute_key=attr_key).groupby('attribute_key').agg(rows=('event_id','size'),event_ids=('event_id','nunique'),timestamps=('timestamp','nunique'))
    repeat=duplicates[duplicates.rows.gt(1)]
    clock=original.timestamp.to_numpy();finite=np.isfinite(clock)
    metadata=[]
    for field in context_names+['syslog_facility','syslog_severity','acl_clause','quoted_acl_name','hash_pair']:
        for (value,cl),n in ledger.groupby([field,'truth'],dropna=False).size().items():
            metadata.append(dict(field=field,value=str(value),truth=int(cl),original_rows=int(n)))
    signature_members=ledger.groupby('facts_key').agg(original_rows=('row_position','size'),classes=('truth','nunique'),
        roots=('root','nunique'),ordered_residuals=('ordered_identity_clock_removed_key','nunique'),context_states=('unqualified_context_state_key','nunique'))
    mixed=signature_members[signature_members.classes.gt(1)]
    OUT.mkdir();ledger.to_parquet(OUT/'all_ASA_context_ledger.parquet',index=False)
    query.to_parquet(OUT/'all_outer_legal_condition_support.parquet',index=False)
    mixed.to_parquet(OUT/'mixed_facts_context_capacity.parquet')
    pd.DataFrame(metadata).to_parquet(OUT/'all_context_state_class_masses.parquet',index=False)
    result=dict(status='raw_context_coherence_and_legal_class_condition_audit_only',latest_actual_classifier='V146',
        official_rows=offset,ASA_original_rows=len(ledger),new_model_forwards=0,new_gradients=0,new_fits=0,new_updates=0,
        quality_acceptance=False,issue_solved=False,next_classifier_training_registered=False,
        available_official_columns=file.schema.names,
        missing_separate_fields=['protocol','action','severity','vendor_event_code','attack_technique','attack_chain_id','label_rationale','acl_policy','packet_identifier'],
        projections=reports,context_state_class_masses=metadata,
        same_complete_facts_mixed_keys=len(mixed),same_complete_facts_mixed_original_rows=int(mixed.original_rows.sum()),
        mixed_facts_multiple_ordered_residuals=int(mixed.ordered_residuals.gt(1).sum()),
        mixed_facts_multiple_unqualified_context_states=int(mixed.context_states.gt(1).sum()),
        time_and_record_exposure=dict(finite_timestamps=int(finite.sum()),missing_timestamps=int((~finite).sum()),
            unique_timestamp_values=int(original.timestamp.nunique()),distinct_available_attribute_groups=len(duplicates),
            duplicate_attribute_groups=len(repeat),original_rows_in_duplicate_attribute_groups=int(repeat.rows.sum()),
            duplicate_attribute_excess_rows=int((repeat.rows-1).sum()),
            interpretation='Same available attributes with distinct event IDs cannot alone establish packet identity, original rate, generated duplication or independent context; retain original classification frequency.'),
        grammar_matched_original_rows=len(parsed),known_578_context_state_mass=ledger[h][context_names].value_counts(dropna=False).rename('original_rows').reset_index().to_dict('records'),
        limits=['No new threat label, classifier input, conditional label-copy rule or trained classifier is produced.',
            'Observed projection collisions/floors do not prove official data impossible or task-optimal performance.',
            'Ordered identity/clock-removed residual still contains unresolved interface/ACL naming and formatting; lower collisions do not grant threat semantics.',
            'Missing/null/empty/product and cross-field equality states diagnose provenance. Their correlation is not authorization to use them as threat truth.',
            'No actual ACL policy, packet identity, incident linkage or rationale is provided; no temporal/identity rate feature is asserted valid.',
            'All known, unknown, ICMP, mixed and previously correct original rows remain; source role casefold issue remains separately versioned, historical inputs unchanged.',
            'Full and HELD results are descriptive development diagnostics only, not field/key/weight/checkpoint selection.'],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'training/v75_views.py',ROOT/'training/experiment_review.py',TRACE,OFFICIAL,
            ROOT/'artifacts/v150_field_readout_diagnostic_20261001/result_audit.json',ROOT/'artifacts/v150_independent_result_review_20261001/audit.json',
            ROOT/'artifacts/v147_independent_conditional_support_20261001/all_known_port_conditional_support.parquet']})
    save(OUT/'audit.json',result)
    print(json.dumps(dict(status=result['status'],ASA_rows=len(ledger),projections=reports[:5],time_and_record_exposure=result['time_and_record_exposure'],
        context_masses=[r for r in metadata if r['field'] in ['src_ip_record_body_relation','dst_ip_record_body_relation','src_port_record_body_relation','syslog_severity']],
        new_fits=0),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
