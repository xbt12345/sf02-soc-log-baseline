"""Incremental grammar/predicate qualification, not another field-readout fit.

Reuse source-verified V53 captures; parse only the 5854 added original rows.
All values remain evidence, no class rules or input changes are generated.
"""
import hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import read,sha,check_bindings

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v153_raw_residual_delta_20261001'
OLD=ROOT/'evidence/2026-09-14/v53_raw_audit'
TRACE=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
CURRENT=ROOT/'artifacts/v151_current_context_view_20261001/all_ASA_current_ordered_view_ledger.parquet'
CONTEXT=ROOT/'artifacts/v151_context_coherence_20261001/all_ASA_context_ledger.parquet'
# V53's executed whole-message grammar, including all quoted ACL and hashes.
PAT=re.compile(r'^(?P<header>.*?)(?P<action>Deny)\s+(?P<protocol>tcp|udp|icmp6?|[0-9]+)\s+src\s+(?P<src>\S+)\s+dst\s+(?P<dst>\S+)(?:\s+\(type\s+(?P<icmp_type>\S+),\s*code\s+(?P<icmp_code>\S+)\))?\s+by\s+(?P<aclword>[A-Za-z0-9_-]+[-_]group)\s+"(?P<acl>[^"]*)"\s+\[(?P<hashes>0x[0-9a-f]+,\s*0x[0-9a-f]+)\]\s*$',re.I|re.S)
BODY_FIELDS=['header','action','protocol','src','dst','icmp_type','icmp_code','aclword','acl','hashes']
COMMAND=re.compile(r'(?:^|\n)\s*(?:[\w.-]+\(config[^)]*\)#\s*)?access-group\s+\S+\s+(in|out)\s+interface\s+\S+',re.I)

def digest(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists(),'Never overwrite executed grammar evidence'
    oldaudit=read(OLD/'summary.json');check_bindings(oldaudit['source_bindings'])
    fidelity=read(ROOT/'artifacts/v149_independent_input_fidelity_20261001/audit.json');check_bindings(fidelity['source_sha256'])
    trace=pd.read_parquet(TRACE).sort_values('row_position').reset_index(drop=True)
    old=pd.read_parquet(OLD/'raw_fields.parquet').sort_values('row_position').reset_index(drop=True)
    current=pd.read_parquet(CURRENT).set_index('row_position')
    contexts=pd.read_parquet(CONTEXT).set_index('row_position')
    assert len(trace)==112807 and len(old)==106953 and old.row_position.is_unique
    indexed=trace.set_index('row_position');prior=indexed.loc[old.row_position]
    assert np.array_equal(prior.raw_message.map(digest),old.raw_sha256)
    assert np.array_equal(prior.truth,old.label_index)
    assert set(old.row_position)<=set(trace.row_position)
    delta=trace[~trace.row_position.isin(old.row_position)]
    assert len(delta)==5854
    records=[];spans=[];unmatched=[]
    for row in delta.itertuples(index=False):
        raw=row.raw_message;m=PAT.fullmatch(raw)
        if m is None:
            unmatched.append(dict(row_position=int(row.row_position),raw_sha256=digest(raw)))
            continue
        rec=m.groupdict();rec.update(row_position=int(row.row_position),root=int(row.root),fold=int(row.fold),truth=int(row.truth),
            raw_sha256=digest(raw),facts_json=row.facts_json)
        f=json.loads(row.facts_json)
        assert f['action']=='deny' and f['outcome']=='blocked' and f['transport_protocol']==m['protocol'].casefold()
        for side in ['src','dst']:
            zone,endpoint=m[side].split(':',1)
            token=endpoint.rsplit('/',1)[1] if '/' in endpoint else ''
            observed=re.fullmatch(r'[0-9]+',token) is not None and 0<=int(token)<=65535
            expected=int(token) if observed else 65536
            if m['protocol'].casefold() in ('tcp','udp'):assert f[side+'_port_fixed']==expected
        # Exact complete span ledger; syntax gaps are retained, not silently
        # removed or assigned a semantic label. Only newly added rows parsed.
        intervals=[(m.start(k),m.end(k),k) for k in BODY_FIELDS if m.start(k)>=0]
        intervals.sort();last=0;pieces=[]
        for start,end,kind in intervals:
            assert start>=last
            if start>last:pieces.append((last,start,'syntax'))
            pieces.append((start,end,kind));last=end
        if last<len(raw):pieces.append((last,len(raw),'syntax'))
        assert ''.join(raw[a:b] for a,b,_ in pieces)==raw
        assert sum(b-a for a,b,_ in pieces)==len(raw)
        for a,b,kind in pieces:spans.append(dict(row_position=int(row.row_position),start=a,end=b,kind=kind,literal=raw[a:b]))
        records.append(rec)
    assert not unmatched,'Unmatched new rows need an explicit preserved grammar version, not dropped rows'
    new=pd.DataFrame(records)
    captures=pd.concat([old[['row_position']+BODY_FIELDS],new[['row_position']+BODY_FIELDS]],ignore_index=True).sort_values('row_position')
    assert np.array_equal(captures.row_position,trace.row_position)
    captures['root']=trace.root.to_numpy();captures['fold']=trace.fold.to_numpy();captures['truth']=trace.truth.to_numpy()
    captures['audit_population']=np.where(captures.row_position.isin(old.row_position),'verified_V53_capture','new_original_delta')
    captures['raw_sha256']=trace.raw_message.map(digest).to_numpy()
    captures['explicit_access_group_binding_present']=trace.raw_message.map(lambda s:bool(COMMAND.search(s))).to_numpy()
    # Separate the pre-Deny identifier from the actual V124-removed header.
    prefix=[]
    for pos,header in zip(captures.row_position,captures.header):
        end=int(current.loc[pos,'current_header_end'])
        assert header==indexed.loc[pos,'raw_message'][:len(header)] and end<=len(header)
        prefix.append(header[end:].strip())
    captures['post_clock_pre_Deny_literal']=prefix
    captures['known_578_cohort']=contexts.loc[captures.row_position,'known_578_cohort'].to_numpy()
    assert int(captures.known_578_cohort.sum())==578
    all_constant=dict(hash_pairs=captures.hashes.nunique()==1,acl_clause_word=captures.aclword.nunique()==1,
        action=captures.action.str.casefold().nunique()==1,event_identifier_literal=captures.post_clock_pre_Deny_literal.nunique()==1)
    inventory=[]
    for col in ['protocol','aclword','acl','hashes','post_clock_pre_Deny_literal']:
        for value,g in captures.groupby(col,dropna=False):
            inventory.append(dict(field=col,value=str(value),original_rows=len(g),roots=int(g.root.nunique()),
                new_delta_rows=int(g.audit_population.eq('new_original_delta').sum())))
    OUT.mkdir();new.to_parquet(OUT/'new_5854_raw_captures.parquet',index=False)
    pd.DataFrame(spans).to_parquet(OUT/'new_5854_complete_raw_spans.parquet',index=False)
    captures.to_parquet(OUT/'all_original_capture_provenance.parquet',index=False)
    pd.DataFrame(inventory).to_parquet(OUT/'raw_component_inventory.parquet',index=False)
    delta_classes=[dict(truth=int(c),fold=int(f),original_rows=int(n)) for (c,f),n in delta.groupby(['truth','fold']).size().items()]
    result=dict(status='incremental_raw_predicate_inventory_executed_no_new_classifier',latest_actual_classifier='V146',
        original_ASA_rows=112807,V53_original_captures_reused_and_raw_gold_verified=106953,new_original_rows_parsed=5854,
        new_rows_roots=int(delta.root.nunique()),new_rows_classes_by_fold=delta_classes,
        new_rows_unmatched=0,new_span_coverage_exact=True,all_unknown_ICMP_original_rows_retained=True,
        new_rows_in_known_578=int(new.row_position.isin(captures[captures.known_578_cohort].row_position).sum()),
        complete_grammar_inherited_scope='V53 raw-hash verified captures plus new delta under the same whole-message grammar; not new all-population field-readout or model replay.',
        component_constant_states=all_constant,hash_pair_values=captures.hashes.unique().tolist(),
        acl_label_values=captures.acl.unique().tolist(),post_clock_pre_Deny_literal_values=captures.post_clock_pre_Deny_literal.unique().tolist(),
        explicit_access_group_binding_rows=int(captures.explicit_access_group_binding_present.sum()),
        new_non_grammar_reason_flags_hitcount_rows=0,
        novel_trusted_threat_predicate_qualified=False,next_classifier_training_registered=False,
        new_model_forwards=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,issue_solved=False,
        candidate_rejections=[dict(candidate='ACL label suffix in/out as actual policy direction',
            evidence='Quoted ACL labels present, no explicit access-group ACL in/out interface binding command or policy context in these log rows.',
            decision='Do not infer actual configured direction from a label suffix; no new direction teacher registered.'),
            dict(candidate='Extra Deny reason, TCP flags or hitcount outside the 13 fields',
            evidence='All new 5854 rows completely match inherited Deny/protocol/endpoint/optional ICMP/ACL/hash grammar; no additional clause remains.',
            decision='No new feature extraction or field-reconstruction trial justified by this claimed missing clause.'),
            dict(candidate='Constant hash pairs or event marker as threat condition',
            evidence='Original component inventory retains literal values and provenance; constants or anonymous labels do not supply independent M/S conditions.',
            decision='No hash/anonymous prefix classifier or pseudo-label teacher generated.')],
        limits=['No claim that all possible semantics or external context is exhausted; preserve full raw strings and ambiguous names.',
            'Matched grammar establishes the observed clause inventory, not original vendor message number or threat rationale.',
            'V53 and current fold schemes differ; only original row/raw/gold identity and grammar captures reused, never old role support counts.',
            'New delta has only two isolation roots in one fold, not new independent supervision for the original 578.',
            'ACL label is inference-available text, but its operational meaning is unverified; no literal-name routing or classification fit.'],
        primary_reference=dict(title='Cisco access-group command: ACL label and in/out binding are separate arguments',
            url='https://www.cisco.com/c/en/us/td/docs/security/asa/asa-cli-reference/A-H/asa-command-ref-A-H/aa-ac-commands.html'),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),OLD/'summary.json',OLD/'raw_fields.parquet',TRACE,CURRENT,CONTEXT,
            ROOT/'artifacts/v149_independent_input_fidelity_20261001/audit.json',ROOT/'docs/V53_ASA_EXECUTION.md',
            ROOT/'docs/V152_NEXT_TRAINING_DECISION_AND_CONDITIONAL_PLAN.md']})
    save(OUT/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','limits','candidate_rejections']},ensure_ascii=False))

if __name__=='__main__':main()
