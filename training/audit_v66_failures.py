"""Trace every v65 fit-side row back to official bytes and characterize evidence gaps."""
import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from train_v65_rank_heads import load_fit,Inputs,ROOT,DATA
from prepare_v61 import normalize,ASA,ACL
from v61_common import FIELDS,MISSING,read,save,sha
from v66_selection import select_both


def key(values):return hashlib.sha256(repr(values).encode()).hexdigest()


def conflict(frame,keys):
    d=pd.DataFrame({'key':keys,'label':frame.label.to_numpy()})
    table=d.groupby(['key','label']).size().unstack(fill_value=0)
    mixed=table.gt(0).sum(1).gt(1)
    return {'mixed_groups':int(mixed.sum()),'rows_in_mixed_groups':int(table[mixed].to_numpy().sum()),
            'finite_sample_minimum_row_errors':int((table.sum(1)-table.max(1)).sum())},d.key.map(mixed).to_numpy()


def raw_read(frame):
    wanted={int(pos):i for i,pos in enumerate(frame.row_position)};messages=[None]*len(frame)
    counters=defaultdict(int);product_counts=defaultdict(int);offset=0
    columns=['event_id','message_sanitized','label_binary','product_name']
    for b in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=65536,columns=columns):
        d=b.to_pandas();labels=d.label_binary.fillna('').astype(str)
        for label,n in labels.value_counts().items():counters['all/'+label]+=int(n)
        primary=d.message_sanitized.str.contains(r'\bDeny\s+(?:tcp|udp|icmp6?)\s+src\s+',regex=True,case=False,na=False)
        control=d.message_sanitized.str.contains(r'TCP\s+\S+\s+denied by ACL from ',regex=True,case=False,na=False)
        for route,mask in [('asa_deny_grammar',primary),('acl_control_grammar',control)]:
            for label,n in labels[mask].value_counts().items():counters[route+'/'+label]+=int(n)
        for pos in wanted.keys() & range(offset,offset+len(d)):
            i=wanted[pos];row=d.iloc[pos-offset]
            assert row.event_id==frame.event_id.iloc[i]
            assert ['benign','malicious','suspicious'][int(frame.label.iloc[i])]==row.label_binary
            messages[i]=row.message_sanitized
        offset+=len(d)
    assert all(isinstance(t,str) for t in messages)
    return messages,dict(counters)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    frame,context=load_fit();v65=ROOT/'artifacts/v65_rank_heads_20260920'
    config=read(DATA/'configuration.json');assert sha(DATA/'private_join_audit.parquet')==config['data_bindings']['private_join_audit.parquet']
    positions=pd.read_parquet(DATA/'records.parquet',columns=['row_position']).row_position
    meta=pd.read_parquet(DATA/'private_join_audit.parquet');meta.index=positions
    meta=meta.loc[frame.row_position].reset_index(drop=True)
    messages,counts=raw_read(frame);safe=[];nuisance=[];residual=[]
    for i,(raw,f) in enumerate(zip(messages,frame[FIELDS].to_dict('records'))):
        text,m=normalize(raw,f);assert text==frame.text.iloc[i]
        for name in meta.columns:assert m[name]==meta[name].iloc[i]
        parsed=ASA.fullmatch(raw.strip());pri=re.match(r'^<(\d+)>',raw)
        severity=int(pri[1])%8 if pri else -1
        sp=m['source_port'];dp=m['destination_port']
        opaque=lambda t:bool(t) and not bool(re.fullmatch(r'\d+',t))
        s={'same_address':m['source']==m['destination'],'same_nonempty_port':bool(sp) and sp==dp,
           'src_port_opaque':opaque(sp),'dst_port_opaque':opaque(dp),'syslog_severity':severity}
        safe.append(s)
        n={'facility':int(pri[1])//8 if pri else -1,'src_zone':'control','dst_zone':'control','policy':'control'}
        if parsed:
            n.update(src_zone=parsed['src'].split(':')[0],dst_zone=parsed['dst'].split(':')[0],policy=parsed['acl'])
            assert parsed['hashes'].replace(' ','')=='0x0,0x0'
        nuisance.append(n)
        residual.append({'row_position':int(frame.row_position.iloc[i]),'raw_sha256':m['raw_hash'],
                         'raw_body_sha256':key(raw[len(m['header']):]),
                         'normalized_sha256':key(text),'src_redacted':opaque(sp),'dst_redacted':opaque(dp)})
    safe=pd.DataFrame(safe);nuisance=pd.DataFrame(nuisance);residual=pd.DataFrame(residual)
    baseline=pd.read_parquet(v65/'H0_guarded_oof.parquet');cand=pd.read_parquet(v65/'H1_selected_oof.parquet')
    for d in [baseline,cand]:assert np.array_equal(d.row_position,frame.row_position)
    facts=frame[FIELDS].astype(str).to_numpy();single=[key(tuple(row)) for row in frame[['text']+FIELDS].to_numpy()]
    single_safe=[key((k,tuple(s))) for k,s in zip(single,safe.to_numpy())]
    unique_text=pd.Index(pd.read_parquet(v65/'unique_text.parquet').text);codes=unique_text.get_indexer(frame.text)
    assert (codes>=0).all();features=np.load(v65/'upstream_text_features.npy')
    audit=frame[['row_position','label','group','fold','behavior']].copy()
    audit['H0_pred']=baseline[['p_B','p_M','p_S']].to_numpy().argmax(1)
    audit['H1_pred']=cand[['p_B','p_M','p_S']].to_numpy().argmax(1)
    audit['empty_context']=(context['neighbors']>=0).sum(1)==0
    audit['neighbor_slots_used']=(context['neighbors']>=0).sum(1)
    audit[['src_redacted','dst_redacted']]=residual[['src_redacted','dst_redacted']]
    audit['single_key']=single;folds={};selection={}
    buckets=defaultdict(list)
    for i,m in meta.iterrows():buckets[(m.namespace,m.source)].append(i)
    # Full relation entries before the cap, excluding all copies of the current symbolic state.
    all_entries=[None]*len(frame);full_counts=np.zeros(len(frame),int)
    for members in buckets.values():
        states=defaultdict(list)
        for i in members:
            m=meta.iloc[i];states[(tuple(facts[i]),m.destination,m.source_port,m.destination_port)].append(i)
        for own,targets in states.items():
            entries={(other[0],int(other[1]==own[1]),int(bool(own[3]) and other[3]==own[3])) for other in states if other!=own}
            entries=tuple(sorted(entries))
            for i in targets:all_entries[i]=entries;full_counts[i]=len(entries)
    audit['uncapped_neighbor_entries']=full_counts;audit['cap_removed_entries']=full_counts>32
    uncapped_keys=[key((single[i],context['stats'][i].tobytes(),all_entries[i])) for i in range(len(frame))]
    for fold in range(3):
        fit=np.flatnonzero(frame.fold.ne(fold));val=np.flatnonzero(frame.fold.eq(fold));inputs=Inputs(frame,context,features,codes,fit,torch.device('cpu'))
        encoded=inputs.facts.numpy()
        for name,idx in [('source',5),('destination',6)]:audit.loc[val,name+'_port_OOV']=encoded[val,idx]==1
        keys=inputs.views
        for name,values in [('single',single),('single_plus_safe_residual',single_safe),('actual_model',keys),('uncapped_legal_context',uncapped_keys),('literal_body_identity_diagnostic',residual.raw_body_sha256)]:
            stat,flags=conflict(frame.iloc[val],np.asarray(values)[val]);folds.setdefault(str(fold),{})[name]=stat
            if name=='actual_model':audit.loc[val,'validation_input_conflict']=flags
        fit_support=pd.DataFrame({'key':keys[fit],'label':frame.label.iloc[fit].to_numpy(),'group':frame.group.iloc[fit].to_numpy()}).drop_duplicates()
        for label in [1,2]:
            support=fit_support[fit_support.label.eq(label)].groupby('key').group.nunique()
            audit.loc[val,'training_exact_view_'+str(label)+'_sources']=pd.Series(keys[val]).map(support).fillna(0).to_numpy(dtype=int)
        folder=v65/f'fold{fold}'
        selection[str(fold)]=select_both({n:read(folder/n/'curve.json') for n in ['H0','H1']},read(folder/'references_before_H1.json')['references'])
        print('AUDITED_FOLD',fold,flush=True)
    summaries=[]
    hard=frame.src_role.eq('outside')&frame.dst_role.eq('dmz')&frame.transport_protocol.isin(['tcp','udp'])
    for name,mask in [('all',np.ones(len(frame),bool)),('hard_S',hard&frame.label.eq(2)),('failed_hard_S',hard&frame.label.eq(2)&audit.H1_pred.ne(2)),('normal',frame.label.eq(0))]:
        d=audit[mask]
        summaries.append({'slice':name,'rows':len(d),'sources':int(d.group.nunique()),
           'H1_errors':int(d.H1_pred.ne(d.label).sum()),'empty_context':int(d.empty_context.sum()),
           'cap_removed_entries':int(d.cap_removed_entries.sum()),'source_port_OOV':int(d.source_port_OOV.sum()),
           'destination_port_OOV':int(d.destination_port_OOV.sum()),'src_redacted':int(d.src_redacted.sum()),'dst_redacted':int(d.dst_redacted.sum()),
           'validation_input_conflict':int(d.validation_input_conflict.sum()),
           'no_same_label_training_exact_view':int(d['training_exact_view_2_sources'].eq(0).sum()) if name.endswith('S') else None})
    audit.to_parquet(a.out/'row_diagnosis.parquet',index=False)
    residual.to_parquet(a.out/'raw_roundtrip_receipts.parquet',index=False)
    safe.to_parquet(a.out/'safe_residual.parquet',index=False)
    nuisance.to_parquet(a.out/'nuisance_private_diagnostic.parquet',index=False)
    pd.DataFrame(summaries).to_csv(a.out/'failure_slices.csv',index=False)
    save(a.out/'selection_replay.json',selection)
    # Candidate residuals partition the existing legal single-event view only if new information exists.
    new_partitions=int(pd.DataFrame({'old':single,'new':single_safe}).groupby('old').new.nunique().gt(1).sum())
    summary={'version':'v66-root-cause-audit','rows_replayed_from_official':len(frame),'official_counts':counts,
       'official_train_sha256':sha(ROOT/'data/official/train.parquet'),
       'normal_rows_fraction':float(frame.label.eq(0).mean()),'safe_residual_new_single_view_partitions':new_partitions,
       'safe_residual_cardinalities':{c:int(safe[c].nunique()) for c in safe},'slices':summaries,'conflicts_by_fold':folds,
       'selection_fixed_by_replay':{k:{n:{'status':v['status'],'selected_step':None if v['selected'] is None else v['selected']['step']} for n,v in d.items()} for k,d in selection.items()},
       'semantic_limits':'Official redaction cannot be reversed from supplied data. Literal body and nuisance distinctions are identity/environment diagnostics, not qualified attack evidence. Conflict bounds are empirical and view-specific; exact support absence is not proof nonlinear generalization is impossible.',
       'quality_acceptance':False,'new_training_executed':False,'source_sha256':sha(__file__)}
    save(a.out/'audit.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
