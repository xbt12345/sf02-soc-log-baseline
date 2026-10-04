"""Whole-population legal-TRAIN fine-support audit; no fitting or label routing."""
import json
import re
import numpy as np
import pandas as pd
from v142_runtime import ROOT,OUT as RUN,read,save,sha
from v131_evaluate import full_folds
from v123_support_ladder_audit import known,COARSE

OUT=ROOT/'artifacts/v143_fine_support_evidence_20261001'


def key(f,fields):
    return json.dumps({k:f.get(k) for k in fields},sort_keys=True,separators=(',',':'))


def main():
    if OUT.exists():raise FileExistsError('Preserve completed evidence')
    asa=pd.read_parquet(RUN/'ASA_prediction_ledger.parquet')
    trace=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    t=pd.read_parquet(trace,columns=['row_position','facts_json'])
    d=asa.merge(t,on='row_position',validate='one_to_one');facts=d.facts_json.map(json.loads)
    source=ROOT/'data/official/train.parquet'
    raw=pd.read_parquet(source,columns=['product_name','message_sanitized','src_port','label_binary'])
    y=raw.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(np.int8)
    assert len(raw)==2056871 and np.array_equal(d.truth,y[d.row_position])
    d['parameter_observed']=facts.map(known)
    d['body_src_port_observed']=facts.map(lambda f:0<=f.get('src_port_fixed',65536)<=65535)
    d['record_src_port_observed']=raw.src_port.iloc[d.row_position].map(lambda s:bool(re.fullmatch(r'\d{1,5}',s or '')) and int(s)<=65535).to_numpy()
    d['wrong']=d.pred_S2.ne(d.truth)
    levels={'destination':['action','outcome','transport_protocol','src_role','dst_role','dst_port_fixed','dst_port_range','icmp_type','icmp_code'],
            'behavior':list(COARSE),'transport_destination':['transport_protocol','dst_port_fixed','icmp_type','icmp_code']}
    summary={};contrast=[]
    for level,fields in levels.items():
        col=level+'_key';d[col]=facts.map(lambda f:key(f,fields))
        for c in [1,2]:
            for what in ['rows','roots']:d[f'{level}_{c}_{what}']=0
        d[level+'_train_majority']=-1
        for f in range(3):
            q=d.fold.eq(f);train=d[~q]
            assert not(set(train.root)&set(d.loc[q,'root']))
            # Known and unknown pools stay separate even when a sentinel key matches.
            if level!='behavior':train=train[train.parameter_observed]
            for c in [1,2]:
                counts=train[train.truth.eq(c)].groupby(col).agg(rows=('truth','size'),roots=('root','nunique'))
                for what in ['rows','roots']:d.loc[q,f'{level}_{c}_{what}']=d.loc[q,col].map(counts[what]).fillna(0).astype(int)
            counts=train.groupby([col,'truth']).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
            majority=pd.Series(np.where(counts[1]>counts[2],1,np.where(counts[2]>counts[1],2,-1)),index=counts.index)
            d.loc[q,level+'_train_majority']=d.loc[q,col].map(majority).fillna(-1).astype(int)
            # Legal TRAIN identity/label collision floor describes projection ambiguity, not model error.
            mixed=counts[(counts[1]>0)&(counts[2]>0)]
            paired=train.groupby([col,'truth']).root.nunique().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
            dual=paired[(paired[1]>=2)&(paired[2]>=2)]
            contrast.append(dict(fold=f,level=level,legal_original_rows=len(train),keys=len(counts),both_label_keys=len(mixed),
                projection_minimum_error_rows=int((mixed.sum(1)-mixed.max(1)).sum()),both_classes_two_roots_keys=len(dual),
                dual_root_original_rows=int(train[col].isin(dual.index).sum())))
        same=np.where(d.truth.eq(1),d[f'{level}_1_rows'],d[f'{level}_2_rows'])
        roots=np.where(d.truth.eq(1),d[f'{level}_1_roots'],d[f'{level}_2_roots'])
        other=np.where(d.truth.eq(1),d[f'{level}_2_rows'],d[f'{level}_1_rows'])
        bucket=np.select([same==0,roots==1],['no_same_class','one_same_source'],default='multiple_same_sources')
        if level!='behavior':bucket=np.where(d.parameter_observed,bucket,'unobserved_parameter_not_fine_support')
        d[level+'_bucket']=bucket;d[level+'_opposite_present']=other>0
        d[level+'_majority_agrees']=d[level+'_train_majority'].eq(d.truth)
        s=d.groupby(['truth',level+'_bucket']).agg(population=('truth','size'),errors=('wrong','sum'),sources=('root','nunique'),
                 opposite_class_rows_with_support=(level+'_opposite_present','sum'),legal_train_majority_agrees=(level+'_majority_agrees','sum')).reset_index()
        s['error_rate']=s.errors/s.population;summary[level]=s.to_dict('records')
    availability=d.groupby(['truth','parameter_observed','body_src_port_observed','record_src_port_observed']).agg(
        population=('truth','size'),errors=('wrong','sum'),sources=('root','nunique')).reset_index()
    # Inspect all VPC originals, preserve malformed/unknown rows, never invent ASA zone roles.
    full=full_folds();assert np.array_equal(full.row_position,np.arange(len(raw)))
    ids=np.flatnonzero(raw.product_name.eq('AWS VPC Security'));vpc=[]
    for i in ids:
        parts=raw.message_sanitized.iloc[i].split();p=port=None;action=None
        if len(parts)>=14:
            p=int(parts[7]) if parts[7].isdigit() else None
            port=int(parts[6]) if parts[6].isdigit() and 0<=int(parts[6])<=65535 else None
            action=parts[-2] if parts[-2] in ['REJECT','ACCEPT'] else None
        vpc.append(dict(row_position=int(i),fold=int(full.proposed_fold.iloc[i]),root=int(full.root.iloc[i]),truth=int(y[i]),
            protocol={6:'tcp',17:'udp',1:'icmp'}.get(p),destination_port=port,action=action,
            raw_parse_observed=p is not None and port is not None and action is not None))
    v=pd.DataFrame(vpc);cross=[]
    for f in range(3):
        train=v[v.fold.ne(f)];held=d[d.fold.eq(f)]
        assert not(set(train.root)&set(held.root))
        usable=train[train.raw_parse_observed & train.protocol.isin(['tcp','udp']) & train.action.eq('REJECT')]
        support=set(zip(usable.protocol,usable.destination_port))
        mask=np.array([(a.get('transport_protocol'),a.get('dst_port_fixed')) in support and known(a) for a in facts[held.index]])
        cross.append(dict(fold=f,legal_VPC_rows=len(train),legal_VPC_usable_reject_rows=len(usable),all_VPC_labels_S=bool(train.truth.eq(2).all()),
            ASA_protocol_destination_matching_rows=int(mask.sum()),ASA_matching_S_rows=int((mask&held.truth.eq(2)).sum()),
            ASA_matching_M_rows=int((mask&held.truth.eq(1)).sum()),ASA_matching_wrong_rows=int((mask&held.wrong).sum()),
            exact_fine_semantic_support_rows=0,reason='VPC has no observed ASA src/dst interface roles; protocol and port overlap is not an identical fine behavior.'))
    report=dict(status='fine_support_evidence_only',latest_actual_training='V142',new_classifier_fits=0,new_parameter_updates=0,
        official_population=len(raw),ASA_population=len(d),VPC_population=len(v),support_summaries=summary,
        legal_TRAIN_projection_contrasts=contrast,raw_parameter_availability=availability.to_dict('records'),cross_format_support=cross,
        second_issue_solved=False,third_issue_solved=False,quality_acceptance=False,
        limits=['Whole ASA population includes correct controls; held truth used only for post-fit diagnosis, never weights or candidate selection.',
                'Unknown parameters are kept; sentinel equality cannot establish concrete parameter support.',
                'TRAIN majority is a diagnostic projection, not deployed routing or a new classifier fit.',
                'Same-class support counts and error rates do not establish causal error attribution.',
                'Cross-format protocol/port overlap lacks equivalent ASA roles and opposite-class VPC support; not authorized as synthetic labeled fine-support expansion.'],
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [__import__('pathlib').Path(__file__),trace,source,RUN/'ASA_prediction_ledger.parquet']})
    OUT.mkdir();d.to_parquet(OUT/'fine_support_population.parquet',index=False);v.to_parquet(OUT/'VPC_observed_originals.parquet',index=False)
    save(OUT/'audit.json',report)
    print(json.dumps({k:report[k] for k in ['support_summaries','legal_TRAIN_projection_contrasts','raw_parameter_availability','cross_format_support']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
