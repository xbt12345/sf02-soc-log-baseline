"""Zero-fit pairing-key relaxation audit. Model inputs and predictions stay fixed."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT/'artifacts/v143_fine_support_evidence_20261001'
RUN = ROOT/'artifacts/v142_second_layer_training_20261001'
OUT = ROOT/'artifacts/v143_independent_pair_review_20261001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def port(s):
    if isinstance(s,str) and re.fullmatch(r'[0-9]{1,5}',s) and int(s)<=65535:
        return int(s)
    return None


def regime(p):
    return 'missing' if p is None else 'system' if p<1024 else 'user' if p<49152 else 'dynamic'


def key(f, policy):
    v=dict(f)
    value=v.get('src_port_fixed',65536)
    if policy=='dynamic_source_numeric' and 49152<=value<=65535:
        v['src_port_fixed']='KNOWN_DYNAMIC_VALUE'
    if policy=='user_dynamic_source_numeric' and 1024<=value<=65535:
        v['src_port_fixed']='KNOWN_'+regime(value).upper()+'_VALUE'
    return json.dumps(v,sort_keys=True,separators=(',',':'))


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed audit')
    original=ROOT/'data/official/train.parquet'
    raw=pd.read_parquet(original,columns=['label_binary','src_port','product_name','message_sanitized'])
    y=raw.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(np.int8)
    d=pd.read_parquet(PREV/'fine_support_population.parquet')
    assert len(raw)==2056871 and len(d)==112807 and not d.row_position.duplicated().any()
    assert np.array_equal(y[d.row_position],d.truth)
    facts=d.facts_json.map(json.loads)
    d['protocol']=facts.map(lambda f:f.get('transport_protocol'))
    policies=['exact_full_facts','dynamic_source_numeric','user_dynamic_source_numeric']
    for policy in policies:
        d[policy]=facts.map(lambda f:key(f,policy))
    d['full_key']=d.exact_full_facts
    d['body_record_source_port_same']=np.array([
        port(raw.src_port.iloc[i])==f.get('src_port_fixed')
        for i,f in zip(d.row_position,facts)
    ])
    capacity,held_coverage,eligible=[] ,[],[]
    for scope in ['known_destination','known_both_transport_ports']:
        observed=d.parameter_observed.copy()
        if scope=='known_both_transport_ports':
            observed &= d.protocol.isin(['tcp','udp']) & d.body_src_port_observed
        for fold in range(3):
            train=d[d.fold.ne(fold)&observed].copy()
            held=d[d.fold.eq(fold)&observed].copy()
            assert not (set(train.root)&set(held.root))
            for policy in policies:
                counts=train.groupby([policy,'truth']).agg(rows=('truth','size'),roots=('root','nunique'),
                    locals=('local','nunique'),full_keys=('full_key','nunique')).reset_index()
                positives=counts[counts.roots.ge(2)&counts.locals.ge(2)]
                mass=counts.pivot(index=policy,columns='truth',values='rows').fillna(0).reindex(columns=[1,2],fill_value=0)
                roots=counts.pivot(index=policy,columns='truth',values='roots').fillna(0).reindex(columns=[1,2],fill_value=0)
                floor=int((mass.sum(1)-mass.max(1)).sum())
                exact=train.groupby(['full_key','truth']).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
                exact_floor=int((exact.sum(1)-exact.max(1)).sum())
                for cl in [1,2]:
                    pos=positives[positives.truth.eq(cl)]
                    mixed=pos[ pos[policy].isin(mass.index[(mass[1]>0)&(mass[2]>0)]) ]
                    dual=pos[ pos[policy].isin(roots.index[(roots[1]>=2)&(roots[2]>=2)]) ]
                    capacity.append(dict(training_role=fold,scope=scope,policy=policy,truth=cl,
                        legal_original_rows=len(train),positive_groups=len(pos),positive_rows=int(pos.rows.sum()),
                        positive_groups_also_opposite_label=len(mixed),positive_groups_dual_class_multisource=len(dual),
                        projection_minimum_error_rows=floor,exact_projection_minimum_error_rows=exact_floor,
                        added_projection_conflict_floor=floor-exact_floor))
                    for v in pos.to_dict('records'):
                        eligible.append(dict(training_role=fold,scope=scope,policy=policy,**v))
                    h=held[held.truth.eq(cl)].copy()
                    h['same_class_positive_group']=h[policy].isin(set(pos[policy]))
                    h['opposite_class_present']=h[policy].isin(set(mass.index[mass[3-cl]>0]))
                    for proto,hh in h.groupby('protocol'):
                        covered=hh.same_class_positive_group
                        held_coverage.append(dict(fold=fold,scope=scope,policy=policy,truth=cl,protocol=proto,
                            population=len(hh),errors=int(hh.wrong.sum()),
                            supported_rows=int(covered.sum()),supported_errors=int((covered&hh.wrong).sum()),
                            supported_correct=int((covered&~hh.wrong).sum()),
                            supported_also_opposite_label=int((covered&hh.opposite_class_present).sum())))

    # Fixed, explicitly unsafe pooling rule is evaluated only as a counterexample.
    # The published classifier and every model probability stay unchanged.
    vpc=pd.read_parquet(PREV/'VPC_observed_originals.parquet')
    assert not vpc.row_position.duplicated().any() and np.array_equal(vpc.truth,y[vpc.row_position])
    parse_checks=[]
    for row in vpc.itertuples(index=False):
        parts=raw.message_sanitized.iloc[row.row_position].split()
        assert len(parts)==14, 'Default format identity needs independent review'
        expected_proto={'6':'tcp','17':'udp','1':'icmp'}.get(parts[7])
        expected_dst=port(parts[6])
        assert (row.protocol is None or pd.isna(row.protocol)) if expected_proto is None else row.protocol==expected_proto
        assert pd.isna(row.destination_port) if expected_dst is None else row.destination_port==expected_dst
        assert row.action==parts[12] and parts[12] in ['ACCEPT','REJECT']
        parse_checks.append(row.row_position)
    assert len(parse_checks)==10620
    counter=[]
    for fold in range(3):
        train=vpc[vpc.fold.ne(fold)&vpc.raw_parse_observed&vpc.protocol.isin(['tcp','udp'])&vpc.action.eq('REJECT')]
        h=d[d.fold.eq(fold)]
        assert not (set(train.root)&set(h.root))
        supporting=set(zip(train.protocol,train.destination_port.astype(int)))
        match=np.array([row.parameter_observed and (json.loads(row.facts_json).get('transport_protocol'),
               json.loads(row.facts_json).get('dst_port_fixed')) in supporting for row in h.itertuples(index=False)])
        before=h.pred_S2.to_numpy();after=before.copy();after[match]=2
        for cl in [1,2]:
            truth=h.truth.to_numpy();clmask=truth==cl
            counter.append(dict(fold=fold,truth=cl,matched_rows=int((match&clmask).sum()),
                before_errors=int((clmask&(before!=truth)).sum()),after_errors=int((clmask&(after!=truth)).sum()),
                repairs=int((clmask&(before!=truth)&(after==truth)).sum()),
                new_errors=int((clmask&(before==truth)&(after!=truth)).sum())))

    both=d.protocol.isin(['tcp','udp'])&d.body_src_port_observed&d.parameter_observed
    report=dict(status='zero_fit_pairing_definition_and_pooling_counterexample_review',
        latest_actual_training='V142',new_classifier_fits=0,new_parameter_updates=0,
        model_inputs_or_predictions_modified=False,pair_key_is_not_model_input=True,
        legal_TRAIN_capacity=capacity,HELD_coverage_diagnosis_only=held_coverage,
        known_both_transport_port_rows=int(both.sum()),
        known_both_ports_body_record_equal_rows=int((both&d.body_record_source_port_same).sum()),
        independently_checked_default_VPC_rows=len(parse_checks),
        unsafe_VPC_label_pooling_counterexample=counter,
        limitations=['Changing a pairing selection key does not remove any classifier input or establish source-port label invariance.',
            'Positive pairs are real legal-TRAIN same-class records, not generated records or added labels.',
            'Additional projection conflicts forbid interpreting these keys as a class lookup table.',
            'HELD labels are retrospective diagnosis only, not pair construction, weights or candidate selection.',
            'The VPC experiment is a fixed-rule counterexample, not proof that all cross-format representation learning is impossible.',
            'Counts and overlap do not prove that contrastive training will improve classification or cover truly missing fine behaviors.'],
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__),original,PREV/'fine_support_population.parquet',PREV/'VPC_observed_originals.parquet',RUN/'final_delivery.json']})
    OUT.mkdir()
    pd.DataFrame(eligible).to_parquet(OUT/'legal_TRAIN_positive_group_profiles.parquet',index=False)
    pd.DataFrame(held_coverage).to_parquet(OUT/'held_coverage_diagnosis.parquet',index=False)
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],legal_TRAIN_capacity=capacity,
        unsafe_VPC_label_pooling_counterexample=counter),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
