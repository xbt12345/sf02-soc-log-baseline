"""Recount source context and attribute groups without models or learned fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from experiment_review import read,sha,check_bindings

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/v151_context_coherence_20261001'
CURRENT=ROOT/'artifacts/v151_current_context_view_20261001'

def main():
    target=BASE/'verification.json';assert not target.exists()
    a=read(BASE/'audit.json');b=read(CURRENT/'audit.json');check_bindings(a['source_sha256']);check_bindings(b['source_sha256'])
    ledger=pd.read_parquet(BASE/'all_ASA_context_ledger.parquet').sort_values('row_position').reset_index(drop=True)
    positions=ledger.row_position.to_numpy();parts=[];offset=0;file=pq.ParquetFile(ROOT/'data/official/train.parquet')
    for batch in file.iter_batches(batch_size=32768):
        d=batch.to_pandas();pos=positions[(positions>=offset)&(positions<offset+len(d))]
        if len(pos):
            rows=d.iloc[pos-offset].copy();rows['row_position']=pos;parts.append(rows)
        offset+=len(d)
    original=pd.concat(parts).sort_values('row_position').reset_index(drop=True)
    assert offset==2056871 and len(original)==len(ledger)==112807
    assert np.array_equal(original.row_position,ledger.row_position) and np.array_equal(original.label_binary.map({'benign':0,'malicious':1,'suspicious':2}),ledger.truth)
    attrs=[c for c in file.schema.names if c not in ['event_id','label_binary']]
    # Exact available-attribute equality checks the pandas 64-bit grouping;
    # fingerprint equality itself is never accepted as event identity.
    actual_unique=original[attrs].drop_duplicates()
    assert len(actual_unique)==a['time_and_record_exposure']['distinct_available_attribute_groups']==50863
    exact_hash=pd.util.hash_pandas_object(actual_unique,index=False)
    assert exact_hash.nunique()==len(actual_unique),'Hash collision would invalidate group counts'
    allhash=pd.util.hash_pandas_object(original[attrs],index=False).to_numpy()
    assert np.array_equal(allhash,ledger.available_event_attribute_hash)
    assert np.array_equal(original.event_id.nunique(),112807)
    for c in ['product_name','vendor_name']:
        assert original.loc[ledger.truth.eq(1),c].eq('').all()
        assert original.loc[ledger.truth.eq(2),c].map(lambda v:isinstance(v,str) and len(v.strip())>0).all()
    assert ledger.syslog_severity.eq(4).all()
    assert ledger.loc[ledger.truth.eq(2),'syslog_facility'].eq(20).all()
    query=pd.read_parquet(BASE/'all_outer_legal_condition_support.parquet')
    assert len(query)==112807 and not query.row_position.duplicated().any()
    assert set(query.row_position)==set(ledger.row_position)
    current=pd.read_parquet(CURRENT/'all_ASA_current_ordered_view_ledger.parquet')
    assert np.array_equal(current.row_position,ledger.row_position) and int(current.old_header_gap.sum())==682
    assert b['current_header_date_clock_invariance_all_rows'] and b['current_text_byte_input_max_abs_gap']==0.0
    # Applicable ICMP is retained in the raw-grammar population, not misreported
    # as port missing or silently removed from context consistency denominators.
    facts=ledger.facts_json.map(json.loads);proto=facts.map(lambda f:f['transport_protocol'])
    assert proto.eq('icmp').sum()==2373
    assert ledger.loc[proto.eq('icmp'),'src_port_record_body_relation'].eq('not_comparable').all()
    result=dict(status='actual_context_source_and_versioned_view_verified',official_rows=offset,ASA_rows=len(ledger),
        new_model_forwards=0,new_gradients=0,new_fits=0,new_updates=0,exact_available_attribute_groups=len(actual_unique),
        fingerprint_hash_collisions_detected=0,all_original_event_ids_unique=True,
        product_vendor_missing_states_exact=True,all_severity4=True,all_ICMP_retained=True,
        original_legal_support_query_population_exact=True,current_text_byte_replay_zero_gap=True,
        v1_partial_clock_name_requires_qualification=True,v1_artifacts_preserved=True,
        no_classifier_predictions_generated=True,quality_acceptance=False,issue_solved=False,
        verified_scope='Official attributes/labels, full original populations, hash collision check, descriptive context groups and executed versioned text replay receipts. No model or class-quality validation.',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),BASE/'audit.json',BASE/'all_ASA_context_ledger.parquet',
            BASE/'all_outer_legal_condition_support.parquet',CURRENT/'audit.json',CURRENT/'all_ASA_current_ordered_view_ledger.parquet',ROOT/'data/official/train.parquet']})
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False))

if __name__=='__main__':main()
