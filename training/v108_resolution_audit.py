"""Separate real body facts from record source-port uniqueness; no ablation fit."""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from run_v75 import ROOT, save, sha
from v75_views import BYTE_FEATURES, matrix_hashes
from v75_metadata import encode, parse_port
from v108_root_evidence_audit import DEST, PREV, N1, conflict


def main():
    target=DEST/'resolution_and_metadata_audit.json'; assert not target.exists()
    d=pd.read_parquet(DEST/'support_and_error_ledger.parquet')
    raw=pd.read_parquet(DEST/'ASA_raw_fact_prediction_trace.parquet')
    assert np.array_equal(raw.row_position,d.row_position)
    positions=d.row_position.to_numpy();offset=0;values=[]
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=32768,columns=['src_port']):
        a=np.searchsorted(positions,offset);b=np.searchsorted(positions,offset+len(batch))
        if a!=b:values.extend(batch.take(positions[a:b]-offset).column(0).to_pylist())
        offset+=len(batch)
    parsed=[json.loads(s) for s in raw.facts_json]
    message=np.array([f.get('src_port_fixed',65536) for f in parsed])
    meta,key,extra=encode(values,message)
    n2=sparse.load_npz(PREV/'N2_ASA.npz')
    assert (meta!=n2[d.local.to_numpy(),-18:]).nnz==0
    record=np.array([65536 if parse_port(v)[0] is None else parse_port(v)[0] for v in values])
    both=(message<65536)&(record<65536);mismatch=both&(message!=record)
    results={}
    for view,x in [('N1',sparse.load_npz(N1)),('N2',n2)]:
        for scope,xx in [('body_text_and_parsed_facts',x[:,:-18]),('parsed_body_facts',x[:,BYTE_FEATURES:-18]),('record_src_port',x[:,-18:])]:
            hashes=np.array(matrix_hashes(xx),dtype=object);d['resolution_key']=hashes[d.local]
            ct=d.groupby(['resolution_key','truth']).size().unstack(fill_value=0)
            mixed=ct.index[(ct>0).sum(axis=1)>1]
            rr=conflict(d,'resolution_key')
            rr['regressed_S_in_mixed_groups']=int((d.resolution_key.isin(mixed)&(d.truth==2)&(d.N1_TabM25==2)&(d.N2_TabM25!=2)).sum())
            rr['equal_input_groups_crossing_existing_folds']=int((d.groupby('resolution_key').fold.nunique()>1).sum())
            results[view+'_'+scope]=rr
    cases=json.loads((DEST/'top20_regression_case_triples.json').read_text())
    lookup=dict(zip(positions,values));examples=[]
    for c in cases[:5]:
        x={'query_local':c['query_local'],'regressed_S_rows':c['regressed_S_rows']}
        for name in ('query','nearest_fit_M','nearest_fit_S'):
            r=c[name];x[name]={'row_position':r['row_position'],'truth':r['truth'],
                'record_src_port':str(lookup[r['row_position']]),'message_src_port':r['facts'].get('src_port_fixed'),
                'N2_text_equals_query':r['N2_text']==c['query']['N2_text'],
                'parsed_facts_equal_query':r['facts']==c['query']['facts']}
        examples.append(x)
    result={'status':'input_resolution_audit_no_fit','source_sha256':sha(__file__),
        'classifier_fits':0,'metadata_reconstruction_matches_all_ASA':True,
        'feature_blocks':{'byte_1_2grams':BYTE_FEATURES,'parsed_body_facts':n2.shape[1]-BYTE_FEATURES-18,'record_src_port':18},
        'record_and_body_port':{'both_visible':int(both.sum()),'disagree':int(mismatch.sum()),
             'body_unknown_record_visible':int(((message==65536)&(record<65536)).sum()),
             'note':'Distinct observations; disagreement does not establish which is true or that either is malicious evidence.'},
        'resolution_comparisons':results,'case_metadata':examples,
        'limitations':['Dropping source-port coordinates is a diagnostic projection, not an approved model input.',
                      'Low full-input collision error can reflect source-port uniqueness rather than stable security meaning.',
                      'New projected-input collisions cross existing folds: an ablation train/test comparison would need new equal-input closure and matched baselines.',
                      'No field was removed from any classifier; no performance improvement is claimed.']}
    save(target,result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
