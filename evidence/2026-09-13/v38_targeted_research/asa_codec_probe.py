"""Encoder distinguishability only: no trained quality claim, no target fit.

The fixed 17-bit code represents the actual unsigned port domain plus missing,
not an ordinal rank assigned to categories observed during fit.
"""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
import joblib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
WORK=ROOT/'artifacts/v37_cloud_20260913T072309Z'
sys.path.insert(0,str(WORK/'runtime'))
import v37_learning as learning

def encode_domain(v):
    a=np.asarray(v,dtype=np.uint32)
    assert np.all(a<=65536)
    return ((a[:,None]>>np.arange(17,dtype=np.uint32))&1).astype(np.uint8)

def identities(x):
    x=x.copy().tocsr();x.eliminate_zeros();x.sort_indices();hs=[]
    for i in range(x.shape[0]):
        a,b=x.indptr[i:i+2]
        hs.append(hashlib.sha256(x.indices[a:b].astype('<i8').tobytes()+x.data[a:b].astype('<f8').tobytes()).hexdigest())
    return np.asarray(hs,dtype=object)

def stats(keys,back,y):
    codes,unique=pd.factorize(keys,sort=False);rc=codes[back]
    cnt=np.bincount(rc*3+y,minlength=len(unique)*3).reshape(-1,3);mixed=(cnt>0).sum(1)>1
    return {'distinct_vectors':len(unique),'mixed_vectors':int(mixed.sum()),'mixed_rows':int(cnt[mixed].sum()),
            'minimum_total_errors':int((cnt.sum(1)-cnt.max(1)).sum())}

def main():
    full=np.arange(65537,dtype=np.uint32);bits=encode_domain(full)
    restored=(bits.astype(np.uint32)*(1<<np.arange(17,dtype=np.uint32))).sum(1)
    assert np.array_equal(restored,full)
    p=ROOT/'artifacts/v37_prepared_r13_20260913'
    d=pq.read_table(p/'prepared.parquet',columns=['row_position','b1','facts','label_index']).to_pandas()
    roles=pq.read_table(p/'protocol.parquet')['asa_hard'].to_numpy();d=d[roles==3];y=d.label_index.to_numpy()
    back,keys=pd.factorize(pd.MultiIndex.from_frame(d[['b1','facts']]),sort=False);texts=[k[0] for k in keys];facts=[json.loads(k[1]) for k in keys]
    b=joblib.load(WORK/'work/models/asa_hard_B2_REPAIRED/model.joblib')
    x=sparse.hstack([b['tfidf'].transform(texts),b['facts'].transform(facts)],format='csr')
    arrays=[];coarse=[]
    for field in ['src_port_category','dst_port_category']:
        vals=[]
        for f in facts:
            value=65536 if field not in f else int(f[field].rsplit('|',1)[1])
            assert 0<=value<=65536;vals.append(value)
        vals=np.array(vals);arrays.append(sparse.csr_matrix(encode_domain(vals),dtype=np.float64))
        labels=np.where(vals==65536,3,np.where(vals<=1023,0,np.where(vals<=49151,1,2)))
        coarse.append(sparse.csr_matrix((np.ones(len(vals)),(np.arange(len(vals)),labels)),shape=(len(vals),4)))
    variants={'frozen_encoder':x,'plus_source_fixed17':sparse.hstack([x,arrays[0]],format='csr'),
      'plus_destination_fixed17':sparse.hstack([x,arrays[1]],format='csr'),
      'plus_both_fixed17':sparse.hstack([x,*arrays],format='csr'),
      'plus_coarse_port_ranges_only':sparse.hstack([x,*coarse],format='csr')}
    result={name:stats(identities(mat),back,y) for name,mat in variants.items()}
    # Stateless hashing is not a learned solution for unseen symbolic values.
    dictionaries=[learning.fact_dictionary(f) for f in facts]
    hx=FeatureHasher(n_features=2**20,input_type='dict',alternate_sign=True).transform(dictionaries)
    result['plus_hashed_facts_2pow20']=stats(identities(sparse.hstack([x,hx],format='csr')),back,y)
    result['full_preencoder_information']=stats(np.array([json.dumps([t,f],sort_keys=True) for t,f in zip(texts,facts)],object),back,y)
    assert result['plus_both_fixed17']['distinct_vectors']==result['full_preencoder_information']['distinct_vectors']
    assert result['plus_both_fixed17']['minimum_total_errors']==result['full_preencoder_information']['minimum_total_errors']
    # General mathematical example: a reserved but untrained one-hot column has
    # zero learned coefficient, so representation distinction alone changes no score.
    example={'known_coef':[1.0,0.0,0.0],'unseen_1':[0,1,0],'unseen_2':[0,0,1]}
    example['scores']=[float(np.dot(example['known_coef'],example[k])) for k in ['unseen_1','unseen_2']]
    out={'scope':'Already-inspected ASA hard, representation audit only; no SOC classifier fitted or accuracy improvement claimed',
         'full_domain_roundtrip_cases':65537,'full_domain_roundtrip_passed':True,'added_dimensions_both_ports':34,
         'uses_target_labels_to_define_encoding':False,'uses_target_labels_to_measure_collision_floor':True,
         'results':result,'untrained_category_example':example,
         'cautions':['Bit codes have arbitrary metric geometry; not a security-semantic embedding.',
                     'Missingness remains observable; not asserted harmless.',
                     'Exact source port may encode environment artifacts; this is an audit/control candidate only.',
                     'Hashing may collide and unseen buckets have no learned meaning.']}
    (OUT/'asa_codec_probe.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
