"""Full-input, label-free replay of frozen v48/v51 development classifiers.

No training, answers, threshold fitting, IP lookup or product routing. The two
classifiers remain development models; this module does not promote a model.
Run with the original sklearn 1.9 runtime (.venv), not .venv-v61.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/'artifacts/v51_fact_residual_20260914'
INPUT_COLUMNS = ['event_id','timestamp','pipeline','src_ip','dst_ip','src_port',
                 'src_host','dst_host','username','message_sanitized','product_name','vendor_name']
LABELS = np.array(['benign','malicious','suspicious'])


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')


def load_bound():
    if sklearn.__version__!='1.9.0':raise RuntimeError('Use original sklearn 1.9.0 runtime')
    config=read(FROZEN/'configuration.json');bindings={}
    for name,h in config['runtime_bindings'].items():
        p=FROZEN/'runtime'/name
        assert sha(p)==h,name
        bindings[p.relative_to(ROOT).as_posix()]=h
    sys.path.insert(0,str(FROZEN/'runtime'))
    import v48_input as adapter
    import v51_residual as residual
    p=FROZEN/'pressure/model.joblib'
    receipt=read(FROZEN/'pressure/complete.json')
    assert sha(p)==receipt['model.joblib']
    bindings[p.relative_to(ROOT).as_posix()]=sha(p)
    bundle=joblib.load(p)
    parent=ROOT/'artifacts/v48_information_repair_r2_20260914/pressure/model.joblib'
    assert sha(parent)==bundle['base_sha256']
    base=joblib.load(parent)
    bindings[parent.relative_to(ROOT).as_posix()]=sha(parent)
    for k in ['coef_','intercept_','classes_']:
        np.testing.assert_array_equal(getattr(base['model'],k),getattr(bundle['base']['model'],k))
    np.testing.assert_array_equal(base['model'].classes_,[0,1,2])
    return bundle,adapter,residual,bindings


def infer_messages(bundle,adapter,residual,messages):
    parsed=[adapter.prepare_record({'message_sanitized':s}) for s in messages]
    facts=[p['facts'] for p in parsed];texts=[p['text'] for p in parsed]
    for f in facts:
        assert not any(s.startswith('unencoded') for s in adapter.fact_dispositions(f).values())
    base=bundle['base'];tx=base['text_encoder'].transform(texts);fx=base['fact_encoder'].transform(facts)
    x=sparse.hstack([tx,fx],format='csr');gate=residual.applicable(facts)
    q0=base['model'].predict_proba(x)
    q1=residual.predict_matrix(base['model'],bundle['residual'],x,gate)
    assert np.isfinite(q1).all() and q1.shape==(len(messages),3)
    np.testing.assert_allclose(q1.sum(axis=1),1,atol=1e-12)
    np.testing.assert_array_equal(q0[~gate],q1[~gate])
    x.eliminate_zeros();x.sort_indices()
    views=[]
    for i in range(x.shape[0]):
        start,end=x.indptr[i:i+2]
        h=hashlib.sha256(x.indices[start:end].astype('<i8').tobytes())
        h.update(x.data[start:end].astype('<f8').tobytes());h.update(bytes([int(gate[i])]))
        views.append(h.hexdigest())
    meta=pd.DataFrame({'route':[p['route'] for p in parsed],
        'protocol':[f.get('transport_protocol','') for f in facts],
        'action':[f.get('action','') for f in facts],
        'src_role':[f.get('src_role','') for f in facts],
        'dst_role':[f.get('dst_role','') for f in facts],
        'residual_applied':gate,'fact_dict_empty':[not f for f in facts],
        'effective_input_sha256':views,
        'text_encoded_zero':np.asarray(tx.getnnz(axis=1)==0),
        'body_sha256':[hashlib.sha256(p['text'].encode()).hexdigest() for p in parsed]})
    return q0,q1,meta


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=8192)
    args=ap.parse_args();source=args.input.resolve();out=args.out.resolve()
    schema=pq.ParquetFile(source).schema_arrow.names
    if set(schema)!=set(INPUT_COLUMNS):raise ValueError('Only the twelve official unlabeled input columns are accepted')
    if out.exists():raise FileExistsError('Refuse overwriting a prior/partial run')
    bundle,adapter,residual,bindings=load_bound();out.mkdir(parents=True)
    (out/'inference_source.py').write_bytes(Path(__file__).read_bytes())
    reg={'scope':'Full official-input development replay, not blind test or model promotion',
         'models':['v48_pressure','v51_pressure'],'decision':'three_class_argmax','new_fits':0,
         'selection':'Fixed retained v51 pressure and its exact v48 parent before answer access; no checkpoint/threshold search',
         'input_path':str(source),'input_sha256':sha(source),'input_columns':schema,
         'source_sha256':{Path(__file__).relative_to(ROOT).as_posix():sha(__file__)},
         'model_and_runtime_sha256':bindings,'python':sys.version,'sklearn':sklearn.__version__}
    save(out/'inference_contract.json',reg)
    seen=set();writer=None;offset=0;start=time.monotonic();parity_rows=0;routes=set()
    with threadpool_limits(limits=4):
        for batch in pq.ParquetFile(source).iter_batches(batch_size=args.batch_size,use_threads=False):
            df=batch.to_pandas()
            if df.event_id.isna().any() or df.event_id.duplicated().any() or any(x in seen for x in df.event_id):
                raise ValueError('Missing or duplicated input event_id')
            seen.update(df.event_id.tolist())
            # Exact repeated messages can share inference; this changes no row weights.
            messages=df.message_sanitized.fillna('').astype(str)
            inverse,unique=pd.factorize(messages,sort=False)
            q0,q1,meta=infer_messages(bundle,adapter,residual,unique.tolist())
            # Compare actual original raw-message entry on new routes and one
            # deterministic record per batch, never select by a label or score.
            take={0}
            for i,r in enumerate(meta.route):
                if r not in routes:take.add(i);routes.add(r)
            ix=sorted(take);records=[{'message_sanitized':unique[i]} for i in ix]
            qr,_=residual.classify_records(bundle,records)
            br,_=adapter.classify_records(bundle['base'],records)
            np.testing.assert_allclose(qr,q1[ix],rtol=0,atol=1e-12)
            np.testing.assert_allclose(br,q0[ix],rtol=0,atol=1e-12)
            # Add and change every unused metadata field: classification must be identical.
            variants=[dict(r,**{k:'changed-metadata' for k in INPUT_COLUMNS if k!='message_sanitized'}) for r in records]
            qv,_=residual.classify_records(bundle,variants)
            np.testing.assert_array_equal(qr,qv);parity_rows+=len(ix)
            result=meta.iloc[inverse].reset_index(drop=True)
            result.insert(0,'event_id',df.event_id.to_numpy())
            result.insert(0,'row_position',np.arange(offset,offset+len(df)))
            result['missing_product']=df.product_name.fillna('').astype(str).str.strip().eq('').to_numpy()
            result['empty_message']=messages.str.strip().eq('').to_numpy()
            for name,q in [('v48',q0[inverse]),('v51',q1[inverse])]:
                result[name+'_pred']=q.argmax(axis=1)
                for i,label in enumerate(['B','M','S']):result[name+'_p_'+label]=q[:,i]
                pd.DataFrame({'event_id':df.event_id.to_numpy(),'pred_label':LABELS[q.argmax(axis=1)]}).to_csv(
                    out/f'res_{name}_development.csv',mode='w' if offset==0 else 'a',header=offset==0,index=False)
            table=pa.Table.from_pandas(result,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(out/'predictions.parquet',table.schema,compression='zstd')
            writer.write_table(table);offset+=len(df)
            if offset%131072<len(df):print(json.dumps({'rows':offset,'seconds':round(time.monotonic()-start,1)}),flush=True)
    if writer:writer.close()
    assert offset==pq.ParquetFile(source).metadata.num_rows==len(seen)
    assert sha(source)==reg['input_sha256']
    for p,h in bindings.items():assert sha(ROOT/p)==h
    save(out/'inference_complete.json',{'rows':offset,'unique_ids':len(seen),'new_fits':0,'answers_read':False,
        'canonical_entry_and_metadata_parity_rows':parity_rows,'routes':sorted(routes),
        'seconds':time.monotonic()-start,'model_quality_accepted':False,
        'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'inference_complete':True,'rows':offset,'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
