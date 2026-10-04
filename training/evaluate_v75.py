"""Full official-input inference first; inspected-answer regression separately."""
import argparse
import functools
import json
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from run_v75 import ROOT,OUT,read,save,sha,adapter,metrics,NAMES
from v75_views import view,byte_matrix
from v75_metadata import encode as metadata_encode


def infer():
    dest=OUT/'official_replay'
    if dest.exists():raise FileExistsError(dest)
    receipt=read(OUT/'full/complete.json');cfg=read(OUT/'configuration.json')
    for name,h in receipt['outputs'].items():assert sha(OUT/'full'/name)==h
    for name,h in cfg['source_sha256'].items():assert sha(ROOT/'training'/name)==h
    v=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');model=joblib.load(OUT/'full/FULL.joblib')
    kind=cfg['arms'][receipt['full_reference_arm']][0]
    inp=ROOT/'data/official/valid_input.parquet';pf=pq.ParquetFile(inp)
    from v73_inference import INPUT_COLUMNS
    assert set(pf.schema_arrow.names)==set(INPUT_COLUMNS)
    dest.mkdir();start=time.monotonic();offset=0;writer=None
    @functools.lru_cache(maxsize=2048)
    def parse(s):return v.prepare_record({'message_sanitized':s})
    save(dest/'contract.json',{'answers_read':False,'input_sha256':sha(inp),'model_sha256':sha(OUT/'full/FULL.joblib'),
        'kind':kind,'capacity_only':receipt['capacity_only'],'source_sha256':sha(__file__),'new_fits':0})
    for batch in pf.iter_batches(batch_size=4096,use_threads=False):
        df=batch.to_pandas();messages=df.message_sanitized.fillna('').astype(str)
        inv,unique=pd.factorize(messages,sort=False)
        parsed=[parse(s) for s in unique];facts=[p['facts'] for p in parsed]
        texts=[p['text'] for p in parsed] if kind=='old' else [view(s)[0] for s in unique]
        fx=normalize(enc.transform(facts).astype(np.float32),norm='l2',copy=False)
        x=sparse.hstack([byte_matrix(texts),fx],format='csr',dtype=np.float32)
        x=x[inv]
        if kind=='new':
            mp=np.array([f.get('src_port_fixed',65536) for f in facts])[inv]
            extra,_,_=metadata_encode(df.src_port.tolist(),mp)
            x=sparse.hstack([x,extra],format='csr',dtype=np.float32)
        q=model.predict_proba(x);pred=q.argmax(1)
        assert np.isfinite(q).all();np.testing.assert_allclose(q.sum(1),1,atol=1e-5)
        data={'row_position':np.arange(offset,offset+len(df)),'event_id':df.event_id.to_numpy(),
              'route':np.array([p['route'] for p in parsed])[inv],'pred':pred.astype(np.int8)}
        for k in range(3):data['p'+str(k)]=q[:,k]
        t=pa.Table.from_pydict(data)
        if writer is None:writer=pq.ParquetWriter(dest/'predictions.parquet',t.schema,compression='zstd')
        writer.write_table(t)
        pd.DataFrame({'event_id':df.event_id,'pred_label':np.asarray(NAMES)[pred]}).to_csv(dest/'res_development.csv',
            mode='w' if offset==0 else 'a',header=offset==0,index=False)
        offset+=len(df)
        if offset%262144==0:print(json.dumps({'stage':'official_inference','rows':offset,'seconds':round(time.monotonic()-start,1)}),flush=True)
    writer.close();assert offset==pf.metadata.num_rows
    ids=pd.read_parquet(dest/'predictions.parquet',columns=['event_id']).event_id
    assert not ids.isna().any() and ids.is_unique
    save(dest/'complete.json',{'rows':offset,'new_fits':0,'answers_read':False,'seconds':time.monotonic()-start,
        'output_sha256':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}})
    print(json.dumps({'complete':True,'rows':offset,'seconds':time.monotonic()-start}))


def score():
    dest=OUT/'official_replay';rec=read(dest/'complete.json')
    for n,h in rec['output_sha256'].items():assert sha(dest/n)==h
    p=pd.read_parquet(dest/'predictions.parquet')
    ans=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    assert len(p)==len(ans)==2014052
    d=p.merge(ans,on='event_id',validate='one_to_one',how='left',sort=False)
    assert d.label_binary.isin(NAMES).all()
    baseline=pd.read_parquet(ROOT/'artifacts/v73_full_task_20260921_r2/predictions.parquet',columns=['event_id','v51_pred'])
    d=d.merge(baseline,on='event_id',validate='one_to_one',how='left',sort=False)
    y=d.label_binary.map(dict(zip(NAMES,range(3)))).to_numpy();pred=d.pred.to_numpy();before=d.v51_pred.to_numpy()
    def met(mask,pred):
        cm=np.zeros((3,3),np.int64);np.add.at(cm,(y[mask],pred[mask]),1);return metrics(cm)
    allrows=np.ones(len(d),bool);a=met(allrows,pred);b=met(allrows,before)
    slices=[]
    for route,z in d.groupby('route'):
        ix=z.index.to_numpy();mask=np.zeros(len(d),bool);mask[ix]=True
        slices.append({'route':route,'before':met(mask,before),'after':met(mask,pred),
            'fixed':int(((before!=y)&(pred==y)&mask).sum()),'broken':int(((before==y)&(pred!=y)&mask).sum())})
    gates={'errors_at_most_11173':a['errors']<=11173,'M_recall_plus_5pp':a['recall'][1]>=b['recall'][1]+.05,
        'macro_f1_increases':a['macro_f1']>b['macro_f1'],'S_recall_not_lower':a['recall'][2]>=b['recall'][2],
        'S_f1_not_lower':a['f1'][2]>=b['f1'][2],'normal_false_alerts_at_most_34':a['normal_false_alerts']<=34}
    save(dest/'scoring.json',{'scope':'Already inspected official private-answer DEVELOPMENT regression; no blind validation or selection after scoring.',
        'before':b,'after':a,'gates':gates,'all_full_task_gates_passed':all(gates.values()),'quality_promoted':False,
        'fixed':int(((before!=y)&(pred==y)).sum()),'broken':int(((before==y)&(pred!=y)).sum()),'slices':slices,
        'source_sha256':sha(__file__),'predictions_sha256':sha(dest/'predictions.parquet')})
    d.loc[(before!=y)|(pred!=y),['row_position','event_id','route','label_binary','v51_pred','pred']].to_parquet(dest/'errors.parquet',index=False)
    print(json.dumps({'before':b,'after':a,'gates':gates},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['infer','score']);a=p.parse_args()
    infer() if a.stage=='infer' else score()
