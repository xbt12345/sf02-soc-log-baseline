"""One prebound development replay to measure actual rare-class recovery; no tuning."""
import argparse
import json
import functools
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from scipy.special import expit
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, BATCH, NAMES, read, save, sha, adapter, metrics
from v78_boundary import DEST
from v75_views import view, byte_matrix
from v75_corrective import stable
from v75_metadata import encode

TARGET=DEST/'frozen_development'
MODEL_NAMES=['K_base','O_sgd','O_lbfgs','shadow']


def infer():
    assert not TARGET.exists();TARGET.mkdir()
    models={n:joblib.load(DEST/(n+'.joblib')) for n in MODEL_NAMES}
    inp=ROOT/'data/official/valid_input.parquet';pf=pq.ParquetFile(inp)
    from v73_inference import INPUT_COLUMNS
    assert set(pf.schema_arrow.names)==set(INPUT_COLUMNS)
    contract={'new_fits':0,'thresholds_fitted':0,'answers_read_during_inference':False,
        'purpose':'All four already frozen experimental models, including failed shadow, replayed once to measure whether internal repairs transfer to the 6226 and 2664 known difficulties. This is not candidate selection, not full-data retraining, and cannot override failed gates.',
        'model_sha256':{n:sha(DEST/(n+'.joblib')) for n in MODEL_NAMES},'input_sha256':sha(inp),'source_sha256':sha(__file__),
        'shadow_already_failed':not all(read(DEST/'shadow.json')['gates'].values()),'prior_private_answer_inspection':True}
    save(TARGET/'contract.json',contract)
    v=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');parse=functools.lru_cache(maxsize=2048)(lambda s:v.prepare_record({'message_sanitized':s}))
    offset=0;writer=None;start=time.monotonic()
    with threadpool_limits(limits=4):
        for batch in pf.iter_batches(batch_size=BATCH,use_threads=False):
            df=batch.to_pandas();inv,unique=pd.factorize(df.message_sanitized.fillna('').astype(str),sort=False)
            parsed=[parse(s) for s in unique];facts=[p['facts'] for p in parsed]
            fx=normalize(enc.transform(facts).astype(np.float32),norm='l2',copy=False)
            text=byte_matrix([stable(view(s)[0]) for s in unique]);x=sparse.hstack([text,fx],format='csr',dtype=np.float32)[inv]
            extra,_,_=encode(df.src_port.tolist(),np.array([f.get('src_port_fixed',65536) for f in facts])[inv]);x=sparse.hstack([x,extra],format='csr',dtype=np.float32)
            d={'row_position':np.arange(offset,offset+len(df)),'event_id':df.event_id.to_numpy(),'route':np.array([p['route'] for p in parsed])[inv]}
            for name,model in models.items():
                z=model.decision_function(x) if hasattr(model,'decision_function') else x@model['coef']+model['intercept']
                assert np.isfinite(z).all();d[name+'_pred']=z.argmax(1).astype(np.int8)
                for k in range(3):d[name+'_z'+str(k)]=z[:,k]
            t=pa.Table.from_pydict(d)
            if writer is None:writer=pq.ParquetWriter(TARGET/'predictions.parquet',t.schema,compression='zstd')
            writer.write_table(t);offset+=len(df)
            if offset%262144==0:print({'stage':'frozen_development','rows':offset,'seconds':round(time.monotonic()-start,1)},flush=True)
    writer.close();assert offset==2014052
    save(TARGET/'inference_receipt.json',{'rows':offset,'new_fits':0,'answers_read':False,'seconds':time.monotonic()-start,
        'prediction_sha256':sha(TARGET/'predictions.parquet'),'contract_sha256':sha(TARGET/'contract.json')})


def score():
    receipt=read(TARGET/'inference_receipt.json');assert sha(TARGET/'predictions.parquet')==receipt['prediction_sha256']
    p=pd.read_parquet(TARGET/'predictions.parquet');a=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    d=p.merge(a,on='event_id',how='left',validate='one_to_one',sort=False);assert len(d)==2014052 and d.label_binary.isin(NAMES).all()
    y=d.label_binary.map(dict(zip(NAMES,range(3)))).to_numpy();totals={};cells=[];paired=[]
    for name in MODEL_NAMES:
        pred=d[name+'_pred'].to_numpy();totals[name]=metrics(np.bincount(y*3+pred,minlength=9).reshape(3,3))
        for route in ['ALL']+sorted(d.route.unique().tolist()):
            take=np.ones(len(d),bool) if route=='ALL' else d.route.eq(route).to_numpy();cm=np.bincount(y[take]*3+pred[take],minlength=9).reshape(3,3)
            for k in range(3):
                n=int(cm[k].sum());pp=int(cm[:,k].sum());tp=int(cm[k,k]);cells.append({'model':name,'route':route,'label':NAMES[k],
                    'support':n,'correct':tp,'errors':n-tp,'recall':tp/n if n else None,'precision':tp/pp if pp else None,'f1':2*tp/(n+pp) if n+pp else None,
                    **{'pred_'+NAMES[j]:int(cm[k,j]) for j in range(3)}})
    for before,after in [('O_sgd','O_lbfgs'),('O_lbfgs','shadow')]:
        p0=d[before+'_pred'].to_numpy();p1=d[after+'_pred'].to_numpy()
        for route in ['ALL']+sorted(d.route.unique().tolist()):
            for k in range(3):
                take=(y==k)&(True if route=='ALL' else d.route.eq(route).to_numpy())
                paired.append({'before':before,'after':after,'route':route,'label':NAMES[k],'support':int(take.sum()),
                    'fixed':int((take&(p0!=y)&(p1==y)).sum()),'broken':int((take&(p0==y)&(p1!=y)).sum())})
    pd.DataFrame(cells).to_csv(TARGET/'classwise.csv',index=False);pd.DataFrame(paired).to_csv(TARGET/'paired.csv',index=False)
    save(TARGET/'scoring.json',{'new_fits':0,'quality_acceptance':False,'scope':'Frozen inspected-answer DEVELOPMENT diagnostic only; no post-score selection or threshold fitting. All models are partial-pool diagnostic fits, not official full-data models.',
        'totals':totals,'source_sha256':sha(__file__),'predictions_sha256':sha(TARGET/'predictions.parquet'),'answers_sha256':sha(ROOT/'data/official/valid_answer_private.parquet')})
    print(json.dumps(totals,indent=2));print(pd.DataFrame(cells).query("route in ['asa','unsupported','vpc_v2'] and label != 'benign'").to_string(index=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['infer','score']);a=ap.parse_args();{'infer':infer,'score':score}[a.stage]()
