"""Prediction-equivalent unique-row calculation, then frozen development scoring."""
import functools
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from v79_execute import DEST, selected_spec, check, emit
from run_v75 import ROOT, OUT, sha, save, metrics, adapter, NAMES
from v75_views import view,byte_matrix
from v75_corrective import stable
from v75_metadata import encode
from v78_denial_adapter_v4 import parse


def infer():
    check();selected,_,_=selected_spec();model=joblib.load(DEST/(selected+'.joblib'))
    v=adapter();enc=joblib.load(OUT/'facts_encoder.joblib')
    @functools.lru_cache(maxsize=8192)
    def prepare(s):
        p=v.prepare_record({'message_sanitized':s});patch=parse(s) if p['route']=='unsupported' else None
        return p['route'],patch['facts'] if patch else p['facts'],bool(patch),stable(view(s)[0])
    save(DEST/'replay_execution_binding.json',{'source_sha256':sha(__file__),'selected':selected,'selection_changed':False,
         'model_sha256':sha(DEST/(selected+'.joblib')),'input_sha256':sha(ROOT/'data/official/valid_input.parquet'),
         'change':'Factorize identical raw and cooked strings per batch; identical per-row parsing, facts, text and src_port encoding. Stopped slower inference before scoring.',
         'answers_read':False})
    writer=None;offset=0;activated=0;start=time.monotonic();parity=[]
    with threadpool_limits(limits=4):
        for batch in pq.ParquetFile(ROOT/'data/official/valid_input.parquet').iter_batches(batch_size=4096,columns=['event_id','message_sanitized','src_port'],use_threads=False):
            df=batch.to_pandas();messages=df.message_sanitized.fillna('').astype(str);inv,unique=pd.factorize(messages,sort=False)
            prepared=[prepare(s) for s in unique];facts=[p[1] for p in prepared]
            fx=normalize(enc.transform(facts).astype(np.float32),copy=False)
            # Byte matrix on unique cooked strings, raw records still retain independent facts.
            ti,tu=pd.factorize(np.array([p[3] for p in prepared],dtype=object),sort=False)
            text=byte_matrix(tu.tolist())[ti]
            block=sparse.hstack([text,fx],format='csr')[inv]
            meta,_,_=encode(df.src_port.tolist(),np.array([f.get('src_port_fixed',65536) for f in facts])[inv])
            xx=sparse.hstack([block,meta],format='csr');z=xx@model['coef']+model['intercept']
            if offset==0 or offset%262144==0:
                # Independent original non-factorized path, without answer access.
                take=np.arange(min(32,len(df)));rawfacts=[prepared[inv[i]][1] for i in take]
                rawfx=normalize(enc.transform(rawfacts).astype(np.float32),copy=False)
                rawmeta,_,_=encode(df.src_port.iloc[take].tolist(),[f.get('src_port_fixed',65536) for f in rawfacts])
                direct=sparse.hstack([byte_matrix([stable(view(messages.iloc[i])[0]) for i in take]),rawfx,rawmeta],format='csr')
                delta=direct-xx[take];maxdelta=float(abs(delta.data).max()) if delta.nnz else 0
                assert maxdelta==0;np.testing.assert_allclose(direct@model['coef']+model['intercept'],z[take],rtol=0,atol=1e-12)
                parity.append({'offset':offset,'rows':len(take),'feature_delta':maxdelta})
            assert np.isfinite(z).all()
            routes=np.array([p[0] for p in prepared])[inv];activated+=int(np.array([p[2] for p in prepared])[inv].sum())
            table=pa.Table.from_pydict({'event_id':df.event_id.to_numpy(),'route':routes,'prediction':z.argmax(1).astype(np.int8)})
            if writer is None:writer=pq.ParquetWriter(DEST/'development_predictions.parquet',table.schema,compression='zstd')
            writer.write_table(table);offset+=len(df)
            if offset%32768==0:emit(stage='frozen_development_inference',rows=offset,seconds=round(time.monotonic()-start,1))
    writer.close();assert offset==2014052
    save(DEST/'development_inference_receipt.json',{'selected':selected,'answers_read':False,'rows':offset,'adapter_activated':activated,
        'prediction_sha256':sha(DEST/'development_predictions.parquet'),'model_sha256':sha(DEST/(selected+'.joblib')),
        'seconds':time.monotonic()-start,'factorization_parity':parity})


def score():
    from run_v75 import read
    receipt=read(DEST/'development_inference_receipt.json');assert receipt['prediction_sha256']==sha(DEST/'development_predictions.parquet')
    selected=receipt['selected'];predictions=pd.read_parquet(DEST/'development_predictions.parquet');answers=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    d=predictions.merge(answers,on='event_id',validate='one_to_one',sort=False);assert len(d)==2014052
    y=d.label_binary.map(dict(zip(NAMES,range(3)))).to_numpy();assert d.label_binary.isin(NAMES).all();p=d.prediction.to_numpy()
    met=metrics(np.bincount(y*3+p,minlength=9).reshape(3,3));table=[];old=ROOT/'artifacts/v78_boundary_20260922'
    prior=pd.read_parquet(old/'frozen_development/predictions.parquet',columns=['event_id','O_sgd_pred']);b=prior.O_sgd_pred.to_numpy().copy()
    patch=pd.read_parquet(old/'denial_adapter_v4/changed_predictions.parquet');patch=patch[patch.dataset.eq('development')]
    b[patch.row_position.to_numpy()]=patch.O_sgd_pred.to_numpy();assert np.array_equal(d.event_id,prior.event_id)
    for route in ['ALL']+sorted(d.route.unique()):
        take=np.ones(len(d),bool) if route=='ALL' else d.route.eq(route).to_numpy();cm=np.bincount(y[take]*3+p[take],minlength=9).reshape(3,3);mm=metrics(cm)
        for k in range(3):
            cell=take&(y==k);table.append({'route':route,'label':NAMES[k],'support':int(cm[k].sum()),'correct':int(cm[k,k]),'recall':mm['recall'][k],'precision':mm['precision'][k],
                   'pred_B':int(cm[k,0]),'pred_M':int(cm[k,1]),'pred_S':int(cm[k,2]),'fixed':int((cell&(b!=y)&(p==y)).sum()),'broken':int((cell&(b==y)&(p!=y)).sum())})
    pd.DataFrame(table).to_csv(DEST/'development_classwise.csv',index=False)
    gates={'errors_le4340':met['errors']<=4340,'M_tp_ge11729':met['cm'][1][1]>=11729,'M_precision':met['precision'][1]>=.8666876277314888,
           'S_recall':met['recall'][2]>=.9568110421253123,'S_precision':met['precision'][2]>=.932814044903176,'normal_fp_le34':met['normal_false_alerts']<=34}
    save(DEST/'development_regression.json',{'selected':selected,'metrics':met,'gates':gates,'passed':all(gates.values()),
           'scope':'Previously inspected target; frozen diagnostic selection before inference, no tuning on target labels',
           'fixed':int(((b!=y)&(p==y)).sum()),'broken':int(((b==y)&(p!=y)).sum())})
    emit(stage='development_scored',metrics=met,gates=gates)


if __name__=='__main__':infer();score()
