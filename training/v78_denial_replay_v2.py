"""Frozen post-result syntax repair diagnostic on ALL records, no label-based gate."""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, read, save, sha, adapter, metrics, NAMES
from v78_boundary import DEST
from v78_denial_adapter_v2 import parse, self_check
from v75_views import view, byte_matrix
from v75_corrective import stable
from v75_metadata import encode

TARGET=DEST/'denial_adapter_v2'
MODELS=['K_base','O_sgd','O_lbfgs','shadow']


def infer():
    assert not TARGET.exists();TARGET.mkdir();checks=self_check()
    save(TARGET/'contract.json',{'new_fits':0,'thresholds_fitted':0,'scope':'Adaptive post-result parser hypothesis. Only literal syslog grammar + old unsupported route activates. No true labels read in parsing/inference; no source/product/time ID rule and no inferred interface roles.',
        'activation':'old parser route unsupported AND strict literal denial or documented PAN-OS CSV grammar','changed_blocks':'structured facts and record-vs-message port conflict flag only; raw byte text identical',
        'source_sha256':sha(__file__),'adapter_sha256':sha(ROOT/'training/v78_denial_adapter_v2.py'),'self_checks':checks,
        'model_sha256':{n:sha(DEST/(n+'.joblib')) for n in MODELS},'quality_promotion_allowed':False})
    v=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');models={n:joblib.load(DEST/(n+'.joblib')) for n in MODELS}
    ledger=[];counts={};start=time.monotonic()
    with threadpool_limits(limits=4):
        for dataset,filename in [('train','train.parquet'),('development','valid_input.parquet')]:
            offset=0;total=0
            for b in pq.ParquetFile(ROOT/'data/official'/filename).iter_batches(batch_size=8192,columns=['event_id','message_sanitized','src_port'],use_threads=False):
                df=b.to_pandas();chosen=[];facts=[];audits=[]
                for j,s in enumerate(df.message_sanitized):
                    candidate=parse(s)
                    if candidate is None:continue
                    prior=v.prepare_record({'message_sanitized':s})
                    if prior['route']!='unsupported':continue
                    chosen.append(j);facts.append(candidate['facts']);audits.append(candidate['audit'])
                if chosen:
                    sub=df.iloc[chosen];texts=[stable(view(s)[0]) for s in sub.message_sanitized]
                    fx=normalize(enc.transform(facts).astype(np.float32),norm='l2',copy=False)
                    extra,_,_=encode(sub.src_port.tolist(),[f['src_port_fixed'] for f in facts])
                    x=sparse.hstack([byte_matrix(texts),fx,extra],format='csr',dtype=np.float32)
                    predicted={name:(model.decision_function(x) if hasattr(model,'decision_function') else x@model['coef']+model['intercept']).argmax(1) for name,model in models.items()}
                    for k,j in enumerate(chosen):
                        ledger.append({'dataset':dataset,'row_position':offset+j,'event_id':str(df.event_id.iloc[j]),'facts_json':json.dumps(facts[k],sort_keys=True),
                            'audit_json':json.dumps(audits[k]),**{name+'_pred':int(a[k]) for name,a in predicted.items()}})
                    total+=len(chosen)
                offset+=len(df)
            counts[dataset]={'scanned_rows':offset,'changed_rows':total};print(dataset,counts[dataset],flush=True)
    pd.DataFrame(ledger).to_parquet(TARGET/'changed_predictions.parquet',index=False)
    save(TARGET/'inference_receipt.json',{'new_fits':0,'answers_read':False,'counts':counts,'seconds':time.monotonic()-start,
        'changed_sha256':sha(TARGET/'changed_predictions.parquet'),'nonmatching_rows':'Prediction remains original frozen model result by construction; no other records transformed.'})


def score():
    rec=read(TARGET/'inference_receipt.json');assert sha(TARGET/'changed_predictions.parquet')==rec['changed_sha256']
    ch=pd.read_parquet(TARGET/'changed_predictions.parquet');p=pd.read_parquet(DEST/'frozen_development/predictions.parquet');a=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    d=p.merge(a,on='event_id',validate='one_to_one');y=d.label_binary.map(dict(zip(NAMES,range(3)))).to_numpy()
    changed=ch[ch.dataset.eq('development')].copy();ix=changed.row_position.to_numpy();np.testing.assert_array_equal(d.event_id.iloc[ix].to_numpy(),changed.event_id.to_numpy())
    totals={};cells=[]
    for name in MODELS:
        before=d[name+'_pred'].to_numpy();after=before.copy();after[ix]=changed[name+'_pred'].to_numpy()
        totals[name]=metrics(np.bincount(y*3+after,minlength=9).reshape(3,3))
        for route in ['ALL','unsupported']:
            take=np.ones(len(d),bool) if route=='ALL' else d.route.eq(route).to_numpy();cm=np.bincount(y[take]*3+after[take],minlength=9).reshape(3,3)
            for k in range(3):
                n=int(cm[k].sum());pp=int(cm[:,k].sum());tp=int(cm[k,k]);cell=take&(y==k)
                cells.append({'model':name,'route':route,'label':NAMES[k],'support':n,'correct':tp,'recall':tp/n if n else None,'precision':tp/pp if pp else None,
                    'fixed':int((cell&(before!=y)&(after==y)).sum()),'broken':int((cell&(before==y)&(after!=y)).sum())})
    pd.DataFrame(cells).to_csv(TARGET/'classwise.csv',index=False)
    save(TARGET/'scoring.json',{'new_fits':0,'quality_acceptance':False,'scope':'Adaptive inspected-development diagnostic; parser proposal not externally validated, failed parent shadow gate unchanged.',
        'changed_true_label_counts':d.label_binary.iloc[ix].value_counts().to_dict(),'totals':totals,'source_sha256':sha(__file__)})
    print(pd.DataFrame(cells).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['infer','score']);a=p.parse_args();{'infer':infer,'score':score}[a.stage]()
