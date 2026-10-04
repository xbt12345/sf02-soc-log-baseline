"""Full F diagnostic only: failed internal M-recall guard, no promotion."""
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
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,OUT,BATCH,NAMES,check,read,save,sha,log,adapter,matrices,features,new_model,load_sparse,metrics
from v75_corrective import DEST as CORRECTIVE,stable
from v75_views import view,byte_matrix
from v75_metadata import encode
DEST=OUT/'F_diagnostic'


def fit():
    c=check();r=pd.read_parquet(OUT/'rows.parquet');m=matrices();m['new_text']=load_sparse(CORRECTIVE/'text')
    if DEST.exists():raise FileExistsError(DEST)
    prior=read(CORRECTIVE/'complete.json');assert prior['selected'] is None
    DEST.mkdir();save(DEST/'contract.json',{'kind':'F full fit for diagnostic replay only',
        'reason':'Rare slice gains but failed predeclared internal malicious recall protection; no promotion and no changed gate.',
        'answers_read':False,'quality_acceptance':False,'source_sha256':sha(__file__),
        'corrective_contract_sha256':sha(CORRECTIVE/'contract.json'),'parent_configuration_sha256':sha(OUT/'configuration.json'),
        'weight_formula':'max(1,min(32,1000/n_route_class_full_train))','epochs':c['epochs'],
        'no_tuning_after_private_scoring':True})
    counts=r.groupby(['route','label_index']).size().to_dict();y=r.label_index.to_numpy()
    w=np.array([max(1,min(32,1000/counts[(route,int(label))])) for route,label in zip(r.route,y)],np.float32)
    save(DEST/'fit_support.json',[{'route':k[0],'label':NAMES[k[1]],'original_rows':int(n),
        'weight':max(1,min(32,1000/n))} for k,n in counts.items()])
    model=new_model(c);rng=np.random.default_rng(c['seed']);exposures=0;start=time.monotonic()
    with threadpool_limits(limits=4):
        for ep in range(c['epochs']):
            seen=0
            for beg in rng.permutation(np.arange(0,len(r),BATCH)):
                ix=np.arange(beg,min(beg+BATCH,len(r)));rng.shuffle(ix)
                model.partial_fit(features(r,ix,m,'new'),y[ix],classes=np.arange(3),sample_weight=w[ix]);seen+=len(ix)
            assert seen==len(r);exposures+=seen
            log(stage='F_full_diagnostic_epoch',epoch=ep+1,seconds=time.monotonic()-start)
    assert counts[('native_flow',1)]==32596;joblib.dump(model,DEST/'model.joblib')
    save(DEST/'fit_complete.json',{'new_fits':1,'rows':len(r),'exposures':exposures,'native_flow_malicious_rows':32596,
        'duplicates_removed':0,'answers_read':False,'quality_acceptance':False,'seconds':time.monotonic()-start,
        'model_sha256':sha(DEST/'model.joblib'),'contract_sha256':sha(DEST/'contract.json')})


def infer():
    contract=read(DEST/'contract.json');assert contract['source_sha256']==sha(__file__)
    rec=read(DEST/'fit_complete.json');assert sha(DEST/'model.joblib')==rec['model_sha256']
    assert not (DEST/'predictions.parquet').exists()
    v=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');model=joblib.load(DEST/'model.joblib')
    inp=ROOT/'data/official/valid_input.parquet';pf=pq.ParquetFile(inp)
    from v73_inference import INPUT_COLUMNS
    assert set(pf.schema_arrow.names)==set(INPUT_COLUMNS)
    offset=0;writer=None;start=time.monotonic()
    @functools.lru_cache(maxsize=2048)
    def parse(s):return v.prepare_record({'message_sanitized':s})
    for batch in pf.iter_batches(batch_size=BATCH,use_threads=False):
        df=batch.to_pandas();inv,unique=pd.factorize(df.message_sanitized.fillna('').astype(str),sort=False)
        parsed=[parse(s) for s in unique];facts=[p['facts'] for p in parsed]
        texts=[stable(view(s)[0]) for s in unique]
        fx=normalize(enc.transform(facts).astype(np.float32),norm='l2',copy=False)
        x=sparse.hstack([byte_matrix(texts),fx],format='csr',dtype=np.float32)[inv]
        mp=np.array([f.get('src_port_fixed',65536) for f in facts])[inv]
        extra,_,_=encode(df.src_port.tolist(),mp);x=sparse.hstack([x,extra],format='csr',dtype=np.float32)
        q=model.predict_proba(x);pred=q.argmax(1)
        assert np.isfinite(q).all();np.testing.assert_allclose(q.sum(1),1,atol=1e-5)
        data={'row_position':np.arange(offset,offset+len(df)),'event_id':df.event_id.to_numpy(),
            'route':np.asarray([p['route'] for p in parsed])[inv],'pred':pred.astype(np.int8)}
        for k in range(3):data['p'+str(k)]=q[:,k]
        t=pa.Table.from_pydict(data)
        if writer is None:writer=pq.ParquetWriter(DEST/'predictions.parquet',t.schema,compression='zstd')
        writer.write_table(t)
        pd.DataFrame({'event_id':df.event_id,'pred_label':np.asarray(NAMES)[pred]}).to_csv(DEST/'res_development.csv',
            mode='w' if offset==0 else 'a',header=offset==0,index=False)
        offset+=len(df)
        if offset%524288==0:log(stage='F_diagnostic_infer',rows=offset,seconds=time.monotonic()-start)
    writer.close();assert offset==2014052
    save(DEST/'inference_complete.json',{'rows':offset,'answers_read':False,'new_fits':0,'seconds':time.monotonic()-start,
        'input_sha256':sha(inp),'outputs_sha256':{n:sha(DEST/n) for n in ['predictions.parquet','res_development.csv']}})
    log(stage='F_diagnostic_inference_complete',rows=offset)


def score():
    rec=read(DEST/'inference_complete.json')
    for n,h in rec['outputs_sha256'].items():assert sha(DEST/n)==h
    p=pd.read_parquet(DEST/'predictions.parquet');ans=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    assert len(p)==len(ans)==2014052
    d=p.merge(ans,on='event_id',validate='one_to_one',how='left',sort=False);assert d.label_binary.isin(NAMES).all()
    before=pd.read_parquet(OUT/'official_replay/predictions.parquet',columns=['event_id','pred']).rename(columns={'pred':'D_pred'})
    d=d.merge(before,on='event_id',validate='one_to_one',how='left',sort=False)
    y=d.label_binary.map(dict(zip(NAMES,range(3)))).to_numpy();a=d.pred.to_numpy();b=d.D_pred.to_numpy()
    def met(ix,pred):
        cm=np.bincount(y[ix]*3+pred[ix],minlength=9).reshape(3,3);return metrics(cm)
    totals={k:met(np.arange(len(d)),pr) for k,pr in [('D',b),('F',a)]};cells=[]
    for route,z in d.groupby('route'):
        ix=z.index.to_numpy();cells.append({'route':route,'D':met(ix,b),'F':met(ix,a),
            'fixed':int(((b[ix]!=y[ix])&(a[ix]==y[ix])).sum()),'broken':int(((b[ix]==y[ix])&(a[ix]!=y[ix])).sum())})
    csv=pd.read_csv(DEST/'res_development.csv');original=pd.read_parquet(ROOT/'data/official/valid_input.parquet',columns=['event_id'])
    np.testing.assert_array_equal(csv.event_id,p.event_id);np.testing.assert_array_equal(original.event_id,p.event_id)
    np.testing.assert_array_equal(csv.pred_label,np.asarray(NAMES)[p.pred]);assert p.event_id.is_unique
    baseline=read(OUT/'official_replay/scoring.json')['before'];f=totals['F']
    gates={'errors_at_most_11173':f['errors']<=11173,'M_recall_plus_5pp':f['recall'][1]>=baseline['recall'][1]+.05,
        'macro_f1_increases':f['macro_f1']>baseline['macro_f1'],'S_recall_not_lower':f['recall'][2]>=baseline['recall'][2],
        'S_f1_not_lower':f['f1'][2]>=baseline['f1'][2],'normal_false_alerts_at_most_34':f['normal_false_alerts']<=34}
    result={'scope':'Inspected-answer full DEVELOPMENT diagnostic, internal F gate failed; no blind or promotion claim.',
        'totals':totals,'slices':cells,'quality_acceptance':False,'whole_task_gates':gates,
        'csv_ids_labels_and_original_order_match':True,'source_sha256':sha(__file__)}
    save(DEST/'scoring.json',result);print(json.dumps({'totals':totals,'gates':gates},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','infer','score']);a=p.parse_args()
    {'fit':fit,'infer':infer,'score':score}[a.stage]()
