"""Official-only four-arm execution, disk-backed features, exact row exposure."""
import argparse
import collections
import functools
import gc
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
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import v75_views as views

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v75_four_arm_20260921_r2'
FROZEN=ROOT/'artifacts/v51_fact_residual_20260914'
NAMES=['benign','malicious','suspicious']
BATCH=4096


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def log(**x):print(json.dumps(x,ensure_ascii=False),flush=True)


def adapter():
    cfg=read(FROZEN/'configuration.json')
    for name,h in cfg['runtime_bindings'].items():assert sha(FROZEN/'runtime'/name)==h
    sys.path.insert(0,str(FROZEN/'runtime'))
    import v48_input
    return v48_input


class SparseWriter:
    def __init__(self,stem,width):
        self.stem=Path(stem);self.width=width;self.rows=0;self.nnz=0
        self.data=open(str(stem)+'.data','wb');self.indices=open(str(stem)+'.indices','wb')
        self.ptr=open(str(stem)+'.indptr','wb');np.array([0],dtype='<i8').tofile(self.ptr)
    def append(self,x):
        x=x.tocsr();x.sort_indices()
        x.data.astype('<f4').tofile(self.data);x.indices.astype('<i4').tofile(self.indices)
        (x.indptr[1:].astype('<i8')+self.nnz).tofile(self.ptr)
        self.rows+=x.shape[0];self.nnz+=x.nnz
    def close(self):
        for f in (self.data,self.indices,self.ptr):f.close()
        save(str(self.stem)+'.json',{'shape':[self.rows,self.width],'nnz':self.nnz})


def load_sparse(stem):
    m=read(str(stem)+'.json')
    return sparse.csr_matrix((np.memmap(str(stem)+'.data',dtype='<f4',mode='r'),
        np.memmap(str(stem)+'.indices',dtype='<i4',mode='r'),
        np.memmap(str(stem)+'.indptr',dtype='<i8',mode='r')),shape=m['shape'],copy=False)


def components(body,source):
    parent=np.arange(int(body.max())+1,dtype=np.int32)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    first={}
    pairs=pd.DataFrame({'b':body,'s':source});pairs=pairs[pairs.s>=0].drop_duplicates()
    for b,s in pairs.itertuples(index=False,name=None):
        if s in first:
            a=find(b);c=find(first[s])
            if a!=c:parent[max(a,c)]=min(a,c)
        else:first[s]=b
    for a in range(len(parent)):parent[a]=find(a)
    return parent[body]


def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();v=adapter();start=time.monotonic()
    rpath=ROOT/'artifacts/v39_local_r2_20260913/prepared/rows.parquet'
    ppath=rpath.parent/'projections.parquet';train=ROOT/'data/official/train.parquet'
    oldreceipt=read(ROOT/'artifacts/v74_train_support_audit_20260921/support.json')
    assert sha(train)==oldreceipt['input_sha256']['data/official/train.parquet']
    assert sha(rpath)==oldreceipt['input_sha256'][rpath.relative_to(ROOT).as_posix()]
    r=pd.read_parquet(rpath,columns=['row_position','event_id','label_index','route','body_group','projection_id','original_empty'])
    p=pd.read_parquet(ppath,columns=['text','facts']);newp=[];patchmap={}
    fitpath=FROZEN/'pressure/training_manifest.parquet'
    fitset=set(pd.read_parquet(fitpath,columns=['row_position']).row_position)
    r['old_fit']=r.row_position.isin(fitset)
    newids=np.empty(len(r),np.int32);sourceids=np.full(len(r),-1,np.int32)
    source_map={};textmap={};pending=[];ledger_writer=None;text_writer=None
    sw=SparseWriter(OUT/'new_text',views.BYTE_FEATURES)
    span_counts=collections.Counter();route_raw=collections.Counter();offset=0
    transform=functools.lru_cache(maxsize=2048)(views.view)
    for b in pq.ParquetFile(train).iter_batches(batch_size=BATCH,use_threads=False,
            columns=['event_id','label_binary','src_ip','message_sanitized']):
        df=b.to_pandas(); rr=r.iloc[offset:offset+len(df)]
        np.testing.assert_array_equal(rr.event_id,df.event_id)
        np.testing.assert_array_equal(rr.label_index,df.label_binary.map(dict(zip(NAMES,range(3)))))
        traces=[]
        for j,(raw,src) in enumerate(zip(df.message_sanitized,df.src_ip)):
            pos=offset+j;raw=None if pd.isna(raw) else raw
            cooked,a=transform(raw);key=hashlib.sha256(cooked.encode()).digest()
            if key not in textmap:
                tid=len(textmap);textmap[key]=tid;pending.append((tid,cooked))
            tid=textmap[key];newids[pos]=tid
            if isinstance(src,str) and src.strip():
                if src not in source_map:source_map[src]=len(source_map)
                sourceids[pos]=source_map[src]
            route=r.route.iat[pos];route_raw[(route,'null' if raw is None else 'empty' if not raw else 'nonempty')]+=1
            for a0,b0,kind in a['spans']:span_counts[kind]+=b0-a0
            traces.append({'row_position':pos,'raw_is_null':a['is_null'],'raw_sha256':a['sha256'],
                'raw_length':a['length'],'spans_json':json.dumps(a['spans']), 'new_text_id':tid})
            if route in ('native_flow','native_firewall'):
                actual=v.prepare_record({'message_sanitized':raw});oldid=int(r.projection_id.iat[pos])
                facts=v.old.canonical(actual['facts'])
                if actual['text']!=p.text.iat[oldid] or facts!=v.old.canonical(json.loads(p.facts.iat[oldid])):
                    k=(actual['text'],facts)
                    if k not in patchmap:patchmap[k]=len(p)+len(newp);newp.append({'text':k[0],'facts':k[1]})
                    r.at[pos,'projection_id']=patchmap[k]
        t=pa.Table.from_pylist(traces)
        if ledger_writer is None:ledger_writer=pq.ParquetWriter(OUT/'raw_ledger.parquet',t.schema,compression='zstd')
        ledger_writer.write_table(t)
        if pending:
            sw.append(views.byte_matrix([x[1] for x in pending]))
            t=pa.Table.from_pylist([{'text_id':x[0],'text':x[1]} for x in pending])
            if text_writer is None:text_writer=pq.ParquetWriter(OUT/'text_dictionary.parquet',t.schema,compression='zstd')
            text_writer.write_table(t);pending=[]
        offset+=len(df)
        if offset%131072==0:log(stage='prepare',rows=offset,unique_texts=len(textmap),seconds=round(time.monotonic()-start,1))
    ledger_writer.close();text_writer.close();sw.close();transform.cache_clear();textmap.clear()
    assert offset==len(r)
    r['new_text_id']=newids;r['source_symbol']=sourceids
    r['component']=components(r.body_group.to_numpy(),sourceids)
    u=np.unique(r.component);fold_map={int(k):int(hashlib.sha256(('v75:'+str(k)).encode()).hexdigest()[:8],16)%5 for k in u}
    r['fold']=r.component.map(fold_map).astype(np.int8)
    # One-component route/class slices remain fit-only in this diagnostic fold;
    # no invented cross-source holdout. The complete component moves together.
    forced=set()
    for _,z in r.groupby(['route','label_index']):
        c=z.component.unique()
        if len(c)==1:forced.add(int(c[0]))
    r.loc[r.component.isin(forced),'fold']=-1
    r['is_validation']=r.fold.eq(0)
    assert r.is_validation.any() and (~r.is_validation).any()
    assert r.groupby('body_group').is_validation.nunique().max()==1
    assert r.loc[r.source_symbol>=0].groupby('source_symbol').is_validation.nunique().max()==1
    assert set(r.loc[r.is_validation,'label_index'])=={0,1,2}
    r.to_parquet(OUT/'rows.parquet',index=False)
    p=pd.concat([p,pd.DataFrame(newp)],ignore_index=True);p.to_parquet(OUT/'projections.parquet',index=False)
    enc=v.old.SemanticFacts()
    # Fixed schema and row normalization, no validation-fitted scale/vocabulary.
    fw=None;ow=SparseWriter(OUT/'old_text',views.BYTE_FEATURES)
    for a in range(0,len(p),512):
        z=p.iloc[a:a+512];fx=enc.transform([json.loads(t) for t in z.facts]).astype(np.float32)
        fx=normalize(fx,norm='l2',copy=False)
        if fw is None:fw=SparseWriter(OUT/'facts',fx.shape[1])
        fw.append(fx);ow.append(views.byte_matrix(z.text.tolist()))
    fw.close();ow.close();joblib.dump(enc,OUT/'facts_encoder.joblib')
    support=[]
    for (route,y),z in r.groupby(['route','label_index']):
        support.append({'route':route,'label':NAMES[int(y)],'official_rows':len(z),
            'components':int(z.component.nunique()),'source_symbols':int(z.loc[z.source_symbol>=0,'source_symbol'].nunique()),
            'old_fit_rows':int((z.old_fit&~z.is_validation).sum()),'full_fit_rows':int((~z.is_validation).sum()),
            'validation_rows':int(z.is_validation.sum()),'fit_only':not bool(z.is_validation.any())})
    (OUT/'runtime').mkdir()
    for name in ['run_v75.py','v75_views.py']:(OUT/'runtime'/name).write_bytes((ROOT/'training'/name).read_bytes())
    cfg={'version':'v75-four-arm-1','new_classifier_family':'SGD logistic one-vs-rest, equal original row exposure',
        'feature_contract':'Fixed collision-free UTF8 byte 1/2-gram counts + frozen-schema normalized facts; no OOV or truncation, NOT sequence-lossless',
        'feature_deviation_from_v74':'Memory-bounded exact byte ngrams chosen instead of fitted character vocabulary; all four arms share this encoder; no hash collisions.',
        'arms':{'A':['old','old'],'B':['new','old'],'C':['old','full'],'D':['new','full']},
        'epochs':6,'alpha':1e-6,'eta0':0.05,'seed':751,'batch_size':BATCH,
        'selection':'C/D must beat A validation errors and macro-F1 without class-recall or normal-FP regression; choose smallest errors. Otherwise full D is capacity-only, not promotion.',
        'full_fit':'All official rows including all 32596 native_flow malicious, no duplicates removed; internal fold scores precede this fit.',
        'support':support,'forced_fit_only_components':sorted(forced),'rows':len(r),'validation_rows':int(r.is_validation.sum()),
        'span_character_counts':dict(span_counts),'raw_message_states':[{'route':k[0],'state':k[1],'rows':n} for k,n in route_raw.items()],
        'unknown_fields':'Residual may still carry unrecognized identity/decision proxies; full invariance not claimed.',
        'input_sha256':{x.relative_to(ROOT).as_posix():sha(x) for x in [train,rpath,ppath,fitpath]},
        'source_sha256':{name:sha(OUT/'runtime'/name) for name in ['run_v75.py','v75_views.py']},
        'prepared_sha256':{x.name:sha(x) for x in OUT.iterdir() if x.is_file()},
        'answers_read':False,'seconds':time.monotonic()-start}
    save(OUT/'configuration.json',cfg);log(stage='prepared',rows=len(r),validation_rows=int(r.is_validation.sum()),native_flow=int(r.route.eq('native_flow').sum()),seconds=cfg['seconds'])


def check():
    c=read(OUT/'configuration.json')
    for n,h in c['source_sha256'].items():assert sha(ROOT/'training'/n)==h
    for n,h in c['prepared_sha256'].items():assert sha(OUT/n)==h,n
    return c


def matrices():return {n:load_sparse(OUT/n) for n in ['old_text','new_text','facts','metadata']}


def features(r,idx,m,kind):
    pid=r.projection_id.to_numpy()[idx]
    tid=pid if kind=='old' else r.new_text_id.to_numpy()[idx]
    blocks=[m[kind+'_text'][tid],m['facts'][pid]]
    if kind=='new':blocks.append(m['metadata'][idx])
    return sparse.hstack(blocks,format='csr',dtype=np.float32)


def metrics(cm):
    cm=np.asarray(cm);s=cm.sum(1);p=cm.sum(0);d=cm.diagonal()
    rec=np.divide(d,s,out=np.zeros(3),where=s>0);pr=np.divide(d,p,out=np.zeros(3),where=p>0)
    f1=np.divide(2*d,s+p,out=np.zeros(3),where=(s+p)>0)
    return {'rows':int(cm.sum()),'errors':int(cm.sum()-d.sum()),'accuracy':float(d.sum()/cm.sum()) if cm.sum() else None,
        'macro_f1':float(f1.mean()) if (s>0).all() else None,'support':s.tolist(),
        'recall':[float(rec[i]) if s[i] else None for i in range(3)],'precision':[float(pr[i]) if p[i] else None for i in range(3)],
        'f1':f1.tolist(),'normal_false_alerts':int(cm[0,1:].sum()),'cm':cm.tolist()}


def score(models,r,m,indices,arms):
    cm={a:np.zeros((3,3),np.int64) for a in models}
    for start in range(0,len(indices),BATCH):
        ix=indices[start:start+BATCH];y=r.label_index.to_numpy()[ix]
        for kind in ['old','new']:
            wanted=[a for a in models if arms[a][0]==kind]
            if not wanted:continue
            x=features(r,ix,m,kind)
            for a in wanted:np.add.at(cm[a],(y,models[a].predict(x)),1)
    return {a:metrics(v) for a,v in cm.items()}


def new_model(c):
    return SGDClassifier(loss='log_loss',penalty='l2',alpha=c['alpha'],learning_rate='constant',
        eta0=c['eta0'],average=True,random_state=c['seed'],shuffle=False)


def train(full=False):
    c=check();r=pd.read_parquet(OUT/'rows.parquet');m=matrices();start=time.monotonic()
    target=OUT/('full' if full else 'four_arm')
    if target.exists():raise FileExistsError(target)
    target.mkdir()
    if full:
        previous=read(OUT/'four_arm/complete.json');chosen=previous['selected'] or 'D'
        arms={'FULL':c['arms'][chosen]};mask={'FULL':np.ones(len(r),bool)}
    else:
        arms=c['arms'];mask={a:(~r.is_validation.to_numpy())&(r.old_fit.to_numpy() if pool=='old' else True) for a,(_,pool) in arms.items()}
    models={a:new_model(c) for a in arms};counts={a:0 for a in arms};curves=[]
    val=np.flatnonzero(r.is_validation);y=r.label_index.to_numpy();rng=np.random.default_rng(c['seed'])
    block_starts=np.arange(0,len(r),BATCH)
    with threadpool_limits(limits=4):
        for epoch in range(c['epochs']):
            seen={a:0 for a in arms}
            for bi,beg in enumerate(rng.permutation(block_starts)):
                ix=np.arange(beg,min(beg+BATCH,len(r)));rng.shuffle(ix)
                for kind in ['old','new']:
                    wanted=[a for a in arms if arms[a][0]==kind]
                    if not wanted:continue
                    # Do not transform rows unused by every model of this view.
                    keep=np.logical_or.reduce([mask[a][ix] for a in wanted]);jx=ix[keep]
                    if not len(jx):continue
                    x=features(r,jx,m,kind)
                    for a in wanted:
                        take=mask[a][jx]
                        if take.any():models[a].partial_fit(x[take],y[jx[take]],classes=np.arange(3));seen[a]+=int(take.sum())
                if bi and bi%150==0:log(stage='full_fit' if full else 'four_fit',epoch=epoch+1,blocks=bi,seconds=round(time.monotonic()-start,1))
            for a in arms:assert seen[a]==int(mask[a].sum());counts[a]+=seen[a]
            measure=score(models,r,m,val,arms) if not full else None
            curves.append({'epoch':epoch+1,'exposures':seen,'validation':measure})
            save(target/'progress.json',curves)
            log(stage='full_epoch' if full else 'four_epoch',epoch=epoch+1,validation=measure,seconds=round(time.monotonic()-start,1))
        for a,model in models.items():joblib.dump(model,target/(a+'.joblib'))
        # Complete real-row predictions enable per-route and fit/holdout audits.
        writer=None
        for beg in range(0,len(r),BATCH):
            ix=np.arange(beg,min(beg+BATCH,len(r)));d={'row_position':ix}
            for kind in ['old','new']:
                wanted=[a for a in models if arms[a][0]==kind]
                if not wanted:continue
                x=features(r,ix,m,kind)
                for a in wanted:
                    prob=models[a].predict_proba(x);d[a+'_pred']=prob.argmax(1).astype(np.int8)
                    for k in range(3):d[a+'_p'+str(k)]=prob[:,k].astype(np.float32)
            t=pa.Table.from_pydict(d)
            if writer is None:writer=pq.ParquetWriter(target/'predictions.parquet',t.schema,compression='zstd')
            writer.write_table(t)
        writer.close()
    selected=None;gates={}
    if not full:
        scores=curves[-1]['validation'];a=scores['A']
        for name in ['C','D']:
            z=scores[name]
            gates[name]={'errors_lower':z['errors']<a['errors'],'macro_f1_higher':z['macro_f1']>a['macro_f1'],
                'class_recalls_not_lower':all(zi>=ai for zi,ai in zip(z['recall'],a['recall'])),
                'normal_false_alerts_not_higher':z['normal_false_alerts']<=a['normal_false_alerts']}
        passed=[n for n,g in gates.items() if all(g.values())]
        if passed:selected=min(passed,key=lambda n:scores[n]['errors'])
    support=[]
    for a in arms:
        for (route,yi),z in r[mask[a]].groupby(['route','label_index']):
            support.append({'arm':a,'route':route,'label':NAMES[int(yi)],'fit_rows':len(z),'epochs':c['epochs']})
    if full:
        assert mask['FULL'].all() and int(((r.route=='native_flow')&(r.label_index==1)).sum())==32596
    result={'new_fits':len(models),'full_fit':full,'selected':selected,'gates':gates,'curves':curves,'fit_support':support,
        'total_exposures':counts,'all_original_rows_weight_one':True,'duplicates_removed':0,
        'full_reference_arm':chosen if full else None,'capacity_only':bool(full and previous['selected'] is None),
        'quality_promotion':False,'answers_read':False,'seconds':time.monotonic()-start,
        'configuration_sha256':sha(OUT/'configuration.json'),'outputs':{p.name:sha(p) for p in target.iterdir() if p.is_file()}}
    save(target/'complete.json',result);log(stage='training_complete',full=full,new_fits=len(models),selected=selected,seconds=result['seconds'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train','full']);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    else:train(args.stage=='full')
