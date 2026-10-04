"""Two predeclared follow-up controls; no private answers and no changed labels."""
import argparse
import json
import re
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from run_v75 import (ROOT,OUT,BATCH,NAMES,check,read,save,sha,log,matrices,
                     features,score,new_model,SparseWriter,load_sparse)
from v75_views import byte_matrix,BYTE_FEATURES

DEST=OUT/'corrective'
EMBEDDED=re.compile(r'(?:USER|HOST|CRED|ORG)-[0-9]+(?:-[0-9]+)*')


def stable(text):
    # Only explicit synthetic identity tokens. Prefix/suffix and all other
    # numbers survive. Original spans and original file remain untouched.
    return EMBEDDED.sub(' <IDENTITY> ',text)


def prepare():
    check()
    if DEST.exists():raise FileExistsError(DEST)
    DEST.mkdir();start=time.monotonic()
    c=read(OUT/'configuration.json')
    contract={'arms':{'E':'D with remaining embedded synthetic identity tokens masked',
                      'F':'E plus fit-only rare format/class risk weighting'},
        'weight':'max(1,min(32,1000/n_fit_cell)); format/class cell counted from fit rows only; applied to ALL classes including normal',
        'weight_is_not_new_data':True,'original_rows_removed':0,'epochs':c['epochs'],
        'selection':'Fixed last epoch. Require lower errors and higher macro-F1, every class recall >= D, normal false alerts <= D. No checkpoint/threshold search.',
        'validation':'Same inspected internal development roles as four arms; additional selection is adaptive, not blind.',
        'answers_read':False,'source_sha256':sha(__file__),'parent_configuration_sha256':sha(OUT/'configuration.json')}
    save(DEST/'contract.json',contract)
    writer=SparseWriter(DEST/'text',BYTE_FEATURES);changed=0;count=0;overlay=None
    for batch in pq.ParquetFile(OUT/'text_dictionary.parquet').iter_batches(batch_size=1024,use_threads=False):
        df=batch.to_pandas();texts=[];entries=[]
        for tid,text in zip(df.text_id,df.text):
            matches=list(EMBEDDED.finditer(text));new=stable(text);texts.append(new)
            if matches:
                changed+=1
                entries.append({'text_id':int(tid),'parent_view_sha256':__import__('hashlib').sha256(text.encode()).hexdigest(),
                    'derived_view_sha256':__import__('hashlib').sha256(new.encode()).hexdigest(),
                    'parent_view_spans_json':json.dumps([[m.start(),m.end()] for m in matches])})
        writer.append(byte_matrix(texts));count+=len(df)
        if entries:
            t=pa.Table.from_pylist(entries)
            if overlay is None:overlay=pq.ParquetWriter(DEST/'identity_overlay.parquet',t.schema,compression='zstd')
            overlay.write_table(t)
    writer.close()
    if overlay is not None:overlay.close()
    save(DEST/'prepared.json',{'dictionary_rows':count,'changed_views':changed,'seconds':time.monotonic()-start,
        'files_sha256':{p.name:sha(p) for p in DEST.iterdir() if p.is_file()}})
    log(stage='corrective_prepared',views=count,changed=changed,seconds=time.monotonic()-start)


def train():
    c=check();contract=read(DEST/'contract.json');assert sha(__file__)==contract['source_sha256']
    for n,h in read(DEST/'prepared.json')['files_sha256'].items():assert sha(DEST/n)==h
    if (DEST/'complete.json').exists():raise FileExistsError(DEST/'complete.json')
    r=pd.read_parquet(OUT/'rows.parquet');m=matrices();m['new_text']=load_sparse(DEST/'text')
    mask=~r.is_validation.to_numpy();val=np.flatnonzero(~mask);y=r.label_index.to_numpy()
    counts=r[mask].groupby(['route','label_index']).size().to_dict()
    w=np.array([max(1,min(32,1000/counts[(route,int(label))])) if take else 0
                for route,label,take in zip(r.route,y,mask)],np.float32)
    support=[]
    for (route,label),n in counts.items():support.append({'route':route,'label':NAMES[label],'original_fit_rows':int(n),
        'weight':max(1,min(32,1000/n)),'effective_mass':n*max(1,min(32,1000/n))})
    save(DEST/'fit_weights.json',support)
    models={a:new_model(c) for a in ['E','F']};arms={a:['new','full'] for a in models}
    rng=np.random.default_rng(c['seed']);start=time.monotonic();curves=[];exposures={a:0 for a in models}
    with threadpool_limits(limits=4):
        for epoch in range(c['epochs']):
            seen=0
            for beg in rng.permutation(np.arange(0,len(r),BATCH)):
                ix=np.arange(beg,min(beg+BATCH,len(r)));rng.shuffle(ix);ix=ix[mask[ix]]
                if not len(ix):continue
                x=features(r,ix,m,'new')
                models['E'].partial_fit(x,y[ix],classes=np.arange(3))
                models['F'].partial_fit(x,y[ix],classes=np.arange(3),sample_weight=w[ix])
                seen+=len(ix)
            assert seen==int(mask.sum())
            for a in models:exposures[a]+=seen
            scores=score(models,r,m,val,arms);curves.append({'epoch':epoch+1,'validation':scores})
            save(DEST/'progress.json',curves);log(stage='corrective_epoch',epoch=epoch+1,validation=scores,seconds=time.monotonic()-start)
        for a,model in models.items():joblib.dump(model,DEST/(a+'.joblib'))
        writer=None
        for beg in range(0,len(r),BATCH):
            ix=np.arange(beg,min(beg+BATCH,len(r)));x=features(r,ix,m,'new');d={'row_position':ix}
            for a,model in models.items():d[a+'_pred']=model.predict(x).astype(np.int8)
            t=pa.Table.from_pydict(d)
            if writer is None:writer=pq.ParquetWriter(DEST/'predictions.parquet',t.schema,compression='zstd')
            writer.write_table(t)
        writer.close()
    base=read(OUT/'four_arm/complete.json')['curves'][-1]['validation']['D'];gates={}
    for a,z in scores.items():gates[a]={'fewer_errors':z['errors']<base['errors'],'macro_f1_higher':z['macro_f1']>base['macro_f1'],
        'recalls_not_lower':all(x>=b for x,b in zip(z['recall'],base['recall'])),
        'normal_false_alerts_not_higher':z['normal_false_alerts']<=base['normal_false_alerts']}
    eligible=[a for a,g in gates.items() if all(g.values())]
    save(DEST/'complete.json',{'new_fits':2,'selected':min(eligible,key=lambda a:scores[a]['errors']) if eligible else None,
        'gates':gates,'curves':curves,'total_original_exposures':exposures,'answers_read':False,'quality_promoted':False,
        'seconds':time.monotonic()-start,'files_sha256':{p.name:sha(p) for p in DEST.iterdir() if p.is_file()}})
    log(stage='corrective_complete',gates=gates)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train']);a=p.parse_args()
    prepare() if a.stage=='prepare' else train()
