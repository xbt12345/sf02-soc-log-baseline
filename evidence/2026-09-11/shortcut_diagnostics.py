from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
LABELS=['benign','malicious','suspicious']
cols=['timestamp','pipeline','product_name','src_host','src_ip','dst_ip','username','dst_host']
train=pd.read_parquet(ROOT/'data/official/train.parquet',columns=cols+['label_binary'])
valid=pd.read_parquet(ROOT/'data/official/valid_input.parquet',columns=['event_id']+cols)
ans=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
valid=valid.merge(ans,on='event_id',validate='one_to_one')
ytrain=pd.Categorical(train.label_binary,categories=LABELS).codes
yvalid=pd.Categorical(valid.label_binary,categories=LABELS).codes
result={'scope':'Diagnostic rules fit to training only; no threshold selection on validation answers. Not proposed as a validated final detector.','splits':{},'experiments':{}}
for name,df in [('train',train),('valid',valid)]:
    item={'by_label':{},'null_vs_label':{}}
    for c in ['dst_host','username','src_ip','dst_ip']:
        status=np.select([df[c].isna(),df[c].eq('')],['NULL','EMPTY'],default='VALUE')
        item['null_vs_label'][c]=pd.crosstab(status,df.label_binary).to_dict(orient='index')
    for label,g in df.groupby('label_binary'):
        ts=pd.to_datetime(g.timestamp,unit='s',utc=True)
        item['by_label'][label]={'rows':len(g),'time_min':str(ts.min()),'time_max':str(ts.max()),'days':ts.dt.strftime('%Y-%m-%d').value_counts().head(8).to_dict(),'nonempty_source_ips':g.src_ip.fillna('').replace('',np.nan).nunique(),'nonempty_src_hosts':g.src_host.fillna('').replace('',np.nan).nunique()}
    result['splits'][name]=item
    print(name,json.dumps(item,ensure_ascii=False,default=str),flush=True)

def metrics(y,p):
    return {'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0),'confusion_matrix':confusion_matrix(y,p,labels=[0,1,2]).tolist(),'report':classification_report(y,p,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)}

product_lookup=train.groupby('product_name').label_binary.agg(lambda x:x.value_counts().idxmax()).to_dict()
for name,df,y in [('train',train,ytrain),('valid',valid,yvalid)]:
    p=df.product_name.map(product_lookup).fillna('benign')
    result['experiments']['product_lookup_'+name]=metrics(y,pd.Categorical(p,categories=LABELS).codes)
    p=p.mask(df.dst_host.isna(),'malicious')
    result['experiments']['null_then_product_'+name]=metrics(y,pd.Categorical(p,categories=LABELS).codes)

for features in [['timestamp'],['timestamp','product_name']]:
    xtrain=train[features].copy()
    xvalid=valid[features].copy()
    if 'product_name' in features:
        cats=sorted(train.product_name.unique())
        xtrain['product_name']=pd.Categorical(xtrain.product_name,categories=cats).codes
        xvalid['product_name']=pd.Categorical(xvalid.product_name,categories=cats).codes
    tree=DecisionTreeClassifier(max_depth=4 if len(features)>1 else 2,min_samples_leaf=20,random_state=42)
    tree.fit(xtrain,ytrain)
    key='tree_'+'_'.join(features)
    result['experiments'][key]={'train':metrics(ytrain,tree.predict(xtrain)),'valid':metrics(yvalid,tree.predict(xvalid)),'tree':export_text(tree,feature_names=features),'product_codebook':cats if 'product_name' in features else None}

(OUT/'shortcut_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps(result['experiments'],ensure_ascii=False,default=str),flush=True)
