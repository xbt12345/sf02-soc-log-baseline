from pathlib import Path
import hashlib
import json
import sys
import time
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.common import load_config, prepare_route_a_features

start=time.perf_counter()
cfg=load_config(ROOT/'config.yaml')
model_path=ROOT/'artifacts/models/route_a_catboost.cbm'
model=CatBoostClassifier()
model.load_model(str(model_path))
print('Frozen model loaded; no fitting or tuning',flush=True)
data_path=ROOT/'data/official/valid_input.parquet'
df=pd.read_parquet(data_path,columns=[cfg['features']['id_col']]+cfg['features']['categorical_cols']+cfg['features']['datetime_cols'])
answers=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
df=df.merge(answers,on='event_id',how='left',validate='one_to_one')
assert df.label_binary.notna().all() and len(df)==len(answers)
y=df.label_binary
x,cat=prepare_route_a_features(df,cfg)
print('Data ready: '+str(x.shape),flush=True)
predict_start=time.perf_counter()
proba=model.predict_proba(x,thread_count=4)
pred=np.array(model.classes_)[proba.argmax(axis=1)]
predict_sec=time.perf_counter()-predict_start
labels=list(model.classes_)
y=pd.Series(pd.Categorical(y,categories=labels).codes,index=df.index)
pred=proba.argmax(axis=1)
codes=list(range(len(labels)))
importance=model.get_feature_importance()
metrics={'rows':len(df),'labels_order':labels,'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,labels=codes,average='macro'),'classification_report':classification_report(y,pred,labels=codes,target_names=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,pred,labels=codes).tolist(),'prediction_counts':pd.Series(np.array(labels)[pred]).value_counts().to_dict(),'per_class_AP':{label:average_precision_score(y.eq(i),proba[:,i]) for i,label in enumerate(labels)},'roc_auc_ovr_macro':roc_auc_score(y,proba,labels=codes,multi_class='ovr',average='macro'),'predict_seconds':predict_sec,'total_seconds':time.perf_counter()-start,'model_sha256':hashlib.file_digest(model_path.open('rb'),'sha256').hexdigest(),'model_tree_count':model.tree_count_,'model_features':model.feature_names_,'feature_importance':dict(sorted(zip(model.feature_names_,map(float,importance)),key=lambda t:-t[1])),'model_params':model.get_all_params(),'by_pipeline':{},'scope':'Frozen existing model, full current local validation, no fitting or threshold tuning. This is not an untouched final test.'}
for pipeline,idx in df.groupby('pipeline',dropna=False).groups.items():
    metrics['by_pipeline'][str(pipeline)]={'rows':len(idx),'label_counts':df.label_binary.loc[idx].value_counts().to_dict(),'macro_f1':f1_score(y.loc[idx],pred[idx],labels=codes,average='macro',zero_division=0),'classification_report':classification_report(y.loc[idx],pred[idx],labels=codes,target_names=labels,output_dict=True,zero_division=0)}
metrics['majority_baseline']={'accuracy':accuracy_score(y,np.repeat(labels.index('benign'),len(y))),'macro_f1':f1_score(y,np.repeat(labels.index('benign'),len(y)),labels=codes,average='macro',zero_division=0)}
error_df=df.loc[y.ne(pred),['product_name','pipeline','label_binary']].copy()
error_df['pred_label']=np.array(labels)[pred[y.ne(pred)]]
metrics['errors_by_product']=error_df.groupby(['product_name','label_binary','pred_label'],dropna=False).size().reset_index(name='rows').to_dict(orient='records')
mal_code=labels.index('malicious')
metrics['malicious_score_diagnostics']={'min_true_malicious_score':float(proba[y.eq(mal_code),mal_code].min()),'max_nonmalicious_score':float(proba[y.ne(mal_code),mal_code].max()),'note':'Observed separation only; no threshold selected from these answers.'}
(Path(__file__).parent/'existing_model_full_valid.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps(metrics,ensure_ascii=False,default=str),flush=True)
