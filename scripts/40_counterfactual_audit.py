"""Frozen baseline representation stress audit; no fitting or threshold tuning."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common import load_config, prepare_route_a_features

def main():
    cfg=load_config(ROOT/'config.yaml')
    path=ROOT/'data/official/valid_input.parquet'
    columns=['event_id']+cfg['features']['categorical_cols']+cfg['features']['datetime_cols']
    full=pd.read_parquet(path,columns=columns)
    df=full.sample(n=min(100_000,len(full)),random_state=20260911).reset_index(drop=True)
    del full
    answers=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    df=df.merge(answers,on='event_id',validate='one_to_one',how='left')
    assert df.label_binary.notna().all()
    model_path=ROOT/'artifacts/models/route_a_catboost.cbm'
    model=CatBoostClassifier().load_model(str(model_path))
    labels=list(model.classes_)
    y=pd.Categorical(df.label_binary,categories=labels).codes
    cat_cols=cfg['features']['categorical_cols']
    entities=['src_ip','dst_ip','src_host','dst_host','username']

    def perturb(data, mode):
        out=data.copy()
        if mode in ['null_to_empty','combined_representation']:
            for col in cat_cols:
                out[col]=out[col].fillna('')
        if mode in ['shift_364_days','combined_representation']:
            out.timestamp=out.timestamp+364*86400
        if mode in ['entity_relabel','combined_representation']:
            mapping={v:'ENTITY_'+hashlib.sha256(v.encode()).hexdigest()[:24] for col in entities for v in out[col].dropna().unique() if v!=''}
            assert len(set(mapping.values()))==len(mapping)
            for col in entities:
                nonempty=out[col].notna() & out[col].ne('')
                out.loc[nonempty,col]=out.loc[nonempty,col].map(mapping)
        if mode=='source_ablation':
            for col in ['pipeline','product_name','vendor_name']:
                out[col]='__SOURCE_HIDDEN__'
        return out

    results={}
    base_pred=None
    for mode in ['unchanged','null_to_empty','shift_364_days','entity_relabel','combined_representation','source_ablation']:
        x,_=prepare_route_a_features(perturb(df,mode),cfg)
        pred=model.predict_proba(x,thread_count=4).argmax(axis=1)
        if base_pred is None:
            base_pred=pred
        normal=y==labels.index('benign')
        metrics={'macro_f1':float(f1_score(y,pred,labels=range(3),average='macro',zero_division=0)), 'flip_count':int((pred!=base_pred).sum()),'flip_fraction':float((pred!=base_pred).mean()),'benign_false_alerts_per_10000':float((pred[normal]!=labels.index('benign')).mean()*10000),'classification_report':classification_report(y,pred,labels=range(3),target_names=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,pred,labels=range(3)).tolist()}
        results[mode]=metrics
        print(mode,json.dumps({k:metrics[k] for k in ['macro_f1','flip_count','benign_false_alerts_per_10000']}),flush=True)
    report={'scope':'100000-row fixed natural sample; representation stress only, not independent external performance. Source ablation removes potentially useful information and is not an invariance requirement. Other transforms preserve event contents used by this frozen structural model except serialization, identity naming and absolute dates.','rows':len(df),'seed':20260911,'class_order':labels,'counts':df.label_binary.value_counts().to_dict(),'model_sha256':hashlib.file_digest(model_path.open('rb'),'sha256').hexdigest(),'sample_id_sha256':hashlib.sha256('\n'.join(sorted(df.event_id)).encode()).hexdigest(),'results':results}
    out=ROOT/'evidence/2026-09-11/representation_stress.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out)

if __name__=='__main__':
    main()
