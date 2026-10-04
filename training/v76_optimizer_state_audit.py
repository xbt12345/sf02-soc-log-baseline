"""Read-only saved averaged vs last iterate diagnostic; never selects a model."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,OUT,read,save,sha,matrices,features,load_sparse,BATCH,NAMES

DEST=ROOT/'evidence/2026-09-22/v76_review'

def main():
    r=pd.read_parquet(OUT/'rows.parquet');m=matrices();preds={};source={}
    for arm,subdir in [('D','four_arm'),('E','corrective'),('F','corrective')]:
        path=OUT/subdir/(arm+'.joblib');model=joblib.load(path);source[arm]=sha(path)
        if arm=='E':m['new_text']=load_sparse(OUT/'corrective/text')
        saved=pd.read_parquet(OUT/subdir/'predictions.parquet',columns=[arm+'_pred'])[arm+'_pred'].to_numpy()
        standard=np.empty(len(r),np.int8)
        with threadpool_limits(limits=4):
            for start in range(0,len(r),BATCH):
                ix=np.arange(start,min(start+BATCH,len(r)));x=features(r,ix,m,'new')
                np.testing.assert_array_equal(model.predict(x),saved[ix])
                standard[ix]=(x@model._standard_coef.T+model._standard_intercept).argmax(axis=1)
        preds[arm]=(saved,standard)
    records=[]
    for (route,label,hold),z in r.groupby(['route','label_index','is_validation']):
        ix=z.index.to_numpy()
        for arm,(avg,last) in preds.items():records.append({'arm':arm,'route':route,'label':NAMES[int(label)],
            'role':'validation' if hold else 'fit','rows':len(ix),
            'averaged_errors':int((avg[ix]!=label).sum()),'last_iterate_errors':int((last[ix]!=label).sum())})
    table=pd.DataFrame(records);table.to_csv(DEST/'optimizer_states.csv',index=False)
    rare=table[table.route.isin(['unsupported','bounded_payload','windows_message','authentication'])&table.label.ne('benign')]
    result={'new_fits':0,'saved_averaged_predictions_all_rows_match':True,'rows_per_model':len(r),
        'scope':'Last iterate is a diagnostic of averaging sensitivity, not a certified optimizer minimum or a selected checkpoint. No private target predictions or scores recomputed.',
        'rare_cells':rare.to_dict('records'),'source_sha256':sha(__file__),'model_sha256':source,
        'table_sha256':sha(DEST/'optimizer_states.csv'),'quality_acceptance':False}
    save(DEST/'optimizer_state_audit.json',result);print(rare.to_string(index=False))

if __name__=='__main__':main()
