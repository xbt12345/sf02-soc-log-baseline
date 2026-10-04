"""Single prespecified diagnostic intervention on fixed v3.7 models. No refitting."""
import json,sys,gc
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import joblib
from scipy import sparse
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'artifacts/v37_cloud_20260913T072309Z/runtime'))
from run_v36_train import string_codes
import v37_learning as learning
work=ROOT/'artifacts/v37_cloud_20260913T072309Z/work';prep=ROOT/'artifacts/v37_prepared_r13_20260913'
rows=[]
for task in ['known_dev','source_ad','source_duo','source_waf','asa_hard']:
    f=work/'models'/(task+'_B2_REPAIRED');b=joblib.load(f/'model.joblib');t=pq.read_table(f/'evaluation.parquet')
    pos=t['row_position'].to_numpy();y=t['label_index'].to_numpy()
    texts,tc=string_codes(prep/'prepared.parquet',pos,'b1');facts,fc=string_codes(prep/'prepared.parquet',pos,'facts')
    pairs,back=np.unique(tc.astype(np.int64)*len(facts)+fc,return_inverse=True)
    x=sparse.hstack([b['tfidf'].transform([texts[i] for i in pairs//len(facts)]),b['facts'].transform([facts[i] for i in pairs%len(facts)])],format='csr')
    names=list(b['tfidf'].names())+list(b['facts'].names());indexes={i for i,n in enumerate(names) if n=='category:category=authentication'}
    x.data[np.isin(x.indices,list(indexes))]=0
    p=b['model'].predict_proba(x)[back];tau=next(po for po in b['policies'] if po['alpha']==.001)['empirical_worst_source_global']
    rows.append({'task':task,'intervention':'zero authentication category feature only; fixed coefficients and threshold',
        'primary':learning.cm_metrics(y,learning.gated_predictions(p,tau)),'risk':learning.risk_metrics(y,p,tau),
        'new_model_trained':False,'deployable_repair_proven':False})
    del x,b,texts,facts;gc.collect()
(OUT/'category_ablation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))
