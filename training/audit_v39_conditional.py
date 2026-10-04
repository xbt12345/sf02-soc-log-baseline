"""Conditional ranking diagnostic of saved predictions, not a tuned model."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/v38_local_r1_20260913'
sys.path.insert(0,str(RUN/'frozen_training_runtime'))
import numpy as np,pandas as pd,pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from run_v38_train import negative_control
OUT=ROOT/'evidence/2026-09-13/v39_review'
def main():
    target=OUT/'conditional_ranking.json'
    if target.exists():raise FileExistsError('Preserve prior run')
    rows=pq.read_table(RUN/'prepared/rows.parquet').to_pandas();p=pq.read_table(RUN/'prepared/projections.parquet').to_pandas()
    x,_=negative_control([json.loads(v) for v in p.C_BOTH]);ids,_=pd.factorize(np.asarray([bytes(a.astype('u1')) for a in x.toarray()],dtype=object),sort=False)
    d=pq.read_table(RUN/'equal_classifiers_attempt2/C_BOTH/evaluation.parquet').to_pandas()
    d['mask']=ids[rows.loc[rows.inner_role==2,'projection_id']]
    res=[]
    for (route,mask),a in d.groupby(['route','mask']):
        classes=sorted(a.label_index.unique())
        for ix,c in enumerate(classes):
            for k in classes[ix+1:]:
                s=a[a.label_index.isin([c,k])];names=['p_benign','p_malicious','p_suspicious']
                margin=np.log(s[names[c]].to_numpy().clip(1e-300))-np.log(s[names[k]].to_numpy().clip(1e-300))
                res.append({'route':route,'mask':int(mask),'positive_class':int(c),'negative_class':int(k),
                    'positive_rows':int((s.label_index==c).sum()),'negative_rows':int((s.label_index==k).sum()),
                    'positive_legacy_groups':int(s[s.label_index==c].union_group.nunique()),
                    'negative_legacy_groups':int(s[s.label_index==k].union_group.nunique()),
                    'auc_of_saved_class_margin':float(roc_auc_score(s.label_index==c,margin)),
                    'constant_mask_only_margin_auc':0.5})
    result={'scope':'Inspected development conditional ranking; neither accuracy nor causal attribution; correlated rows and few legacy groups.',
        'slices':res,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
