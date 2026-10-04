"""Retrain only the simple normal gate after pre-model class-support split repair."""
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from train_v65_rank_heads import load_fit,ROOT
from v61_common import read,save,sha


def main():
    root=ROOT/'artifacts/v66_issue_resolution_20260920';out=root/'supported_normal_gate'
    assert not out.exists();out.mkdir()
    f,_=load_fit();m=pd.read_parquet(root/'supported_fold_manifest.parquet')
    pd.testing.assert_frame_equal(f[['row_position','label','group','body_group']],m[['row_position','label','group','body_group']])
    v65=ROOT/'artifacts/v65_rank_heads_20260920';r=read(v65/'text_cache_receipt.json')
    assert sha(v65/'upstream_text_features.npy')==r['features_sha256']
    codes=pd.Index(pd.read_parquet(v65/'unique_text.parquet').text).get_indexer(f.text)
    X=np.load(v65/'upstream_text_features.npy')[codes];y=f.label.eq(0).astype(int).to_numpy()
    save(out/'preregistered.json',{'source_sha256':sha(__file__),'manifest_sha256':sha(root/'supported_fold_manifest.parquet'),
      'method':'Same unweighted standardized logistic normal gate C=1, lbfgs, max_iter=500, threshold=.5; three actual refits. No M/S head and no reused old-fold labels/features from trained encoders.',
      'reason':'Verify normal gate with normal sources present in every validation fold. Chosen unweighted variant is simpler and already tied with balanced on original controls.',
      'quality_acceptance':False})
    scores=np.zeros(len(f));reports=[]
    with threadpool_limits(limits=4),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for fold in range(3):
            train=np.flatnonzero(m.fold.ne(fold));val=np.flatnonzero(m.fold.eq(fold))
            model=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(C=1.,solver='lbfgs',max_iter=500,random_state=20260920))])
            model.fit(X[train],y[train]);p=model.predict_proba(X[val])[:,1];scores[val]=p
            path=out/f'fold{fold}.joblib';joblib.dump(model,path)
            np.testing.assert_array_equal(joblib.load(path).predict_proba(X[val])[:,1],p)
            reports.append({'fold':fold,'normal_support':int(y[val].sum()),'normal_errors':int(((p<.5)&(y[val]==1)).sum()),
              'threat_rows':int((y[val]==0).sum()),'threat_to_B':int(((p>=.5)&(y[val]==0)).sum()),'model_sha256':sha(path)})
            print(reports[-1],flush=True)
    d=m.copy();d['normal_score']=scores;d.to_parquet(out/'oof.parquet',index=False)
    save(out/'analysis.json',{'folds':reports,'normal_control_repair_passed':all(x['normal_errors']==0 and x['threat_to_B']==0 for x in reports),
      'quality_acceptance':False,'normal_source_groups':int(f[y==1].group.nunique()),
      'scope':'Gate-only class-support replication. Original source groups and controls reused; not independent external validation, not a new M/S result, not full SOC acceptance.'})


if __name__=='__main__':main()
