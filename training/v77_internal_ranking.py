"""Stored-prediction internal ranking and refit-protocol diagnosis; zero fitting."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from run_v75 import ROOT, OUT, read, save, sha

DEST=ROOT/'evidence/2026-09-22/v77_attribution'


def main():
    paths=[OUT/'rows.parquet',OUT/'four_arm/predictions.parquet',OUT/'full/predictions.parquet']
    r,inner,full=[pd.read_parquet(p) for p in paths]
    np.testing.assert_array_equal(r.row_position,inner.row_position)
    np.testing.assert_array_equal(r.row_position,full.row_position)
    out=[]
    for route,z in r[r.is_validation].groupby('route'):
        ms=z.label_index.ne(0);ix=z.index[ms].to_numpy();y=(z.loc[ms,'label_index']==1).to_numpy()
        for model,p,prefix in [('D_inner',inner,'D'),('D_full',full,'FULL')]:
            if len(np.unique(y))!=2:continue
            q=p.loc[ix,prefix+'_p1'].to_numpy()/np.maximum(p.loc[ix,prefix+'_p1'].to_numpy()+p.loc[ix,prefix+'_p2'].to_numpy(),1e-30)
            out.append({'route':route,'model':model,'rows':len(ix),'M_rows':int(y.sum()),'S_rows':int((~y).sum()),
                'M_vs_S_auroc':float(roc_auc_score(y,q)),'M_average_precision':float(average_precision_score(y,q)),
                'full_has_fitted_these_rows':model=='D_full'})
    pd.DataFrame(out).to_csv(DEST/'internal_ranking.csv',index=False)
    models={}
    for name in ['four_arm/D.joblib','full/FULL.joblib','corrective/F.joblib','F_diagnostic/model.joblib']:
        p=OUT/name;m=joblib.load(p);np.testing.assert_array_equal(m.classes_,[0,1,2])
        models[name]={'classes':m.classes_.tolist(),'n_features':m.n_features_in_,'sample_updates':int(m.t_-1),
                      'alpha':m.alpha,'eta0':m.eta0,'learning_rate':m.learning_rate,'average':m.average,'model_sha256':sha(p)}
    save(DEST/'internal_ranking_receipt.json',{'new_fits':0,'source_sha256':sha(__file__),
         'inputs_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},
         'output_sha256':sha(DEST/'internal_ranking.csv'),'models':models,
         'scope':'D_inner uses original heldout rows; D_full fitted the same rows and is diagnostic resubstitution only. Float32 stored probabilities, not new logits.'})
    print(pd.DataFrame(out).to_string(index=False))


if __name__=='__main__':main()
