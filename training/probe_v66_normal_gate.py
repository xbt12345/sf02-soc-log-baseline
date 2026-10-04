"""A bounded normal-control repair; no claim to solve ASA M/S or full SOC benign detection."""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from train_v65_rank_heads import load_fit,ROOT,measure
from v61_common import read,save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    v65=ROOT/'artifacts/v65_rank_heads_20260920'
    frame,_=load_fit();cache=read(v65/'text_cache_receipt.json')
    assert sha(v65/'upstream_text_features.npy')==cache['features_sha256']
    texts=pd.Index(pd.read_parquet(v65/'unique_text.parquet').text)
    codes=texts.get_indexer(frame.text);assert (codes>=0).all()
    X=np.load(v65/'upstream_text_features.npy')[codes];y=frame.label.eq(0).astype(int).to_numpy()
    base=pd.read_parquet(v65/'H0_guarded_oof.parquet');assert np.array_equal(base.row_position,frame.row_position)
    basep=base[['p_B','p_M','p_S']].to_numpy();basepred=basep.argmax(1)
    registration={'hypothesis':'A separately optimized linear normal gate can repair the rare normal controls without damaging threat rows; this does not address M/S transfer.',
      'arms':['unweighted_gate','balanced_gate'],'C':1.,'solver':'lbfgs','max_iter':500,'threshold':.5,
      'features':'Same frozen upstream text vectors; StandardScaler fits training side only.',
      'folds':'Exactly original v65 three subject folds. No threshold or hyperparameter search.',
      'composition':'Gate score >= .5 => B; otherwise retain H0 guarded decision exactly.',
      'acceptance':'All 12 heldout normal controls correct and no added threat-to-B errors. Passing is only a narrow control repair, not quality acceptance.',
      'limitations':'Controls from only three source symbols and one grammar; fold 1 has no normal validation support. Balanced score is not a calibrated deployment probability.',
      'source_sha256':sha(__file__),'feature_sha256':cache['features_sha256']}
    save(a.out/'preregistered.json',registration)
    out={};details=[]
    with threadpool_limits(limits=4),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for arm,weight in [('unweighted_gate',None),('balanced_gate','balanced')]:
            scores=np.zeros(len(frame));pred=basepred.copy()
            for fold in range(3):
                train=np.flatnonzero(frame.fold.ne(fold));valid=np.flatnonzero(frame.fold.eq(fold))
                model=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(C=1.,solver='lbfgs',max_iter=500,class_weight=weight,random_state=20260920))])
                model.fit(X[train],y[train]);p=model.predict_proba(X[valid])[:,1];scores[valid]=p
                path=a.out/f'{arm}_fold{fold}.joblib';joblib.dump(model,path)
                again=joblib.load(path).predict_proba(X[valid])[:,1]
                np.testing.assert_array_equal(p,again)
                pred[valid[p>=.5]]=0
                details.append({'arm':arm,'fold':fold,'train_normal':int(y[train].sum()),'valid_normal':int(y[valid].sum()),
                   'normal_missed':int(((p<.5)&(y[valid]==1)).sum()),'threat_as_normal':int(((p>=.5)&(y[valid]==0)).sum()),
                   'iterations':int(model.named_steps['classifier'].n_iter_.max()),'model_sha256':sha(path),'reload_probability_max_diff':float(abs(p-again).max())})
                print(arm,fold,details[-1],flush=True)
            d=frame[['row_position','label','group','fold']].copy();d['normal_score']=scores;d['pred']=pred
            d.to_parquet(a.out/f'{arm}_oof.parquet',index=False)
            metric=measure(frame,np.eye(3)[pred]);normal_errors=int(((pred!=0)&(y==1)).sum())
            added=int(((pred==0)&(y==0)&(basepred!=0)).sum())
            out[arm]={'metrics':metric,'normal_errors':normal_errors,'added_threat_to_B':added,
                      'M_S_decisions_changed':int(((pred!=basepred)&(y==0)).sum()),
                      'control_repair_passed':normal_errors==0 and added==0}
    result={'status':'normal_control_gate_probes_executed','models':out,'folds':details,
            'baseline_metrics':measure(frame,basep),'quality_acceptance':False,'new_M_S_training':False,
            'full_SOC_benign_coverage_validated':False,'source_sha256':sha(__file__)}
    save(a.out/'analysis.json',result);print('GATE_PROBES_COMPLETE',out,flush=True)


if __name__=='__main__':main()
