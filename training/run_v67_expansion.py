"""Official historical development data reclassified as auxiliary training.

Previous selection/evaluation roles are retired for this branch, never reused
as test scores. Each original held-out source remains completely excluded.
"""
import argparse
from pathlib import Path
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from run_v67_specialist import PARAMS,hard_mask,threshold_from_inner
from run_v67_targeted import design,counts
from train_v65_rank_heads import load_fit, ROOT, DATA
from v61_common import read,save,sha


def expanded_data():
    f,c=load_fit();allf=pd.read_parquet(DATA/'records.parquet')
    aux=allf[allf.role.ne('fit')].copy()
    assert not set(aux.group)&set(f.group)
    assert not set(aux.body_group)&set(f.body_group)
    ix=pd.Index(allf.row_position).get_indexer(aux.row_position)
    with np.load(DATA/'context.npz') as z:
        extra_stats=z['stats'][ix]
        # Validate that every old context neighbor belongs to its own whole source.
        neighbors=z['neighbors'][ix];ok=neighbors>=0
        assert ((allf.group.to_numpy()[neighbors.clip(min=0)]==aux.group.to_numpy()[:,None])|~ok).all()
    aux['fold']=-1
    return pd.concat([f,aux],ignore_index=True),{'stats':np.r_[c['stats'],extra_stats]},len(f)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    f,c,n=expanded_data();hard=hard_mask(f);m=pd.read_parquet(ROOT/'artifacts/v67_targeted_20260920/manifest.parquet')
    assert np.array_equal(f.row_position.iloc[:n],m.row_position)
    base=m.baseline_pred_with_normal_gate.to_numpy();target=m.target_578.to_numpy()
    aux=f.iloc[n:];arms=['coarse','numeric']
    reg={'version':'v67-official-support-expansion-1','source_sha256':sha(__file__),
        'helpers':{name:sha(ROOT/'training'/name) for name in ['run_v67_specialist.py','run_v67_targeted.py','train_v65_rank_heads.py']},
        'official_prepared_sha256':sha(DATA/'records.parquet'),'context_sha256':sha(DATA/'context.npz'),
        'auxiliary_rows':len(aux),'auxiliary_sources':int(aux.group.nunique()),'original_target_sources_excluded':True,
        'retired_roles':['v61 selection','v61 evaluation'],'role_change':'Those already observed records become auxiliary training only. Their prior historical scores stay historical; no evaluation on them here.',
        'parameters':PARAMS,'arms':arms,'fits_planned':24,'class_weight':None,
        'calibration':'Same shared <=1% M-source loss in every inner-fold/protocol cell; 3 source-held-out inner folds. Default .5 also reported, not selected after testing.',
        'quality_acceptance':False,'adaptive_development':True,'scope':'Fixed 578 target and all 59640 original development rows; no external or future holdout claims'}
    save(a.out/'preregistered.json',reg)
    f[['row_position','event_id','label','group','body_group','role','fold']].to_parquet(a.out/'expanded_manifest.parquet',index=False)
    preds={arm:base.copy() for arm in arms};defaults={arm:base.copy() for arm in arms}
    scores={arm:np.full(n,np.nan) for arm in arms};details={};t=time.monotonic()
    with threadpool_limits(limits=4):
        for fold in range(3):
            tr=np.flatnonzero(f.fold.ne(fold)&hard);va=np.flatnonzero(f.fold.eq(fold)&hard)
            assert (va<n).all();assert not set(f.group.iloc[tr])&set(f.group.iloc[va])
            assert not set(f.body_group.iloc[tr])&set(f.body_group.iloc[va])
            inner=np.full(len(tr),-1,int)
            for k,(_,v) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=20260920+fold).split(np.zeros(len(tr)),f.label.iloc[tr],f.group.iloc[tr])):inner[v]=k
            im=f.iloc[tr][['row_position','label','group','body_group']].copy();im['inner_fold']=inner
            folder=a.out/f'fold{fold}';folder.mkdir();im.to_parquet(folder/'inner_manifest.parquet',index=False)
            details[str(fold)]={}
            for arm in arms:
                sub=folder/arm;sub.mkdir();nested=np.full(len(tr),np.nan)
                feature_arm='context_'+arm
                for k in range(3):
                    ti=tr[inner!=k];vi=tr[inner==k]
                    assert not set(f.group.iloc[ti])&set(f.group.iloc[vi])
                    x,enc,names,cat=design(f,c,feature_arm,ti)
                    model=HistGradientBoostingClassifier(**PARAMS,categorical_features=cat)
                    model.fit(x[ti],f.label.iloc[ti].eq(2));nested[inner==k]=model.predict_proba(x[vi])[:,1]
                    joblib.dump({'model':model,'encoder':enc,'names':names,'train_positions':f.row_position.iloc[ti].to_numpy()},sub/f'inner{k}.joblib')
                threshold,cal=threshold_from_inner(f.iloc[tr].reset_index(drop=True),nested,inner)
                im2=im.copy();im2['score']=nested;im2.to_parquet(sub/'calibration_oof.parquet',index=False);save(sub/'calibration.json',cal)
                x,enc,names,cat=design(f,c,feature_arm,tr)
                model=HistGradientBoostingClassifier(**PARAMS,categorical_features=cat)
                model.fit(x[tr],f.label.iloc[tr].eq(2));score=model.predict_proba(x[va])[:,1]
                scores[arm][va]=score;preds[arm][va]=(score>=threshold).astype(int)+1;defaults[arm][va]=(score>=.5).astype(int)+1
                joblib.dump({'model':model,'encoder':enc,'names':names,'threshold':threshold,'train_positions':f.row_position.iloc[tr].to_numpy()},sub/'model.joblib')
                details[str(fold)][arm]={'threshold':threshold,'target_fixed':int((preds[arm][va][target[va]]==2).sum()),
                    'calibration_S_recall':float((nested[f.label.iloc[tr].eq(2)]>=threshold).mean())}
                print('EXPANSION',fold,arm,details[str(fold)][arm],flush=True)
    result={'quality_acceptance':False,'fits_completed':24,'seconds':time.monotonic()-t,'folds':details,
        'baseline':counts(f.iloc[:n],base,base,target),'arms':{}}
    for arm in arms:
        assert np.array_equal(preds[arm][~hard[:n]],base[~hard[:n]])
        o=f.iloc[:n][['row_position','label','group','fold']].copy();o['hard_score']=scores[arm];o['pred']=preds[arm];o['default_pred']=defaults[arm]
        o.to_parquet(a.out/(arm+'_oof.parquet'),index=False)
        result['arms'][arm]={'nested':counts(f.iloc[:n],preds[arm],base,target),'default':counts(f.iloc[:n],defaults[arm],base,target)}
    save(a.out/'analysis.json',result);print('COMPLETE',result['arms'],flush=True)


if __name__=='__main__':main()
