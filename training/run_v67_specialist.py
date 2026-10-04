"""Adaptive follow-up: nested source-held-out hard-behavior specialists.

No target identities or outer labels used by fit/calibration. Fixed inner
M-source budget in each protocol and each fold; all score ties move together.
"""
import argparse
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from run_v67_targeted import ROOT, AUDIT, OLD, design, counts
from train_v65_rank_heads import load_fit, measure
from v61_common import read,save,sha

ARMS=['coarse_plain','numeric_plain','coarse_balanced','numeric_balanced']
PARAMS=dict(learning_rate=.05,max_iter=500,max_leaf_nodes=15,min_samples_leaf=10,
            l2_regularization=1.,early_stopping=False,random_state=20260920)


def hard_mask(f):
    return (f.src_role.eq('outside')&f.dst_role.eq('dmz')&f.transport_protocol.isin(['tcp','udp'])).to_numpy()


def threshold_from_inner(f, scores, inner):
    """Lowest threshold satisfying EVERY protocol/fold <= 1% M-source errors.

    The all-M candidate (+inf) is legitimate and leads to no S changes.
    S labels do not select a threshold; they only describe calibration recall.
    """
    cells=[];thresholds=[]
    for k in sorted(set(inner)):
        for protocol in ['tcp','udp']:
            mask=(inner==k)&f.transport_protocol.eq(protocol).to_numpy()&f.label.eq(1).to_numpy()
            part=f[mask];source_counts=part.groupby('group').size()
            assert len(source_counts)>0
            weights=1/len(source_counts)/part.group.map(source_counts).to_numpy()
            s=scores[mask];r=pd.DataFrame({'score':s,'mass':weights}).groupby('score').mass.sum().sort_index(ascending=False)
            # Moving threshold below a tied block is forbidden if it breaches the cap.
            crossed=r.cumsum().gt(.01+1e-12)
            boundary=float(np.nextafter(r.index[np.flatnonzero(crossed)[0]],np.inf)) if crossed.any() else -np.inf
            thresholds.append(boundary)
            cells.append({'inner_fold':int(k),'protocol':protocol,'M_sources':len(source_counts),'M_rows':len(part),'cell_threshold':boundary})
    threshold=max(thresholds)
    for cell in cells:
        mask=(inner==cell['inner_fold'])&f.transport_protocol.eq(cell['protocol']).to_numpy()&f.label.eq(1).to_numpy()
        part=f[mask].copy();part['alert']=scores[mask]>=threshold
        cell['M_source_error_rate_at_shared_threshold']=float(part.groupby('group').alert.mean().mean())
        assert cell['M_source_error_rate_at_shared_threshold']<=.01+1e-12
    return threshold,{'threshold':threshold,'cells':cells,'scope':'Shared hard-behavior threshold; constraint estimated from inner OOF sources, not a population guarantee'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    f,c=load_fit();oldrun=ROOT/'artifacts/v67_targeted_20260920'
    manifest=pd.read_parquet(oldrun/'manifest.parquet');assert np.array_equal(f.row_position,manifest.row_position)
    target=manifest.target_578.to_numpy();base=manifest.baseline_pred_with_normal_gate.to_numpy()
    hard=hard_mask(f);assert (f.label[hard]>0).all()
    registration={'version':'v67-specialist-nested-1','adaptive_followup_to':sha(oldrun/'analysis.json'),
        'parameters':PARAMS,'arms':ARMS,'fits_planned':48,'primary':'coarse_plain',
        'training':'Hard TCP/UDP outside->dmz only, original other-two source folds; retains row multiplicity. No target-row or validation-error mining.',
        'calibration':'Each outer fit side: 3 new source-only inner folds. Lowest threshold with <=1% M-source error in EVERY inner-fold x protocol. No outer-label threshold search.',
        'scope':'Core hard M/S development; same fixed normal gate and all nonhard predictions retained. Outer data already adaptively observed.',
        'continuation':'Same registered v67 fixed-control requirements; no arm selected or promoted using outer answers.',
        'source_sha256':sha(__file__),'helpers_sha256':sha(ROOT/'training/run_v67_targeted.py'),
        'manifest_sha256':sha(oldrun/'manifest.parquet'),'quality_acceptance':False}
    save(a.out/'preregistered.json',registration)
    predictions={arm:base.copy() for arm in ARMS};scores={arm:np.full(len(f),np.nan) for arm in ARMS}
    folds={};started=time.monotonic()
    with threadpool_limits(limits=4):
        for fold in range(3):
            tr=np.flatnonzero(f.fold.ne(fold)&hard);va=np.flatnonzero(f.fold.eq(fold)&hard)
            inner=np.full(len(tr),-1,int)
            cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20260920+fold)
            for k,(_,v) in enumerate(cv.split(np.zeros(len(tr)),f.label.iloc[tr],f.group.iloc[tr])):inner[v]=k
            assert f.iloc[tr].assign(inner=inner).groupby('group').inner.nunique().max()==1
            folder=a.out/f'fold{fold}';folder.mkdir()
            im=f.iloc[tr][['row_position','group','label','body_group']].copy();im['inner_fold']=inner
            im.to_parquet(folder/'inner_manifest.parquet',index=False)
            folds[str(fold)]={}
            for arm in ARMS:
                sub=folder/arm;sub.mkdir();nested=np.full(len(tr),np.nan)
                feature_arm='context_numeric' if arm.startswith('numeric') else 'context_coarse'
                weight='balanced' if arm.endswith('balanced') else None
                for k in range(3):
                    ti=tr[inner!=k];vi=tr[inner==k]
                    assert not set(f.group.iloc[ti])&set(f.group.iloc[vi])
                    assert not set(f.body_group.iloc[ti])&set(f.body_group.iloc[vi])
                    x,encoder,names,catmask=design(f,c,feature_arm,ti)
                    model=HistGradientBoostingClassifier(**PARAMS,categorical_features=catmask,class_weight=weight)
                    model.fit(x[ti],f.label.iloc[ti].eq(2))
                    nested[inner==k]=model.predict_proba(x[vi])[:,1]
                    joblib.dump({'model':model,'encoder':encoder,'names':names,'train_positions':f.row_position.iloc[ti].to_numpy()},sub/f'inner{k}.joblib')
                assert np.isfinite(nested).all()
                threshold,calibration=threshold_from_inner(f.iloc[tr].reset_index(drop=True),nested,inner)
                im2=im.copy();im2['score']=nested;im2.to_parquet(sub/'calibration_oof.parquet',index=False)
                save(sub/'calibration.json',calibration)
                x,encoder,names,catmask=design(f,c,feature_arm,tr)
                model=HistGradientBoostingClassifier(**PARAMS,categorical_features=catmask,class_weight=weight)
                model.fit(x[tr],f.label.iloc[tr].eq(2));score=model.predict_proba(x[va])[:,1]
                scores[arm][va]=score;predictions[arm][va]=(score>=threshold).astype(int)+1
                joblib.dump({'model':model,'encoder':encoder,'names':names,'threshold':threshold,'train_positions':f.row_position.iloc[tr].to_numpy()},sub/'model.joblib')
                fit_pred=model.predict_proba(x[tr])[:,1]>=threshold
                detail={'threshold':threshold,'fit_hard_S_recall':float(fit_pred[f.label.iloc[tr].eq(2)].mean()),
                    'inner_hard_S_recall':float((nested[f.label.iloc[tr].eq(2)]>=threshold).mean()),
                    'effects':counts(f.iloc[va],predictions[arm][va],base[va],target[va])}
                folds[str(fold)][arm]=detail
                print('SPECIALIST',fold,arm,detail,flush=True)
    results={'baseline':counts(f,base,base,target),'arms':{},'folds':folds,'seconds':time.monotonic()-started,
             'quality_acceptance':False,'fits_completed':48}
    for arm,pred in predictions.items():
        o=f[['row_position','label','group','fold']].copy();o['hard_score']=scores[arm];o['pred']=pred
        o.to_parquet(a.out/(arm+'_oof.parquet'),index=False)
        effect=counts(f,pred,base,target);issues=[];ref=results['baseline']
        if effect['target_fixed']<58:issues.append('target_repairs_below_58')
        if effect['S']['source_recall']<ref['S']['source_recall']+.05-1e-12:issues.append('S_source_gain_below_5pp')
        if effect['M']['errors']>ref['M']['errors']+.01*ref['M']['rows']:issues.append('M_row_loss')
        if effect['M']['source_recall']<ref['M']['source_recall']-.01-1e-12:issues.append('M_source_loss')
        fold_metrics={}
        for k in range(3):
            ix=np.flatnonzero(f.fold.eq(k));dv=f.iloc[ix].reset_index(drop=True)
            cm=measure(dv,np.eye(3)[pred[ix]]);rm=measure(dv,np.eye(3)[base[ix]])
            fold_metrics[str(k)]={'candidate':cm,'reference':rm}
            if cm['M_source_recall']<rm['M_source_recall']-.01-1e-12:issues.append(str(k)+'/M_source_loss')
            if cm['ASA']['macro_f1_M_S']<rm['ASA']['macro_f1_M_S']-.005-1e-12:issues.append(str(k)+'/macro_loss')
            for p in ['tcp','udp']:
                if cm['hard'][p]['1']<rm['hard'][p]['1']-.01-1e-12:issues.append(str(k)+'/'+p+'/M_source_loss')
                if cm['hard'][p]['2']<rm['hard'][p]['2']-1e-12:issues.append(str(k)+'/'+p+'/S_source_loss')
        assert np.array_equal(pred[~hard],base[~hard])
        results['arms'][arm]={'all':effect,'continuation_passed':not issues,'rejection_reasons':issues,'fold_metrics':fold_metrics}
    save(a.out/'analysis.json',results)
    print('COMPLETE',{a:v['all'] for a,v in results['arms'].items()},flush=True)


if __name__=='__main__':main()
