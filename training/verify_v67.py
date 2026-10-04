"""Reload all fitted estimators; independently check folds, thresholds and effects."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from train_v65_rank_heads import ROOT,load_fit
from run_v67_targeted import design,counts
from run_v67_specialist import hard_mask,threshold_from_inner
from run_v67_expansion import expanded_data
from v61_common import read,save,sha


def assert_close(a,b):
    np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
    return float(np.abs(np.asarray(a)-np.asarray(b)).max(initial=0))


def check_categories(bundle,f,train):
    names=bundle['names'];cols=[n for n in names if n in ['action','outcome','transport_protocol','src_role','dst_role','src_port_range','dst_port_range','icmp_type','icmp_code','icmp_message','icmp_unreachable']]
    for col,categories in zip(cols,bundle['encoder'].categories_):assert set(categories)==set(f[col].iloc[train])


def independent_threshold(f,scores,inner):
    # Separate validation algorithm: enumerate all attainable global thresholds
    # and test each using precomputed per-cell cumulative source mass.
    boundaries=np.r_[np.unique(scores),np.nextafter(np.unique(scores),np.inf),np.inf]
    eligible=np.ones(len(boundaries),bool)
    for k in range(3):
        for protocol in ['tcp','udp']:
            mask=(inner==k)&f.transport_protocol.eq(protocol).to_numpy()&f.label.eq(1).to_numpy()
            d=f[mask];ct=d.groupby('group').size();w=1/len(ct)/d.group.map(ct).to_numpy()
            s=scores[mask];order=np.argsort(s);s=s[order];w=w[order];prefix=np.r_[0,np.cumsum(w)]
            error=prefix[-1]-prefix[np.searchsorted(s,boundaries,side='left')]
            eligible&=error<=.01+1e-12
    return float(boundaries[eligible].min())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    f,c=load_fit();ef,ec,n=expanded_data();manifest=pd.read_parquet(ROOT/'artifacts/v67_targeted_20260920/manifest.parquet')
    base=manifest.baseline_pred_with_normal_gate.to_numpy();target=manifest.target_578.to_numpy();hard=hard_mask(f)
    models=0;rows=0;maximum=0.;checks=[];stress=[]
    with threadpool_limits(limits=4):
        for stage,frame,context in [('targeted',f,c),('specialist',f,c),('expansion',ef,ec)]:
            run=ROOT/f'artifacts/v67_{stage}_20260920';reg=read(run/'preregistered.json');result=read(run/'analysis.json')
            if stage=='targeted':assert reg['sources']['run_v67_targeted.py']==sha(ROOT/'training/run_v67_targeted.py')
            else:assert reg['source_sha256']==sha(ROOT/f'training/run_v67_{stage}.py')
            index=pd.Index(frame.row_position)
            for arm in reg['arms']:
                oof=pd.read_parquet(run/(arm+'_oof.parquet'));assert np.array_equal(oof.row_position,f.row_position)
                scorecol='p_S_given_threat' if stage=='targeted' else 'hard_score'
                for fold in range(3):
                    feature_arm=arm if stage=='targeted' else ('context_numeric' if arm.startswith('numeric') else 'context_coarse')
                    folder=run/f'fold{fold}'/arm
                    valid=np.flatnonzero(f.fold.eq(fold)&(True if stage=='targeted' else hard))
                    bundle=joblib.load(folder/'model.joblib');tr=index.get_indexer(bundle['train_positions']);assert (tr>=0).all()
                    expected=np.flatnonzero(frame.fold.ne(fold)&(frame.label.gt(0) if stage=='targeted' else hard_mask(frame)))
                    assert np.array_equal(tr,expected)
                    assert not set(frame.group.iloc[tr])&set(f.group.iloc[valid]);assert not set(frame.body_group.iloc[tr])&set(f.body_group.iloc[valid])
                    check_categories(bundle,frame,tr)
                    x,_,names,_=design(frame,context,feature_arm,tr,bundle['encoder']);assert names==bundle['names']
                    score=bundle['model'].predict_proba(x[valid])[:,1]
                    maximum=max(maximum,assert_close(score,oof[scorecol].iloc[valid]));models+=1;rows+=len(valid)
                    threshold=.5 if stage=='targeted' else bundle['threshold']
                    pred=(score>=threshold).astype(int)+1
                    if stage=='targeted':
                        gate=pd.read_parquet(ROOT/'artifacts/v66_issue_resolution_20260920/normal_gate/unweighted_gate_oof.parquet')
                        pred[gate.normal_score.iloc[valid].ge(.5)]=0
                    assert np.array_equal(pred,oof.pred.iloc[valid])
                    if stage!='targeted':
                        cal=pd.read_parquet(folder/'calibration_oof.parquet');ci=index.get_indexer(cal.row_position)
                        assert np.array_equal(ci,tr);inner=cal.inner_fold.to_numpy()
                        assert frame.iloc[ci].assign(inner=inner).groupby('group').inner.nunique().max()==1
                        for k in range(3):
                            mb=joblib.load(folder/f'inner{k}.joblib');ti=index.get_indexer(mb['train_positions']);vi=ci[inner==k]
                            assert np.array_equal(ti,ci[inner!=k]);assert not set(frame.group.iloc[ti])&set(frame.group.iloc[vi])
                            assert not set(frame.body_group.iloc[ti])&set(frame.body_group.iloc[vi]);check_categories(mb,frame,ti)
                            xi,*_=design(frame,context,feature_arm,ti,mb['encoder'])
                            check=mb['model'].predict_proba(xi[vi])[:,1]
                            maximum=max(maximum,assert_close(check,cal.score[inner==k]));models+=1;rows+=len(vi)
                        threshold2=independent_threshold(frame.iloc[ci].reset_index(drop=True),cal.score.to_numpy(),inner)
                        assert threshold2==threshold
                        assert threshold_from_inner(frame.iloc[ci].reset_index(drop=True),cal.score.to_numpy(),inner)[0]==threshold
                    # Full nuisance-field intervention through the allowlisted input builder.
                    spoof=frame.copy();spoof['timestamp']='2099-01-01';spoof['product_name']='FAKE';spoof['src_ip']='10.9.8.7';spoof['dst_ip']='10.8.7.6'
                    spoof['event_id']='FAKE';spoof['group']=-100;spoof['label']=0;spoof['text']='FAKE RAW TEXT'
                    nx,*_=design(spoof,context,feature_arm,tr,bundle['encoder'])
                    np.testing.assert_array_equal(nx,x)
                    # Sensitivity only: modified corpus extent, NOT a valid label-preserving attack.
                    if stage=='expansion':
                        for factor in [.5,2.]:
                            modified=x[valid].copy()
                            for col in ['log_other_fact_states','log_other_destinations','log_other_dst_port_symbols','log_other_src_port_symbols']:
                                j=names.index(col);modified[:,j]=np.log1p(np.expm1(modified[:,j])*factor)
                            mp=(bundle['model'].predict_proba(modified)[:,1]>=threshold).astype(int)+1
                            stress.append({'arm':arm,'fold':fold,'count_factor':factor,'changed_decisions':int((mp!=pred).sum()),
                                'target_fixed':int((mp[target[valid]]==2).sum()),'reference_target_fixed':int((pred[target[valid]]==2).sum()),
                                'scope':'Feature sensitivity, not a fully realized alternative corpus or invariance requirement'})
                actual=counts(f,oof.pred.to_numpy(),base,target)
                expected=result['arms'][arm]['all' if stage!='expansion' else 'nested']
                assert actual==expected
                if stage!='targeted':assert np.array_equal(oof.pred[~hard],base[~hard])
                checks.append({'stage':stage,'arm':arm,'predictions_replayed':True,'effects_recomputed':True})
    previous=read(ROOT/'evidence/2026-09-20/v66_issue_resolution/delivery.json')
    binding=previous['artifact_sha256']
    if isinstance(binding,dict):
        for name,digest in binding.items():assert sha(ROOT/name)==digest
    else:raise TypeError('Unexpected binding schema')
    assert models==84
    save(a.out/'verification.json',{'verification_passed':True,'models_restored':models,'prediction_rows_replayed':rows,
        'max_probability_difference':maximum,'prior_v66_bound_files_unchanged':len(binding),'checks':checks,
        'source_and_body_disjoint':True,'training_only_categories_verified':True,'inner_thresholds_independently_enumerated':18,
        'nuisance_feature_invariance_verified':True,'count_sensitivity':stress,'quality_acceptance':False,'source_sha256':sha(__file__)})
    print('VERIFIED',models,rows,maximum,flush=True)


if __name__=='__main__':main()
