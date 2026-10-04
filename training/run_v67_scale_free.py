"""Targeted count-feature ablation after measured count sensitivity.

Same expanded coarse specialist, same source folds and fixed learner, remove
four absolute log counts only; train-only per-protocol dual M budgets.
"""
import argparse
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from run_v67_targeted import design,counts
from run_v67_specialist import PARAMS,hard_mask
from run_v67_expansion import expanded_data
from run_v67_dual_budget import threshold
from train_v65_rank_heads import ROOT,measure
from v61_common import read,save,sha

DROP=['log_other_fact_states','log_other_destinations','log_other_dst_port_symbols','log_other_src_port_symbols']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    f,c,n=expanded_data();target_manifest=pd.read_parquet(ROOT/'artifacts/v67_targeted_20260920/manifest.parquet')
    target=target_manifest.target_578.to_numpy();base=target_manifest.baseline_pred_with_normal_gate.to_numpy();hard=hard_mask(f)
    save(a.out/'preregistered.json',{'version':'v67-count-ablation-1','source_sha256':sha(__file__),'parameters':PARAMS,
        'drop':DROP,'fixed_control':'expanded coarse with dual per-protocol M budgets',
        'fits_planned':12,'hypothesis':'Previously repaired hard S decisions depend on corpus size; test whether removing absolute counts improves transfer.',
        'split':'Reuse exact expanded fold inner manifests; all official auxiliary records stay training only.',
        'calibration':'Per protocol, <=1% M row and source errors in every inner fold. No outer-label choice.',
        'quality_acceptance':False,'adaptive_development':True})
    idx=pd.Index(f.row_position);pred=base.copy();scores=np.full(n,np.nan);details={};t0=time.monotonic()
    def inputs(train,encoder=None):
        x,e,names,cat=design(f,c,'context_coarse',train,encoder)
        keep=[i for i,k in enumerate(names) if k not in DROP]
        return x[:,keep],e,[names[i] for i in keep],[cat[i] for i in keep]
    with threadpool_limits(limits=4):
        for fold in range(3):
            im=pd.read_parquet(ROOT/f'artifacts/v67_expansion_20260920/fold{fold}/inner_manifest.parquet')
            tr=idx.get_indexer(im.row_position);inner=im.inner_fold.to_numpy();va=np.flatnonzero(f.fold.eq(fold)&hard)
            assert not set(f.group.iloc[tr])&set(f.group.iloc[va]);assert not set(f.body_group.iloc[tr])&set(f.body_group.iloc[va])
            folder=a.out/f'fold{fold}';folder.mkdir();nested=np.full(len(tr),np.nan)
            for k in range(3):
                ti=tr[inner!=k];vi=tr[inner==k];x,enc,names,cat=inputs(ti)
                model=HistGradientBoostingClassifier(**PARAMS,categorical_features=cat)
                model.fit(x[ti],f.label.iloc[ti].eq(2));nested[inner==k]=model.predict_proba(x[vi])[:,1]
                joblib.dump({'model':model,'encoder':enc,'names':names,'train_positions':f.row_position.iloc[ti].to_numpy()},folder/f'inner{k}.joblib')
            im['score']=nested;im['transport_protocol']=f.transport_protocol.iloc[tr].to_numpy();im.to_parquet(folder/'calibration_oof.parquet',index=False)
            thresholds={p:threshold(im,p) for p in ['tcp','udp']}
            x,enc,names,cat=inputs(tr);model=HistGradientBoostingClassifier(**PARAMS,categorical_features=cat)
            model.fit(x[tr],f.label.iloc[tr].eq(2));score=model.predict_proba(x[va])[:,1];scores[va]=score
            for p in ['tcp','udp']:
                mask=f.transport_protocol.iloc[va].eq(p).to_numpy();pred[va[mask]]=(score[mask]>=thresholds[p]).astype(int)+1
            joblib.dump({'model':model,'encoder':enc,'names':names,'thresholds':thresholds,'train_positions':f.row_position.iloc[tr].to_numpy()},folder/'model.joblib')
            details[str(fold)]={'thresholds':thresholds,'effects':counts(f.iloc[va],pred[va],base[va],target[va])}
            print('SCALE_FREE',fold,details[str(fold)],flush=True)
    o=f.iloc[:n][['row_position','label','group','fold']].copy();o['pred']=pred;o['hard_score']=scores;o.to_parquet(a.out/'oof.parquet',index=False)
    effect=counts(f.iloc[:n],pred,base,target);issues=[];ref=counts(f.iloc[:n],base,base,target)
    if effect['target_fixed']<58:issues.append('target_repairs_below_58')
    if effect['S']['source_recall']<ref['S']['source_recall']+.05-1e-12:issues.append('S_source_gain_below_5pp')
    fold_metrics={}
    for fold in range(3):
        ix=np.flatnonzero(f.fold.eq(fold));dv=f.iloc[ix].reset_index(drop=True)
        cm=measure(dv,np.eye(3)[pred[ix]]);rm=measure(dv,np.eye(3)[base[ix]])
        fold_metrics[str(fold)]={'candidate':cm,'reference':rm}
        if cm['M_source_recall']<rm['M_source_recall']-.01-1e-12:issues.append(str(fold)+'/M_source_loss')
        if cm['ASA']['recall_B_M_S'][1]<rm['ASA']['recall_B_M_S'][1]-.01-1e-12:issues.append(str(fold)+'/M_row_loss')
        if cm['ASA']['macro_f1_M_S']<rm['ASA']['macro_f1_M_S']-.005-1e-12:issues.append(str(fold)+'/macro_loss')
        for p in ['tcp','udp']:
            if cm['hard'][p]['1']<rm['hard'][p]['1']-.01-1e-12:issues.append(str(fold)+'/'+p+'/M_source_loss')
            if cm['hard'][p]['2']<rm['hard'][p]['2']-1e-12:issues.append(str(fold)+'/'+p+'/S_source_loss')
    assert np.array_equal(pred[~hard[:n]],base[~hard[:n]])
    result={'all':effect,'folds':details,'fold_metrics':fold_metrics,'fits_completed':12,'seconds':time.monotonic()-t0,
        'continuation_passed':not issues,'rejection_reasons':issues,'quality_acceptance':False}
    save(a.out/'analysis.json',result);print('COMPLETE',effect,issues,flush=True)


if __name__=='__main__':main()
