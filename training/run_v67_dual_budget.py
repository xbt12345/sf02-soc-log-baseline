"""Match calibration constraints to both row and source M protection.

Keep protocol separation; same saved models. No outer labels for thresholding.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from train_v65_rank_heads import ROOT,load_fit,measure
from run_v67_targeted import counts
from run_v67_specialist import hard_mask
from v61_common import read,save,sha


def threshold(cal,protocol):
    limits=[]
    for fold in range(3):
        d=cal[cal.transport_protocol.eq(protocol)&cal.inner_fold.eq(fold)&cal.label.eq(1)]
        assert len(d)>0 and d.group.nunique()>0
        sizes=d.groupby('group').size()
        for weights in [np.full(len(d),1/len(d)),1/len(sizes)/d.group.map(sizes).to_numpy()]:
            grouped=pd.DataFrame({'s':d.score.to_numpy(),'w':weights}).groupby('s').w.sum().sort_index(ascending=False)
            crossing=np.flatnonzero(grouped.cumsum().gt(.01+1e-12))
            limits.append(float(np.nextafter(grouped.index[crossing[0]],np.inf)) if len(crossing) else -np.inf)
    return max(limits)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    save(a.out/'preregistered.json',{'version':'v67-dual-M-budget-1','source_sha256':sha(__file__),
        'arms':['coarse','numeric'],'new_fits':0,'threshold_policy':'Protocol-specific; every inner fold must have <=1% M errors BOTH per row and averaged per source. Lowest feasible threshold, all tied scores together.',
        'reason':'Source-only cap allowed a few large M sources to dominate new row errors. Align with previously stated joint protection.',
        'quality_acceptance':False,'adaptive_development':True,'outer_answer_selection':False})
    f,c=load_fit();hard=hard_mask(f);manifest=pd.read_parquet(ROOT/'artifacts/v67_targeted_20260920/manifest.parquet')
    target=manifest.target_578.to_numpy();base=manifest.baseline_pred_with_normal_gate.to_numpy()
    allf=pd.read_parquet(ROOT/'artifacts/v61_source_factorial_20260914_r2/records.parquet').set_index('row_position')
    root=ROOT/'artifacts/v67_expansion_20260920';result={'quality_acceptance':False,'arms':{}}
    for arm in ['coarse','numeric']:
        o=pd.read_parquet(root/(arm+'_oof.parquet'));assert np.array_equal(o.row_position,f.row_position)
        pred=base.copy();thresholds={};checks=[]
        for fold in range(3):
            cal=pd.read_parquet(root/f'fold{fold}'/arm/'calibration_oof.parquet')
            cal['transport_protocol']=allf.loc[cal.row_position,'transport_protocol'].to_numpy()
            assert not set(cal.group)&set(f[f.fold.eq(fold)].group)
            thresholds[str(fold)]={}
            for p in ['tcp','udp']:
                t=threshold(cal,p);thresholds[str(fold)][p]=t
                ix=hard&f.fold.eq(fold).to_numpy()&f.transport_protocol.eq(p).to_numpy()
                pred[ix]=(o.hard_score[ix]>=t).to_numpy(int)+1
                for inner in range(3):
                    d=cal[cal.inner_fold.eq(inner)&cal.transport_protocol.eq(p)&cal.label.eq(1)].copy()
                    d['error']=d.score.ge(t);row=float(d.error.mean());source=float(d.groupby('group').error.mean().mean())
                    assert max(row,source)<=.01+1e-12
                    checks.append({'outer_fold':fold,'inner_fold':inner,'protocol':p,'M_row_error':row,'M_source_error':source})
        assert np.array_equal(pred[~hard],base[~hard])
        o['pred']=pred;o.to_parquet(a.out/(arm+'_oof.parquet'),index=False)
        eff=counts(f,pred,base,target);ref=counts(f,base,base,target);reasons=[]
        if eff['target_fixed']<58:reasons.append('target_repairs_below_58')
        if eff['S']['source_recall']<ref['S']['source_recall']+.05-1e-12:reasons.append('S_source_gain_below_5pp')
        folds={}
        for fold in range(3):
            ix=np.flatnonzero(f.fold.eq(fold));dv=f.iloc[ix].reset_index(drop=True)
            cm=measure(dv,np.eye(3)[pred[ix]]);rm=measure(dv,np.eye(3)[base[ix]])
            folds[str(fold)]={'candidate':cm,'reference':rm,'effects':counts(dv,pred[ix],base[ix],target[ix])}
            if cm['M_source_recall']<rm['M_source_recall']-.01-1e-12:reasons.append(str(fold)+'/M_source_loss')
            if cm['ASA']['recall_B_M_S'][1]<rm['ASA']['recall_B_M_S'][1]-.01-1e-12:reasons.append(str(fold)+'/M_row_loss')
            if cm['ASA']['macro_f1_M_S']<rm['ASA']['macro_f1_M_S']-.005-1e-12:reasons.append(str(fold)+'/macro_loss')
            for p in ['tcp','udp']:
                if cm['hard'][p]['1']<rm['hard'][p]['1']-.01-1e-12:reasons.append(str(fold)+'/'+p+'/M_source_loss')
                if cm['hard'][p]['2']<rm['hard'][p]['2']-1e-12:reasons.append(str(fold)+'/'+p+'/S_source_loss')
        result['arms'][arm]={'all':eff,'thresholds':thresholds,'inner_checks':checks,'folds':folds,'continuation_passed':not reasons,'rejection_reasons':reasons}
        print(arm,eff,reasons,flush=True)
    save(a.out/'analysis.json',result)


if __name__=='__main__':main()
