"""Fix cross-protocol score-scale coupling, using inner labels only.

No retraining. Two expanded and two non-expanded plain specialist controls.
Thresholds cannot be chosen from outer predictions or target membership.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from train_v65_rank_heads import ROOT,load_fit,measure
from run_v67_targeted import counts
from run_v67_specialist import hard_mask
from v61_common import read,save,sha


def calibrated_threshold(cal,protocol):
    part=cal[cal.transport_protocol.eq(protocol)]
    scores=part.score.to_numpy();choices=np.r_[np.unique(scores),np.nextafter(np.unique(scores),np.inf),np.inf]
    allowed=np.ones(len(choices),bool)
    for fold in range(3):
        d=part[part.inner_fold.eq(fold)&part.label.eq(1)]
        assert d.group.nunique()>0
        size=d.groupby('group').size();weights=1/len(size)/d.group.map(size).to_numpy()
        order=np.argsort(d.score.to_numpy());s=d.score.to_numpy()[order];w=weights[order]
        cumulative=np.r_[0,np.cumsum(w)]
        loss=cumulative[-1]-cumulative[np.searchsorted(s,choices,side='left')]
        allowed&=loss<=.01+1e-12
    return float(choices[allowed].min())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    f,c=load_fit();manifest=pd.read_parquet(ROOT/'artifacts/v67_targeted_20260920/manifest.parquet')
    target=manifest.target_578.to_numpy();base=manifest.baseline_pred_with_normal_gate.to_numpy();hard=hard_mask(f)
    sources={'specialist_coarse':('specialist','coarse_plain'),'specialist_numeric':('specialist','numeric_plain'),
             'expanded_coarse':('expansion','coarse'),'expanded_numeric':('expansion','numeric')}
    # Frozen before reading outer scores: no per-row model union or S-only rescue.
    save(a.out/'preregistered.json',{'version':'v67-protocol-threshold-repair-1','source_sha256':sha(__file__),
        'arms':sources,'training_fits_added':0,'decision_rule':'Separate TCP and UDP threshold, each from 3 inner source folds; <=1% M-source error in each inner fold. Both M/S outputs replace base inside hard branch.',
        'motivation':'All six expanded shared thresholds are set by UDP, despite lower admissible TCP thresholds.',
        'primary':'expanded_coarse','quality_acceptance':False,'adaptive_development':True,
        'continuation':'Reuse originally registered v67 target>=58, S-source gain>=5pp, per-fold M row/source/macro and per-protocol M budget and no hard-S decline; normal boundary unchanged.',
        'source_receipts':{s:sha(ROOT/f'artifacts/v67_{stage}_20260920/preregistered.json') for s,(stage,_) in sources.items()}})
    allf=pd.read_parquet(ROOT/'artifacts/v61_source_factorial_20260914_r2/records.parquet').set_index('row_position')
    results={'baseline':counts(f,base,base,target),'arms':{},'quality_acceptance':False,'new_models_trained':0}
    for name,(stage,arm) in sources.items():
        run=ROOT/f'artifacts/v67_{stage}_20260920';o=pd.read_parquet(run/(arm+'_oof.parquet'))
        assert np.array_equal(o.row_position,f.row_position)
        pred=base.copy();thresholds={};cells=[]
        for fold in range(3):
            folder=run/f'fold{fold}'/arm;cal=pd.read_parquet(folder/'calibration_oof.parquet')
            cal['transport_protocol']=allf.loc[cal.row_position,'transport_protocol'].to_numpy()
            assert not set(cal.group)&set(f.loc[f.fold.eq(fold),'group'])
            thresholds[str(fold)]={}
            for protocol in ['tcp','udp']:
                threshold=calibrated_threshold(cal,protocol);thresholds[str(fold)][protocol]=threshold
                use=f.fold.eq(fold)&hard&f.transport_protocol.eq(protocol)
                pred[use]=(o.hard_score[use]>=threshold).to_numpy(int)+1
                # Explicitly demonstrate retained inner constraints after splitting.
                for inner in range(3):
                    subset=cal[cal.inner_fold.eq(inner)&cal.transport_protocol.eq(protocol)].copy()
                    subset['alert']=subset.score.ge(threshold)
                    by_source=subset[subset.label.eq(1)].groupby('group').alert.mean()
                    assert by_source.mean()<=.01+1e-12
                    cells.append({'outer_fold':fold,'inner_fold':inner,'protocol':protocol,'M_source_error':float(by_source.mean()),
                        'S_source_recall':float(subset[subset.label.eq(2)].groupby('group').alert.mean().mean()),
                        'S_sources_recalled':int(subset[subset.label.eq(2)&subset.alert].group.nunique())})
        assert np.array_equal(pred[~hard],base[~hard])
        out=f[['row_position','label','group','fold']].copy();out['pred']=pred;out['hard_score']=o.hard_score
        out.to_parquet(a.out/(name+'_oof.parquet'),index=False)
        eff=counts(f,pred,base,target);ref=results['baseline'];issues=[]
        if eff['target_fixed']<58:issues.append('target_repairs_below_58')
        if eff['S']['source_recall']<ref['S']['source_recall']+.05-1e-12:issues.append('S_source_gain_below_5pp')
        if eff['M']['errors']>ref['M']['errors']+.01*ref['M']['rows']:issues.append('M_row_loss')
        if eff['M']['source_recall']<ref['M']['source_recall']-.01-1e-12:issues.append('M_source_loss')
        folds={}
        for fold in range(3):
            ix=np.flatnonzero(f.fold.eq(fold));dv=f.iloc[ix].reset_index(drop=True)
            cm=measure(dv,np.eye(3)[pred[ix]]);rm=measure(dv,np.eye(3)[base[ix]])
            folds[str(fold)]={'effects':counts(dv,pred[ix],base[ix],target[ix]),'candidate':cm,'reference':rm}
            if cm['M_source_recall']<rm['M_source_recall']-.01-1e-12:issues.append(str(fold)+'/M_source_loss')
            if cm['ASA']['recall_B_M_S'][1]<rm['ASA']['recall_B_M_S'][1]-.01-1e-12:issues.append(str(fold)+'/M_row_loss')
            if cm['ASA']['macro_f1_M_S']<rm['ASA']['macro_f1_M_S']-.005-1e-12:issues.append(str(fold)+'/macro_loss')
            for protocol in ['tcp','udp']:
                if cm['hard'][protocol]['1']<rm['hard'][protocol]['1']-.01-1e-12:issues.append(str(fold)+'/'+protocol+'/M_source_loss')
                if cm['hard'][protocol]['2']<rm['hard'][protocol]['2']-1e-12:issues.append(str(fold)+'/'+protocol+'/S_source_loss')
        results['arms'][name]={'all':eff,'thresholds':thresholds,'inner_constraint_checks':cells,'folds':folds,
                             'continuation_passed':not issues,'rejection_reasons':issues}
        print(name,eff,issues,flush=True)
    save(a.out/'analysis.json',results)


if __name__=='__main__':main()
