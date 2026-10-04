"""Verify decision refinements and publish a single complete row ledger."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train_v65_rank_heads import ROOT,load_fit,Inputs
from run_v67_targeted import counts
from run_v67_specialist import hard_mask
from v61_common import read,save,sha


def independent_threshold(cal,protocol,dual):
    boundaries=[]
    for fold in range(3):
        d=cal[cal.transport_protocol.eq(protocol)&cal.inner_fold.eq(fold)&cal.label.eq(1)]
        ct=d.groupby('group').size();source=1/len(ct)/d.group.map(ct).to_numpy()
        for weight in [source]+([np.ones(len(d))/len(d)] if dual else []):
            s=d.score.to_numpy();order=np.argsort(s);s=s[order];w=weight[order];c=np.r_[0,np.cumsum(w)]
            candidates=np.r_[np.unique(s),np.nextafter(np.unique(s),np.inf),np.inf]
            mass=c[-1]-c[np.searchsorted(s,candidates,side='left')]
            boundaries.append(float(candidates[mass<=.01+1e-12].min()))
    return max(boundaries)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    f,c=load_fit();manifest=pd.read_parquet(ROOT/'artifacts/v67_targeted_20260920/manifest.parquet')
    target=manifest.target_578.to_numpy();base=manifest.baseline_pred_with_normal_gate.to_numpy();hard=hard_mask(f)
    ledger=pd.read_parquet(ROOT/'artifacts/v67_row_review_20260920/target_578_ledger.parquet')
    assert np.array_equal(ledger.row_position,f.row_position[target])
    allf=pd.read_parquet(ROOT/'artifacts/v61_source_factorial_20260914_r2/records.parquet').set_index('row_position')
    tests=[];effects=[]
    for stage in ['protocol','dual_budget']:
        root=ROOT/f'artifacts/v67_{stage}_20260920';r=read(root/'analysis.json');reg=read(root/'preregistered.json')
        source_name='refine_v67_protocol.py' if stage=='protocol' else 'run_v67_dual_budget.py'
        assert reg['source_sha256']==sha(ROOT/'training'/source_name)
        for arm,detail in r['arms'].items():
            if stage=='protocol':oldstage,oldarm=reg['arms'][arm]
            else:oldstage,oldarm='expansion',arm
            source=ROOT/f'artifacts/v67_{oldstage}_20260920'
            o=pd.read_parquet(root/(arm+'_oof.parquet'));old=pd.read_parquet(source/(oldarm+'_oof.parquet'))
            assert np.array_equal(o.row_position,f.row_position)
            np.testing.assert_allclose(o.hard_score,old.hard_score,equal_nan=True,atol=0,rtol=0)
            pred=base.copy()
            for fold in range(3):
                cal=pd.read_parquet(source/f'fold{fold}'/oldarm/'calibration_oof.parquet')
                cal['transport_protocol']=allf.loc[cal.row_position,'transport_protocol'].to_numpy()
                assert not set(cal.group)&set(f[f.fold.eq(fold)].group)
                for p in ['tcp','udp']:
                    t=independent_threshold(cal,p,stage=='dual_budget')
                    assert t==detail['thresholds'][str(fold)][p]
                    mask=hard&f.fold.eq(fold).to_numpy()&f.transport_protocol.eq(p).to_numpy()
                    pred[mask]=(o.hard_score[mask]>=t).to_numpy(int)+1
            assert np.array_equal(pred,o.pred)
            assert counts(f,pred,base,target)==detail['all']
            assert not detail['continuation_passed']
            ledger[stage+'_'+arm+'_pred']=pred[target]
            effects.append({'candidate':stage+'_'+arm,**detail['all']})
            tests.append({'stage':stage,'arm':arm,'thresholds_recomputed':True,'all_row_decisions_recomputed':True})
    # Find actual training-side M counterexamples for all ten cross-fold conflicts.
    old=ROOT/'artifacts/v65_rank_heads_20260920';features=np.load(old/'upstream_text_features.npy')
    codes=pd.Index(pd.read_parquet(old/'unique_text.parquet').text).get_indexer(f.text);pairs=[]
    cross=target&manifest.training_exact_view_1_sources.gt(0).to_numpy()
    for fold in range(3):
        train=np.flatnonzero(f.fold.ne(fold));inputs=Inputs(f,c,features,codes,train,torch.device('cpu'))
        for i in np.flatnonzero(cross&f.fold.eq(fold).to_numpy()):
            match=train[(inputs.views[train]==inputs.views[i])&f.label.iloc[train].eq(1).to_numpy()]
            assert len(match)>0;j=int(match[0]);assert f.group.iloc[i]!=f.group.iloc[j]
            pairs.append({'S_row_position':int(f.row_position.iloc[i]),'M_training_row_position':int(f.row_position.iloc[j]),
                'S_group':int(f.group.iloc[i]),'M_training_group':int(f.group.iloc[j]),'fold':fold,
                'S_text':f.text.iloc[i],'M_text':f.text.iloc[j],'exact_model_input_sha256':str(inputs.views[i])})
    assert len(pairs)==10
    pd.DataFrame(pairs).to_csv(a.out/'ten_cross_fold_counterexamples.csv',index=False,encoding='utf-8-sig')
    pred_columns=[col for col in ledger if col.endswith('_pred') and col not in ['H0_pred','H1_pred']]
    ledger['any_candidate_fixed_diagnostic_only']=ledger[pred_columns].eq(2).any(axis=1)
    ledger['accepted_repair']=False
    ledger['resolution_status']=np.where(ledger.any_candidate_fixed_diagnostic_only,
        'candidate_repair_rejected_due_to_collateral_or_source_gate','unresolved_in_every_tested_candidate')
    ledger.to_parquet(a.out/'target_578_ledger.parquet',index=False)
    ledger.to_csv(a.out/'target_578_ledger.csv',index=False,encoding='utf-8-sig')
    source_rows=ledger.groupby(['fold','group']).agg(rows=('row_position','size'),any_candidate_repairs=('any_candidate_fixed_diagnostic_only','sum'),accepted_repairs=('accepted_repair','sum')).reset_index()
    source_rows.to_csv(a.out/'target_162_sources.csv',index=False)
    source_checks=read(ROOT/'artifacts/v67_verification_r2_20260920/verification.json')
    save(a.out/'verification.json',{'verification_passed':True,'model_replay':source_checks['models_restored'],
        'decision_refinement_checks':tests,'cross_fold_counterexample_pairs_verified':10,
        'target_rows':len(ledger),'target_sources':int(ledger.group.nunique()),
        'any_candidate_union_NOT_A_MODEL':int(ledger.any_candidate_fixed_diagnostic_only.sum()),
        'accepted_repairs':0,'accepted_remaining':578,'quality_acceptance':False,
        'source_sha256':sha(__file__)})
    print('FINAL_LEDGER',len(ledger),int(ledger.any_candidate_fixed_diagnostic_only.sum()),'accepted=0',flush=True)


if __name__=='__main__':main()
