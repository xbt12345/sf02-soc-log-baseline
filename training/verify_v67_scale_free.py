"""Actual scale-free model reload and final merged ledger, no fitting."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from train_v65_rank_heads import ROOT
from run_v67_expansion import expanded_data
from run_v67_targeted import design,counts
from run_v67_specialist import hard_mask
from finish_v67_evidence import independent_threshold
from v61_common import read,save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    run=ROOT/'artifacts/v67_scale_free_20260920';reg=read(run/'preregistered.json')
    assert reg['source_sha256']==sha(ROOT/'training/run_v67_scale_free.py')
    f,c,n=expanded_data();idx=pd.Index(f.row_position);hard=hard_mask(f)
    o=pd.read_parquet(run/'oof.parquet');manifest=pd.read_parquet(ROOT/'artifacts/v67_targeted_20260920/manifest.parquet')
    base=manifest.baseline_pred_with_normal_gate.to_numpy();target=manifest.target_578.to_numpy()
    replay=base.copy();models=0;rows=0
    def inputs(bundle,train):
        x,enc,names,cat=design(f,c,'context_coarse',train,bundle['encoder'])
        keep=[j for j,name in enumerate(names) if name not in reg['drop']]
        assert [names[j] for j in keep]==bundle['names']
        categorical=names[:len(enc.categories_)]
        for col,values in zip(categorical,enc.categories_):assert set(values)==set(f[col].iloc[train])
        return x[:,keep]
    with threadpool_limits(limits=4):
        for fold in range(3):
            folder=run/f'fold{fold}';cal=pd.read_parquet(folder/'calibration_oof.parquet');ci=idx.get_indexer(cal.row_position)
            assert np.array_equal(ci,np.flatnonzero(f.fold.ne(fold)&hard))
            inner=cal.inner_fold.to_numpy()
            for k in range(3):
                b=joblib.load(folder/f'inner{k}.joblib');tr=idx.get_indexer(b['train_positions']);vi=ci[inner==k]
                assert np.array_equal(tr,ci[inner!=k]);assert not set(f.group.iloc[tr])&set(f.group.iloc[vi])
                assert not set(f.body_group.iloc[tr])&set(f.body_group.iloc[vi])
                x=inputs(b,tr);p=b['model'].predict_proba(x[vi])[:,1]
                np.testing.assert_allclose(p,cal.score[inner==k],atol=0,rtol=0);models+=1;rows+=len(vi)
            b=joblib.load(folder/'model.joblib');tr=idx.get_indexer(b['train_positions']);va=np.flatnonzero(f.fold.eq(fold)&hard)
            assert np.array_equal(tr,ci);assert not set(f.group.iloc[tr])&set(f.group.iloc[va])
            x=inputs(b,tr);p=b['model'].predict_proba(x[va])[:,1]
            np.testing.assert_allclose(p,o.hard_score.iloc[va],atol=0,rtol=0);models+=1;rows+=len(va)
            for protocol in ['tcp','udp']:
                t=independent_threshold(cal,protocol,True);assert t==b['thresholds'][protocol]
                mask=f.transport_protocol.iloc[va].eq(protocol).to_numpy()
                replay[va[mask]]=(p[mask]>=t).astype(int)+1
    assert np.array_equal(replay,o.pred)
    assert counts(f.iloc[:n],replay,base,target)==read(run/'analysis.json')['all']
    assert not read(run/'analysis.json')['continuation_passed']
    ledger=pd.read_parquet(ROOT/'artifacts/v67_final_evidence_20260920/target_578_ledger.parquet')
    assert np.array_equal(ledger.row_position,f.row_position.iloc[:n][target])
    ledger['scale_free_pred']=replay[target]
    ledger['any_candidate_fixed_diagnostic_only']|=ledger.scale_free_pred.eq(2)
    ledger['accepted_repair']=False
    ledger['resolution_status']=np.where(ledger.any_candidate_fixed_diagnostic_only,
        'candidate_repair_rejected_due_to_collateral_or_source_gate','unresolved_in_every_tested_candidate')
    ledger.to_csv(a.out/'target_578_ledger.csv',index=False,encoding='utf-8-sig')
    ledger.to_parquet(a.out/'target_578_ledger.parquet',index=False)
    ledger.groupby(['fold','group']).agg(rows=('row_position','size'),any_candidate_repairs=('any_candidate_fixed_diagnostic_only','sum'),accepted_repairs=('accepted_repair','sum')).reset_index().to_csv(a.out/'target_162_sources.csv',index=False)
    summary={'verification_passed':True,'models_restored':models,'rows_replayed':rows,'max_probability_difference':0,
        'all_v67_models_verified':84+models,'all_v67_prediction_rows_replayed':1196622+rows,
        'thresholds_independently_recomputed':6,'target_rows':len(ledger),'target_sources':int(ledger.group.nunique()),
        'any_candidate_union_NOT_A_MODEL':int(ledger.any_candidate_fixed_diagnostic_only.sum()),
        'accepted_repairs':0,'accepted_remaining':578,'quality_acceptance':False,'source_sha256':sha(__file__)}
    save(a.out/'verification.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
