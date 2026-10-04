"""Retain actual collateral rows and parent audit; no classifier execution."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v138_runtime import save
OUT=ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001'

def main():
    assert not (OUT/'residual_transfer_review.json').exists()
    parent=ROOT/'artifacts/v155_independent_endpoint_audit_v2_20261001/audit.json'
    audit=read(parent);check_bindings(audit['source_sha256'])
    assert audit['official_rows']==2056871 and audit['own_model_forwards']==audit['own_gradients']==audit['own_fits']==audit['own_updates']==0
    source=OUT/'ASA_prediction_ledger.parquet';rows=pd.read_parquet(source)
    reg=rows[rows.pred_V146_A.eq(rows.truth)&rows.pred_B.ne(rows.truth)].copy()
    fix=rows[rows.pred_V146_A.ne(rows.truth)&rows.pred_B.eq(rows.truth)].copy()
    assert len(reg)==16 and reg.truth.eq(1).all() and len(fix)==0
    for f in range(3):
        mask=reg.fold.eq(f);local=reg.loc[mask,'local'].to_numpy()
        for arm in ['A','B']:
            q=np.load(OUT/f'fold{f}_{arm}/sealed_all_prob.npy')
            for c in range(3):reg.loc[mask,f'{arm}_p{c}']=q[local,c]
    assert np.array_equal(reg.B_p2.gt(reg.B_p1),reg.pred_B.eq(2))
    path=OUT/'new_M_regressions_vs_V146_A.parquet';reg.to_parquet(path,index=False)
    result=dict(status='actual_16_collateral_rows_preserved_no_new_fit_qualified',regressions=16,repairs=0,
        canonical_inputs=int(reg.canonical_key.nunique()),actual_local_inputs=int(reg.local.nunique()),
        by_fold={str(k):int(v) for k,v in reg.groupby('fold').size().items()},by_root={str(k):int(v) for k,v in reg.groupby('root').size().items()},
        parent_status=audit['status'],parent_fixed_cohorts=audit['fixed_cohorts'],
        new_model_forwards=0,new_gradients=0,new_fits=0,new_updates=0,
        decision='Current fixed neighborhood configuration rejected. These rows describe residual failure, not label/frequency/guard changes or automatic qualification of a new factor.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),parent,source,path]})
    save(OUT/'residual_transfer_review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','parent_fixed_cohorts','decision']},ensure_ascii=False))

if __name__=='__main__':main()
