"""Saved per-row local progress pressure test, zero model or gradient calls."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,sha,read,check_bindings
from v159_float64_repeat_policy_v2 import SEGMENTS
OUT=ROOT/'artifacts/v161_saved_progress_trajectory_pressure_review_20261002'
DIAG=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002'
TAIL=ROOT/'artifacts/v161_cached_direction_backtrack_tail_20261002'
COHORT=ROOT/'artifacts/v161_independent_frozen_error_cohort_review_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();reports=[];files={Path(__file__).resolve(),ROOT/'training/v159_float64_repeat_policy_v2.py'}
    for role,j in [(0,10),(1,14),(2,20)]:
        baseline=DIAG/f'role{role}/baseline_error_class1_repeat0';probe=DIAG/f'role{role}/round0/probe{j}' if role!=2 else TAIL/'role2/probe20'
        paths=[baseline/'OOF_original_rows.parquet',probe/'OOF_original_rows.parquet',probe/'probe.json',COHORT/f'role{role}/fixed_pure_error_targets.parquet',DIAG/f'role{role}/round0/polished_direction.npy'];files|=set(paths)
        b=pd.read_parquet(paths[0]);a=pd.read_parquet(paths[1]);assert np.array_equal(b.row_position,a.row_position) and np.array_equal(b.truth,a.truth)
        targets=pd.read_parquet(paths[3]);mask=b.row_position.isin(targets.row_position).to_numpy();truth=b.truth.to_numpy(np.int64);rival=b.pred.to_numpy(np.int64);ii=np.arange(len(b));bl=b[['logp0','logp1','logp2']].to_numpy();al=a[['logp0','logp1','logp2']].to_numpy()
        margin=(bl[ii,truth]-bl[ii,rival])[mask];delta=((al-bl)[ii,truth]-(al-bl)[ii,rival])[mask]
        assert np.all(margin<0) and int(a.loc[mask,'pred'].eq(a.loc[mask,'truth']).sum())==0
        needed=np.full(len(margin),np.inf);positive=delta>0;needed[positive]=-margin[positive]/delta[positive]
        ledger=b.loc[mask,['row_position','local','root','truth']].copy();ledger['baseline_wrong_margin']=margin;ledger['actual_safe_step_margin_delta']=delta;ledger['same_progress_linear_steps_to_old_rival']=needed;ledger.to_parquet(OUT/f'role{role}_local_progress_original_rows.parquet',index=False)
        byclass=[]
        for cls in [1,2]:
            mm=ledger.truth.eq(cls).to_numpy();n=needed[mm];byclass.append(dict(class_id=cls,original_target_rows=int(mm.sum()),improved_rows=int((delta[mm]>0).sum()),median_current_wrong_margin=float(np.median(margin[mm])),median_actual_safe_delta=float(np.median(delta[mm])),median_local_constant_progress_steps=float(np.median(n)),within_100_same_progress_steps=int((n<=100).sum()),actual_classification_repairs=0))
        direction=np.load(paths[4]);step=read(paths[2])['step'];segments=[]
        for name,start,end in SEGMENTS:
            update=step*direction[start:end];segments.append(dict(segment=name,nonzero_coordinates=int(np.count_nonzero(update)),update_infinity_norm=float(np.max(np.abs(update))),update_L2=float(np.linalg.norm(update))))
        reports.append(dict(role=role,step=step,classes=byclass,first_finite_candidate_parameter_increment=segments))
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};check_bindings(binding)
    result=dict(status='actual_saved_safe_progress_requires_trajectory_or_nonlinear_feasibility_evidence_before_long_fit',reports=reports,scope='Per-original-row constant-local-progress extrapolation only. Future directions, curvature, hidden learning and rates can change; this is neither a 100-iteration failure proof nor actual future mastery.',no_new_fixed_threshold_or_loss=True,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,quality_acceptance=False,source_sha256=binding)
    (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],reports=reports,official_calls=0)))

if __name__=='__main__':main()
