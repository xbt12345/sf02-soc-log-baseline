"""Saved-result challenge: independent paired effect, risk frontier and gaps. No fit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from run_v71_resolution import ROOT, ARMS, load, hard
from verify_v69_support_control import independent_cut
from v61_common import read, save, sha


def audit(out):
    f,c,m=load();full=f.set_index('row_position').loc[m.row_position].reset_index();h=hard(full)
    ds={a:pd.read_parquet(out/f'{a}_budget0.01_oof.parquet') for a in ARMS}
    for d in ds.values():assert np.array_equal(d.row_position,m.row_position)
    full['Q']=ds['Q_old_bins'].pred.to_numpy();full['R']=ds['R_observed'].pred.to_numpy()
    full['Q_correct']=full.Q.eq(full.label);full['R_correct']=full.R.eq(full.label)
    source=full[h].groupby(['label','fold','group']).agg(rows=('label','size'),Q_accuracy=('Q_correct','mean'),R_accuracy=('R_correct','mean')).reset_index()
    source['delta']=source.R_accuracy-source.Q_accuracy;source.to_csv(out/'paired_source_effects.csv',index=False,encoding='utf-8-sig')
    s=source[source.label.eq(2)];gain=float(s.delta.mean());rng=np.random.default_rng(20260921);sums=np.zeros(5000)
    for fold,z in s.groupby('fold'):
        delta=z.delta.to_numpy(); draw=rng.integers(0,len(delta),size=(5000,len(delta)))
        sums+=delta[draw].sum(axis=1)
    ci=np.quantile(sums/len(s),[.025,.975]);ref=read(out/'analysis.json')['R_vs_Q']
    np.testing.assert_allclose(ci,ref['descriptive_bootstrap_95pct'],atol=1e-14,rtol=0)
    assert abs(gain-ref['source_recall_gain'])<1e-14
    assert abs(float(s.delta.drop(s.delta.idxmax()).mean())-ref['gain_after_removing_best_source'])<1e-14
    cells=[]
    for fold in range(3):
        for proto in ['tcp','udp']:
            row={'fold':fold,'protocol':proto}
            for arm in ARMS:
                z=pd.read_parquet(out/f'fold{fold}'/arm/'evaluation.parquet');z=z[z.transport_protocol.eq(proto)].copy()
                cut=independent_cut(z,.01);z['pred']=np.where(z.score>=cut,2,1)
                ss=z[z.label.eq(2)];mm=z[z.label.eq(1)]
                row[arm+'_oracle_S_source']=float(ss.pred.eq(2).groupby(ss.group).mean().mean())
                row[arm+'_oracle_M_row_error']=float(mm.pred.eq(2).mean())
                row[arm+'_oracle_M_source_error']=float(mm.pred.eq(2).groupby(mm.group).mean().mean())
                assert max(row[arm+'_oracle_M_row_error'],row[arm+'_oracle_M_source_error'])<=.01+1e-12
                sizes=z.groupby(['label','group']).label.transform('size');n=z.groupby('label').group.nunique()
                w=1/sizes/z.label.map(n)
                row[arm+'_pAUC']=float(roc_auc_score(z.label.eq(2),z.score,sample_weight=w,max_fpr=.01))
                # S labels do not choose calibration thresholds: changing every S
                # to an irrelevant non-M code leaves M calibration evidence intact.
                cd=pd.read_parquet(out/f'fold{fold}'/arm/'calibration.parquet');cd=cd[cd.transport_protocol.eq(proto)].copy()
                cc=independent_cut(cd,.01);cd.loc[cd.label.eq(2),'label']=0
                assert independent_cut(cd,.01)==cc
            cells.append(row)
    frontier=float(np.mean([z['R_observed_oracle_S_source']-z['Q_old_bins_oracle_S_source'] for z in cells]))
    pauc=float(np.mean([z['R_observed_pAUC']-z['Q_old_bins_pAUC'] for z in cells]))
    assert abs(frontier-ref['mean_six_cell_oracle_S_source_gain'])<1e-12
    assert abs(pauc-ref['mean_six_cell_pAUC_gain'])<1e-12
    p=pd.read_parquet(out/'collision_pair_paths.parquet')
    supplemental=read(ROOT/'docs/review_v70/supplemental_collision_fields.json')
    t=pd.DataFrame(supplemental['targets']).set_index('row_position')
    hit=p.groupby('S_row_position').agg(any_path_difference=('R_same_all_tree_paths',lambda v:bool((~v).any())),
          any_S_above_M=('R_score_orders_S_above_M','any')).join(t[['group','empty_context','difference_types']])
    hit['type']=hit.difference_types.map('|'.join)
    types=hit.groupby(['type','empty_context']).agg(rows=('group','size'),sources=('group','nunique'),
          targets_with_path_difference=('any_path_difference','sum'),targets_with_any_better_order=('any_S_above_M','sum')).reset_index()
    fit=pd.read_parquet(out/'newly_conflicted_training_S.parquet')
    fitgap=fit.groupby('fold').agg(rows=('row_position','size'),sources=('group','nunique'),
        Q_mean_loss=('Q_old_bins_logloss','mean'),R_mean_loss=('R_observed_logloss','mean'),
        Q_correct_at_frozen_cut=('Q_old_bins_correct_primary','sum'),R_correct_at_frozen_cut=('R_observed_correct_primary','sum')).reset_index()
    target=full[m.target_578.to_numpy()].copy()
    target['original_input_status']=np.where(m.loc[m.target_578,'training_exact_view_1_sources'].to_numpy()>0,'old_exact_fit_M',np.where(m.loc[m.target_578,'empty_context'].to_numpy(),'novel_no_context','novel_with_context'))
    slices=target.groupby('original_input_status').agg(rows=('label','size'),Q_correct=('Q_correct','sum'),R_correct=('R_correct','sum')).reset_index()
    assert len(target)==578 and sum(slices.rows)==578
    old_analysis=read(ROOT/'artifacts/v69_support_control_20260921/analysis.json')
    history={'v69_B':old_analysis['arms']['B_small_sources']['budgets']['0.01']['historical_effects']}
    for arm in ARMS:history[arm]=read(out/'analysis.json')['arms'][arm]['budgets']['0.01']['historical_effects']
    inputs=[out/'analysis.json',out/'mechanism.json',out/'collision_pair_paths.parquet',out/'newly_conflicted_training_S.parquet',ROOT/'docs/review_v70/supplemental_collision_fields.json']
    inputs += [out/f'{a}_budget0.01_oof.parquet' for a in ARMS]
    result={'scope':'Post-fit descriptive audit only; no new fits, no threshold deployment or model selection',
        'independent_source_bootstrap_matches':True,'independent_six_cell_frontier_matches':True,'calibration_S_label_irrelevance_checks':12,
        'paired_source_effect':ref,'fixed_score_oracle_cells_NOT_DEPLOYABLE':cells,
        'collision_72_path_breakdown':types.to_dict('records'),'newly_conflicted_fit_S':fitgap.to_dict('records'),
        'target_578_original_status_slices':slices.to_dict('records'),'historical_regression_not_matched_model_family':history,
        'collision_72_Q_correct':int(target.loc[target.row_position.isin(hit.index),'Q_correct'].sum()),
        'collision_72_R_correct':int(target.loc[target.row_position.isin(hit.index),'R_correct'].sum()),
        'new_fits':0,'source_sha256':sha(__file__),'input_sha256':{str(v.relative_to(ROOT)):sha(v) for v in inputs}}
    save(out/'adversarial_audit.json',result)
    print(json.dumps({k:result[k] for k in ['independent_source_bootstrap_matches','independent_six_cell_frontier_matches','collision_72_path_breakdown','newly_conflicted_fit_S','target_578_original_status_slices','collision_72_R_correct']},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();audit(args.out.resolve())
