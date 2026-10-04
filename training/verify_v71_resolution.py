"""Independent decisions, physical model replay, and real collision path audit.

No fitting. The fixed 72 targets are diagnostics, never training inputs/weights.
"""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from run_v71_resolution import ROOT, OLD, ARMS, PORTS, load, matrices, ids, collisions, bindings, xgb
from run_v69_support_control import BUDGETS, roles, hard
from verify_v69_support_control import independent_cut
from v61_common import read, save, sha


def verify(out):
    bindings(out)
    f,c,m=load(); original=f.set_index('row_position').loc[m.row_position].reset_index()
    target=m.target_578.to_numpy();base=m.baseline_pred_with_normal_gate.to_numpy();h=hard(original)
    summary=read(out/'analysis.json');replays=0;cuts_checked=0
    all_predictions={(a,b):base.copy() for a in ARMS for b in BUDGETS}
    mechanism=[];pair_rows=[];fit_rows=[];old_bindings=read(ROOT/'evidence/2026-09-21/v69_support_control/delivery.json')['artifact_sha256']
    supplemental=read(ROOT/'docs/review_v70/supplemental_collision_fields.json')
    affected={t['row_position'] for t in supplemental['targets']}
    with threadpool_limits(limits=4):
        for fold in range(3):
            rs=pd.read_parquet(out/f'fold{fold}_roles.parquet')
            fresh,fc,fe=roles(f,fold);fit=rs.B_small_sources.to_numpy();cal=rs.calibration.to_numpy();ev=rs.evaluation.to_numpy()
            assert np.array_equal(fit,fresh['B_small_sources']) and np.array_equal(cal,fc) and np.array_equal(ev,fe)
            for a,b in [(fit,cal),(fit,ev),(cal,ev)]:
                for field in ['group','body_group']: assert not set(f.loc[a,field])&set(f.loc[b,field])
            encoding=joblib.load(out/f'fold{fold}_encoding.joblib');oldbundle=joblib.load(ROOT/encoding['old_model_path'])
            assert sha(ROOT/encoding['old_model_path'])==encoding['old_model_sha256']
            x,_,names,_,_=matrices(f,c,oldbundle,fit,encoder=encoding['onehot'])
            catfields=[n for n in __import__('v61_common').FIELDS if n not in PORTS]
            for col,vocab in zip(catfields,encoding['onehot'].categories_):assert set(vocab)==set(f.loc[fit,col].astype(str))
            with np.load(out/f'fold{fold}_matrices.npz') as z:
                for arm in ARMS:np.testing.assert_allclose(x[arm],z[arm],equal_nan=True,rtol=0,atol=0)
            model={};scores={};cfg={};fold_diag={'fold':fold,'arms':{}}
            for arm in ARMS:
                sub=out/f'fold{fold}'/arm;model[arm]=xgb.Booster();model[arm].load_model(sub/'model.ubj');model[arm].set_param({'nthread':4})
                assert sha(sub/'model.ubj')==read(sub/'fit_receipt.json')['model_sha256']
                assert model[arm].num_boosted_rounds()==500
                cfg[arm]=read(sub/'thresholds.json');scores[arm]={}
                for role,mask in [('fit',fit),('calibration',cal),('evaluation',ev)]:
                    d=pd.read_parquet(sub/(role+'.parquet'));assert np.array_equal(d.row_position,f.loc[mask,'row_position'])
                    assert np.array_equal(d.label,f.loc[mask,'label'])
                    p=model[arm].predict(xgb.DMatrix(x[arm][mask],nthread=4)).astype(np.float64)
                    np.testing.assert_array_equal(p,d.score);replays+=len(d)
                    scores[arm][role]=d
                ed=scores[arm]['evaluation'];ix=pd.Index(m.row_position).get_indexer(ed.row_position)
                for budget in BUDGETS:
                    for proto in ['tcp','udp']:
                        cd=scores[arm]['calibration'];cd=cd[cd.transport_protocol.eq(proto)]
                        cut=independent_cut(cd,budget);assert cut==cfg[arm][str(budget)][proto]['threshold'];cuts_checked+=1
                        mask=ed.transport_protocol.eq(proto).to_numpy()
                        all_predictions[(arm,budget)][ix[mask]]=np.where(ed.score.to_numpy()[mask]>=cut,2,1)
                trees=model[arm].trees_to_dataframe();splits=trees[trees.Feature.ne('Leaf')]
                binsplit={}
                for port in PORTS:
                    j=names.index(port);raw=x['R_observed'][fit,j];q=x['Q_old_bins'][fit,j]
                    limits=pd.DataFrame({'raw':raw,'bin':q}).dropna().groupby('bin').raw.agg(['min','max'])
                    thresholds=splits.loc[splits.Feature.eq('f'+str(j)),'Split'].to_numpy()
                    binsplit[port]={'split_nodes':len(thresholds),'nodes_potentially_splitting_old_bins':int(sum(((limits['min'].to_numpy()<t)&(limits['max'].to_numpy()>=t)).any() for t in thresholds)) if arm=='R_observed' else None}
                fold_diag['arms'][arm]={'numeric_port_splits':binsplit,'total_split_nodes':len(splits),'fit_full_view_collisions':collisions(x[arm][fit],f.loc[fit,'label'])[0]}
            # All target-M pairs matching Q's full input. R differs only in ports.
            qkey=ids(x['Q_old_bins']);rkey=ids(x['R_observed'])
            targets=ev & f.row_position.isin(affected).to_numpy(); pairs=[]
            for i in np.flatnonzero(targets):
                match=np.flatnonzero(fit & f.label.eq(1).to_numpy() & (qkey==qkey[i]));assert len(match)
                for j in match:
                    assert rkey[i]!=rkey[j]
                    pairs.append((i,j))
            u=np.unique(np.array(pairs).ravel());lookup={int(v):i for i,v in enumerate(u)}
            leaves={a:model[a].predict(xgb.DMatrix(x[a][u],nthread=4),pred_leaf=True) for a in ARMS}
            ps={a:model[a].predict(xgb.DMatrix(x[a][u],nthread=4)).astype(float) for a in ARMS}
            for i,j in pairs:
                a,b=lookup[i],lookup[j]
                assert np.array_equal(leaves['Q_old_bins'][a],leaves['Q_old_bins'][b])
                assert ps['Q_old_bins'][a]==ps['Q_old_bins'][b]
                cut=cfg['R_observed']['0.01'][f.transport_protocol.iloc[i]]['threshold']
                pair_rows.append({'fold':fold,'S_row_position':int(f.row_position.iloc[i]),'M_fit_row_position':int(f.row_position.iloc[j]),
                    'S_group':int(f.group.iloc[i]),'M_group':int(f.group.iloc[j]),'Q_same_all_tree_paths':True,
                    'R_same_all_tree_paths':bool(np.array_equal(leaves['R_observed'][a],leaves['R_observed'][b])),
                    'R_different_tree_count':int((leaves['R_observed'][a]!=leaves['R_observed'][b]).sum()),
                    'R_S_score':float(ps['R_observed'][a]),'R_M_fit_score':float(ps['R_observed'][b]),
                    'R_score_orders_S_above_M':bool(ps['R_observed'][a]>ps['R_observed'][b]),
                    'R_both_pair_labels_correct_at_primary':bool(ps['R_observed'][a]>=cut and ps['R_observed'][b]<cut)})
            cr,mr=collisions(x['R_observed'][fit],f.loc[fit,'label']);cq,mq=collisions(x['Q_old_bins'][fit],f.loc[fit,'label'])
            newly=mq & ~mr & f.loc[fit,'label'].eq(2).to_numpy()
            for k,i in enumerate(np.flatnonzero(fit)):
                if not newly[k]:continue
                row={'fold':fold,'row_position':int(f.row_position.iloc[i]),'group':int(f.group.iloc[i]),'protocol':f.transport_protocol.iloc[i]}
                for arm in ARMS:
                    p=float(scores[arm]['fit'].score.iloc[k]);cut=cfg[arm]['0.01'][row['protocol']]['threshold']
                    row[arm+'_score']=p;row[arm+'_logloss']=-float(np.log(max(p,1e-15)));row[arm+'_correct_primary']=p>=cut
                fit_rows.append(row)
            mechanism.append(fold_diag)
    for (arm,budget),pred in all_predictions.items():
        d=pd.read_parquet(out/f'{arm}_budget{budget}_oof.parquet');np.testing.assert_array_equal(pred,d.pred)
        np.testing.assert_array_equal(pred[~h],base[~h])
        a=summary['arms'][arm]['budgets'][str(budget)]
        assert confusion_matrix(original.label,pred,labels=[0,1,2]).tolist()==a['confusion_B_M_S']
        assert int((pred[target]==2).sum())==a['historical_effects']['target_fixed']
        for y,name in [(0,'B'),(1,'M'),(2,'S')]:
            mask=original.label.eq(y).to_numpy();rr=a['historical_effects'][name]
            assert int((mask&(pred!=y)).sum())==rr['errors']
            assert int((mask&(base!=y)&(pred==y)).sum())==rr['fixed']
            assert int((mask&(base==y)&(pred!=y)).sum())==rr['broken']
    assert len(pair_rows)==788
    pp=pd.DataFrame(pair_rows);assert pp.S_row_position.nunique()==72 and pp.S_group.nunique()==36
    pp.to_parquet(out/'collision_pair_paths.parquet',index=False)
    pd.DataFrame(fit_rows).to_parquet(out/'newly_conflicted_training_S.parquet',index=False)
    save(out/'mechanism.json',{'folds':mechanism,'target_M_pairs':len(pp),'targets':72,'target_sources':36,
        'pairs_R_distinct_paths':int((~pp.R_same_all_tree_paths).sum()),
        'targets_R_any_distinct_paths':int(pp.loc[~pp.R_same_all_tree_paths,'S_row_position'].nunique()),
        'pairs_R_score_orders_S_above_M':int(pp.R_score_orders_S_above_M.sum()),
        'pairs_R_both_labels_correct_primary':int(pp.R_both_pair_labels_correct_at_primary.sum()),
        'scope':'Target S vs TRAINING M pairs; mechanism diagnosis only, not an independent held-out pair accuracy.'})
    for path,digest in old_bindings.items():assert sha(ROOT/path)==digest,path
    for path,digest in read(ROOT/'evidence/2026-09-21/v70_bottleneck_review/verification.json')['artifact_sha256'].items():
        if path.startswith('docs/review_v70/') or path=='docs/V70_BOTTLENECK_REVIEW_AND_PLAN.md':assert sha(ROOT/path)==digest,path
    save(out/'verification.json',{'all_checks_passed':True,'models_reloaded':6,'prediction_rows_replayed':replays,'maximum_probability_difference':0.,
        'independent_threshold_checks':cuts_checked,'full_decision_tables_recomputed':len(all_predictions),
        'original_v69_bound_files_unchanged':len(old_bindings),'v70_audit_and_plan_unchanged':True,
        'source_and_body_role_isolation':True,'train_only_onehot':True,'matrices_recreated_from_sources':True,
        'post_calibration_refits':0,'quality_acceptance':False,'scope':'Integrity, implementation and mechanism verification, not model quality acceptance'})
    print(json.dumps(read(out/'verification.json'),indent=2),flush=True)
    print(json.dumps(read(out/'mechanism.json'),indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();verify(args.out.resolve())
