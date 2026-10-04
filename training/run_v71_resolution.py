"""Frozen paired exact-tree experiment: observed ports vs old fit-only bins.

Official previously inspected development only. No feature/weight/threshold
search, no post-calibration fit. Prepare must complete before six actual fits.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.runtime/v71'))
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from threadpoolctl import threadpool_limits
from run_v69_support_control import load, roles, hard, threshold, rates, ranking, BUDGETS
from run_v67_targeted import design, counts
from v61_common import read, save, sha, FIELDS

OLD = ROOT / 'artifacts/v69_support_control_20260921'
ARMS = ['Q_old_bins', 'R_observed']
PARAMS = dict(objective='binary:logistic', tree_method='exact', max_depth=4,
              eta=.05, reg_lambda=1., reg_alpha=0., gamma=0., base_score=.5,
              min_child_weight=0., subsample=1., colsample_bytree=1.,
              scale_pos_weight=1., nthread=4, seed=20260921,
              eval_metric='logloss', validate_parameters=True)
ROUNDS = 500
PORTS = ['src_port_fixed', 'dst_port_fixed']


def ids(x):
    v = np.array(x, dtype=np.float64, copy=True)
    v[np.isnan(v)] = np.inf
    v[v == 0] = 0
    return np.unique(v, axis=0, return_inverse=True)[1]


def collisions(x, y):
    k = ids(x)
    a = np.bincount(k, weights=np.asarray(y) == 1)
    b = np.bincount(k, weights=np.asarray(y) == 2)
    mixed = (a > 0) & (b > 0)
    return {'minimum_empirical_errors': int(np.minimum(a, b).sum()),
            'mixed_S_rows': int(((np.asarray(y) == 2) & mixed[k]).sum()),
            'unique_views': int(len(a))}, mixed[k]


def matrices(f, context, bundle, fit, encoder=None):
    oldx, _, oldnames, cat = design(f, context, 'context_numeric', np.flatnonzero(fit), encoder=bundle['encoder'])
    model = bundle['model']
    xp = model._preprocess_X(oldx, reset=False)
    xb = model._bin_mapper.transform(xp)
    remap = [n for n, c in zip(oldnames, cat) if c] + [n for n, c in zip(oldnames, cat) if not c]
    cats = [n for n, c in zip(oldnames, cat) if c]
    z = f[cats].astype(str)
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        encoder.fit(z.loc[fit])
    oh = encoder.transform(z)
    unknown = np.column_stack([~z[n].isin(v).to_numpy() for n, v in zip(cats, encoder.categories_)]).astype(np.float32)
    numeric = [n for n, c in zip(oldnames, cat) if not c]
    nr = oldx[:, [oldnames.index(n) for n in numeric]].astype(np.float32)
    nq = nr.copy()
    for j, n in enumerate(PORTS):
        assert numeric[j] == n
        # Restore missingness: a missing-bin code must never become a large port.
        q = xb[:, remap.index(n)].astype(np.float32)
        q[np.isnan(oldx[:, oldnames.index(n)])] = np.nan
        nq[:, j] = q
        np.testing.assert_array_equal(np.isnan(nr[:, j]), np.isnan(nq[:, j]))
        good = ~np.isnan(nr[:, j])
        np.testing.assert_array_equal(nr[good, j].astype(np.float64), oldx[good, oldnames.index(n)])
    names = encoder.get_feature_names_out(cats).tolist() + [n + '_unknown' for n in cats] + numeric
    both = {'R_observed': np.c_[oh, unknown, nr], 'Q_old_bins': np.c_[oh, unknown, nq]}
    qold = oldx.copy()
    for j, n in enumerate(PORTS): qold[:, oldnames.index(n)] = nq[:, j]
    return both, encoder, names, oldx, qold


def prepare(out):
    assert not out.exists(), 'Never overwrite a previous or partial run'
    assert xgb.__version__ == '3.0.5' and sklearn.__version__ == '1.6.1'
    out.mkdir(parents=True)
    f, c, m = load()
    sources = [Path(__file__), ROOT/'training/run_v69_support_control.py', ROOT/'training/run_v67_targeted.py',
               ROOT/'training/v61_common.py', ROOT/'training/prepare_v61.py', ROOT/'training/verify_v71_resolution.py',
               ROOT/'docs/V70_BOTTLENECK_REVIEW_AND_PLAN.md']
    inputs = [ROOT/'artifacts/v61_source_factorial_20260914_r2'/n for n in ['records.parquet','context.npz','configuration.json']]
    inputs += [ROOT/'artifacts/v67_targeted_20260920/manifest.parquet', ROOT/'docs/review_v70/binning_audit.json',
               ROOT/'docs/review_v70/supplemental_collision_fields.json', ROOT/'.runtime/v71_install.json']
    oldaudit = read(ROOT/'docs/review_v70/binning_audit.json')
    audits = []
    for fold in range(3):
        rolepath = OLD/f'fold{fold}_roles.parquet'
        r = pd.read_parquet(rolepath)
        fit = r.B_small_sources.to_numpy(); cal = r.calibration.to_numpy(); ev = r.evaluation.to_numpy()
        fresh, fc, fe = roles(f, fold)
        assert np.array_equal(r.row_position, f.row_position)
        assert np.array_equal(fit, fresh['B_small_sources']) and np.array_equal(cal, fc) and np.array_equal(ev, fe)
        path = OLD/f'fold{fold}/B_small_sources/model.joblib'; bundle = joblib.load(path)
        assert np.array_equal(bundle['fit_positions'], f.loc[fit,'row_position'])
        assert np.array_equal(bundle['calibration_positions'], f.loc[cal,'row_position'])
        inputs += [rolepath, path]
        x, enc, names, raw, quantized = matrices(f, c, bundle, fit)
        use = fit | cal | ev
        # Float32 and common one-hot must not introduce new complete-view aliases.
        for arm, original in [('R_observed', raw), ('Q_old_bins', quantized)]:
            before, after = ids(original[use]), ids(x[arm][use])
            assert np.unique(np.c_[before,after],axis=0).shape[0] == len(np.unique(before)) == len(np.unique(after))
        # Forbidden metadata never enters either arm. Group-conditioned context is
        # held fixed here; this is not a claim that context lacks source proxies.
        altered = f.copy()
        for field in ['label','group','event_id','timestamp','product_name','src_ip','dst_ip','text']:
            altered[field] = 'adversarial-change'
        xx, *_ = matrices(altered, c, bundle, fit, encoder=enc)
        for arm in ARMS: np.testing.assert_allclose(x[arm], xx[arm], equal_nan=True, rtol=0, atol=0)
        ixport = [names.index(n) for n in PORTS]
        ixother = [i for i in range(len(names)) if i not in ixport]
        np.testing.assert_allclose(x['R_observed'][:,ixother], x['Q_old_bins'][:,ixother], equal_nan=True, rtol=0, atol=0)
        r[['row_position','label','group','body_group','fold','calibration','evaluation','B_small_sources']].to_parquet(out/f'fold{fold}_roles.parquet',index=False)
        joblib.dump({'onehot':enc, 'feature_names':names, 'port_columns':ixport, 'old_model_path':str(path.relative_to(ROOT)),
                     'old_model_sha256':sha(path)},out/f'fold{fold}_encoding.joblib')
        np.savez_compressed(out/f'fold{fold}_matrices.npz', **x)
        target = ev & f.row_position.isin(m.loc[m.target_578,'row_position']).to_numpy()
        entry = {'fold':fold,'fit_rows':int(fit.sum()),'calibration_rows':int(cal.sum()),'evaluation_rows':int(ev.sum()),'arms':{}}
        ref = next(z for z in oldaudit['models'] if z['fold']==fold and z['arm']=='B_small_sources')
        for arm, expected in [('R_observed','exact_design'),('Q_old_bins','post_binning')]:
            cs, _ = collisions(x[arm][fit], f.loc[fit,'label'])
            assert cs['minimum_empirical_errors'] == ref[expected]['minimum_empirical_row_errors']
            key = ids(x[arm][fit|target]); jf = fit[fit|target]; jy=f.label.to_numpy()[fit|target]
            targets_same_m = np.isin(key[~jf], key[jf & (jy==1)])
            count = int(targets_same_m.sum())
            assert count == ref['target']['same_exact_fit_M_rows' if arm=='R_observed' else 'same_binned_fit_M_rows']
            entry['arms'][arm] = {**cs,'target_same_fit_M_rows':count}
        audits.append(entry)
    save(out/'preflight.json', {'folds':audits,'forbidden_metadata_invariance':True,'float32_full_view_partition_preserved':True,
         'only_two_port_columns_differ':True,'old_B_roles_exactly_reused':True,'same_missingness':True,'new_training_fits':0})
    reg = {'version':'v71-port-resolution-paired-exact-1','arms':ARMS,'parameters':PARAMS,'rounds':ROUNDS,'fits_planned':6,
           'sample_weight':'All original rows weight 1; no target mining or class reweighting','post_calibration_refit':False,
           'budgets':BUDGETS,'primary_budget':.01,'bootstrap':{'seed':20260921,'replicates':5000,'unit':'paired S source within fold'},
           'continuation_gate':'primary paired S source CI lower>0; delete best source point gain>0; mean six-cell source pAUC gain>0; mean six-cell own-evaluation joint M<=1% oracle S source gain>0; oracle is diagnostic only',
           'eligibility_additional_gate':'all six primary frozen-cut evaluation M row/source error<=1%; independent whole SOC validation still required',
           'role_contract':'Previously inspected official source-symbol development; roles unchanged from v69 B; no blind/external claim',
           'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'xgboost':xgb.__version__},
           'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
           'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
           'prepared_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    save(out/'preregistered.json',reg)
    print('PREPARED',json.dumps(audits),flush=True)


def bindings(out):
    r = read(out/'preregistered.json')
    for key in ['source_sha256','input_sha256']:
        for path,h in r[key].items(): assert sha(ROOT/path)==h, path
    for path,h in r['prepared_sha256'].items(): assert sha(out/path)==h,path
    assert r['parameters']==PARAMS and r['rounds']==ROUNDS
    return r


def train(out):
    bindings(out)
    assert not (out/'fold0').exists(), 'No implicit retraining'
    f, c, m = load(); t0=time.monotonic(); fits=0
    with threadpool_limits(limits=4):
        for fold in range(3):
            rs=pd.read_parquet(out/f'fold{fold}_roles.parquet')
            masks={'fit':rs.B_small_sources.to_numpy(),'calibration':rs.calibration.to_numpy(),'evaluation':rs.evaluation.to_numpy()}
            with np.load(out/f'fold{fold}_matrices.npz') as z:
                for arm in ARMS:
                    x=z[arm]; sub=out/f'fold{fold}'/arm;sub.mkdir(parents=True)
                    fit=masks['fit']; dtrain=xgb.DMatrix(x[fit],label=f.loc[fit,'label'].eq(2).astype(int),nthread=4)
                    history={}; start=time.monotonic()
                    booster=xgb.train(PARAMS,dtrain,num_boost_round=ROUNDS,evals=[(dtrain,'fit')],evals_result=history,verbose_eval=100)
                    fits+=1; assert booster.num_boosted_rounds()==ROUNDS
                    booster.save_model(sub/'model.ubj')
                    save(sub/'booster_config.json',json.loads(booster.save_config()))
                    save(sub/'fit_history.json',history)
                    for role,mask in masks.items():
                        score=booster.predict(xgb.DMatrix(x[mask],nthread=4)).astype(np.float64)
                        assert np.isfinite(score).all()
                        d=f.loc[mask,['row_position','label','group','fold','transport_protocol']].copy()
                        d['score']=score;d.to_parquet(sub/(role+'.parquet'),index=False)
                    cd=pd.read_parquet(sub/'calibration.parquet');cuts={}
                    for budget in BUDGETS:
                        cuts[str(budget)]={}
                        for proto in ['tcp','udp']:
                            d=cd[cd.transport_protocol.eq(proto)];cut=threshold(d,d.score,budget)
                            rr=rates(d,np.where(d.score.ge(cut),2,1))
                            assert max(1-rr['M']['row_recall'],1-rr['M']['source_recall'])<=budget+1e-12
                            cuts[str(budget)][proto]={'threshold':cut,'calibration':rr}
                    save(sub/'thresholds.json',cuts)
                    save(sub/'fit_receipt.json',{'fit_rows':int(fit.sum()),'fit_sources':int(f.loc[fit,'group'].nunique()),
                         'seconds':time.monotonic()-start,'rounds':ROUNDS,'model_sha256':sha(sub/'model.ubj'),
                         'preregistration_sha256':sha(out/'preregistered.json'),'calibration_after_freeze':True,'post_calibration_fits':0})
                    print('FIT_COMPLETE',fold,arm,round(time.monotonic()-start,2),flush=True)
    save(out/'training_complete.json',{'actual_new_fits':fits,'seconds':time.monotonic()-t0,'platform_used':False,'external_data':False})


def paired(f, r, q):
    s=f[f.label.eq(2)].copy();mask=f.label.eq(2).to_numpy()
    s['delta']=(np.asarray(r)[mask]==2).astype(float)-(np.asarray(q)[mask]==2).astype(float)
    g=s.groupby(['fold','group']).delta.mean().reset_index();values=g.delta.to_numpy()
    rng=np.random.default_rng(20260921);rep=np.zeros(5000)
    for _,z in g.groupby('fold'):
        a=z.delta.to_numpy();rep+=rng.choice(a,(5000,len(a)),replace=True).sum(axis=1)
    rep/=len(g)
    return {'source_recall_gain':float(values.mean()),'descriptive_bootstrap_95pct':np.quantile(rep,[.025,.975]).tolist(),
            'gain_after_removing_best_source':float(np.delete(values,np.argmax(values)).mean()),
            'sources_improved':int((values>0).sum()),'sources_worsened':int((values<0).sum()),'source_count':len(values)}


def analyze(out):
    bindings(out); assert read(out/'training_complete.json')['actual_new_fits']==6
    f,c,m=load(); original=f.set_index('row_position').loc[m.row_position].reset_index()
    h=hard(original);target=m.target_578.to_numpy();base=m.baseline_pred_with_normal_gate.to_numpy()
    result={'actual_new_fits':6,'quality_acceptance':False,'accepted_repairs':0,'accepted_remaining':578,
            'scope':'Adaptive official source-symbol development; frozen nonhard branch; no external or blind test', 'arms':{}}
    primary={}; ranks={}; oracles={}; curves=[]; geometry=[]; losses=[]
    ledger=m[target].copy()
    for arm in ARMS:
        score=np.full(len(m),np.nan);cuts={}; rk=[]; oracle=[]
        for fold in range(3):
            sub=out/f'fold{fold}'/arm; cuts[fold]=read(sub/'thresholds.json')
            for role in ['fit','calibration','evaluation']:
                d=pd.read_parquet(sub/(role+'.parquet'))
                if role=='evaluation':
                    ix=pd.Index(m.row_position).get_indexer(d.row_position);assert (ix>=0).all();score[ix]=d.score
                for proto,z in d.groupby('transport_protocol'):
                    cut=cuts[fold]['0.01'][proto]['threshold'];oc=threshold(z,z.score,.01)
                    row={'arm':arm,'fold':fold,'role':role,'protocol':proto,'frozen_cut_rates':rates(z,np.where(z.score>=cut,2,1)),
                         'own_role_oracle_rates':rates(z,np.where(z.score>=oc,2,1)),'ranking':ranking(z,z.score)}
                    geometry.append(row)
                    if role=='evaluation': rk.append(row['ranking']);oracle.append(row['own_role_oracle_rates']['S']['source_recall'])
                if role=='fit':
                    y=d.label.eq(2).to_numpy();p=np.clip(d.score.to_numpy(),1e-15,1-1e-15)
                    loss=-(y*np.log(p)+(1-y)*np.log1p(-p)); d=d.assign(loss=loss)
                    for label,z in d.groupby('label'):
                        losses.append({'fold':fold,'arm':arm,'label':int(label),'rows':len(z),'sources':int(z.group.nunique()),
                                       'mean_loss':float(z.loss.mean()),'total_loss_fraction':float(z.loss.sum()/loss.sum())})
        assert np.isfinite(score[h]).all() and np.isnan(score[~h]).all()
        ranks[arm]=rk;oracles[arm]=oracle;effects={}
        for budget in BUDGETS:
            pred=base.copy();cells=[]
            for fold in range(3):
                for proto in ['tcp','udp']:
                    ix=h & original.fold.eq(fold).to_numpy() & original.transport_protocol.eq(proto).to_numpy()
                    cut=cuts[fold][str(budget)][proto]['threshold'];pred[ix]=np.where(score[ix]>=cut,2,1)
                    rr=rates(original[ix],pred[ix]);cells.append({'fold':fold,'protocol':proto,**rr})
                    curves.append({'arm':arm,'budget':budget,'fold':fold,'protocol':proto,'M_row_error':1-rr['M']['row_recall'],
                                   'M_source_error':1-rr['M']['source_recall'],'S_row_recall':rr['S']['row_recall'],'S_source_recall':rr['S']['source_recall']})
            effects[str(budget)]={'historical_effects':counts(original,pred,base,target),'hard_rates':rates(original[h],pred[h]),'cells':cells,
                                 'confusion_B_M_S':confusion_matrix(original.label,pred,labels=[0,1,2]).tolist(),
                                 'classification':classification_report(original.label,pred,labels=[0,1,2],output_dict=True,zero_division=0)}
            pd.DataFrame({'row_position':m.row_position,'label':m.label,'group':m.group,'fold':m.fold,'hard_score':score,'pred':pred}).to_parquet(out/f'{arm}_budget{budget}_oof.parquet',index=False)
            if budget==.01:primary[arm]=pred;ledger[arm+'_pred']=pred[target]
        result['arms'][arm]={'budgets':effects}
    comp=paired(original[h],primary['R_observed'][h],primary['Q_old_bins'][h])
    comp['mean_six_cell_pAUC_gain']=float(np.mean([r['source_standardized_pAUC_1pct']-q['source_standardized_pAUC_1pct'] for r,q in zip(ranks['R_observed'],ranks['Q_old_bins'])]))
    comp['mean_six_cell_oracle_S_source_gain']=float(np.mean(np.array(oracles['R_observed'])-np.array(oracles['Q_old_bins'])))
    comp['mechanism_evidence']=bool(comp['descriptive_bootstrap_95pct'][0]>0 and comp['gain_after_removing_best_source']>0 and comp['mean_six_cell_pAUC_gain']>0 and comp['mean_six_cell_oracle_S_source_gain']>0)
    cells=result['arms']['R_observed']['budgets']['0.01']['cells']
    comp['outer_M_budget_met']=all(max(1-x['M']['row_recall'],1-x['M']['source_recall'])<=.01+1e-12 for x in cells)
    comp['eligible_for_further_validation']=comp['mechanism_evidence'] and comp['outer_M_budget_met']
    comp['matched_effects']=counts(original,primary['R_observed'],primary['Q_old_bins'],target)
    result['R_vs_Q']=comp
    for arm in ARMS:
        old=pd.read_parquet(OLD/'B_small_sources_budget0.01_oof.parquet');assert np.array_equal(old.row_position,m.row_position)
        result['arms'][arm]['vs_old_B_regression_only']=counts(original,primary[arm],old.pred.to_numpy(),target)
    ledger['accepted_repair']=False;ledger.to_csv(out/'target_578_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(curves).to_csv(out/'fixed_budget_curves.csv',index=False,encoding='utf-8-sig')
    save(out/'role_geometry.json',geometry);save(out/'terminal_loss.json',losses);save(out/'analysis.json',result)
    print(json.dumps({'R_vs_Q':comp,'primary':{a:result['arms'][a]['budgets']['0.01']['historical_effects'] for a in ARMS}},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','train','analyze']);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();{'prepare':prepare,'train':train,'analyze':analyze}[args.phase](args.out.resolve())
