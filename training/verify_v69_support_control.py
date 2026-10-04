"""Independent threshold/decision audit and saved-model replay. No fit calls."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from run_v69_support_control import ROOT, ARMS, BUDGETS, load, roles, hard
from run_v67_targeted import design
from v61_common import read, save, sha


def independent_cut(d, budget):
    m = d[d.label.eq(1)]
    assert len(m)
    v = m.score.to_numpy(); order = np.argsort(v); v = v[order]
    sizes = m.groupby('group').size()
    limits = []
    candidates = np.unique(np.r_[0., v, np.nextafter(v, np.inf)])
    for weights in [np.ones(len(m))/len(m), 1/len(sizes)/m.group.map(sizes).to_numpy()]:
        sums = np.r_[0., np.cumsum(weights[order])]
        errors = sums[-1] - sums[np.searchsorted(v, candidates, side='left')]
        limits.append(float(candidates[errors <= budget+1e-12].min()))
    return max(limits)


def main(out):
    reg = read(out/'preregistered.json')
    for name, digest in reg['source_sha256'].items(): assert sha(ROOT/'training'/name) == digest
    for name, digest in reg['inputs_sha256'].items(): assert sha(ROOT/name) == digest
    for name, digest in reg['prepared_sha256'].items(): assert sha(out/name) == digest
    f, c, m = load(); original = f.set_index('row_position').loc[m.row_position].reset_index()
    labels = original.label.to_numpy(); target = m.target_578.to_numpy(); base = m.baseline_pred_with_normal_gate.to_numpy()
    h = hard(original); analysis = read(out/'analysis.json'); replayed = 0; models = 0; cut_checks = 0; maxdiff = 0.
    decisions = {(a,b): base.copy() for a in ARMS for b in BUDGETS}; encoded_rows = []; cal_gap = []
    with threadpool_limits(limits=4):
        for fold in range(3):
            rs = pd.read_parquet(out/f'fold{fold}_roles.parquet'); train, cal, va = roles(f, fold)
            assert np.array_equal(rs.row_position, f.row_position)
            assert np.array_equal(rs.calibration, cal) and np.array_equal(rs.evaluation, va)
            for arm in ARMS:
                assert np.array_equal(rs[arm], train[arm]); sub = out/f'fold{fold}'/arm
                before = sha(sub/'model.joblib'); bundle = joblib.load(sub/'model.joblib')
                fit = train[arm]; ix = np.flatnonzero(fit)
                assert np.array_equal(bundle['fit_positions'], f.loc[fit, 'row_position'])
                assert np.array_equal(bundle['calibration_positions'], f.loc[cal, 'row_position'])
                assert bundle['preregistered_sha256'] == sha(out/'preregistered.json')
                assert bundle['model'].n_iter_ == 500 and not bundle['model'].early_stopping
                x, enc, names, cat = design(f, c, 'context_numeric', ix, encoder=bundle['encoder'])
                assert names == bundle['names']
                # Add forbidden metadata adversarially; allowlisted features must not change.
                altered = f.copy()
                for field in ['label','group','event_id','timestamp','product_name','text','src_ip','dst_ip']:
                    altered[field] = 'adversarial-change'
                xx, *_ = design(altered, c, 'context_numeric', ix, encoder=enc)
                np.testing.assert_allclose(x, xx, equal_nan=True, rtol=0, atol=0)
                # Verify fitted category vocabularies contain fit-side values only.
                categorical = [n for n, iscat in zip(names, cat) if iscat]
                for column, vocabulary in zip(categorical, enc.categories_):
                    assert set(vocabulary) == set(f.loc[fit, column])
                saved = {}
                for role, mask in [('fit',fit),('calibration',cal),('evaluation',va)]:
                    d = pd.read_parquet(sub/(role+'.parquet')); saved[role] = d
                    assert np.array_equal(d.row_position, f.loc[mask,'row_position'])
                    assert np.array_equal(d.label, f.loc[mask,'label'])
                    p = bundle['model'].predict_proba(x[mask])[:,1]
                    np.testing.assert_allclose(p,d.score,rtol=0,atol=0)
                    maxdiff = max(maxdiff,float(np.max(np.abs(p-d.score))))
                    replayed += len(d)
                assert sha(sub/'model.joblib') == before == read(sub/'fit_receipt.json')['model_sha256']
                models += 1
                cd = saved['calibration']; ed = saved['evaluation']; cuts=read(sub/'thresholds.json')
                idx = pd.Index(m.row_position).get_indexer(ed.row_position); assert (idx>=0).all()
                for budget in BUDGETS:
                    for proto in ['tcp','udp']:
                        cc = cd[cd.transport_protocol.eq(proto)]
                        threshold = independent_cut(cc,budget)
                        assert threshold == cuts[str(budget)][proto]['threshold']; cut_checks += 1
                        local = ed.transport_protocol.eq(proto).to_numpy()
                        decisions[(arm,budget)][idx[local]] = np.where(ed.score.to_numpy()[local]>=threshold,2,1)
                        if budget == .01:
                            row = {'fold':fold,'arm':arm,'protocol':proto,'threshold':threshold}
                            for name,d in [('calibration',cc),('evaluation',ed[local])]:
                                md=d[d.label.eq(1)].copy();md['error']=md.score.ge(threshold)
                                row[name+'_M_row_error']=float(md.error.mean())
                                row[name+'_M_source_error']=float(md.groupby('group').error.mean().mean())
                            cal_gap.append(row)
                # Actual v69 design-matrix trace for every target in this outer fold.
                view = pd.util.hash_pandas_object(pd.DataFrame(x),index=False).astype(str)
                fit_table = pd.DataFrame({'view':view[fit], 'label':f.label[fit], 'group':f.group[fit]})
                supports = {y:fit_table[fit_table.label.eq(y)].groupby('view').group.nunique() for y in [1,2]}
                targets = va & f.row_position.isin(m.loc[target,'row_position']).to_numpy()
                for i in np.flatnonzero(targets):
                    same = fit & view.eq(view[i]).to_numpy()
                    if same.any():
                        # Hash is merely an index; assert full matrix equality for all matches.
                        np.testing.assert_allclose(x[same],np.broadcast_to(x[i],x[same].shape),equal_nan=True,rtol=0,atol=0)
                    item={'fold':fold,'arm':arm,'row_position':int(f.row_position.iloc[i]),'group':int(f.group.iloc[i]),
                          'matrix_sha256':__import__('hashlib').sha256(x[i].tobytes()).hexdigest(),
                          'same_actual_matrix_train_M_sources':int(supports[1].get(view[i],0)),
                          'same_actual_matrix_train_S_sources':int(supports[2].get(view[i],0))}
                    item.update({n:float(v) for n,v in zip(names,x[i])})
                    encoded_rows.append(item)
    assert models==9 and cut_checks==90 and len(encoded_rows)==3*578
    for (arm,budget),pred in decisions.items():
        saved=pd.read_parquet(out/f'{arm}_budget{budget}_oof.parquet')
        assert np.array_equal(pred,saved.pred)
        assert np.array_equal(pred[~h],base[~h])
        a=analysis['arms'][arm]['budgets'][str(budget)]
        assert confusion_matrix(labels,pred,labels=[0,1,2]).tolist()==a['confusion_B_M_S']
        assert int((pred[target]==2).sum())==a['historical_effects']['target_fixed']
        for y,name in [(0,'B'),(1,'M'),(2,'S')]:
            mask=labels==y; rr=a['historical_effects'][name]
            assert int((mask&(pred!=labels)).sum())==rr['errors']
            assert int((mask&(base!=labels)&(pred==labels)).sum())==rr['fixed']
            assert int((mask&(base==labels)&(pred!=labels)).sum())==rr['broken']
    pd.DataFrame(encoded_rows).to_parquet(out/'target_actual_model_inputs.parquet',index=False)
    pd.DataFrame(cal_gap).to_csv(out/'calibration_transfer.csv',index=False,encoding='utf-8-sig')
    old=read(ROOT/'evidence/2026-09-20/v67_target_578/delivery.json')
    for name,digest in old['artifact_sha256'].items(): assert sha(ROOT/name)==digest, name
    receipt={'verification_passed':True,'models_restored':models,'prediction_rows_replayed':replayed,
             'max_probability_difference':maxdiff,'independently_recomputed_thresholds':cut_checks,
             'all_15_full_decision_tables_recomputed':True,'source_and_body_group_roles_disjoint':True,
             'fitted_vocabulary_fit_only':True,'forbidden_field_matrix_invariance':True,
             'actual_target_matrix_trace_rows':len(encoded_rows),'v67_bound_files_unchanged':len(old['artifact_sha256']),
             'quality_acceptance':False,'model_promoted':False,'new_fits_during_verification':0,
             'verifier_sha256':sha(__file__)}
    save(out/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();main(a.out.resolve())
