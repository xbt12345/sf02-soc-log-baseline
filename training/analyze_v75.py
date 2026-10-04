"""Independent real-row scoring and staged information/capacity diagnostics."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from run_v75 import ROOT,OUT,read,save,sha,metrics,matrices,features,load_sparse,NAMES,BATCH
from v75_views import matrix_hashes


def cm(y,p):
    x=np.zeros((3,3),np.int64);np.add.at(x,(y,p),1);return x


def floor(keys,y,mask):
    d=pd.DataFrame({'key':keys[mask],'label':y[mask]})
    t=d.groupby(['key','label']).size().unstack(fill_value=0)
    return {'rows':int(mask.sum()),'input_states':len(t),'mixed_states':int(((t>0).sum(1)>1).sum()),
        'minimum_errors':int((t.sum(1)-t.max(1)).sum()),
        'scope':'This representation and inspected train population only; not all available-field or physical-world Bayes error.'}


def ids(stem):
    x=load_sparse(OUT/stem);hashes=[]
    for a in range(0,x.shape[0],512):hashes.extend(matrix_hashes(x[a:a+512]))
    return pd.factorize(np.asarray(hashes,dtype='S32'),sort=False)[0].astype(np.uint64)


def main():
    target=OUT/'analysis'
    if (target/'summary.json').exists():raise FileExistsError(target)
    target.mkdir(exist_ok=True)
    cfg=read(OUT/'configuration.json');receipt=read(OUT/'four_arm/complete.json')
    for name,h in receipt['outputs'].items():assert sha(OUT/'four_arm'/name)==h
    r=pd.read_parquet(OUT/'rows.parquet');p=pd.read_parquet(OUT/'four_arm/predictions.parquet')
    np.testing.assert_array_equal(r.row_position,p.row_position)
    y=r.label_index.to_numpy();val=r.is_validation.to_numpy();fit=~val
    results={};table=[];paired=[]
    for a in cfg['arms']:
        pred=p[a+'_pred'].to_numpy();trainmask=fit&(r.old_fit.to_numpy() if cfg['arms'][a][1]=='old' else True)
        results[a]={'fit':metrics(cm(y[trainmask],pred[trainmask])), 'validation':metrics(cm(y[val],pred[val]))}
        assert results[a]['validation']['cm']==receipt['curves'][-1]['validation'][a]['cm']
        for route in sorted(r.route.unique()):
            for label in range(3):
                mask=r.route.eq(route).to_numpy()&(y==label)
                fmask=mask&trainmask;vmask=mask&val
                table.append({'arm':a,'route':route,'label':NAMES[label],'official_rows':int(mask.sum()),
                    'fit_rows':int(fmask.sum()),'fit_errors':int((pred[fmask]!=y[fmask]).sum()),
                    'validation_rows':int(vmask.sum()),'validation_errors':int((pred[vmask]!=y[vmask]).sum()),
                    'validation_recall':float((pred[vmask]==y[vmask]).mean()) if vmask.any() else None,
                    'status':'no_official_support' if not mask.any() else 'fit_only_no_holdout_evidence' if not vmask.any() else 'observed_holdout'})
    pd.DataFrame(table).to_csv(target/'all_formats_all_classes.csv',index=False,encoding='utf-8-sig')
    for a,b in [('A','B'),('A','C'),('C','D'),('B','D'),('A','D')]:
        ap=p[a+'_pred'].to_numpy();bp=p[b+'_pred'].to_numpy()
        for route in ['ALL']+sorted(r.route.unique()):
            mask=val if route=='ALL' else val&r.route.eq(route).to_numpy()
            paired.append({'before':a,'after':b,'route':route,'rows':int(mask.sum()),
                'fixed':int(((ap!=y)&(bp==y)&mask).sum()),'broken':int(((ap==y)&(bp!=y)&mask).sum())})
    pd.DataFrame(paired).to_csv(target/'paired_changes.csv',index=False,encoding='utf-8-sig')
    # Recompute actual model input identities, not just parser IDs.
    fids=ids('facts');oldids=ids('old_text');newids=ids('new_text')
    pid=r.projection_id.to_numpy();tid=r.new_text_id.to_numpy()
    oldkey=(fids[pid]<<np.uint64(32))|oldids[pid]
    meta=np.load(OUT/'metadata_code.npy').astype(np.uint64)
    def combined(textids):
        a=np.column_stack([fids[pid],textids,meta]).astype('<u8',copy=False)
        return pd.factorize(a.view('S24').reshape(-1),sort=False)[0]
    newkey=combined(newids[tid])
    viewkey=combined(tid.astype(np.uint64))
    raw=pd.read_parquet(OUT/'raw_ledger.parquet',columns=['raw_sha256']).raw_sha256.fillna('NULL')
    rawids=pd.factorize(raw,sort=False)[0]
    raw_with_meta=pd.factorize(np.column_stack([rawids.astype(np.uint64),meta]).astype('<u8').view('S16').reshape(-1),sort=False)[0]
    floors={name:floor(key,y,np.ones(len(r),bool)) for name,key in [
        ('exact_message_only',rawids),('exact_message_plus_observed_record_port',raw_with_meta),
        ('old_actual_input',oldkey),('new_view_with_facts',viewkey),('new_actual_input',newkey)]}
    # Train-only exact-input majority reference: a diagnostic, not a promoted
    # lookup model. Unknown inputs use D; never read a held-out label to predict.
    t=pd.DataFrame({'key':newkey[fit],'label':y[fit]}).groupby(['key','label']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    majority=t.idxmax(axis=1);pure=(t>0).sum(1)==1
    lookup=pd.Series(newkey).map(majority);known=lookup.notna().to_numpy()
    pref=p.D_pred.to_numpy().copy();pref[known]=lookup[known].astype(int)
    pure_lookup=pd.Series(newkey).map(pure).fillna(False).to_numpy(dtype=bool)
    dwrong=p.D_pred.to_numpy()!=y
    capacity={'validation_reference':metrics(cm(y[val],pref[val])),
        'validation_known_input_rows':int((val&known).sum()),
        'D_errors_on_pure_fit_inputs_with_matching_label':int((val&pure_lookup&(pref==y)&dwrong).sum()),
        'D_errors_on_unseen_actual_inputs':int((val&~known&dwrong).sum()),
        'D_fit_errors':int((fit&dwrong).sum()),
        'fitted_input_empirical_floor':floor(newkey,y,fit),
        'scope':'Pure matching fit-input errors implicate classifier/optimization. Unseen-input errors alone do not prove insufficient data or insufficient capacity.'}
    difficulty=ROOT/'artifacts/v67_targeted_20260920/manifest.parquet'
    targets=pd.read_parquet(difficulty,columns=['row_position','target_578'])
    positions=targets.loc[targets.target_578,'row_position'];mask=r.row_position.isin(positions).to_numpy()
    assert int(mask.sum())==578
    historic={'original_targets':578,'new_fit_members':int((mask&fit).sum()),'new_validation_members':int((mask&val).sum()),
        'D_errors_on_new_validation_members':int((mask&val&dwrong).sum()),'D_errors_on_new_fit_members':int((mask&fit&dwrong).sum()),
        'accepted_historical_repairs':0,'scope':'Changed roles; cannot promote these counts as original 578 blind repairs.'}
    # Reload all four artifacts and compare independent raw matrix entry scores.
    m=matrices();chosen=set()
    for _,z in r.groupby(['route','label_index']):
        for hold in [False,True]:chosen.update(z.loc[z.is_validation.eq(hold),'row_position'].head(3).tolist())
    chosen=sorted(chosen)
    for a,(kind,_) in cfg['arms'].items():
        model=joblib.load(OUT/'four_arm'/(a+'.joblib'));q=model.predict_proba(features(r,chosen,m,kind))
        np.testing.assert_allclose(q,p.loc[chosen,[a+'_p0',a+'_p1',a+'_p2']],atol=1e-6,rtol=1e-6)
        np.testing.assert_array_equal(q.argmax(1),p.loc[chosen,a+'_pred'])
    result={'new_optimizer_fits':4,'all_format_class_cells':len(table),'results':results,'paired':paired,
        'representation_floors':floors,'capacity_diagnostic':capacity,'historical_578':historic,
        'reloaded_models_checked':4,'replay_rows_each':len(chosen),'source_isolation':bool(r[r.source_symbol>=0].groupby('source_symbol').is_validation.nunique().max()==1),
        'body_isolation':bool(r.groupby('body_group').is_validation.nunique().max()==1),'answers_read':False,
        'quality_promoted':False,'source_sha256':sha(__file__), 'prediction_sha256':sha(OUT/'four_arm/predictions.parquet')}
    save(target/'summary.json',result)
    print(json.dumps({k:result[k] for k in ['results','representation_floors','capacity_diagnostic','historical_578']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
