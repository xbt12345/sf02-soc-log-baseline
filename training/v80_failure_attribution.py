"""Frozen-model causal-boundary review; no new fitting or threshold search."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,OUT,read,save,sha,load_sparse,metrics

DEST=ROOT/'artifacts/v80_attribution_20260927'
OLD=ROOT/'artifacts/v78_boundary_20260922'
LAST=ROOT/'artifacts/v79_execution_20260927'


def main():
    if (DEST/'capacity_and_support.json').exists():raise FileExistsError('completed audit exists')
    DEST.mkdir(exist_ok=True)
    from v79_execute import rows
    r=rows();fid=np.load(LAST/'row_feature_id.npy',mmap_mode='r');x=load_sparse(LAST/'X')
    registration=read(LAST/'registration.json')
    assert sha(ROOT/'training/v79_execute.py')==registration['source_sha256']
    p=pd.read_parquet(OLD/'frozen_development/predictions.parquet',columns=['event_id','route','O_sgd_pred','O_lbfgs_pred'])
    patch=pd.read_parquet(OLD/'denial_adapter_v4/changed_predictions.parquet');patch=patch[patch.dataset.eq('development')]
    ix=patch.row_position.to_numpy();np.testing.assert_array_equal(p.event_id.iloc[ix].to_numpy(),patch.event_id)
    answers=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    d=p[['event_id','route']].merge(answers,on='event_id',validate='one_to_one',sort=False)
    y=d.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();assert np.isfinite(y).all()
    new=pd.read_parquet(LAST/'development_predictions.parquet');np.testing.assert_array_equal(p.event_id,new.event_id)
    predictions={}
    for name in ['O_sgd','O_lbfgs']:
        a=p[name+'_pred'].to_numpy().copy();a[ix]=patch[name+'_pred'].to_numpy();predictions['v78_'+name]=a
    predictions['v79_B_ovr']=new.prediction.to_numpy()
    totals={k:metrics(confusion_matrix(y,a,labels=[0,1,2])) for k,a in predictions.items()}
    changes=[]
    for before,after in [('v78_O_sgd','v78_O_lbfgs'),('v78_O_lbfgs','v79_B_ovr'),('v78_O_sgd','v79_B_ovr')]:
        a,b=predictions[before],predictions[after]
        for route in ['ALL']+sorted(p.route.unique()):
            take=np.ones(len(y),bool) if route=='ALL' else p.route.eq(route).to_numpy()
            for k in range(3):
                cell=take&(y==k);changes.append({'before':before,'after':after,'route':route,'class':k,'support':int(cell.sum()),
                    'before_wrong':int((cell&(a!=y)).sum()),'after_wrong':int((cell&(b!=y)).sum()),
                    'fixed':int((cell&(a!=y)&(b==y)).sum()),'broken':int((cell&(a==y)&(b!=y)).sum())})
    pd.DataFrame(changes).to_csv(DEST/'paired_changes.csv',index=False)
    old_manifest=pd.read_parquet(OLD/'manifest.parquet',columns=['optimizer_diagnostic'])
    oldfit=old_manifest.optimizer_diagnostic.to_numpy();fit=~r.fold.isin([0,2]).to_numpy()
    from v79_execute import weights
    cc,den,exposure=weights(r,fid,x.shape[0],fit,'B')
    rng=np.random.default_rng(7901);newfit=fit.copy()
    for route in sorted(r.route.unique()):
        pos=np.flatnonzero(fit&r.route.eq(route).to_numpy()&r.label_index.eq(0).to_numpy())
        if len(pos)>2048:newfit[pos]=False;newfit[rng.choice(pos,2048,replace=False)]=True
    assert hashlib.sha256(np.flatnonzero(newfit).astype('<i4').tobytes()).hexdigest()==exposure['selected_rows_sha256']
    sampling={name:int(mask.sum()) for name,mask in [('old_fit_rows',oldfit),('new_fit_rows',newfit),('same_rows',oldfit&newfit),('old_only',oldfit&~newfit),('new_only',newfit&~oldfit)]}
    sampling['nonbenign_symmetric_difference']=int(((oldfit^newfit)&r.label_index.ne(0).to_numpy()).sum())
    enc=joblib.load(OUT/'facts_encoder.joblib');fact_names=list(enc.names())
    selected=read(LAST/'P2_selection.json')['selected']
    assert selected=='P1_B_ovr'
    report={'new_fits':0,'new_calibration_fits':0,'target_is_inspected_development':True,'totals':totals,'sampling':sampling,
            'selected_model':selected,'auxiliary_loss_selected':False,
            'loss_chain':{'old_SGD_to_same_subset_LBFGS':totals['v78_O_lbfgs']['errors']-totals['v78_O_sgd']['errors'],
                          'old_LBFGS_to_new_LBFGS':totals['v79_B_ovr']['errors']-totals['v78_O_lbfgs']['errors'],
                          'total':totals['v79_B_ovr']['errors']-totals['v78_O_sgd']['errors']},
            'interpretation':'The same-subset v78 optimization path comparison accounts for net regression. Do not attribute full target regression to the rejected auxiliary objective. This does not uniquely identify early stopping, averaging, coefficient norm or regularization mechanism.',
            'source_sha256':sha(__file__)}
    save(DEST/'regression_chain.json',report)
    # Exact additive logit-margin decomposition: explanation of the fixed classifier,
    # not proof that any corresponding factual observation causes malicious behavior.
    coef_report={}
    for route,mask,modelname,pair in [
        ('cef_H_benign',r.fold.eq(0)&r.route.eq('cef_fields')&r.label_index.eq(0),'P4_cef_fields_whole_format',(2,0)),
        ('asa_C_suspicious',r.fold.eq(2)&r.route.eq('asa')&r.label_index.eq(2),'P1_B_ovr',(1,2)),
    ]:
        take=np.flatnonzero(mask.to_numpy()); model=joblib.load(LAST/(modelname+'.joblib'))
        a,b=pair;dw=model['coef'][:,a]-model['coef'][:,b];bias=float(model['intercept'][a]-model['intercept'][b])
        xx=x[fid[take]];pred=(xx@model['coef']+model['intercept']).argmax(1);wrong=pred!=r.label_index.to_numpy()[take]
        xx=xx[wrong];n=xx.shape[0];mean=np.asarray(xx.sum(axis=0,dtype=np.float64)).ravel()/n if n else np.zeros(x.shape[1]);contribution=mean*dw
        order=np.argsort(contribution[65792:66269])[::-1]
        blocks={'text':float(contribution[:65792].sum()),'facts':float(contribution[65792:66269].sum()),'record_port':float(contribution[66269:].sum()),'intercept':bias}
        actual=float(np.mean(xx@dw+bias)) if n else None
        assert actual is None or abs(sum(blocks.values())-actual)<1e-5
        coef_report[route]={'model':modelname,'margin_classes':list(pair),'all_rows':len(take),'wrong_rows':n,'mean_margin':actual,'block_contributions':blocks,
           'top_positive_fact_contributions':[{'feature':fact_names[i],'mean_contribution':float(contribution[65792+i])} for i in order[:12]],
           'top_negative_fact_contributions':[{'feature':fact_names[i],'mean_contribution':float(contribution[65792+i])} for i in order[-6:]]}
    save(DEST/'linear_margin_attribution.json',coef_report)
    # Model geometry: a larger coefficient norm is measurable, overfitting is an explanation hypothesis.
    geometry={}
    for name,path in [('v78_SGD',OLD/'O_sgd.joblib'),('v78_LBFGS',OLD/'O_lbfgs.joblib'),('v79_LBFGS',LAST/'P1_B_ovr.joblib')]:
        model=joblib.load(path);w=model.coef_.T if hasattr(model,'coef_') else model['coef']
        geometry[name]={'coef_l2':float(np.linalg.norm(w)),'MS_difference_l2':float(np.linalg.norm(w[:,1]-w[:,2])),
                        'text_l2':float(np.linalg.norm(w[:65792])),'facts_l2':float(np.linalg.norm(w[65792:66269]))}
    save(DEST/'model_geometry.json',geometry)
    # Observability audit: exact input conflicts are not a general prerequisite to test capacity.
    allcounts=np.bincount(fid[fit]*3+r.label_index.to_numpy()[fit],minlength=x.shape[0]*3).reshape(-1,3)
    pred=np.load(LAST/'P1_B_ovr_feature_prediction.npy');err=fit&(pred[fid]!=r.label_index.to_numpy())
    support=r.groupby(['route','label_index']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    support.to_csv(DEST/'full_training_route_class_support.csv')
    capacity={'fit_errors':int(err.sum()),'encoded_MS_conflict_floor':int(np.minimum(allcounts[:,1],allcounts[:,2]).sum()),
              'warning':'Identical ordered residuals in two conflicting feature groups do not rule out nonlinear/order/binding benefits for other errors. P3 was not tested; do not claim capacity has been ruled out.',
              'fit_errors_by_route_class':r[err].groupby(['route','label_index']).size().rename('errors').reset_index().to_dict('records')}
    save(DEST/'capacity_and_support.json',capacity)
    print(json.dumps({'regression_chain':report['loss_chain'],'sampling':sampling,'geometry':geometry,'linear_attribution':coef_report},ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
