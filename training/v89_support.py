"""Five from-scratch support-withdrawal/control fits; no old trained state reused."""
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from v89_common import ROOT,OUT,DEST,LAST,check,read,save,sha,emit,raw_counts,metrics,cm_from_counts,ledger_start,ledger_end
from v79_execute import rows
from v85_protection import baseline_objective,WIDTH


class BudgetStop(Exception):pass


def fit(name):
    reg=check();assert name in reg['support_controls'];spec=read(DEST/'support_design.json')
    if not spec['feasible']:raise RuntimeError('No matched support design; do not replace with an unmatched claim')
    path=DEST/('support_'+name+'_fit.json');assert not path.exists()
    li=ledger_start('classifier_fit','support_'+name);start=time.monotonic()
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=__import__('run_v75').load_sparse(LAST/'X')
    train=np.load(DEST/('support_'+name+'_rows.npy'));evalrows=np.load(DEST/'support_evaluation_rows.npy');target=np.load(DEST/'support_target_rows.npy')
    assert not set(r.component.iloc[train]).intersection(set(r.component.iloc[evalrows]))
    cc=np.bincount(fid[train]*3+y[train],minlength=x.shape[0]*3).reshape(-1,3);used=np.flatnonzero(cc.sum(1));xx=x[used];counts=cc[used]
    initial=np.zeros((WIDTH+1)*3);initial[-3:]=np.log(np.maximum(counts.sum(0)/counts.sum(),1e-9))
    v=[initial.copy()];iterations=[0];lastgrad=[None];logs=[]
    def cb(theta):
        v[0]=theta.copy();iterations[0]+=1
        if iterations[0]%50==0:
            f,g=baseline_objective(theta,xx,counts);lastgrad[0]=g
            entry={'condition':name,'iteration':iterations[0],'objective':f,'gradient_inf':float(np.abs(g).max()),'seconds':time.monotonic()-start}
            logs.append(entry);save(DEST/('support_'+name+'_progress.json'),logs);emit(stage='support_fit',**entry)
        if time.monotonic()-start>reg['support_seconds_per_fit']:raise BudgetStop()
    try:
        res=minimize(baseline_objective,initial,args=(xx,counts),jac=True,method='L-BFGS-B',callback=cb,
            options={'maxiter':1000,'maxcor':10,'gtol':1e-6,'ftol':1e-12,'maxls':30})
        theta=res.x;success=bool(res.success and np.abs(res.jac).max()<=1e-5);message=str(res.message)
    except BudgetStop:
        theta=v[0];success=False;message='registered_wall_budget_reached'
    f,g=baseline_objective(theta,xx,counts);w=theta.reshape(WIDTH+1,3);model={'coef':w[:-1].copy(),'intercept':w[-1].copy()}
    joblib.dump(model,DEST/('support_'+name+'.joblib'))
    unique=np.unique(np.r_[fid[evalrows],fid[target]]);scores=np.asarray(x[unique]@model['coef'])+model['intercept'];pred=scores.argmax(1).astype(np.int8)
    pfull=pred[np.searchsorted(unique,fid[evalrows])];ptarget=pred[np.searchsorted(unique,fid[target])]
    np.savez_compressed(DEST/('support_'+name+'_predictions.npz'),evaluation_rows=evalrows,evaluation_prediction=pfull,target_rows=target,target_prediction=ptarget,
         unique_input_ids=unique,scores=scores)
    cm=lambda ix,pr:np.bincount(y[ix]*3+pr,minlength=9).reshape(3,3)
    result={'status':'completed' if success else 'stopped_unconverged','new_classifier_fits':1,'converged':success,
       'condition':name,'initialization':'fresh zero coefficients and condition-local class prior intercept; no prior trained model',
       'iteration':iterations[0],'objective':f,'gradient_inf':float(np.abs(g).max()),'message':message,
       'training_rows':len(train),'training_class_counts':np.bincount(y[train],minlength=3).tolist(),
       'component_disjoint':True,'all_evaluation':metrics(cm(evalrows,pfull)),'target_evaluation':metrics(cm(target,ptarget)),
       'seconds':time.monotonic()-start,'source_sha256':sha(__file__),'model_sha256':sha(DEST/('support_'+name+'.joblib')),
       'training_rows_sha256':sha(DEST/('support_'+name+'_rows.npy'))}
    save(path,result);ledger_end(li,'completed' if success else 'stopped_unconverged',converged=success,seconds=result['seconds'])
    emit(stage='support_complete',condition=name,converged=success,target=result['target_evaluation'],seconds=result['seconds'])


def summarize():
    reg=check();spec=read(DEST/'support_design.json')
    if not spec['feasible']:
        save(DEST/'support_results.json',{'status':'blocked_by_no_matched_components','fits_executed':0,'reason':spec['rejected']});return
    reports={n:read(DEST/('support_'+n+'_fit.json')) for n in reg['support_controls']};full=np.load(DEST/'support_full_predictions.npz')
    target=full['target_rows'];r=rows();y=r.label_index.to_numpy();rowsout=[]
    for name,rec in reports.items():
        a=np.load(DEST/('support_'+name+'_predictions.npz'));assert np.array_equal(a['target_rows'],target)
        b=full['target_prediction'];p=a['target_prediction']
        for cls in [1,2]:
            take=y[target]==cls
            rowsout.append({'condition':name,'class':cls,'support':int(take.sum()),'correct':int((p[take]==cls).sum()),
                'repairs_vs_full':int(((b[take]!=cls)&(p[take]==cls)).sum()),'regressions_vs_full':int(((b[take]==cls)&(p[take]!=cls)).sum()),
                'components':int(r.component.iloc[target[take]].nunique())})
    pd.DataFrame(rowsout).to_csv(DEST/'support_comparison.csv',index=False)
    contrasts=[]
    for cls in [1,2]:
        a=np.load(DEST/f'support_withdraw_{cls}_predictions.npz')['target_prediction'];b=np.load(DEST/f'support_control_{cls}_predictions.npz')['target_prediction']
        take=y[target]==cls
        difference=(a[take]!=cls).astype(int)-(b[take]!=cls).astype(int)
        components=r.component.iloc[target[take]].to_numpy();bycomponent=pd.DataFrame({'component':components,'extra_errors_withdraw_vs_control':difference}).groupby('component').sum()
        contrasts.append({'withdrawn_class':cls,'support':int(take.sum()),'withdraw_correct':int((a[take]==cls).sum()),
            'control_correct':int((b[take]==cls).sum()),'extra_errors_withdraw_vs_control':int(difference.sum()),
            'component_differences':bycomponent.reset_index().to_dict('records'),
            'uncertainty':'Selected controlled development cases only; too few evaluation components for a broad-confidence generalization claim' if len(bycomponent)<10 else 'Report component distribution, not row-independent CI'})
    save(DEST/'support_results.json',{'status':'completed','actual_classifier_fits':5,'all_conditions_converged':all(a['converged'] for a in reports.values()),
        'controlled_same_class_route_source_fold_row_counts':True,'component_matching_residual':spec['control_matching'],
        'contrasts':contrasts,'conditions':reports,'source_sha256':sha(__file__),
        'scope':'Fresh R0 architecture only; fixed isolated groups; class/ASA/source-fold matched, residual subject-composition confound disclosed. No isolated universal support-causal estimate or claim about every stronger model/natural zero-support error.'})
    emit(stage='support_summary',contrasts=contrasts,converged=all(a['converged'] for a in reports.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','summarize']);p.add_argument('--condition');a=p.parse_args()
    with threadpool_limits(limits=4):fit(a.condition) if a.stage=='fit' else summarize()
