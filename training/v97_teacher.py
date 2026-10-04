"""One shared A+B teacher, no V or historic locked labels loaded."""
import json
import time
import joblib
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, read, save, sha, load_sparse
from v89_common import LAST
from v85_protection import baseline_objective, cm_from_counts
from v97_prepare import DEST


def main():
    assert not (DEST/'teacher_fit.json').exists()
    reg=read(DEST/'registration.json')
    assert reg['source_sha256']==sha(ROOT/'training/v97_prepare.py')
    assert reg['counts_sha256']==sha(DEST/'AB_train_counts.npz')
    with np.load(DEST/'AB_train_counts.npz') as z:counts=z['A'].astype(np.float64)+z['B'].astype(np.float64)
    assert int(counts.sum())==753709
    x=load_sparse(LAST/'X');used=np.flatnonzero(counts.sum(1))
    rows=counts[used];features=x[used];w=np.zeros((x.shape[1]+1,3),np.float64)
    w[-1]=np.log(np.maximum(rows.sum(0)/rows.sum(),1e-9));flat=w.ravel()
    rng=np.random.default_rng(reg['seed']);direction=rng.normal(size=len(flat));direction/=np.linalg.norm(direction)
    eps=1e-5;smallx,smallc=features[:31],rows[:31]
    _,grad=baseline_objective(flat,smallx,smallc)
    fd=(baseline_objective(flat+eps*direction,smallx,smallc)[0]-baseline_objective(flat-eps*direction,smallx,smallc)[0])/(2*eps)
    assert abs(fd-grad@direction)<1e-6
    started=time.monotonic();progress=[]
    def callback(theta):
        progress.append({'iteration':len(progress)+1,'seconds':time.monotonic()-started})
        if len(progress)%25==0:
            loss,g=baseline_objective(theta,features,rows)
            print(json.dumps({'stage':'teacher_progress','iteration':len(progress),'loss':loss,
                'gradient_inf':float(np.abs(g).max()),'seconds':time.monotonic()-started}),flush=True)
    save(DEST/'teacher_started.json',{'stage':'started','source_sha256':sha(__file__),'A_B_rows':int(rows.sum()),'seed':reg['seed']})
    with threadpool_limits(limits=4):
        result=minimize(baseline_objective,flat,args=(features,rows),jac=True,method='L-BFGS-B',callback=callback,
            options={'maxiter':1000,'maxcor':10,'gtol':1e-6,'ftol':1e-12,'maxls':30})
    ww=result.x.reshape(x.shape[1]+1,3);model={'coef':ww[:-1].copy(),'intercept':ww[-1].copy()}
    joblib.dump(model,DEST/'teacher.joblib')
    scores=np.asarray(x@model['coef'])+model['intercept'];np.save(DEST/'teacher_scores.npy',scores)
    pred=scores.argmax(1).astype(np.int8);np.save(DEST/'teacher_prediction.npy',pred)
    metrics={}
    for role in 'AB':
        with np.load(DEST/'AB_train_counts.npz') as z:c=z[role]
        cm=cm_from_counts(c,pred)
        metrics[role]={'rows':int(c.sum()),'class_support':c.sum(0).astype(int).tolist(),
            'class_correct':np.diag(cm).astype(int).tolist(),'errors':int(c.sum()-np.trace(cm))}
    report={'stage':'teacher_fit','status':'fit_executed','actual_classifier_fits':1,'actual_calibration_fits':0,
        'A_B_original_rows_used':int(rows.sum()),'A_B_unique_inputs_used':len(used),
        'A_B_class_rows_used':rows.sum(0).astype(int).tolist(),'V_gradient_rows_used':0,
        'inner_C_H_labels_loaded':False,'gradient_check_error':float(abs(fd-grad@direction)),
        'converged':bool(result.success and np.abs(result.jac).max()<=1e-5),
        'iterations':int(result.nit),'objective':float(result.fun),
        'gradient_inf':float(np.abs(result.jac).max()),'message':str(result.message),
        'seconds':time.monotonic()-started,'training_role_metrics':metrics,'source_sha256':sha(__file__),
        'model_sha256':sha(DEST/'teacher.joblib'),'scores_sha256':sha(DEST/'teacher_scores.npy'),
        'prediction_sha256':sha(DEST/'teacher_prediction.npy')}
    save(DEST/'teacher_progress.json',progress);save(DEST/'teacher_fit.json',report)
    print(json.dumps({'stage':'teacher_complete','converged':report['converged'],'iterations':report['iterations'],
        'gradient_inf':report['gradient_inf'],'A_B_errors':{k:v['errors'] for k,v in metrics.items()},
        'seconds':report['seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
