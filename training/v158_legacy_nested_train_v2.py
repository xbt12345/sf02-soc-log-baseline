"""Fresh nine full-population N1 linear experts; no outer teacher used as OOF."""
import argparse,gc,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import softmax
from threadpoolctl import threadpool_limits
import joblib
from v158_nested_runtime_v2 import ROOT,OUT,PLAN,read,sha,save,review,require
from v158_fusion_contract import inner_fold
from v107_matched_training import ROWS,FOLDS,FID,ASA_IDS,X,PREP
from run_v75 import load_sparse
from v85_protection import baseline_objective

class LegacyGradientBudget(RuntimeError):pass

def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)

def fit(f,j):
    p=require(__file__);folder=OUT/f'legacy_outer{f}_inner{j}';assert not folder.exists()
    order=[(a,b) for a in range(3) for b in range(3)]
    done=[pair for pair in order if (OUT/f'legacy_outer{pair[0]}_inner{pair[1]}'/'fit.json').exists()]
    assert (f,j)==order[len(done)]
    rr=pd.read_parquet(ROWS,columns=['row_position','route','label_index'])
    source=pd.read_parquet(FOLDS,columns=['row_position','root','proposed_fold'])
    assert np.array_equal(rr.row_position,source.row_position)
    inner=source.root.map(lambda z:inner_fold(f,z));mask=source.proposed_fold.ne(f)&inner.ne(j)
    excluded=source.proposed_fold.ne(f)&inner.eq(j)
    assert not set(source.loc[mask,'root'])&set(source.loc[excluded,'root'])
    assert not set(source.loc[mask,'root'])&set(source.loc[source.proposed_fold.eq(f),'root'])
    fid=np.load(FID,mmap_mode='r');base=load_sparse(X);delta=sparse.load_npz(PREP/'N1_delta.npz')
    c=np.bincount(fid[mask]*3+rr.label_index.to_numpy()[mask],minlength=base.shape[0]*3).reshape(-1,3)
    expected=p['legacy_role_budgets'][f*3+j];assert c.sum(0).tolist()==expected['fit_class_mass'] and int(c.sum())==expected['fit_rows']
    ids=np.flatnonzero(c.sum(1));feat=base[ids]+delta[ids];weights=c[ids].astype(np.float64)
    folder.mkdir();save(folder/'started.json',dict(outer=f,excluded_inner=j,fit_rows=int(mask.sum()),class_mass=c.sum(0).tolist(),
        fit_roots=sorted(set(source.loc[mask,'root'])),all_query_and_outer_roots_excluded=True,prior_supervised_model_weights_loaded=False,seal_sha256=sha(OUT/'run_seal_v2.json')))
    theta=np.zeros((66288,3),np.float64);theta[-1]=np.log(np.maximum(weights.sum(0)/weights.sum(),1e-9))
    accepted=theta.ravel().copy();evaluations=iterations=0;start=time.monotonic();history=[]
    log=(folder/'gradient_calls.jsonl').open('w',encoding='utf-8',buffering=1)
    def objective(values):
        nonlocal evaluations
        if evaluations>=1000:raise LegacyGradientBudget('Fixed full-risk gradient budget')
        evaluations+=1;log.write(json.dumps(dict(event='gradient_attempt',call=evaluations,original_class_mass=c.sum(0).tolist()))+'\n')
        value,gradient=baseline_objective(values,feat,weights)
        if not np.isfinite(value) or not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite original N1 risk')
        log.write(json.dumps(dict(event='gradient_completed',call=evaluations,objective=float(value),gradient_inf=float(np.abs(gradient).max())))+'\n')
        return value,gradient
    def callback(values):
        nonlocal accepted,iterations
        accepted=values.copy();iterations+=1
        history.append(dict(iteration=iterations,full_gradients=evaluations,seconds=time.monotonic()-start));save(folder/'progress.json',history)
        if iterations%25==0:emit(stage='nested_legacy_iteration',outer=f,inner=j,iteration=iterations,full_gradients=evaluations)
    with threadpool_limits(limits=4):
        try:
            result=minimize(objective,theta.ravel(),jac=True,method='L-BFGS-B',callback=callback,
                options=dict(maxiter=1000,maxcor=10,gtol=1e-6,ftol=1e-12,maxls=30))
            endpoint=result.x;termination=str(result.message);converged=bool(result.success and np.abs(result.jac).max()<=1e-5)
        except LegacyGradientBudget:
            endpoint=accepted;termination='global_gradient_budget_trial_rolled_back';converged=False
    log.close();w=endpoint.reshape(66288,3);model=dict(coef=w[:-1].copy(),intercept=w[-1].copy());joblib.dump(model,folder/'endpoint.joblib')
    # Query the original N1 ASA input, not the canonical-header input.
    asa=sparse.load_npz(PREP/'N1_ASA.npz');assert asa.shape==(22546,66287)
    parts=[];classifier_prediction_calls=0
    calls=(folder/'ASA_forward_calls.jsonl').open('w',encoding='utf-8',buffering=1)
    for offset in range(0,22546,8192):
        classifier_prediction_calls+=1;calls.write(json.dumps(dict(event='attempt',call=classifier_prediction_calls,offset=offset))+'\n')
        ii=slice(offset,min(offset+8192,22546));parts.append(np.asarray(asa[ii]@model['coef'])+model['intercept'])
        calls.write(json.dumps(dict(event='completed',call=classifier_prediction_calls))+'\n')
    calls.close()
    scores=np.concatenate(parts);assert np.isfinite(scores).all();np.save(folder/'ASA_logits.npy',scores)
    require(__file__)
    save(folder/'fit.json',dict(status='legacy_inner_base_fit_executed',outer=f,excluded_inner=j,new_fits=1,full_gradient_evaluations=evaluations,
        accepted_iterations=iterations,gradient_risk_classifier_forward_calls=evaluations,ASA_classifier_forward_chunks=classifier_prediction_calls,
        gradients_per_call_original_class_mass=c.sum(0).tolist(),termination=termination,converged=converged,selected_by_score=False,query_or_outer_labels_used=0,
        fit_rows=int(mask.sum()),class_mass=c.sum(0).tolist(),seconds=time.monotonic()-start,model_sha256=sha(folder/'endpoint.joblib'),scores_sha256=sha(folder/'ASA_logits.npy'),
        seal_sha256=sha(OUT/'run_seal_v2.json'),quality_acceptance=False,new_task_candidate=False))
    emit(stage='nested_legacy_fit_complete',outer=f,inner=j,full_gradients=evaluations,accepted_iterations=iterations,converged=converged)
    del base,delta,feat,c,weights;gc.collect()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fit','all-legacy']);ap.add_argument('--outer',type=int);ap.add_argument('--inner',type=int);a=ap.parse_args()
    try:
        if a.mode=='all-legacy':
            for f in range(3):
                for j in range(3):fit(f,j)
            receipts=[read(OUT/f'legacy_outer{f}_inner{j}'/'fit.json') for f in range(3) for j in range(3)]
            save(OUT/'legacy_phase_completion.json',dict(status='all_nine_new_full_population_N1_inner_fits_completed',new_fits=9,
                full_gradients=sum(z['full_gradient_evaluations'] for z in receipts),accepted_iterations=sum(z['accepted_iterations'] for z in receipts),
                ASA_classifier_forward_chunks=sum(z['ASA_classifier_forward_chunks'] for z in receipts),quality_acceptance=False))
        else:fit(a.outer,a.inner)
    except Exception as e:
        if OUT.exists():save(OUT/'legacy_execution_failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),entry_sha256=sha(Path(__file__)),no_automatic_restart=True))
        raise
