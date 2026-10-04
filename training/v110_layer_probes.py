"""V110: execute the registered V109 P1/P2 frozen-layer probe design."""
import argparse
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from run_v75 import ROOT, save, sha
from v75_views import BYTE_FEATURES
from v104_phase_b import SparseTabM, csr_tensor, DEVICE, K
from v108_root_evidence_audit import PREV, DEST as OLD

PLAN = ROOT/'artifacts/v109_plan_review_20260929/next_training_contract.json'
DEST = ROOT/'artifacts/v110_layer_probes_20260929'
LEDGER = OLD/'support_and_error_ledger.parquet'
N2 = PREV/'N2_ASA.npz'
FOLDS = ROOT/'artifacts/v106_frozen_audit_20260928/proposed_body_closed_folds.parquet'
FID = ROOT/'artifacts/v79_execution_20260927/row_feature_id.npy'
ROWS = ROOT/'artifacts/v75_four_arm_20260921_r2/rows.parquet'
OFFICIAL = ROOT/'data/official/train.parquet'
LAMBDA = 1e-3


def rel(p): return Path(p).relative_to(ROOT).as_posix()


def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def register():
    assert not DEST.exists(),DEST
    plan=read_json(PLAN)
    assert plan['status']=='plan_only_not_training_registration'
    assert plan['next_stage_max_classifier_fits']==6
    assert [(a['name'],a['dimensions']) for a in plan['arms']]==[('P1',2543),('P2',2543)]
    old=read_json(ROOT/'artifacts/v109_plan_review_20260929/verification.json')
    assert old['all_checks_passed']
    d=pd.read_parquet(LEDGER,columns=['row_position','local','fold','truth','root'])
    assert len(d)==112807 and set(d.fold)=={0,1,2} and set(d.truth)=={1,2}
    f=pd.read_parquet(FOLDS,columns=['row_position','proposed_fold','root'])
    assert len(f)==2056871 and np.array_equal(f.row_position.to_numpy()[d.row_position.to_numpy()],d.row_position.to_numpy())
    assert np.array_equal(f.proposed_fold.to_numpy()[d.row_position.to_numpy()],d.fold.to_numpy())
    assert np.array_equal(f.root.to_numpy()[d.row_position.to_numpy()],d.root.to_numpy())
    inputs=[PLAN,LEDGER,N2,FOLDS,FID,ROWS,OFFICIAL,
      ROOT/'artifacts/v109_plan_review_20260929/verification.json',
      ROOT/'artifacts/v107_matched_training_20260928/evaluation.json',
      ROOT/'artifacts/v106_frozen_audit_20260928/verification_and_N2_preflight.json']
    for k in (0,1,2):
        p=PREV/f'fold{k}_N2_TabM25_seed10201'
        fit=read_json(p/'fit.json')
        assert fit['model_sha256']==sha(p/'model.pt')
        assert fit['prob_sha256']==sha(p/'ASA_input_prob.npy')
        inputs += [p/'model.pt',p/'ASA_input_prob.npy',p/'fit.json',
          PREV/f'fold{k}_N1_teacher'/'scores_all_input_ids.npy']
    reg={'status':'registered_before_any_fit','classifier_fits_at_registration':0,
      'source_sha256':sha(__file__),
      'evaluation_source_sha256':sha(ROOT/'training/v110_layer_probes_evaluate.py'),
      'model_source_sha256':sha(ROOT/'training/v104_phase_b.py'),
      'input_sha256':{rel(p):sha(p) for p in inputs},
      'contract_sha256':sha(PLAN),
      'fold_class_rows':{str(k):np.bincount(d.loc[d.fold==k,'truth'],minlength=3).tolist() for k in (0,1,2)},
      'all_official_rows':2056871,'ASA_rows':112807,'ASA_M_rows':78748,'ASA_S_rows':34059,
      'environment':{'python':sys.version,'numpy':np.__version__,'scipy':importlib.metadata.version('scipy'),
        'pandas':pd.__version__,'torch':torch.__version__,'tabm':importlib.metadata.version('tabm'),
        'device':DEVICE,'architecture':__import__('platform').machine()},
      'no_private_answer_used':True,'old_models_frozen':True,'fit_budget':6,
      'heldout_labels_used_in_fit':False,'external_training_data_used':False}
    DEST.mkdir();save(DEST/'registration.json',reg)
    print(json.dumps({'stage':'registered','sha256':sha(DEST/'registration.json'),
      'fold_class_rows':reg['fold_class_rows']},ensure_ascii=False),flush=True)


def check_registration():
    r=read_json(DEST/'registration.json')
    assert r['status']=='registered_before_any_fit'
    assert sha(__file__)==r['source_sha256']
    assert sha(ROOT/'training/v110_layer_probes_evaluate.py')==r['evaluation_source_sha256']
    assert sha(ROOT/'training/v104_phase_b.py')==r['model_source_sha256']
    for p,digest in r['input_sha256'].items():assert sha(ROOT/p)==digest,p
    return r


def objective(theta, x, m, s, lam=LAMBDA):
    w=theta[:-1];b=theta[-1]
    z=x@w+b
    total=float((m+s).sum())
    value=(m@np.logaddexp(0,z)+s@np.logaddexp(0,-z))/total+lam*(w@w)/2
    residual=((m+s)*expit(z)-s)/total
    grad=np.empty(len(theta),np.float64)
    grad[:-1]=x.T@residual+lam*w
    grad[-1]=residual.sum()
    return float(value),grad


def selftest():
    rng=np.random.default_rng(110)
    x=rng.normal(size=(21,7));m=rng.integers(0,8,size=21).astype('f8');s=rng.integers(0,6,size=21).astype('f8')
    t=rng.normal(size=8)
    val,grad=objective(t,x,m,s)
    for j in range(len(t)):
        delta=np.zeros_like(t);delta[j]=1e-6
        approx=(objective(t+delta,x,m,s)[0]-objective(t-delta,x,m,s)[0])/2e-6
        assert abs(grad[j]-approx)<1e-8,(j,grad[j],approx)
    expanded=np.repeat(x,np.asarray(m+s,dtype=np.int64),axis=0)
    expanded_m=np.concatenate([np.r_[np.ones(int(mm)),np.zeros(int(ss))]
      for mm,ss in zip(m,s)])
    expanded_s=1-expanded_m
    e_value,e_grad=objective(t,expanded,expanded_m,expanded_s)
    assert abs(val-e_value)<1e-12 and np.allclose(grad,e_grad,atol=1e-12,rtol=1e-12)
    print(json.dumps({'stage':'selftest','objective':val,'gradient_max_error_bound':1e-8,
      'weighted_rows_equal_expanded_rows':True}))


def extract(fold,x):
    folder=PREV/f'fold{fold}_N2_TabM25_seed10201'
    checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    assert checkpoint['fold']==fold and checkpoint['view']=='N2' and checkpoint['epoch']==25
    model=SparseTabM().to(DEVICE);model.load_state_dict(checkpoint['state_dict']);model.eval()
    captured={}
    h1=model.first.register_forward_hook(lambda _m,_a,v:captured.update(P1=torch.relu(v)))
    h2=model.second.register_forward_hook(lambda _m,_a,v:captured.update(P2=torch.relu(v)))
    arrays={'P1':[],'P2':[]};prob=[]
    with torch.inference_mode():
        for start in range(0,x.shape[0],512):
            block=x[start:start+512]
            fact=torch.as_tensor(block[:,BYTE_FEATURES:].toarray(),device=DEVICE,dtype=torch.float32)
            z=model(csr_tensor(block,DEVICE),fact)
            prob.append(torch.softmax(z,-1).mean(1).cpu().numpy())
            fv=fact.cpu().numpy()
            for arm in ('P1','P2'):
                layer=captured[arm].flatten(1).cpu().numpy()
                assert layer.shape==(len(fv),K*128)
                arrays[arm].append(np.concatenate([layer,fv],axis=1))
    h1.remove();h2.remove();del model
    if DEVICE=='cuda':torch.cuda.empty_cache()
    replay=np.concatenate(prob)
    saved=np.load(folder/'ASA_input_prob.npy')
    assert np.array_equal(replay.argmax(1),saved.argmax(1))
    assert np.allclose(replay,saved,atol=2e-6,rtol=2e-6)
    result={a:np.concatenate(chunks) for a,chunks in arrays.items()}
    assert all(v.shape==(x.shape[0],2543) and np.isfinite(v).all() for v in result.values())
    assert np.array_equal(result['P1'][:,-495:],result['P2'][:,-495:])
    return result,float(np.max(np.abs(replay-saved)))


def weighted_standardizer(features,counts):
    weight=counts.sum(1).astype(np.float64);total=weight.sum()
    used=weight>0
    train=features[used].astype(np.float64)
    w=weight[used]
    mean=(w@train)/total
    # Weighted central second moment; stable even for sparse binary facts.
    centered=train-mean
    var=(w@(centered*centered))/total
    std=np.sqrt(np.maximum(var,0))
    std[std<1e-12]=1.
    standardized=np.ascontiguousarray(centered/std)
    return used,standardized,mean,std


def fit_one(arm,fold,features,counts,replay_diff):
    folder=DEST/f'fold{fold}_{arm}'
    assert not folder.exists();folder.mkdir()
    started=time.monotonic()
    used,xt,mean,std=weighted_standardizer(features,counts)
    assert int(counts.sum())==int((counts[used]).sum())
    m=counts[used,1].astype(np.float64);s=counts[used,2].astype(np.float64)
    assert m.sum()>0 and s.sum()>0
    init=np.zeros(features.shape[1]+1,np.float64);init[-1]=math.log(float(s.sum()/m.sum()))
    history=[]
    def callback(theta):
        if len(history)%25==0:
            value,grad=objective(theta,xt,m,s)
            print(json.dumps({'stage':'fit_progress','fold':fold,'arm':arm,'iteration':len(history),
              'objective':value,'gradient_inf':float(np.max(np.abs(grad))),
              'seconds':round(time.monotonic()-started,2)}),flush=True)
        history.append(time.monotonic()-started)
    try:
        with threadpool_limits(limits=4):
            result=minimize(objective,init,args=(xt,m,s),jac=True,method='L-BFGS-B',callback=callback,
              options={'maxiter':2000,'maxcor':20,'gtol':1e-7,'ftol':1e-15,'maxls':40})
        val,grad=objective(result.x,xt,m,s)
        converged=bool(np.isfinite(val) and np.isfinite(grad).all() and np.abs(grad).max()<=1e-5)
        # No extra fit or changed hyperparameter is permitted if the contract is missed.
        save(folder/'fit.json',{'status':'fit_executed','classifier_fits':1,'arm':arm,'fold':fold,
          'train_original_rows':int(m.sum()+s.sum()),'train_M_rows':int(m.sum()),'train_S_rows':int(s.sum()),
          'train_unique_input_ids':int(used.sum()),'solver_success':bool(result.success),
          'converged_to_registered_gradient_gate':converged,'iterations':int(result.nit),
          'objective':val,'gradient_inf':float(np.abs(grad).max()),'message':str(result.message),
          'elapsed_seconds':time.monotonic()-started,'source_sha256':sha(__file__),
          'registration_sha256':sha(DEST/'registration.json'),'old_N2_probability_replay_max_abs_difference':replay_diff,
          'heldout_label_used_in_fit':False})
        if not converged:raise RuntimeError(f'Numerical convergence gate failed: {fold} {arm}, grad={np.abs(grad).max()}')
        theta=result.x
        np.savez_compressed(folder/'probe.npz',coef=theta[:-1],intercept=theta[-1],mean=mean,std=std)
        score=np.empty(len(features),np.float32)
        for lo in range(0,len(features),4096):
            hi=min(lo+4096,len(features))
            score[lo:hi]=expit(((features[lo:hi].astype(np.float64)-mean)/std)@theta[:-1]+theta[-1]).astype('f4')
        assert np.isfinite(score).all()
        np.save(folder/'ASA_input_S_probability.npy',score)
        report=read_json(folder/'fit.json')
        report.update(probe_sha256=sha(folder/'probe.npz'),prob_sha256=sha(folder/'ASA_input_S_probability.npy'),
          train_original_correct=int((counts[used,1]*(score[used]<.5)+counts[used,2]*(score[used]>=.5)).sum()))
        save(folder/'fit.json',report)
        print(json.dumps({'stage':'fit_complete','fold':fold,'arm':arm,'converged':converged,
          'iterations':result.nit,'gradient_inf':report['gradient_inf'],
          'train_M':report['train_M_rows'],'train_S':report['train_S_rows'],
          'seconds':round(report['elapsed_seconds'],1)}),flush=True)
    finally:
        del xt


def fit_fold(fold):
    assert fold in (0,1,2)
    reg=check_registration()
    assert all(not (DEST/f'fold{fold}_{a}').exists() for a in ('P1','P2'))
    d=pd.read_parquet(LEDGER,columns=['local','fold','truth'])
    local=d.local.to_numpy(dtype=np.int64);y=d.truth.to_numpy(dtype=np.int8)
    role=d.fold.to_numpy(dtype=np.int8)
    x=sparse.load_npz(N2)
    assert x.shape==(22546,66287)
    assert local.min()==0 and local.max()==x.shape[0]-1
    training=role!=fold
    counts=np.bincount(local[training]*3+y[training],minlength=x.shape[0]*3).reshape(-1,3)
    assert not counts[:,0].any() and int(counts.sum())==int(training.sum())
    assert np.array_equal(counts.sum(0),np.bincount(y[training],minlength=3))
    expected=np.asarray(reg['fold_class_rows']['0'])+np.asarray(reg['fold_class_rows']['1'])+np.asarray(reg['fold_class_rows']['2'])-np.asarray(reg['fold_class_rows'][str(fold)])
    assert np.array_equal(counts.sum(0),expected)
    features,replay_diff=extract(fold,x)
    print(json.dumps({'stage':'extracted','fold':fold,'features':{a:list(v.shape) for a,v in features.items()},
      'replay_max_diff':replay_diff},ensure_ascii=False),flush=True)
    for arm in ('P1','P2'):
        fit_one(arm,fold,features[arm],counts,replay_diff)
    print(json.dumps({'stage':'fold_complete','fold':fold}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['selftest','register','fit-fold'])
    p.add_argument('--fold',type=int)
    a=p.parse_args()
    if a.stage=='selftest':selftest()
    elif a.stage=='register':register()
    else:fit_fold(a.fold)
