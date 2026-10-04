"""Verify finite-input preservation counts and the local loss derivative."""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, read, save, sha, load_sparse
from v79_execute import rows
from v82_capacity import LAST, DEST as OLD

DEST=ROOT/'artifacts/v84_preservation_20260927'


def main():
    if (DEST/'review_receipt.json').exists():raise FileExistsError('Published review is frozen')
    d=read(DEST/'diagnosis.json');assert d['source_sha256']==sha(ROOT/'training/v84_preservation_audit.py')
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    names={a:read(OLD/('primary_'+a+'_fit.json'))['states'][-1]['name'] for a in ['linear','nonlinear']}
    old=np.load(OLD/(names['linear']+'_prediction.npy'))[fid];new=np.load(OLD/(names['nonlinear']+'_prediction.npy'))[fid]
    selected=np.load(OLD/'selected_rows.npy');sel=np.zeros(len(r),bool);sel[selected]=True
    masks={'selected_fit':sel,'fit':~r.fold.isin([0,2]).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    for rec in d['transition_table']:
        take=masks[rec['role']];yc=y[take];a=old[take];b=new[take]
        matrix=np.bincount((a==yc).astype(int)*2+(b==yc).astype(int),minlength=4).reshape(2,2)
        assert int(matrix[0,1])==rec['positive_flips'] and int(matrix[1,0])==rec['negative_flips']
        assert int((a!=b).sum())==rec['changed_decisions']
        assert rec['old_errors']-rec['new_errors']==rec['positive_flips']-rec['negative_flips']
    frame=pd.DataFrame({'fid':fid[selected],'label':y[selected],'old':old[selected]})
    conflicts=0;wrong=0;floor=0;lockfloor=0
    for _,g in frame.groupby('fid'):
        truth=g.label.value_counts();prediction=int(g.old.iloc[0]);pc=int(truth.get(prediction,0))
        floor+=len(g)-int(truth.max());lockfloor+=len(g)-pc if pc else len(g)-int(truth.max())
        if pc and pc<len(g):conflicts+=1;wrong+=len(g)-pc
    f=d['fit_input_lock_feasibility']
    assert (conflicts,wrong,floor,lockfloor)==(f['protected_inputs_with_other_labels'],f['wrong_rows_in_protected_inputs'],
        f['empirical_minimum_errors_with_unrestricted_deterministic_input_lookup'],f['empirical_minimum_errors_keeping_every_old_correct_fit_decision'])
    m=joblib.load(OLD/(names['linear']+'.joblib'));z=np.asarray(x@m['coef'])+m['intercept']
    cohort=np.load(OLD/'fit_457_cohort.npy');ix=cohort[new[cohort]!=2]
    residual=expit(z[fid[ix]]);residual[np.arange(len(ix)),y[ix]]-=1
    gw=np.asarray(x[fid[ix]].T@residual)/len(ix);gb=residual.mean(0)
    norm=np.sqrt(np.square(gw).sum()+np.square(gb).sum());gw/=(-norm);gb/=(-norm)
    result={}
    for cls,name in enumerate(['B','M','S']):
        ix=np.flatnonzero(sel&(old==y)&(y==cls));zz=z[fid[ix]];delta=np.asarray(x[fid[ix]]@gw)+gb
        def loss(scores):return float((np.logaddexp(0,scores).sum(1)-scores[np.arange(len(ix)),y[ix]]).mean())
        eps=1e-4;fd=(loss(zz+eps*delta)-loss(zz-eps*delta))/(2*eps)
        claimed=d['local_gradient_interference']['protected_classes'][name]['directional_protected_loss_change_per_unit_error_descent']
        assert abs(fd-claimed)<1e-8,(name,fd,claimed)
        result[name]={'finite_difference':fd,'analytic':claimed,'absolute_difference':abs(fd-claimed)}
    prior={}
    for p in ['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
              'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json',
              'evidence/2026-09-27/v83_root_review/delivery.json']:
        prior.update(read(ROOT/p)['artifact_sha256'])
    bad=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not bad,bad
    assert sha(ROOT/'data/official/train.parquet')==read(OLD/'supervision_preflight.json')['original_train_sha256']
    out={'status':'passed','source_sha256':sha(__file__),'diagnosis_sha256':sha(DEST/'diagnosis.json'),
         'finite_difference_checks':result,'old_bound_files_rehashed':len(prior),'old_bound_files_changed':bad,
         'selected_fit_input_groups_checked':int(frame.fid.nunique()),'transition_roles_checked':4,
         'raw_train_unchanged':True,'new_classifier_fits':0,'new_calibration_fits':0,
         'scope':'Arithmetic/derivative verification, no new trained model or unseen-data non-regression guarantee.'}
    save(DEST/'verification.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
