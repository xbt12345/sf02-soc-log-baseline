"""Independently check no-fit diagnosis against frozen states and actual fit rows."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, read, save, sha, load_sparse
from v79_execute import rows
from v82_capacity import LAST, DEST as PREV

DEST=ROOT/'artifacts/v83_root_review_20260927'


def main():
    if (DEST/'review_receipt.json').exists():raise FileExistsError('Published review is frozen')
    d=read(DEST/'diagnosis.json');r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy')
    x=load_sparse(LAST/'X');k=np.load(PREV/'primary_kernel.npy',mmap_mode='r')
    selected=np.load(PREV/'selected_rows.npy');active=np.zeros(len(r),bool);active[selected]=True
    c=np.bincount(fid[selected]*3+y[selected],minlength=x.shape[0]*3).reshape(-1,3)
    models={};pred={}
    for a in ['linear','nonlinear']:
        name=read(PREV/('primary_'+a+'_fit.json'))['states'][-1]['name']
        models[a]=joblib.load(PREV/(name+'.joblib'))
        pred[a]=np.load(PREV/(name+'_prediction.npy'))[fid]
    a,b=pred['linear'],pred['nonlinear'];changes=pd.read_parquet(DEST/'changed_rows.parquet')
    ix=changes.row_position.to_numpy();ids=fid[ix]
    # Direct subtraction of parameter vectors, independently of saved margins.
    linear_delta=models['nonlinear']['coef']-models['linear']['coef']
    bias_delta=models['nonlinear']['intercept']-models['linear']['intercept']
    dz=np.asarray(x[ids]@linear_delta)+bias_delta
    assert np.allclose(dz[:,1]-dz[:,2],changes.changed_linear_part_M_minus_S,atol=1e-10,rtol=0)
    for name,m in models.items():
        z=np.asarray(x[ids]@m['coef'])+m['intercept']
        if len(m['kernel_coef']):z+=k[ids]@m['kernel_coef']
        col='old_M_minus_S' if name=='linear' else 'new_M_minus_S'
        assert np.allclose(z[:,1]-z[:,2],changes[col],atol=1e-10,rtol=0)
    changed_correctness=(a==y)!=(b==y)
    assert np.array_equal(np.sort(ix),np.flatnonzero(changed_correctness))
    expected_net={}
    for role,mask in [('fit',~r.fold.isin([0,2]).to_numpy()),('C',r.fold.eq(2).to_numpy()),('H',r.fold.eq(0).to_numpy())]:
        net=int(((a!=y)&mask).sum()-((b!=y)&mask).sum());expected_net[role]=net
        q=changes[changes.role.eq(role)]
        assert int(q.change.eq('repair').sum()-q.change.eq('regression').sum())==net
        assert d['paired_cluster_analysis'][role]['observed_net_repair']==net
    assert expected_net=={'fit':6,'C':30,'H':-14}
    for s in read(DEST/'encoded_support.json'):
        role=(~r.fold.isin([0,2]).to_numpy()) if s['role']=='fit' else r.fold.eq(2 if s['role']=='C' else 0).to_numpy()
        cls=s['class'];present=c[fid,cls]>0;other=(c.sum(1)[fid]-c[fid,cls])>0
        masks={'same_only':present&~other,'mixed':present&other,'opposite_only':~present&other,'unseen':~present&~other}
        take=role&(y==cls)&masks[s['input_support']]
        assert (int(take.sum()),int(((a!=y)&take).sum()),int(((b!=y)&take).sum()))==(s['rows'],s['linear_errors'],s['nonlinear_errors'])
    pairs=read(DEST/'nearest_fit_class_pairs.json');basis=pd.read_csv(DEST/'remaining_S_basis.csv')
    assert len(pairs)==len(basis)==220 and sum(t['S_rows'] for t in pairs)==439
    for p in pairs:
        si,mi=p['S_fit_row'],p['M_fit_row']
        assert active[si] and active[mi] and y[si]==2 and y[mi]==1
        assert c[fid[si],2]>0 and c[fid[si],:2].sum()==0 and b[si]!=2
    gamma=joblib.load(PREV/'primary_basis.joblib').gamma;max_delta=0
    for row in basis.itertuples():
        delta=x[int(row.feature_id)].astype(float)-x[int(row.nearest_M_feature_id)].astype(float)
        distance=float(delta.multiply(delta).sum())
        max_delta=max(max_delta,abs(distance-row.input_squared_distance))
        exact=2*(-np.expm1(-gamma*distance))
        approximate=float(np.square(np.asarray(k[row.feature_id],float)-np.asarray(k[row.nearest_M_feature_id],float)).sum())
        assert np.isclose(approximate/exact,row.retained_RBF_pair_distance_ratio,atol=3e-6,rtol=0)
    assert max_delta<1e-6
    m=models['nonlinear'];z=np.asarray(x[fid[selected]]@m['coef'])+m['intercept']+k[fid[selected]]@m['kernel_coef']
    # Evaluate actual rows directly rather than the aggregated count formula.
    loss=float((np.logaddexp(0,z).sum(1)-z[np.arange(len(selected)),y[selected]]).mean())
    penalty=float(1e-6/2*(np.square(m['coef']).sum()+np.square(m['kernel_coef']).sum()))
    objective=loss+penalty
    assert abs(objective-d['objective_trajectory'][-1]['objective'])<1e-10
    assert abs(objective-read(PREV/'primary_nonlinear_fit.json')['objective'])<1e-10
    assert d['source_sha256']==sha(ROOT/'training/v83_root_diagnosis.py')
    receipts=['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
              'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json']
    prior={}
    for p in receipts:prior.update(read(ROOT/p)['artifact_sha256'])
    failed=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not failed,failed
    original_ok=sha(ROOT/'data/official/train.parquet')==read(PREV/'supervision_preflight.json')['original_train_sha256']
    assert original_ok
    out={'status':'passed','new_classifier_fits':0,'target_answers_read':False,'transition_rows_checked':len(changes),
         'paired_net_repairs':expected_net,'encoded_support_cells_checked':36,'actual_fit_pairs_checked':220,
         'basis_distance_max_recompute_delta':max_delta,'direct_original_row_objective':objective,
         'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':failed,'original_train_unchanged':original_ok,
         'scope':'Numerical diagnosis verification only; no model quality acceptance or new blind validation.',
         'source_sha256':sha(__file__)}
    save(DEST/'verification.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
