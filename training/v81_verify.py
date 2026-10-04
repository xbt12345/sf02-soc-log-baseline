"""Independent saved-model replay plus actual-exposure support crosscheck."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, read, save, sha, load_sparse
from v79_execute import rows, weights
from v81_support_diagnosis import summarize


def main():
    dest=ROOT/'artifacts/v81_diagnosis_20260927';trajectory=dest/'trajectory';last=ROOT/'artifacts/v79_execution_20260927'
    if (dest/'verification.json').exists():raise FileExistsError('verified result exists')
    r=rows();y=r.label_index.to_numpy();fid=np.load(last/'row_feature_id.npy');x=load_sparse(last/'X')
    half=(r.component.to_numpy().astype(np.int64)*2654435761)%2
    fit=~r.fold.isin([0,2]).to_numpy();cal=r.fold.eq(2).to_numpy()
    masks={'fit':fit,'C':cal,'C_half0':cal&(half==0),'C_half1':cal&(half==1)}
    checked=0
    for t in read(trajectory/'trace.json'):
        model=joblib.load(trajectory/(t['name']+'.joblib'))
        p=(x@model['coef']+model['intercept']).argmax(1)[fid]
        np.testing.assert_array_equal(p,np.load(trajectory/(t['name']+'_pred.npy')))
        for k,m in masks.items():
            reported=t['fit'] if k=='fit' else t['C_scores'][k]
            assert confusion_matrix(y[m],p[m],labels=[0,1,2]).tolist()==reported['cm']
        checked+=len(r)
    assert read(trajectory/'complete.json')['source_sha256']==sha(ROOT/'training/v81_optimizer_trajectory.py')
    # The original support audit used all available fit-role rows. Recheck the actual B exposure.
    selected=fit.copy();rng=np.random.default_rng(7901)
    for route in sorted(r.route.unique()):
        pos=np.flatnonzero(fit&r.route.eq(route).to_numpy()&(y==0))
        if len(pos)>2048:selected[pos]=False;selected[rng.choice(pos,2048,replace=False)]=True
    _,_,e=weights(r,fid,x.shape[0],fit,'B')
    import hashlib
    assert hashlib.sha256(np.flatnonzero(selected).astype('<i4').tobytes()).hexdigest()==e['selected_rows_sha256']
    facts=pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts
    canonical=np.array([json.dumps(json.loads(s),sort_keys=True,separators=(',',':')) for s in facts],dtype=object)
    ids,unique=pd.factorize(canonical,sort=False);ids=ids[r.projection_id.to_numpy()]
    pred=np.load(last/'P1_B_ovr_feature_prediction.npy')[fid];audit={}
    for role,mask in [('C',cal),('H',r.fold.eq(0).to_numpy())]:
        take=mask&r.route.eq('asa').to_numpy()
        audit[role]={
            'facts_only':summarize(ids,y,selected,take,pred,len(unique)),
            'complete_encoding':summarize(fid,y,selected,take,pred,x.shape[0])}
    original=read(dest/'support_diagnosis.json')
    same=all(audit[role][kind]['support_buckets']==original['roles'][role]['asa'][kind]['support_buckets'] for role in audit for kind in audit[role])
    save(dest/'actual_exposure_support.json',{'actual_selected_rows':int(selected.sum()),'available_fit_role_rows':int(fit.sum()),'ASA_C_H_buckets_unchanged':same,'roles':audit,
         'scope':'Actual model B exposure, not only available fit-role label support.'})
    support=original['roles']['C']['asa']
    assert support['top_error_components'][0]['errors']==3316
    sums=pd.read_parquet(dest/'fit_error_rows.parquet');assert len(sums)==491
    for name in ['v81_support_diagnosis.py','v81_vpc_evidence.py','v81_flow_preflight.py']:
        payload={'v81_support_diagnosis.py':'support_diagnosis.json','v81_vpc_evidence.py':'vpc_interval_audit.json','v81_flow_preflight.py':'flow_preflight.json'}[name]
        assert read(dest/payload)['source_sha256']==sha(ROOT/'training'/name)
    result={'status':'passed','source_sha256':sha(__file__),'model_states_reloaded':12,'rows_predictions_compared':checked,
        'confusion_matrices_compared':48,'trajectory_classifier_optimization_runs':1,'actual_exposure_crosschecked':True,
        'ASA_support_buckets_unchanged':same,'new_model_quality_passed':False,'scope':'Implementation/provenance verification; no test of unseen real environments.'}
    save(dest/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
