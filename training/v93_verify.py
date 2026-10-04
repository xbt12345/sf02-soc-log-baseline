"""Independent row-level reconciliation, exact path replay and immutable history check."""
import json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from v89_common import ROOT,read,save,sha,data
from v93_regression_audit import DEST,PREV

def main():
    assert not (DEST/'verification.json').exists()
    reg=read(DEST/'registration.json');assert sha(ROOT/'training/v93_regression_audit.py')==reg['source_sha256']
    for p,h in reg['input_sha256'].items():assert sha(ROOT/p)==h,p
    assert read(DEST/'trace_controls.json')['source_sha256']==sha(ROOT/'training/v93_trace_and_controls.py')
    r,y,fid,z0,old,_,fit=data();labels=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();assert np.array_equal(y,labels)
    new=np.load(PREV/'U0R_prediction.npy');inner=r.fold.eq(1).to_numpy();m=np.flatnonzero(inner&(y==1)&(old[fid]==1)&(new[fid]!=1));s=np.flatnonzero(inner&(y==2)&(old[fid]!=2)&(new[fid]==2))
    t=pd.read_parquet(DEST/'inner_500M_58S_trace.parquet');assert np.array_equal(np.sort(t.row_position),np.sort(np.r_[m,s])) and len(m)==500 and len(s)==58
    assert t.R0_id.nunique()==94 and t[t.kind=='regression'].component.nunique()==60
    ids=np.load(PREV/'ASA_input_ids.npy');z=np.asarray(z0[ids]);scores=np.load(PREV/'U0R_ASA_scores.npy');delta=scores-z;op=z.argmax(1)
    x=sparse.load_npz(PREV/'ASA_R0.npz').astype(float);saved=torch.load(PREV/'U0R_model.pt',map_location='cpu',weights_only=True);st=saved['state_dict']
    tx=torch.sparse_csr_tensor(torch.from_numpy(x.indptr.astype(np.int64)),torch.from_numpy(x.indices.astype(np.int64)),torch.from_numpy(x.data),size=x.shape,dtype=torch.float64)
    h=torch.nn.functional.gelu(torch.sparse.mm(tx,st['first.weight'].T),approximate='tanh');h=torch.nn.functional.gelu(h@st['second.weight'].T+st['second.bias'],approximate='tanh')
    replay=z+(h@st['out.weight'].T+st['out.bias']).numpy();assert np.allclose(replay,scores,atol=1e-10,rtol=0)
    # Exhaustive actual argmax on every open interval. Does not reuse event-count recurrence.
    curves=pd.read_csv(DEST/'scalar_path_frontier.csv');summary=read(DEST/'scalar_path_diagnosis.json');checks=0
    for role,mask in [('fit',fit),('inner',inner),('C',r.fold.eq(2).to_numpy()),('H',r.fold.eq(0).to_numpy())]:
        rr=np.flatnonzero(mask&r.route.eq('asa').to_numpy());a=np.searchsorted(ids,fid[rr]);yy=y[rr];oldrow=op[a]
        table=curves[curves.role==role].sort_values('alpha_crossing');cross=table.alpha_crossing.to_numpy();alphas=np.r_[(cross[:-1]+cross[1:])/2,1.]
        baseline=int((oldrow!=yy).sum())
        for it,alpha in enumerate(alphas):
            pp=(z+alpha*delta).argmax(1)[a];row=table.iloc[it]
            assert int((pp!=yy).sum())==row.errors
            for c in [0,1,2]:
                assert int(((oldrow!=yy)&(pp==yy)&(yy==c)).sum())==row['PF'+str(c)]
                assert int(((oldrow==yy)&(pp!=yy)&(yy==c)).sum())==row['NF'+str(c)]
            checks+=1
        assert int(table.errors.min())==summary[role]['min_total_errors_on_scalar_path']
    assert summary['inner']['S_repairs_at_zero_M_regression']==0 and summary['inner']['min_M_regressions_preserving_all_endpoint_S_repairs']==500
    groups=read(DEST/'all_94_input_reconstruction.json');assert len(groups)==94 and sum(len(g['rows']) for g in groups)==558
    # Independently recompute global mixed labels for every target R0.
    population=pd.DataFrame({'fid':fid,'y':labels}).groupby('fid').y.nunique()
    assert (population.loc[t.R0_id]==1).all() and all(not g['same_X_other_label'] and not g['missing_known_facts_rows'] for g in groups)
    receipt_paths=list(read(ROOT/'artifacts/v91_first_principles_20260928/verification.json')['receipt_sha256'])+['evidence/2026-09-28/v91_first_principles/delivery.json','evidence/2026-09-28/v92_evidence_training/delivery.json']
    prior={};receipts={}
    for p in receipt_paths:
        receipts[p]=sha(ROOT/p)
        for f,hsh in read(ROOT/p)['artifact_sha256'].items():assert f not in prior or prior[f]==hsh;prior[f]=hsh
    changed=[p for p,hsh in prior.items() if sha(ROOT/p)!=hsh];assert not changed,changed
    result={'status':'passed','new_classifier_fits':0,'new_calibration_fits':0,'original_labels_checked':len(labels),'inner_changed_rows':len(t),
        'inner_M_regression_inputs':t[t.kind=='regression'].R0_id.nunique(),'inner_M_regression_components':60,'torch_score_replay_max_error':float(np.abs(replay-scores).max()),
        'independent_scalar_interval_role_checks':checks,'all_official_conflicting_input_rows_in_target':0,'known_fact_omission_rows_in_target':0,
        'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,'receipt_sha256':receipts,'quality_acceptance':False,'selected':None,
        'source_sha256':sha(__file__),'scope':'Frozen-model replay, actual original-row flips and exhaustive interpolation diagnosis on observed development; not new model quality or generalization.'}
    save(DEST/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
