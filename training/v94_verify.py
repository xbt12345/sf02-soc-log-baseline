"""Independent torch replay, label reconciliation, support partition and receipts."""
import hashlib,json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from v89_common import ROOT,read,save,sha,data
from v94_mechanism_audit import DEST
from v93_regression_audit import PREV

def main():
    assert not (DEST/'verification.json').exists()
    reg=read(DEST/'registration.json');assert sha(ROOT/'training/v94_mechanism_audit.py')==reg['source_sha256']
    for p,h in reg['input_sha256'].items():assert sha(ROOT/p)==h,p
    r,y,fid,z,old,_,fit=data();original=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();assert np.array_equal(original,y)
    ids=np.load(PREV/'ASA_input_ids.npy');xx=sparse.load_npz(PREV/'ASA_R0.npz').astype(float)
    tx=torch.sparse_csr_tensor(torch.from_numpy(xx.indptr.astype(np.int64)),torch.from_numpy(xx.indices.astype(np.int64)),torch.from_numpy(xx.data),size=xx.shape,dtype=torch.float64)
    state={n:{k:t.double() for k,t in torch.load(PREV/(n+'_model.pt'),map_location='cpu',weights_only=True)['state_dict'].items()} for n in ['U0','U0R']}
    roles={'fit':fit,'inner':r.fold.eq(1).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()};reports=read(DEST/'encoder_head_factorial.json')
    for rec in reports:
        se=state[rec['encoder']];sh=state[rec['head']]
        h=torch.nn.functional.gelu(torch.sparse.mm(tx,se['first.weight'].T),approximate='tanh');h=torch.nn.functional.gelu(h@se['second.weight'].T+se['second.bias'],approximate='tanh')
        sc=np.asarray(z[ids])+(h@sh['out.weight'].T+sh['out.bias']).numpy();p=old.copy();p[ids]=sc.argmax(1);saved=np.load(DEST/(rec['name']+'_prediction.npy'));assert np.array_equal(p,saved)
        for role,mask in roles.items():
            a=old[fid[mask]];b=p[fid[mask]];yy=original[mask];expected=rec['roles'][role]
            assert int((b!=yy).sum())==expected['errors']
            for cl in range(3):
                assert int(((a==cl)&(yy==cl)&(b!=cl)).sum())==expected['NF'][cl]
                assert int(((a!=cl)&(yy==cl)&(b==cl)).sum())==expected['PF'][cl]
    support=read(DEST/'episode_support.json');assert support['source_sha256']==sha(ROOT/'training/v94_support_audit.py')
    a=pd.read_parquet(DEST/'prospective_ASA_roles.parquet');assert len(a)==38771 and len(a.row_position.unique())==38771 and fit[a.row_position].all()
    assert np.array_equal(original[a.row_position],a.label_index)
    assert a.groupby('component').proposed_role.nunique().max()==1
    for c,role in a[['component','proposed_role']].drop_duplicates().itertuples(index=False,name=None):
        u=int(hashlib.sha256(('v94:9301:'+str(c)).encode()).hexdigest()[:16],16)%10;assert role==('A' if u<6 else ('B' if u<8 else 'V'))
    actual=pd.read_csv(DEST/'behavior_support.csv');good3=0;goodall=0
    for b in actual.itertuples():
        subset=a[a.behavior==b.behavior];nM=subset[subset.label_index==1].component.nunique();nS=subset[subset.label_index==2].component.nunique()
        assert nM==b.M_components and nS==b.S_components
        good3+=int(nM>=3 and nS>=3)
        goodall+=int(all(subset[(subset.label_index==cl)&(subset.proposed_role==role)].component.nunique()>0 for cl in [1,2] for role in ['A','B','V']))
    assert good3==5 and goodall==2 and support['groups_both_classes_observed_in_A_B_V']==2
    pop=read(DEST/'component_population.json')
    for entry in pop:
        rr=r[roles[entry['role']]&r.route.eq('asa').to_numpy()];ct=rr.groupby('component').label_index.nunique()
        assert len(rr)==entry['ASA_rows'] and int((ct==1).sum())==entry['pure_components']
    paths=list(read(ROOT/'artifacts/v93_regression_audit_20260928/verification.json')['receipt_sha256'])+['evidence/2026-09-28/v93_regression_audit/delivery.json'];prior={};receipt_hash={}
    for p in paths:
        receipt_hash[p]=sha(ROOT/p)
        for f,h in read(ROOT/p)['artifact_sha256'].items():assert f not in prior or prior[f]==h;prior[f]=h
    changes=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not changes,changes
    result={'status':'passed','original_labels_verified':len(y),'frozen_combinations_replayed':4,'role_metrics_verified':16,'ASA_support_rows':len(a),'behavior_groups_verified':len(actual),
        'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changes,'receipt_sha256':receipt_hash,'new_classifier_fits':0,'new_calibration_fits':0,
        'quality_acceptance':False,'source_sha256':sha(__file__),'scope':'Frozen mechanism and necessary episode support audit. Does not establish causal shortcut use, balanced independent domains, or success of proposed training.'}
    save(DEST/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='receipt_sha256'}),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
