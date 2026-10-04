"""Separate observed training errors, input conflicts and frozen surface sensitivity."""
import hashlib
import json
import re
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from v89_common import ROOT, OUT, data, read, save, sha
from v75_views import view, byte_matrix, BYTE_FEATURES
from v75_corrective import stable
from v96_behavior_contract import observed_behavior
from v92_train import Branch, csr
from v98_frozen_audit import DEST, V97, V92


def main():
    assert not (DEST/'fit_and_surface_audit.json').exists()
    r,y,fid,_,old,_,_=data()
    roles=pd.read_parquet(ROOT/'artifacts/v95_cross_component_training_20260928/full_format_roles.parquet',columns=['role']).role.to_numpy()
    ab=r.route.eq('asa').to_numpy()&np.isin(roles,['A','B'])
    facts=[json.loads(x) for x in pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts]
    keys=np.array([observed_behavior(facts[p])[0] for p in r.projection_id],object)
    group=pd.read_csv(DEST/'supported_behavior_fit_and_transfer.csv')
    target=ab&(y==2)&np.isin(keys,group.behavior.unique())
    pred=np.load(V97/'MAG_prediction.npy');bad=np.flatnonzero(target&(pred[fid]!=y))
    assert target.sum()==63 and len(bad)==6
    teacher=np.load(V97/'teacher_scores.npy',mmap_mode='r')
    ids=np.load(V92/'ASA_input_ids.npy');x=sparse.load_npz(V92/'ASA_R0.npz')
    lookup=np.full(int(fid.max())+1,-1,np.int32);lookup[ids]=np.arange(len(ids))
    state=torch.load(V97/'MAG_model.pt',map_location='cpu',weights_only=True)
    net=Branch(x.shape[1]);net.load_state_dict(state['state_dict']);net.eval()
    lin=joblib.load(V97/'teacher.joblib')
    def score(xx):
        t=np.asarray(xx@lin['coef'])+lin['intercept']
        with torch.no_grad(): z=t+net(csr(xx,'cpu')).double().numpy()
        return t,z
    train_records=[];representatives=set()
    for f in np.unique(fid[bad]):
        ix=np.flatnonzero(target&(fid==f));i=int(ix[0]);representatives.add(i)
        exact_ab=ab&(fid==f)
        exact_all=r.route.eq('asa').to_numpy()&(fid==f)
        t,z=score(x[lookup[f]])
        train_records.append({'R0':int(f),'representative':i,'rows':len(ix),'component':int(r.component.iloc[i]),
            'facts':facts[r.projection_id.iloc[i]],'AB_exact_R0_class_counts':np.bincount(y[exact_ab],minlength=3).tolist(),
            'all_roles_exact_R0_class_counts':np.bincount(y[exact_all],minlength=3).tolist(),
            'teacher_MS_margin':float(t[0,2]-t[0,1]),'MAG_MS_margin':float(z[0,2]-z[0,1])})
    save(DEST/'six_supported_training_errors.json',train_records)
    for rec in read(DEST/'raw_normalized_evidence.json'):representatives.add(rec['row_position'])
    raw={};off=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=32768,columns=['message_sanitized'],use_threads=False):
        needed=[i for i in representatives if off<=i<off+batch.num_rows]
        if needed:
            ss=batch.column(0).to_pylist()
            for i in needed:raw[i]=ss[i-off]
        off+=batch.num_rows
    controls=[]
    # A narrow diagnostic: only literal leftovers directly adjoining an identity placeholder.
    # No real numeric port is removed; facts/metadata retain exactly their original coordinates.
    pat=re.compile(r'(?:CRED|HOST|USER|ORG)-\s*(?=<IDENTITY>)')
    for i in sorted(raw):
        text=stable(view(raw[i])[0]);base=x[lookup[fid[i]]]
        d=base[:,:BYTE_FEATURES]-byte_matrix([text]);assert not d.nnz or abs(d.data).max()<1e-7
        derived=pat.sub('',text)
        if derived==text:continue
        changed=sparse.hstack([byte_matrix([derived]),base[:,BYTE_FEATURES:]],format='csr')
        bt,bz=score(base);ct,cz=score(changed)
        controls.append({'row_position':i,'role':str(roles[i]),'label':int(y[i]),'R0':int(fid[i]),
            'raw_sha256':hashlib.sha256(raw[i].encode()).hexdigest(),
            'original_view':text,'derived_view':derived,'facts_and_metadata_unchanged':True,
            'teacher_original_MS_margin':float(bt[0,2]-bt[0,1]),'teacher_control_MS_margin':float(ct[0,2]-ct[0,1]),
            'MAG_original_MS_margin':float(bz[0,2]-bz[0,1]),'MAG_control_MS_margin':float(cz[0,2]-cz[0,1]),
            'MAG_original_pred':int(bz.argmax(1)[0]),'MAG_control_pred':int(cz.argmax(1)[0])})
    save(DEST/'frozen_placeholder_fragment_controls.json',controls)
    history=read(V97/'MAG_progress.json')
    result={'status':'fit_failure_and_surface_sensitivity_audited_no_fit','source_sha256':sha(__file__),
        'training_supported_S_rows':63,'training_supported_S_errors':6,
        'error_distinct_R0':len(train_records),
        'training_errors_with_AB_exact_R0_other_label':sum(g['rows'] for g in train_records if sum(g['AB_exact_R0_class_counts'][:2])>0),
        'training_errors_with_any_role_exact_R0_other_label':sum(g['rows'] for g in train_records if sum(g['all_roles_exact_R0_class_counts'][:2])>0),
        'MAG_train_errors_step175':history[-2]['ASA_A_B_errors_after_update'],
        'MAG_train_errors_step200':history[-1]['ASA_A_B_errors_after_update'],
        'surface_control_representatives':len(controls),
        'surface_control_prediction_flips':sum(g['MAG_original_pred']!=g['MAG_control_pred'] for g in controls),
        'classifier_fits':0,'calibration_fits':0,
        'scope':'Targeted posthoc frozen controls. No statement that all identity/clock processing is correct, that all input is lossless, or that removing tokens improves unseen classification.'}
    save(DEST/'fit_and_surface_audit.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
