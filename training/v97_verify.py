"""Replay all 3 fits' outputs, selection, historic gates and immutable receipts."""
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from run_v75 import ROOT, OUT, read, save, sha, load_sparse
from v89_common import LAST, data, raw_counts
from v85_protection import cm_from_counts
from v81_training_contract import compare
from v92_train import Branch, csr
from v97_prepare import DEST


def main():
    assert not (DEST/'verification.json').exists()
    started=time.monotonic();reg=read(DEST/'registration.json')
    assert reg['source_sha256']==sha(ROOT/'training/v97_prepare.py')
    assert reg['counts_sha256']==sha(DEST/'AB_train_counts.npz')
    for p,h in reg['input_sha256'].items():assert sha(ROOT/p)==h,p
    r,y,fid,zold,old,sel,fit=data()
    roles=pd.read_parquet(ROOT/'artifacts/v95_cross_component_training_20260928/full_format_roles.parquet',columns=['role']).role.to_numpy()
    assert len(y)==2056871 and len(fid)==len(roles)==len(y)
    with np.load(DEST/'AB_train_counts.npz') as z:
        A=z['A'];B=z['B']
    assert np.array_equal(A,raw_counts(fid,y,roles=='A',len(A)))
    assert np.array_equal(B,raw_counts(fid,y,roles=='B',len(B)))
    assert int(A.sum()+B.sum())==753709 and (A+B).sum(0).tolist()==[709247,36888,7574]
    x=load_sparse(LAST/'X');teacher_fit=read(DEST/'teacher_fit.json');model=joblib.load(DEST/'teacher.joblib')
    assert teacher_fit['source_sha256']==sha(ROOT/'training/v97_teacher.py') and teacher_fit['converged']
    assert sha(DEST/'teacher.joblib')==teacher_fit['model_sha256']
    calc=np.asarray(x@model['coef'])+model['intercept']
    saved=np.load(DEST/'teacher_scores.npy');teacher=np.load(DEST/'teacher_prediction.npy')
    assert np.allclose(calc,saved,rtol=0,atol=1e-10) and np.array_equal(calc.argmax(1),teacher)
    ids=np.load(ROOT/'artifacts/v92_evidence_training_20260928/ASA_input_ids.npy')
    asa=sparse.load_npz(ROOT/'artifacts/v92_evidence_training_20260928/ASA_R0.npz')
    assert (A[ids]+B[ids]).sum(0).tolist()==[0,25940,5349]
    abasa=A[ids]+B[ids]
    teacher_asa_errors=int(abasa.sum()-abasa[np.arange(len(ids)),teacher[ids]].sum())
    assert teacher_asa_errors==398
    init=set();replayed={}
    for arm in ['ERM','MAG']:
        info=read(DEST/f'{arm}_fit.json')
        assert info['source_sha256']==sha(ROOT/'training/v97_branches.py')
        assert info['teacher_model_sha256']==teacher_fit['model_sha256']
        assert info['steps_executed']==200 and info['actual_classifier_fits']==1
        assert not info['V_labels_loaded'] and not info['inner_C_H_labels_loaded']
        assert info['auxiliary_behavior_groups_used']==0 and info['new_ICMP_features'] is False
        assert sha(DEST/f'{arm}_model.pt')==info['model_sha256']
        assert sha(DEST/f'{arm}_prediction.npy')==info['prediction_sha256']
        init.add(info['initial_state_sha256'])
        state=torch.load(DEST/f'{arm}_model.pt',map_location='cpu',weights_only=True)
        branch=Branch(asa.shape[1]);branch.load_state_dict(state['state_dict']);branch.eval()
        expected=teacher.copy();scores=np.empty((len(ids),3),float)
        with torch.no_grad():
            for start in range(0,len(ids),2048):
                end=min(start+2048,len(ids))
                scores[start:end]=saved[ids[start:end]]+branch(csr(asa[start:end],'cpu')).double().numpy()
        expected[ids]=scores.argmax(1)
        assert np.array_equal(expected,np.load(DEST/f'{arm}_prediction.npy'))
        replayed[arm]=len(ids)
        logs=read(DEST/f'{arm}_progress.json')
        assert logs[-1]['step']==200 and len(logs)==9
        assert info['final_train_metrics']==logs[-1]
        assert all(not any('V_' in str(k) or 'inner' in str(k) or 'C_H' in str(k) for k in item.keys()) for item in logs)
    assert len(init)==1
    selection=read(DEST/'selection.json');diagnosis=read(DEST/'diagnosis.json')
    assert selection['source_sha256']==sha(ROOT/'training/v97_select.py')
    assert diagnosis['source_sha256']==sha(ROOT/'training/v97_diagnose.py')
    assert diagnosis['selection_sha256']==sha(DEST/'selection.json')
    assert selection['selected_arm'] is None and diagnosis['selected_arm'] is None
    assert selection['actual_classifier_fits']==3 and selection['actual_calibration_fits']==0
    assert not selection['inner_C_H_labels_loaded'] and not diagnosis['model_promoted']
    V=np.flatnonzero((roles=='V')&r.route.eq('asa').to_numpy())
    assert len(V)==7482
    def confusion(pred):
        cm=np.zeros((3,3),np.int64);np.add.at(cm,(y[V],pred[fid[V]]),1);return cm
    basecm=confusion(teacher)
    assert basecm.tolist()==selection['teacher_and_arms']['teacher']['confusion']
    for arm in ['ERM','MAG']:
        p=np.load(DEST/f'{arm}_prediction.npy');cm=confusion(p)
        assert cm.tolist()==selection['teacher_and_arms'][arm]['confusion']
        assert not compare(basecm,cm,True)['eligible']
        assert selection['teacher_and_arms'][arm]['negative_flips_vs_teacher']==4
    oldrows=old[fid]
    diagnosis_roles=read(DEST/'postselection_role_metrics.json')
    for role,mask in [('inner',r.fold.eq(1).to_numpy()),('C',r.fold.eq(2).to_numpy()),('H',r.fold.eq(0).to_numpy()),('old_full_fit',fit)]:
        oldcounts=raw_counts(fid,y,mask,len(old));cmold=cm_from_counts(oldcounts,old)
        assert int(oldcounts.sum()-np.trace(cmold))==diagnosis_roles[role]['historic_v85']['errors']
        for arm in ['ERM','MAG']:
            p=np.load(DEST/f'{arm}_prediction.npy');full=np.where(r.route.eq('asa').to_numpy(),p[fid],oldrows)
            rows=mask.nonzero()[0];cm=np.zeros((3,3),np.int64);np.add.at(cm,(y[rows],full[rows]),1)
            name=f'{arm}_ASA_plus_old_other';table=diagnosis_roles[role][name]
            assert cm.tolist()==table['confusion']
            assert table['guard_vs_historic']==compare(cmold,cm,True)
            assert table['negative_flips_vs_historic']==int(((oldrows[rows]==y[rows])&(full[rows]!=y[rows])).sum())
    changes=read(DEST/'MAG_V_new_M_errors.json');assert len(changes)==4
    assert len({a['component'] for a in changes})==len({a['R0'] for a in changes})==1
    assert read(DEST/'type3_code13_readout.json')['slice']['2']['MAG_correct']==0
    support=read(DEST/'support_audit.json')
    assert support['source_sha256']==sha(ROOT/'training/v97_support_audit.py')
    Vframe=pd.read_parquet(DEST/'V_ASA_row_support.parquet')
    assert len(Vframe)==len(V) and np.array_equal(np.sort(Vframe.row_position),V)
    vf=Vframe[~Vframe.MAG_correct]
    assert len(vf)==737 and vf.label_index.value_counts().to_dict()=={2:733,1:4}
    assert support['MAG_M_errors']['rows']==4 and support['MAG_S_errors']['rows']==733
    assert support['MAG_S_errors']['same_label_AB_behavior_supported_rows']==12
    assert support['type3_code13_685_S']['rows']==685
    prev=read(ROOT/'evidence/2026-09-28/v96_root_cause_audit/delivery.json')
    prior={}
    receipts=prev['verification']['receipt_sha256']
    receipt_paths=list(receipts)+['evidence/2026-09-28/v95_four_arm_training/delivery.json',
        'evidence/2026-09-28/v96_root_cause_audit/delivery.json']
    for path in dict.fromkeys(receipt_paths):
        if path in receipts:assert sha(ROOT/path)==receipts[path]
        for name,h in read(ROOT/path)['artifact_sha256'].items():
            assert name not in prior or prior[name]==h
            prior[name]=h
    dirty=[name for name,h in prior.items() if sha(ROOT/name)!=h]
    assert not dirty,dirty
    result={'status':'passed','source_sha256':sha(__file__),'official_label_rows_verified':len(y),
        'AB_row_counts_recomputed':True,'AB_ASA_teacher_errors_recomputed':teacher_asa_errors,
        'teacher_scores_replayed':len(teacher),
        'branch_predictions_replayed':replayed,'equal_initial_state':True,
        'V_selection_recomputed':len(V),'historic_role_metrics_recomputed':4,
        'new_M_errors_one_component_one_R0':True,'V_error_support_audit_checked':len(Vframe),
        'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':dirty,
        'prior_delivery_sha256':{p:sha(ROOT/p) for p in dict.fromkeys(receipt_paths)},
        'actual_classifier_fits':3,'actual_calibration_fits':0,'quality_acceptance':False,
        'model_promoted':False,'elapsed_seconds':time.monotonic()-started,
        'scope':'Independent fixed-input teacher and both terminal branch inference, row/class safety and historic hashes; no platform, blind, external transfer or full-task replay.'}
    save(DEST/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='prior_delivery_sha256'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
