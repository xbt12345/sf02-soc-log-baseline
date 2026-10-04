"""Pre-register the v9.7 matched A+B trial; export TRAIN labels only to fitters."""
import json
import time
import numpy as np
import pandas as pd
from run_v75 import ROOT, OUT, read, save, sha
from v89_common import LAST, raw_counts
from v95_prepare import DEST as V95

DEST=ROOT/'artifacts/v97_matched_teacher_20260928'
SEED=9701


def main():
    assert not DEST.exists()
    delivered=read(ROOT/'evidence/2026-09-28/v96_root_cause_audit/delivery.json')
    assert delivered['status']=='protocol_and_experiment_root_audit_completed_no_new_fit'
    assert delivered['verification']['prior_bound_files_rehashed']==1031
    assert not delivered['verification']['prior_bound_files_changed']
    project=read(ROOT/'mcp_readonly/catalog.json')['project']
    assert project['authoritative_direction_id']=='v96-direction-review'
    r=pd.read_parquet(OUT/'rows.parquet',columns=['row_position','component','body_group','source_symbol','fold','route','label_index'])
    fid=np.load(LAST/'row_feature_id.npy',mmap_mode='r')
    roles=pd.read_parquet(V95/'full_format_roles.parquet',columns=['role']).role.to_numpy()
    assert len(r)==len(fid)==len(roles)==2056871
    assert np.all(roles[~r.fold.isin([0,1,2]).to_numpy()]!='L')
    assert np.all(roles[r.fold.isin([0,1,2]).to_numpy()]=='L')
    for group in ['component','body_group','source_symbol']:
        v=r[group].to_numpy();seen=v>=0
        assert pd.DataFrame({'group':v[seen],'role':roles[seen]}).groupby('group').role.nunique().max()==1
    n=int(fid.max())+1;y=r.label_index.to_numpy()
    assert np.isin(y,[0,1,2]).all()
    A=raw_counts(fid,y,roles=='A',n).astype(np.int32)
    B=raw_counts(fid,y,roles=='B',n).astype(np.int32)
    assert int(A.sum())==547330 and int(B.sum())==206379
    assert (A.sum(0)+B.sum(0)).tolist()==[709247,36888,7574]
    V=raw_counts(fid,y,roles=='V',n)
    assert int(V.sum())==168615 and V.sum(0).tolist()==[158597,8672,1346]
    DEST.mkdir()
    np.savez_compressed(DEST/'AB_train_counts.npz',A=A,B=B)
    asa_ids=np.load(ROOT/'artifacts/v92_evidence_training_20260928/ASA_input_ids.npy')
    assert len(asa_ids)==np.unique(asa_ids).size and np.all(asa_ids[:-1]<asa_ids[1:])
    assert (A[asa_ids]+B[asa_ids]).sum(0).tolist()==[0,25940,5349]
    # V labels and historic locked labels are never persisted into the training directory.
    sources=[ROOT/'data/official/train.parquet',OUT/'rows.parquet',LAST/'row_feature_id.npy',
        LAST/'X.data',LAST/'X.indices',LAST/'X.indptr',V95/'full_format_roles.parquet',
        ROOT/'artifacts/v92_evidence_training_20260928/ASA_R0.npz',
        ROOT/'artifacts/v92_evidence_training_20260928/ASA_input_ids.npy',
        ROOT/'docs/V96_ROOT_CAUSE_CORRECTION_AND_TRAINING_PLAN.md']
    reg={'version':'v97-matched-teacher-1','registered_unix':time.time(),'status':'registered_before_fits',
        'seed':SEED,'roles':'Fixed v95 A/B/V component separation; only A+B source labels exported to fitters',
        'teacher':'One A+B exact original-frequency OVR BCE, alpha1e-6, L-BFGS-B maxiter1000',
        'branch':'Same full R0 ASA gate and 64/16 residual architecture; Adam lr0.003, 200 steps, same seed',
        'arms':['ERM','MAG'],'common_loss':'A+B original-frequency ASA cross-entropy plus shared .01 teacher-correct soft margin protection',
        'single_method_difference':'MAG adds .01 raw-frequency mean squared output residual; ERM does not',
        'auxiliary_group_loss':None,'new_ICMP_features':False,'all_nonASA_outputs':'frozen historical v85 reference in postfit comparison',
        'checkpoints':'200 is the only V selection endpoint; intermediate snapshots/training logs exclude V and all locked labels',
        'selection':'Require V-ASA >0 repairs over matched teacher, zero new errors, every supported M/S recall/precision/F1 nondecreasing. If both qualify, choose fewer V-ASA errors then arm name; if neither, no selection.',
        'method_comparison':'Compare both 200-step arms on same V-ASA rows, class outcomes and component bootstrap; bootstrap 95% lower bound >0 required to claim robust MAG gain.',
        'continuation':'After selection receipt, independent inner/C/H+full-fit historical safety audit. No expansion, final fit, promotion or full replay unless original safeguards pass.',
        'max_new_classifier_fits':3,'max_new_calibration_fits':0,'external_data':False,'original_labels_changed':0,
        'A_B_rows':[int(A.sum()),int(B.sum())],'A_B_class_rows':(A.sum(0)+B.sum(0)).astype(int).tolist(),
        'A_B_ASA_class_rows':(A[asa_ids].sum(0)+B[asa_ids].sum(0)).astype(int).tolist(),
        'V_role_rows_without_exporting_labels':int((roles=='V').sum()),
        'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        'counts_sha256':sha(DEST/'AB_train_counts.npz'),'source_sha256':sha(__file__)}
    save(DEST/'registration.json',reg)
    print(json.dumps({'stage':'registered','A_B_rows':reg['A_B_rows'],'A_B_class_rows':reg['A_B_class_rows'],
        'A_B_ASA_class_rows':reg['A_B_ASA_class_rows'],'fit_budget':3},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
