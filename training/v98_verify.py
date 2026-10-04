"""Verify source provenance, frozen controls, row outcomes and historical integrity."""
import hashlib
import json
import re
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from v89_common import ROOT, OUT, data, read, save, sha
from v75_views import view, byte_matrix, BYTE_FEATURES
from v75_corrective import stable
from v98_frozen_audit import DEST, V97, V92
from scipy import sparse


def main():
    started=time.monotonic()
    assert not (ROOT/'evidence/2026-09-28/v98_boundary_audit/delivery.json').exists()
    r,y,fid,_,old,_,_=data();isa=r.route.eq('asa').to_numpy()
    roles=pd.read_parquet(ROOT/'artifacts/v95_cross_component_training_20260928/full_format_roles.parquet',columns=['role']).role.to_numpy().astype(object)
    for name,fold in [('inner',1),('C',2),('H',0)]:roles[r.fold.eq(fold).to_numpy()]=name
    for report,source in [('audit.json','v98_frozen_audit.py'),('fit_and_surface_audit.json','v98_fit_and_artifact_audit.py'),('population_surface_controls.json','v98_surface_population_audit.py')]:
        assert read(DEST/report)['source_sha256']==sha(ROOT/'training'/source)
    pop=read(DEST/'population_surface_controls.json')
    ids=np.load(V92/'ASA_input_ids.npy');x=sparse.load_npz(V92/'ASA_R0.npz')
    pos=np.full(int(fid.max())+1,-1,np.int32);pos[ids]=np.arange(len(ids))
    texts=pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    pat=re.compile(r'((?:CRED|HOST|USER|ORG)-)(\s*)(?=<IDENTITY>)')
    changed=np.array([bool(pat.search(stable(texts.loc[t]))) for t in r.loc[isa,'new_text_id']],bool)
    changed_rows=np.flatnonzero(isa)[changed];needed=set(changed_rows.tolist())
    assert len(needed)==4307 and np.all(y[changed_rows]==2)
    raw_nested=0;off=0;raw_checks=0
    nested=re.compile(r'(?:(?:CRED|HOST|USER|ORG)-){2,}\d+')
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=32768,columns=['message_sanitized','label_binary'],use_threads=False):
        assert np.array_equal(batch.column(1).to_pandas().map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(),y[off:off+batch.num_rows])
        local=[i for i in needed if off<=i<off+batch.num_rows]
        if local:
            strings=batch.column(0).to_pylist()
            for i in local:
                raw=strings[i-off];text=stable(view(raw)[0])
                assert pat.search(text) and text==stable(texts.loc[r.new_text_id.iloc[i]])
                diff=byte_matrix([text])-x[pos[fid[i]],:BYTE_FEATURES]
                assert not diff.nnz or abs(diff.data).max()<1e-7
                raw_nested+=bool(nested.search(raw));raw_checks+=1
        off+=batch.num_rows
    assert off==2056871 and raw_checks==4307
    for rec in read(DEST/'placeholder_provenance_examples.json'):
        text,ledger=view(rec['raw'])
        # JSON stores span tuples as lists; compare their serialized content.
        assert text==rec['view'] and json.loads(json.dumps(ledger))==rec['ledger'] and stable(text)==rec['stable']
    scores=np.load(DEST/'frozen_ASA_scores.npz');assert np.array_equal(scores['ASA_ids'],ids)
    predictions={}
    for name in ['teacher','ERM','MAG']:
        p=np.load(V97/f'{name}_prediction.npy');assert np.array_equal(scores[name].argmax(1),p[ids]);predictions[name]=p
    ab=isa&np.isin(roles,['A','B'])
    counts=np.zeros((len(ids),3),np.int64);np.add.at(counts,(pos[fid[ab]],y[ab]),1)
    bound=read(DEST/'AB_empirical_fit_bound.json')
    assert int((counts.sum(1)-counts.max(1)).sum())==bound['AB_ASA_full_R0_empirical_error_lower_bound']==22
    assert int((counts.sum(1)-counts[np.arange(len(ids)),predictions['MAG'][ids]]).sum())==bound['MAG_AB_errors']==251
    for mode,record in pop['controls'].items():
        control=np.load(DEST/f'surface_{mode}_ASA_prediction.npy')
        assert len(control)==len(ids)
        cp=control[pos[fid[isa]]];baseline=predictions['MAG'][fid[isa]]
        for cell in record['populations']:
            mask=(roles[isa]==cell['role'])&(y[isa]==cell['class']);truth=y[isa][mask]
            a=baseline[mask];b=cp[mask]
            assert cell['rows']==len(truth)
            assert cell['repairs']==int(((a!=truth)&(b==truth)).sum())
            assert cell['regressions']==int(((a==truth)&(b!=truth)).sum())
    vi=np.flatnonzero(isa&(roles=='V'));index=pos[fid[vi]];tt=scores['teacher'][index];mm=scores['MAG'][index]
    for point in read(DEST/'frozen_residual_scale_diagnostic.json')['frontier']:
        p=(tt+point['scale']*(mm-tt)).argmax(1);truth=y[vi]
        assert point['errors']==int((p!=truth).sum())
        assert point['M_correct']==int(((truth==1)&(p==1)).sum())
        assert point['S_correct']==int(((truth==2)&(p==2)).sum())
    for rec in read(DEST/'six_supported_training_errors.json'):
        ii=(fid==rec['R0'])&isa
        assert np.bincount(y[ii],minlength=3).tolist()==rec['all_roles_exact_R0_class_counts']
        assert rec['all_roles_exact_R0_class_counts']==[0,0,2]
    attribution=read(DEST/'teacher_branch_exposure_attribution.json')
    for role,g in attribution.items():
        mask=isa&(roles==role);o=old[fid]==y;t=predictions['teacher'][fid]==y;m=predictions['MAG'][fid]==y
        assert g['historical_correct_final_wrong']==int((mask&o&~m).sum())
        assert g['historical_correct_teacher_lost_still_wrong']==int((mask&o&~t&~m).sum())
        assert g['historical_correct_teacher_kept_branch_lost']==int((mask&o&t&~m).sum())
    prior={};receipts=read(V97/'verification.json')['prior_delivery_sha256'].copy()
    receipts['evidence/2026-09-28/v97_matched_training/delivery.json']=sha(ROOT/'evidence/2026-09-28/v97_matched_training/delivery.json')
    for path,h in receipts.items():
        assert sha(ROOT/path)==h
        for name,digest in read(ROOT/path)['artifact_sha256'].items():
            assert name not in prior or prior[name]==digest
            prior[name]=digest
    dirty=[name for name,h in prior.items() if sha(ROOT/name)!=h];assert not dirty and len(prior)==1099
    result={'status':'passed','source_sha256':sha(__file__),'official_label_rows_checked':off,
        'placeholder_affected_rows_raw_replayed':raw_checks,'affected_rows_with_nested_original_placeholder':raw_nested,
        'frozen_score_inputs_checked_per_model':len(ids),'surface_modes_population_counts_checked':3,
        'six_scale_V_counts_checked':len(vi),'training_nonconflict_errors_rechecked':6,
        'historic_role_attributions_checked':len(attribution),'prior_bound_files_rehashed':len(prior),
        'prior_bound_files_changed':dirty,'receipt_sha256':receipts,'actual_classifier_fits':0,'actual_calibration_fits':0,
        'quality_acceptance':False,'model_promoted':False,'seconds':time.monotonic()-started,
        'scope':'Original labels and all 4307 affected ASA raw views, saved frozen scores/decisions and historical bindings. No new training, blind score, external transfer or full-task replay.'}
    save(DEST/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='receipt_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
