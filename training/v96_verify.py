"""Independent raw-to-actual-R0 check and retrospective evidence verification; zero fits."""
import hashlib
import json
import re
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from v89_common import ROOT, OUT, read, save, sha, data
from run_v75 import adapter
from v75_views import view, byte_matrix
from v75_corrective import stable
from v75_metadata import encode
from v96_icmp_parser import parse, self_check
from v96_protocol_audit import DEST, V95


def main():
    assert not (DEST/'verification.json').exists()
    started=time.monotonic()
    r,y,fid,_,old,_,fit=data()
    rows=pd.read_parquet(DEST/'icmp_rows.parquet').set_index('row_position')
    expected=set(rows.index); found=set(); raw_records=[]; existing_facts=[]; texts=[]
    parser=adapter(); enc=joblib.load(OUT/'facts_encoder.joblib')
    columns=['message_sanitized','label_binary','src_port']; offset=0; total_asa=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=32768,columns=columns,use_threads=False):
        frame=batch.to_pandas(); end=offset+len(frame)
        assert np.array_equal(frame.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(),y[offset:end])
        for j in np.flatnonzero(r.route.iloc[offset:end].eq('asa').to_numpy()):
            total_asa+=1; i=int(offset+j); raw=frame.message_sanitized.iloc[j]
            if not isinstance(raw,str) or re.search(r'\bDeny\s+icmp\b',raw,re.I) is None: continue
            found.add(i); assert i in expected
            rec=rows.loc[i]; p=parse(raw); old_parse=parser.prepare_record(frame.iloc[j].to_dict())
            assert p['format_matched']
            # Independent direct extraction from the source, not the parser's own output.
            m=re.search(r'Deny\s+icmp\s+src\s+([^\s:]+):\S+\s+dst\s+([^\s:]+):\S+\s+\(type\s+(\d+),\s*code\s+(\d+)\)',raw,re.I)
            assert m is not None
            direct={'transport_protocol':'icmp','action':'deny','outcome':'blocked',
                    'src_role':re.sub(r'-\d+$','',m[1].lower()),'dst_role':re.sub(r'-\d+$','',m[2].lower()),
                    'icmp_type':int(m[3]),'icmp_code':int(m[4])}
            assert p['facts']==direct
            assert all(old_parse['facts'].get(k)==v for k,v in direct.items()),i
            assert rec.src_zone_audit==m[1].lower() and rec.dst_zone_audit==m[2].lower()
            assert all(rec[k]==direct[k] for k in ['src_role','dst_role','icmp_type','icmp_code'])
            assert rec.label==y[i] and rec.R0==fid[i] and rec.component==r.component.iloc[i]
            assert hashlib.sha256(raw.encode()).hexdigest()==rec.raw_sha256
            for obs in json.loads(rec.spans_json):
                assert raw[obs['span'][0]:obs['span'][1]]==obs['literal']
            raw_records.append((i,frame.src_port.iloc[j]))
            existing_facts.append(old_parse['facts']);texts.append(stable(view(raw)[0]))
        offset=end
    assert offset==len(y) and found==expected and len(found)==2373 and total_asa==112807
    ii=np.array([a[0] for a in raw_records]); fx=normalize(enc.transform(existing_facts).astype(np.float32),norm='l2')
    mx,_,_=encode([a[1] for a in raw_records],[f.get('src_port_fixed',65536) for f in existing_facts])
    actual=sparse.hstack([byte_matrix(texts),fx,mx],format='csr')
    asa_ids=np.load(ROOT/'artifacts/v92_evidence_training_20260928/ASA_input_ids.npy')
    cached=sparse.load_npz(ROOT/'artifacts/v92_evidence_training_20260928/ASA_R0.npz')
    diff=actual-cached[np.searchsorted(asa_ids,fid[ii])]
    maxdiff=float(np.max(np.abs(diff.data))) if diff.nnz else 0.
    assert maxdiff<1e-7,maxdiff
    names=enc.names().tolist(); coords=[n for n in names if 'icmp' in n.lower()]
    assert 'icmp_code:bit0' in coords and 'icmp_type:bit0' in coords
    target=rows[(rows.icmp_type==3)&(rows.icmp_code==13)]
    assert set(target.role)=={'V'} and target.groupby('label').size().to_dict()=={1:180,2:685}
    roles=pd.read_parquet(V95/'full_format_roles.parquet').role.to_numpy().astype(object)
    for name,k in [('inner',1),('C',2),('H',0)]:roles[r.fold.eq(k).to_numpy()]=name
    assert np.array_equal(rows.role.to_numpy(),roles[rows.index])
    assert not rows[rows.role.eq('B')].label.eq(2).any()
    support=rows.reset_index().groupby(['src_role','dst_role','icmp_type','icmp_code','role','label']).agg(
        rows=('row_position','size'),components=('component','nunique'),inputs=('R0','nunique')).reset_index()
    pd.testing.assert_frame_equal(support,pd.read_csv(DEST/'icmp_semantic_support.csv'),check_dtype=False)
    details=target.groupby(['role','label','component','R0','src_zone_audit','dst_zone_audit']).size().rename('rows').reset_index()
    details.to_csv(DEST/'type3_code13_exact_zone_audit.csv',index=False)
    all_counts=np.zeros((int(fid.max())+1,3),dtype=np.int64);np.add.at(all_counts,(fid,y),1)
    target_conflict=all_counts[fid[target.index]]>0
    representation={'status':'all_2373_raw_records_replayed_to_actual_R0',
        'official_label_rows_verified':len(y),'ASA_rows_checked':total_asa,'ICMP_raw_records_replayed':len(found),
        'R0_fact_values_agree_with_independent_extraction':True,'R0_max_absolute_feature_difference':maxdiff,
        'existing_ICMP_coordinate_names':coords,'target_type3_code13_role_class_rows':{'V_M':180,'V_S':685},
        'target_rows_sharing_full_R0_with_another_label':int((target_conflict.sum(1)>1).sum()),
        'auxiliary_parser_empty_is_not_model_information_absence':True,
        'new_input_feature_training_warranted_by_missing_ICMP_fields':False,
        'source_sha256':sha(__file__),
        'scope':'Raw source to frozen R0 replay, not model improvement. Coarse semantic tuple conflicts do not imply full-input conflicts or irreducible error.'}
    save(DEST/'representation_crosscheck.json',representation)
    op=old[fid];bp=np.load(V95/'teacher_prediction.npy')[fid];npred=np.load(V95/'E10_step200_prediction.npy')[fid]
    attribution=pd.read_csv(DEST/'teacher_branch_error_attribution.csv')
    o=op==y;b=bp==y;n=npred==y
    masks={'inner':r.fold.eq(1).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy(),'fit':fit}
    for row in attribution.itertuples(index=False):
        take=masks[row.role]&r.route.eq(row.route).to_numpy()
        assert row.rows==int(take.sum())
        for key,flag in [('old_errors',~o),('A_teacher_errors',~b),('E10_errors',~n),
            ('old_correct_lost_in_teacher_and_still_wrong',o&~b&~n),('old_correct_lost_by_branch',o&b&~n),
            ('old_correct_teacher_lost_branch_restored',o&~b&n),
            ('branch_corrected_teacher_error',~b&n),('branch_lost_teacher_correct',b&~n)]:
            assert getattr(row,key)==int((take&flag).sum())
    buckets=pd.read_csv(DEST/'empty_bucket_protocol_composition.csv')
    bs=buckets[(buckets.role=='B')&(buckets.label_index==2)]
    assert len(bs)==1 and bs.iloc[0].protocol_observed=='udp' and bs.iloc[0].rows==6
    assert buckets[(buckets.role=='B')&(buckets.label_index==1)].iloc[0].rows==48
    prev=read(V95/'verification.json')['prior_delivery_receipts_sha256']
    paths=list(prev)+['evidence/2026-09-28/v95_four_arm_training/delivery.json']
    bound={};receipts={}
    for p in paths:
        receipts[p]=sha(ROOT/p)
        if p in prev:assert receipts[p]==prev[p]
        for name,h in read(ROOT/p)['artifact_sha256'].items():
            assert name not in bound or bound[name]==h
            bound[name]=h
    changed=[p for p,h in bound.items() if sha(ROOT/p)!=h]
    assert not changed,changed
    scope=read(DEST/'frozen_nonASA_scope_control.json')
    composed=np.where(r.route.eq('asa').to_numpy(),npred,op)
    for row in scope['results']:
        m=masks[row['role']]
        assert row['scope_restored_errors']==int((composed[m]!=y[m]).sum())
        assert row['regressions_vs_E10']==int(((npred[m]==y[m])&(composed[m]!=y[m])).sum())
    contract=read(DEST/'behavior_contract_audit.json')
    assert len(contract['eligible_AB_both_class_groups'])==1
    assert contract['minimum_two_components_per_role_class_groups']==0
    for file,script in [('protocol_audit.json','v96_protocol_audit.py'),('experiment_audit.json','v96_experiment_audit.py'),
                        ('behavior_contract_audit.json','v96_behavior_contract.py')]:
        audit=read(DEST/file);assert audit['source_sha256']==sha(ROOT/'training'/script)
        assert audit['new_classifier_fits']==0 and audit['new_calibration_fits']==0
    assert read(DEST/'protocol_audit.json')['parser_sha256']==sha(ROOT/'training/v96_icmp_parser.py')
    result={'status':'passed','source_sha256':sha(__file__),'raw_records_to_frozen_R0_replayed':len(found),
        'actual_R0_feature_difference_max':maxdiff,'official_labels_verified':len(y),'error_attribution_cells_verified':len(attribution),
        'parser_self_checks':self_check(),'prior_bound_files_rehashed':len(bound),'prior_bound_files_changed':changed,
        'receipt_sha256':receipts,'new_classifier_fits':0,'new_calibration_fits':0,'quality_acceptance':False,
        'seconds':time.monotonic()-started,
        'scope':'Independent original-source/R0 replay, support and error-flow recomputation, frozen gradient diagnosis and historical hashes; no new fit, blind test, full-task replay or external-transfer claim.'}
    save(DEST/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='receipt_sha256'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
