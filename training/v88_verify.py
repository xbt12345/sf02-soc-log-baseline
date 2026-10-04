"""Independently recompute v88 attribution/support and verify unchanged history; no fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_v75 import ROOT, OUT, read, save, sha

DEST=ROOT/'artifacts/v88_root_review_20260927'
PRIOR=ROOT/'artifacts/v87_solver_supervision_r2_20260927/fold1'


def main():
    d=read(DEST/'diagnosis.json')
    assert d['source_sha256']==sha(ROOT/'training/v88_diagnose.py')
    r=pd.read_parquet(OUT/'rows.parquet')
    raw=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary','src_port','message_sanitized'])
    y=raw.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert np.array_equal(r.row_position,np.arange(len(r))) and np.array_equal(y,r.label_index)
    fid=np.load(ROOT/'artifacts/v79_execution_20260927/row_feature_id.npy')
    old=np.load(ROOT/'artifacts/v85_protection_20260927/fold1/teacher_prediction.npy')[fid]
    selected=np.load(PRIOR/'selected_rows.npy');fit=~r.fold.isin([0,1,2]).to_numpy()
    mask=np.zeros(len(r),bool);mask[selected]=True
    assert np.array_equal(mask&(y>0),fit&(y>0))
    assert mask.sum()==d['main_loss_rows']==65996
    assert (mask&(old==y)).sum()==d['main_loss_correct_rows']==65681
    assert (mask&(old!=y)).sum()==d['main_loss_error_rows']==315
    assert (fit&(old==y)).sum()==d['full_protected_correct_rows']==922009
    cc=np.bincount(fid[fit]*3+y[fit],minlength=(int(fid.max())+1)*3).reshape(-1,3)
    old_by_input=np.load(ROOT/'artifacts/v85_protection_20260927/fold1/teacher_prediction.npy')
    free=fit&(old!=y)&(cc[np.arange(len(cc)),old_by_input][fid]==0)
    assert free.sum()==d['unblocked_errors']==293
    assert not (free&((cc>0).sum(1)[fid]>1)).any()
    assert len(np.unique(fid[free]))==146
    changed_counts={}
    for name,record in d['selected_candidate_attribution'].items():
        new=np.load(PRIOR/(name+'_all_prediction.npy'))[fid]
        roles={'fit_full':fit,'inner':r.fold.eq(1).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
        rows=[]
        for role,take in roles.items():
            changed=take&(old!=new)
            frame=pd.DataFrame({'route':r.route[changed].to_numpy(),'class':y[changed],
                 'repairs':(new[changed]==y[changed]).astype(int),'regressions':(old[changed]==y[changed]).astype(int)})
            grouped=frame.groupby(['route','class'])[['repairs','regressions']].sum()
            for (route,cls),val in grouped.iterrows():
                if val.sum():rows.append({'role':role,'route':route,'class':int(cls),**{k:int(v) for k,v in val.items()}})
        expected=[{k:z[k] for k in ['role','route','class','repairs','regressions']} for z in record]
        sort=lambda a: sorted(a,key=lambda z:(z['role'],z['route'],z['class']))
        assert sort(rows)==sort(expected),(name,rows,expected)
        changed_counts[name]=rows
    # Rebuild semantic support from facts, without using v87 support-building code.
    fields=['action','outcome','transport_protocol','src_role','dst_role']
    facts=[json.loads(s) for s in pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts]
    key_by_projection=[tuple(str(f[k]) for k in fields)+(str(f.get('dst_port_fixed','<missing>')),) if all(k in f for k in fields) else None for f in facts]
    supported={}
    for i in selected:
        key=key_by_projection[int(r.projection_id.iloc[i])]
        if key is not None:supported.setdefault((key,int(y[i])),set()).add(int(r.component.iloc[i]))
    bins=pd.read_parquet(PRIOR/'inner_support_bins.parquet')
    assert np.array_equal(bins.row_position,np.flatnonzero(r.fold.eq(1).to_numpy()&(y>0)))
    for b in bins.itertuples(index=False):
        i=b.row_position;key=key_by_projection[int(r.projection_id.iloc[i])]
        n=0 if key is None else len(supported.get((key,int(y[i])),set()))
        cat='missing_behavior_facts' if key is None else 'same_class_zero_support' if n==0 else 'same_class_single_component' if n==1 else 'same_class_multi_component'
        assert b.support_components==n and b.support_bin==cat
        assert b.route==r.route.iloc[i] and b._1==int(y[i])  # class is a reserved Python name
    new=np.load(PRIOR/'A3_epoch060_all_prediction.npy')[fid];ii=bins.row_position.to_numpy()
    b=bins.assign(old_errors=(old[ii]!=y[ii]).astype(int),new_errors=(new[ii]!=y[ii]).astype(int))
    actual=b.groupby(['route','class','support_bin']).agg(rows=('row_position','size'),old_errors=('old_errors','sum'),new_errors=('new_errors','sum')).reset_index()
    expected=pd.read_csv(DEST/'support_and_errors.csv')
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
    for group in d['same_input_conflict_groups']:
        ix=np.flatnonzero(fit&(fid==group['fid']))
        assert np.array_equal(ix,group['row_positions'])
        assert np.bincount(y[ix],minlength=3).tolist()==group['class_counts']==[0,106,22]
        assert raw.src_port.iloc[ix[y[ix]==1]].isna().all()
        assert (raw.src_port.iloc[ix[y[ix]==2]]=='').all()
    for s in d['loss_and_gradient']:
        assert s['independent_head_gradient_max_abs_error']<1e-7
        assert abs(s['independent_main_loss']-s['main_loss'])<1e-6
    # Numeric analytic-gradient comparison is performed in the no-fit diagnosis.
    # This verifier independently checks rows/support/attribution and checks that receipt.
    prior={}
    receipts=['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
       'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json',
       'evidence/2026-09-27/v83_root_review/delivery.json','evidence/2026-09-27/v84_preservation/delivery.json',
       'evidence/2026-09-27/v85_protection/delivery.json','evidence/2026-09-27/v86_boundary_review/delivery.json',
       'evidence/2026-09-27/v87_solver_supervision/delivery.json']
    for path in receipts:prior.update(read(ROOT/path)['artifact_sha256'])
    changed=[p for p,h in prior.items() if sha(ROOT/p)!=h]
    assert not changed,changed
    doc=next((ROOT/'docs/official').glob('附件2*.docx'))
    result={'status':'passed','actual_new_classifier_fits':0,'actual_new_calibration_fits':0,
       'source_sha256':sha(__file__),'diagnosis_sha256':sha(DEST/'diagnosis.json'),
       'support_sha256':sha(DEST/'support_and_errors.csv'),'official_task_sha256':sha(doc),
       'original_labels_checked':len(r),'independent_support_rows_checked':len(bins),
       'candidate_attributions_recomputed':len(changed_counts),
       'free_error_rows':293,'free_error_input_groups':146,'free_error_rows_with_mixed_input_labels':0,
       'analytic_head_gradient_max_abs_error':max(s['independent_head_gradient_max_abs_error'] for s in d['loss_and_gradient']),
       'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,
       'receipt_hashes':{p:sha(ROOT/p) for p in receipts},
       'scope':'Frozen decision attribution and support from original row labels/facts; nullable input checked; analytic gradient receipt checked; all prior bound artifacts rehashed. No optimizer or new model inference replay here.'}
    save(DEST/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
