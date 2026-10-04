"""Read-only diagnosis of frozen v38 failures; no fit or role/label change."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v38_local_r1_20260913'
sys.path.insert(0,str(RUN/'frozen_training_runtime'))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
from scipy import sparse
from run_v38_train import actual_keys,negative_control

OUT=ROOT/'evidence/2026-09-13/v39_review'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def counts(a):return np.bincount(np.asarray(a,dtype=int),minlength=3).tolist()
def table_counts(ids,y,size):return np.bincount(ids.astype(np.int64)*3+y,minlength=size*3).reshape(size,3)

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'remaining_diagnosis.json'
    if target.exists():raise FileExistsError('Preserve prior diagnosis')
    prepared=RUN/'prepared';modeldir=RUN/'equal_classifiers_attempt2/C_BOTH'
    receipt=json.loads((prepared/'complete.json').read_text(encoding='utf-8'))
    bindings={}
    for name in ('rows.parquet','projections.parquet'):
        digest=sha(prepared/name);assert digest==receipt['files'][name];bindings[name]=digest
    mr=json.loads((modeldir/'complete.json').read_text(encoding='utf-8'))
    for name,key in [('model.joblib','model_sha256'),('evaluation.parquet','predictions_sha256')]:
        digest=sha(modeldir/name);assert digest==mr[key];bindings[name]=digest
    rows=pq.read_table(prepared/'rows.parquet').to_pandas()
    proj=pq.read_table(prepared/'projections.parquet').to_pandas()
    ev=pq.read_table(modeldir/'evaluation.parquet').to_pandas()
    assert np.array_equal(rows.loc[rows.inner_role==2,'row_position'],ev.row_position)
    fit=rows.inner_role.isin([0,1]).to_numpy();audit=(rows.inner_role==2).to_numpy()
    facts=[json.loads(v) for v in proj.C_BOTH];mx,names=negative_control(facts)
    maskids,maskkeys=pd.factorize(np.asarray([bytes(v.astype('u1')) for v in mx.toarray()],dtype=object),sort=False)
    ids=rows.projection_id.to_numpy();y=rows.label_index.to_numpy();mask=maskids[ids];nm=len(maskkeys)
    rows['mask_id']=mask
    model=joblib.load(modeldir/'model.joblib')
    x=sparse.hstack([model['text_encoder'].transform(proj.text.tolist()),model['fact_encoder'].transform(facts)],format='csr')
    vid=actual_keys(x);vec=vid[ids];nv=int(vid.max()+1)
    rows['vector_id']=vec
    train_v=table_counts(vec[fit],y[fit],nv);audit_v=table_counts(vec[audit],y[audit],nv)
    train_m=table_counts(mask[fit],y[fit],nm);audit_m=table_counts(mask[audit],y[audit],nm)
    probability=model['model'].predict_proba(x)
    stored=ev[['p_benign','p_malicious','p_suspicious']].to_numpy()
    delta=float(np.abs(probability[ids[audit]]-stored).max());assert delta<1e-10
    rows['pred']=probability[ids].argmax(1).astype('u1')
    r=rows.loc[audit].copy();pred=r.pred.to_numpy()
    # Same-mask contrasts are necessary, not sufficient: semantics and group
    # independence still need examination. No oracle quantity is called a score.
    masks=[]
    for m in range(nm):
        p=int(np.flatnonzero(maskids==m)[0]);s=rows[(mask==m)&(fit|audit)]
        if len(s)==0:continue
        masks.append({'mask_id':m,'fields':[n for n,v in zip(names,mx[p].toarray()[0]) if v],
            'fit_counts':train_m[m].tolist(),'audit_counts':audit_m[m].tolist(),
            'fit_groups_by_class':[int(s[(s.inner_role<2)&(s.label_index==k)].union_group.nunique()) for k in range(3)],
            'audit_groups_by_class':[int(s[(s.inner_role==2)&(s.label_index==k)].union_group.nunique()) for k in range(3)],
            'routes':sorted(s.route.unique()),
            'fit_actual_vectors':int(s[s.inner_role<2].vector_id.nunique()),
            'audit_actual_vectors':int(s[s.inner_role==2].vector_id.nunique()),
            'audit_errors':int(((s.inner_role==2)&(s.label_index!=s.pred)).sum())})
    save('mask_support.json',sorted(masks,key=lambda z:z['audit_errors'],reverse=True))
    route_errors=[]
    for (route,label,prediction),sub in r.groupby(['route','label_index','pred']):
        route_errors.append({'route':route,'label':int(label),'prediction':int(prediction),'rows':len(sub),
            'legacy_union_groups':int(sub.union_group.nunique()),'actual_vectors':int(sub.vector_id.nunique()),
            'mask_ids':sorted(int(v) for v in sub.mask_id.unique())})
    save('route_decisions.json',route_errors)
    # Aggregate exact-vector support and direction of conditional label shifts.
    errors=r[r.label_index!=r.pred].copy();errorgroups=[]
    for (route,label,prediction,group),sub in errors.groupby(['route','label_index','pred','union_group']):
        errorgroups.append({'route':route,'label':int(label),'prediction':int(prediction),'group':int(group),'rows':len(sub),
            'fit_exact_vector_rows':int((train_v[sub.vector_id].sum(1)>0).sum()),
            'fit_exact_vector_same_label_rows':int((train_v[sub.vector_id,int(label)]>0).sum()),
            'row_positions':sub.row_position.head(4).astype(int).tolist()})
    save('error_groups.json',sorted(errorgroups,key=lambda z:z['rows'],reverse=True))
    asa=r[r.route=='asa'];conflict=(audit_v>0).sum(1)>1
    vectors=[]
    for v in sorted(asa.loc[conflict[asa.vector_id],'vector_id'].unique()):
        one=asa[asa.vector_id==v];p=int(one.projection_id.iloc[0])
        vectors.append({'vector_id':int(v),'facts':facts[p],'fit_counts':train_v[v].tolist(),'audit_counts':audit_v[v].tolist(),
            'prediction':int(one.pred.iloc[0]),'audit_union_groups':int(one.union_group.nunique()),
            'row_positions':one.groupby('label_index').row_position.first().astype(int).to_dict()})
    save('asa_conflicts.json',vectors)
    # Inspect legacy components and origins rather than naming them incidents.
    oldpath=ROOT/'artifacts/v37_prepared_r13_20260913/prepared.parquet'
    old=pq.read_table(oldpath,columns=['raw_hash','legacy_group_text','b0','b1','facts']).to_pandas()
    assert len(old)==len(rows)
    rawcodes,_=pd.factorize(old.raw_hash,sort=False);rows['raw_id']=rawcodes
    groupinfo=[]
    selected=rows.loc[(y==1)&fit,'union_group'].value_counts().head(6).index.tolist()
    selected+=errors.union_group.value_counts().head(6).index.tolist()
    for g in sorted(set(selected)):
        ix=np.flatnonzero(rows.union_group.to_numpy()==g);a=rows.iloc[ix];b=old.iloc[ix]
        groupinfo.append({'union_group':int(g),'rows':len(a),'classes':counts(a.label_index),
            'inner_roles':{str(int(k)):int(v) for k,v in a.inner_role.value_counts().items()},
            'routes':a.route.value_counts().to_dict(),'unique_raw_messages':int(b.raw_hash.nunique()),
            'unique_legacy_group_text':int(b.legacy_group_text.nunique()),'unique_b0_text':int(b.b0.nunique()),
            'unique_b1_text':int(b.b1.nunique()),'unique_v38_vectors':int(a.vector_id.nunique()),
            'legacy_text_examples':b.legacy_group_text.drop_duplicates().head(3).tolist(),
            'b1_text_examples':b.b1.drop_duplicates().head(3).tolist()})
    save('legacy_components.json',groupinfo)
    # Raw hash overlap only for nonempty messages; an empty string is no incident identity.
    nonempty=~rows.original_empty.to_numpy();seen_raw=np.unique(rawcodes[fit&nonempty])
    overlap=audit&nonempty&np.isin(rawcodes,seen_raw)
    # Select actual failures and same-source benign records for local inspection.
    positions=set()
    for route,sub in errors.groupby('route'):
        positions.update(sub.drop_duplicates(['label_index','pred','projection_id']).head(12).row_position.astype(int))
    for source in ['Duo','Barracuda WAF']:
        for (_,label),sub in r[r['product']==source].groupby(['product','label_index']):
            positions.update(sub.drop_duplicates('projection_id').head(12).row_position.astype(int))
    raw=[];offset=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized'],use_threads=False):
        chosen=[i-offset for i in positions if offset<=i<offset+len(batch)]
        if chosen:
            for j in sorted(chosen):
                pos=offset+j;record=rows.iloc[pos];p=int(record.projection_id)
                text=batch.column(0)[j].as_py() or ''
                assert hashlib.sha256(text.encode('utf-8')).digest()==old.raw_hash.iloc[pos]
                raw.append({'row_position':pos,'route':record['route'],'label':int(record.label_index),'prediction':int(record.pred),
                    'mask':int(record.mask_id),'raw':text,'text':proj.text.iloc[p],'facts':facts[p],
                    'fit_mask_counts':train_m[int(record.mask_id)].tolist(),
                    'fit_vector_counts':train_v[int(record.vector_id)].tolist()})
        offset+=len(batch)
    save('local_cases.json',raw)
    mixedfit=(train_m>0).sum(1)>1
    summary={'scope':'Existing v38 fit and inspected evaluation diagnostics only. No retraining, new test, role change or label repair.',
        'input_hashes':bindings,'script_sha256':sha(Path(__file__)),'probability_replay_max_difference':delta,
        'fit_rows':int(fit.sum()),'audit_rows':int(audit.sum()),'availability_patterns':nm,
        'audit_rows_in_mixed_fit_masks':int(mixedfit[mask[audit]].sum()),
        'audit_class_counts_in_mixed_fit_masks':counts(y[audit][mixedfit[mask[audit]]]),
        'audit_class_counts_in_single_label_fit_masks':counts(y[audit][(train_m[mask[audit]]>0).sum(1)==1]),
        'audit_class_counts_in_unseen_fit_masks':counts(y[audit][train_m[mask[audit]].sum(1)==0]),
        'nonempty_exact_raw_overlap_between_fit_and_audit_rows':int(overlap.sum()),
        'errors':len(errors),'error_counts_by_true_class':counts(errors.label_index),
        'error_rows_with_fit_exact_vector':int((train_v[errors.vector_id].sum(1)>0).sum()),
        'error_rows_with_fit_vector_same_label':int((train_v[errors.vector_id,errors.label_index]>0).sum()),
        'group_names_are_legacy_projection_components_not_verified_incidents':True,
        'seconds':time.perf_counter()-start}
    save('remaining_diagnosis.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
