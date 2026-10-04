"""Audit proposed method assumptions on inspected data, without training."""
import hashlib,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/v38_local_r1_20260913'
sys.path.insert(0,str(RUN/'frozen_training_runtime'))
import numpy as np,pandas as pd,pyarrow.parquet as pq
OUT=ROOT/'evidence/2026-09-13/v39_methods'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def summary(frame,key):
    counts=pd.crosstab(frame[key],frame.label_index)
    return {'keys':len(counts),'mixed_keys':int(((counts>0).sum(1)>1).sum()),
        'empirical_min_errors':int((counts.sum(1)-counts.max(1)).sum())}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'assumption_probes.json').exists():raise FileExistsError('Preserve prior run')
    plan=ROOT/'docs/V39_NEXT_PLAN.md';(OUT/'plan_before_method_review.md').write_bytes(plan.read_bytes())
    modeldir=RUN/'equal_classifiers_attempt2/C_BOTH'
    receipt=json.loads((modeldir/'complete.json').read_text(encoding='utf-8'))
    assert sha(modeldir/'evaluation.parquet')==receipt['predictions_sha256']
    assert sha(modeldir/'model.joblib')==receipt['model_sha256']
    d=pq.read_table(modeldir/'evaluation.parquet').to_pandas()
    p=d[['p_benign','p_malicious','p_suspicious']].to_numpy();assert np.all(p>0)
    logp=np.log(p);y=d.label_index.to_numpy()
    # Analytic feasibility, with post-inspection target selection. Not a tuned
    # threshold, new model or estimated out-of-sample gain.
    B=y==0;bn_slack=logp[B,0]-logp[B,2]
    delta_need=np.maximum(logp[:,0],logp[:,1])-logp[:,2]
    probes=[]
    for name,m in [('windows_missed_suspicious',(d.route=='windows_message').to_numpy()&(y==2)&(p.argmax(1)==0)),
                   ('Duo_missed_suspicious',(d['product']=='Duo').to_numpy()&(y==2))]:
        lower=float(delta_need[m].max());upper=float(bn_slack.min());eps=np.nextafter(lower,np.inf)
        false=int((bn_slack<eps).sum())
        probes.append({'target':name,'rows':int(m.sum()),'S_logit_offset_required_to_fix_all_strictly_greater_than':lower,
            'S_logit_offset_allowed_with_no_new_benign_errors_strictly_less_than':upper,
            'intervals_overlap':lower<upper,'new_benign_errors_just_above_required_offset':false,
            'normal_rows':int(B.sum()),'scope':'Oracle feasibility diagnostic on already inspected evaluation; not calibration or a deployable decision rule'})
    # Recover all original messages in the main ICMP collision component.
    rows=pq.read_table(RUN/'prepared/rows.parquet').to_pandas()
    relevant=rows[(rows.union_group==2054164)&(rows.inner_role==2)]
    wanted=set(relevant.row_position.astype(int));assert len(wanted)==865
    raw={};offset=0
    for b in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized'],use_threads=False):
        for pos in wanted:
            if offset<=pos<offset+len(b):raw[pos]=b.column(0)[pos-offset].as_py()
        offset+=len(b)
    records=[]
    for _,r in relevant.iterrows():
        original=raw[int(r.row_position)];start=original.lower().find('deny icmp ')
        assert start>=0
        body=original[start:].strip()
        m=re.search(r'\bsrc\s+([^:\s]+):(\S+)\s+dst\s+([^:\s]+):(\S+)',body,re.I);assert m
        # Identity retains only audit evidence. It is not a new feature proposal.
        identity_removed=re.sub(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)','<IP>',body)
        identity_removed=re.sub(r'(?:USER|HOST|ORG|CRED)(?:-\d+)+','<ENTITY>',identity_removed)
        identity_removed=re.sub(r'dmz-?\d+','dmz',identity_removed,flags=re.I)
        records.append({'row_position':int(r.row_position),'label_index':int(r.label_index),
            'raw_hash':hashlib.sha256(original.encode('utf-8')).hexdigest(),
            'body_hash':hashlib.sha256(body.encode('utf-8')).hexdigest(),
            'src_interface':m[1],'dst_interface':m[3],
            'without_identity_hash':hashlib.sha256(identity_removed.encode('utf-8')).hexdigest(),
            'body_schema':identity_removed})
    a=pd.DataFrame(records)
    interface_counts=a.groupby(['src_interface','dst_interface','label_index']).size().reset_index(name='rows').to_dict('records')
    full_r=pq.read_table(RUN/'prepared/rows.parquet',columns=['inner_role','route','label_index','projection_id','union_group']).to_pandas()
    proj=pq.read_table(RUN/'prepared/projections.parquet').to_pandas()
    f=[json.loads(s) for s in proj.C_BOTH]
    maskcols=[];strict=[]
    # Audit whether controlled value learning has support after fixing observed
    # action/protocol/direction. No class labels used to construct these keys.
    from run_v38_train import negative_control
    mx,names=negative_control(f)
    for j,v in enumerate(f):
        maskcols.append(bytes(mx[j].toarray()[0].astype('u1')))
        strict.append(json.dumps({k:v[k] for k in ['action','outcome','transport_protocol','src_role','dst_role','icmp_type','icmp_code'] if k in v},sort_keys=True))
    full_r['mask']=pd.factorize(np.asarray(maskcols,dtype=object),sort=False)[0][full_r.projection_id]
    full_r['context']=np.asarray(strict,dtype=object)[full_r.projection_id]
    fit=full_r[full_r.inner_role.isin([0,1])]
    contrast=[]
    for (route,mask,context),s in fit.groupby(['route','mask','context']):
        if s.label_index.nunique()<2:continue
        cc=s.groupby('label_index').agg(rows=('projection_id','size'),projections=('projection_id','nunique'),legacy_components=('union_group','nunique')).reset_index().to_dict('records')
        contrast.append({'route':route,'mask':int(mask),'fixed_context':json.loads(context),'class_support':cc})
    result={'scope':'Read-only source-bound diagnostics and assumption checks; no model training, no new splits or labels.',
        'input_model_sha256':receipt['model_sha256'],'input_predictions_sha256':receipt['predictions_sha256'],
        'old_plan_sha256':sha(OUT/'plan_before_method_review.md'),'script_sha256':sha(Path(__file__)),
        'global_class_bias_feasibility':probes,
        'ASA_code13':{'rows':len(a),'raw':summary(a,'raw_hash'),'body_keeping_identity':summary(a,'body_hash'),
            'body_removing_identity_and_zone_suffix':summary(a,'without_identity_hash'),'interface_label_counts':interface_counts,
            'identity_removed_body_shapes':a.body_schema.drop_duplicates().tolist(),
            'interpretation':'Raw and body counts are not independent incidents; interface suffix and IP equality can explain dataset labels without validated causal meaning.'},
        'context_matched_fit_support':contrast}
    (OUT/'assumption_probes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pq.write_table(__import__('pyarrow').Table.from_pandas(a.drop(columns='body_schema'),preserve_index=False),OUT/'asa_code13_audit.parquet',compression='zstd')
    print(json.dumps({k:v for k,v in result.items() if k!='context_matched_fit_support'},ensure_ascii=False))
if __name__=='__main__':main()
