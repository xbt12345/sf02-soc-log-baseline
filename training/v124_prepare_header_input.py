"""Prepare and verify B input before any V124 model fit; no optimizer steps."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from v75_views import BYTE_FEATURES, byte_matrix, matrix_hashes
from v124_header import HEADER, old_text, transform

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'artifacts/v124_header_trial_20260929'
TRACE = ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
SOURCE = ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
V122 = ROOT/'artifacts/v122_evidence_review_20260929'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    if DEST.exists():raise FileExistsError('Never overwrite an input trial')
    historical=read(V122/'verification.json')['artifact_sha256']
    for rel,h in historical.items():assert sha(ROOT/rel)==h,rel
    old=sparse.load_npz(SOURCE).tocsr()
    d=pd.read_parquet(TRACE)
    prior=pd.read_parquet(V122/'row_diagnosis.parquet',columns=['row_position','local','root','fold','truth'])
    assert len(d)==len(prior)==112807 and old.shape==(22546,66287)
    for col in ('row_position','local','root','fold','truth'):
        assert np.array_equal(d[col],prior[col]),col
    changed=[]; led=[]
    for i,(raw,local) in enumerate(zip(d.raw_message,d.local)):
        text,span,is_gap=transform(raw)
        if not is_gap:continue
        before=old_text(raw)
        body=raw[span[1]:]
        assert raw[:span[1]]+body==raw and text!=before
        assert (text==transform('<164>Jan 01 2000 01:02:03: '+body)[0]
                ==transform('<164>Dec 31 2001 USER-CRED-45: '+body)[0])
        changed.append((i,int(local),before,text))
        led.append({'row_position':int(d.row_position.iloc[i]),'local':int(local),
                    'header_end':span[1],'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                    'body_sha256':hashlib.sha256(body.encode()).hexdigest(),
                    'old_text_sha256':hashlib.sha256(before.encode()).hexdigest(),
                    'new_text_sha256':hashlib.sha256(text.encode()).hexdigest()})
    assert len(changed)==682 and len({x[1] for x in changed})==289
    q=pd.read_parquet(V122/'clock_gap_rows.parquet')
    # Parquet does not preserve the filtered frame's old positional index.
    assert [int(d.row_position.iloc[x[0]]) for x in changed]==q.row_position.to_list()
    assert [x[2] for x in changed]==q.original_N1_text.to_list()
    assert [x[3] for x in changed]==q.patched_text.to_list()
    assert [int(d.truth.iloc[x[0]]) for x in changed].count(1)==44
    perlocal={}
    for _,local,before,after in changed:
        if local in perlocal:assert perlocal[local]==(before,after)
        else:perlocal[local]=(before,after)
    used=np.array(sorted(perlocal),dtype=np.int32)
    oldbyte=byte_matrix([perlocal[i][0] for i in used])
    diff=oldbyte-old[used,:BYTE_FEATURES]
    assert not diff.nnz or np.max(np.abs(diff.data))<1e-7
    newbyte=byte_matrix([perlocal[i][1] for i in used])
    new=old.tolil(copy=True)
    for local,row in zip(used,newbyte):
        b=row.indices;v=row.data
        facts=old.getrow(int(local))[:,BYTE_FEATURES:]
        new.rows[int(local)]=list(b)+list(facts.indices+BYTE_FEATURES)
        new.data[int(local)]=list(v)+list(facts.data)
    new=new.tocsr();new.sum_duplicates();new.sort_indices()
    assert new.shape==old.shape
    factual=new[:,BYTE_FEATURES:]-old[:,BYTE_FEATURES:]
    assert factual.nnz==0 or np.max(np.abs(factual.data))<1e-8
    nonmodified=np.setdiff1d(np.arange(old.shape[0]),used)
    assert (new[nonmodified]!=old[nonmodified]).nnz==0
    rowlocal=d.local.to_numpy(dtype=np.int32)
    anychanged=np.asarray((new!=old).getnnz(axis=1)>0).ravel()
    affected=np.flatnonzero(anychanged[rowlocal])
    assert affected.tolist()==[x[0] for x in changed]
    hashes=matrix_hashes(new)
    keys,_=pd.factorize([hashes[i] for i in rowlocal])
    frame=d[['row_position','fold','root','truth','local']].copy();frame['key']=keys
    cross=frame.groupby('key').fold.nunique()
    ct=pd.crosstab(frame.key,frame.truth)
    floor=int((ct.sum(1)-ct.max(1)).sum())
    assert floor==28 and cross.max()==1 and frame.key.nunique()==22276
    oldto=frame.groupby('local').key.nunique()
    assert oldto.max()==1
    expect=pd.read_parquet(V122/'patched_header_input_groups.parquet')
    assert np.array_equal(expect.row_position,frame.row_position)
    assert np.array_equal(expect.key,frame.key)
    DEST.mkdir()
    sparse.save_npz(DEST/'B_header_ASA.npz',new,compressed=True)
    pd.DataFrame(led).to_parquet(DEST/'header_span_ledger.parquet',index=False)
    frame.to_parquet(DEST/'B_input_groups.parquet',index=False)
    paths=[Path(__file__),ROOT/'training/v124_header.py',TRACE,SOURCE,V122/'verification.json',
           V122/'clock_gap_rows.parquet',V122/'patched_header_input_groups.parquet',
           ROOT/'training/v75_views.py',ROOT/'training/v75_corrective.py',
           ROOT/'training/v99_normalization_feasibility.py']
    result={'status':'prepared_no_training','all_ASA_rows':len(d),'changed_rows':len(changed),
        'changed_old_local_units':len(used),'M_changed':44,'S_changed':638,
        'event_body_bytes_changed':0,'fact_metadata_nnz_changed':0,
        'input_shape':list(new.shape),'changed_row_positions_match_v122':True,
        'date_and_clock_invariance_checked_rows':682,
        'old_local_splits':int(oldto.gt(1).sum()),'cross_fold_input_keys':int(cross.gt(1).sum()),
        'unique_new_full_inputs':frame.key.nunique(),'empirical_collision_floor':floor,
        'all_original_rows_and_labels_retained':True,'classifier_fits':0,
        'output_sha256':{p.name:sha(p) for p in DEST.iterdir() if p.is_file()},
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        'scope':'Input contract only; no trained model or classification gain.'}
    save(DEST/'input_preflight.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','output_sha256')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
