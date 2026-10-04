"""Build the V125 ordered/shuffled body byte views before training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from v124_header import HEADER
from v75_views import view, matrix_hashes
from v75_corrective import stable
from v99_normalization_feasibility import normalize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v125_order_trial_20260929'
TRACE=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
OLD=ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
PARENT=ROOT/'artifacts/v124_header_trial_20260929/delivery.json'
V125=ROOT/'artifacts/v125_review_20260929/verification.json'
SEED=12501


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def check_receipt(p):
    for rel,h in json.loads(p.read_text(encoding='utf-8'))['artifact_sha256'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Historical artifact changed: '+rel)


def normalize_body(raw):
    m=HEADER.match(raw)
    if m is None:raise ValueError('Unverified ASA header; preserve and stop')
    body=raw[m.end():]
    if not body.startswith('USER-0010-0324 Deny '):raise ValueError('Body boundary changed')
    return normalize(stable(view(body)[0]),'placeholder_cluster'),m.end()


def permuted_bytes(body):
    encoded=np.frombuffer(body.encode('utf-8'),dtype=np.uint8).copy()
    seed=int.from_bytes(hashlib.sha256(str(SEED).encode()+b'|'+body.encode('utf-8')).digest()[:8],'big')
    return encoded[np.random.default_rng(seed).permutation(len(encoded))]


def main():
    if OUT.exists():raise FileExistsError(OUT)
    check_receipt(PARENT);check_receipt(V125)
    d=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth','raw_message'])
    x=sparse.load_npz(OLD)
    assert len(d)==112807 and x.shape==(22546,66287)
    assert np.array_equal(d.local.drop_duplicates().sort_values(),np.arange(x.shape[0]))
    assert d.groupby('local').fold.nunique().max()==d.groupby('local').root.nunique().max()==1
    raw=d.drop_duplicates('local').sort_values('local')
    unique_text=[];heads=[]
    for message in raw.raw_message:
        body,end=normalize_body(message)
        unique_text.append(body);heads.append(end)
    observed={int(loc):(body,end) for loc,body,end in zip(raw.local,unique_text,heads)}
    # Every original row is checked, rather than silently choosing one old-local representative.
    ledger=[]
    for r in d.itertuples(index=False):
        body,end=normalize_body(r.raw_message)
        prior,_=observed[int(r.local)]
        if body!=prior:raise ValueError(f'Old local {r.local} split by ordered body: row {r.row_position}')
        ledger.append((int(r.row_position),int(r.local),int(r.root),int(r.fold),int(r.truth),end,
                       hashlib.sha256(r.raw_message[:end].encode()).hexdigest(),
                       hashlib.sha256(r.raw_message[end:].encode()).hexdigest(),
                       hashlib.sha256(body.encode()).hexdigest()))
    lengths=np.array([len(t.encode()) for t in unique_text],dtype=np.int16)
    assert lengths.min()>0 and lengths.max()<=176
    B=np.zeros((len(unique_text),int(lengths.max())),dtype=np.uint8)
    C=np.zeros_like(B)
    for i,t in enumerate(unique_text):
        b=np.frombuffer(t.encode(),dtype=np.uint8)
        c=permuted_bytes(t)
        B[i,:len(b)]=b;C[i,:len(c)]=c
    # Need byte 0 as a real byte; both arrays store raw uint8 plus an independent length.
    assert all(np.array_equal(np.sort(B[i,:n]),np.sort(C[i,:n])) for i,n in enumerate(lengths))
    assert (B!=C).any(axis=1).all()
    old_hash=np.array([h.hex() for h in matrix_hashes(x)],dtype=object)
    assert d.assign(key=old_hash[d.local.to_numpy()]).groupby('key').fold.nunique().max()==1
    OUT.mkdir()
    np.save(OUT/'ordered_body_bytes.npy',B)
    np.save(OUT/'shuffled_body_bytes.npy',C)
    np.save(OUT/'body_lengths.npy',lengths)
    pd.DataFrame(ledger,columns=['row_position','local','root','fold','truth','header_end',
        'header_sha256','raw_body_sha256','normalized_body_sha256']).to_parquet(OUT/'body_span_ledger.parquet',index=False)
    report={'status':'input_prepared_before_fit','rows':len(d),'old_locals':x.shape[0],
        'fine_body_groups':len(unique_text),'old_local_splits':0,'body_length_min':int(lengths.min()),
        'body_length_max':int(lengths.max()),'truncated_bytes':0,'C_byte_multiset_equal_B':True,
        'C_exact_equal_B_rows':0,'B_C_depend_on_label_root_fold':False,
        'cross_fold_old_input_keys':0,'original_row_mass_preserved':True,
        'data_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [TRACE,OLD,PARENT,V125]},
        'files_sha256':{p.name:sha(p) for p in [OUT/'ordered_body_bytes.npy',OUT/'shuffled_body_bytes.npy',
             OUT/'body_lengths.npy',OUT/'body_span_ledger.parquet']},
        'source_sha256':sha(Path(__file__))}
    (OUT/'input_preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('data_sha256','files_sha256')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
