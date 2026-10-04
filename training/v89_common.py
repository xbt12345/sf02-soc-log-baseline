"""Shared data/receipts for the registered v88-plan execution; no historical mutation."""
import json
import time
import numpy as np
import pandas as pd
from run_v75 import ROOT,OUT,read,save,sha,load_sparse,metrics
from v79_execute import rows
from v85_protection import changes,cm_from_counts

DEST=ROOT/'artifacts/v89_readout_support_20260927'
PRIOR=ROOT/'artifacts/v87_solver_supervision_r2_20260927'
OLD=ROOT/'artifacts/v85_protection_20260927/fold1'
LAST=ROOT/'artifacts/v79_execution_20260927'
DELTA=.01
TOL=1e-8
ALPHA=1e-6


def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)


def data():
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy')
    z=np.load(OLD/'teacher_scores.npy',mmap_mode='r');old=z.argmax(1).astype(np.int8)
    sel=np.load(PRIOR/'fold1/selected_rows.npy');fit=~r.fold.isin([0,1,2]).to_numpy()
    return r,y,fid,z,old,sel,fit


def raw_counts(fid,y,mask,n):
    return np.bincount(fid[mask]*3+y[mask],minlength=n*3).reshape(n,3)


def check():
    reg=read(DEST/'registration.json')
    for p,h in reg['source_bindings'].items():assert sha(ROOT/p)==h,p
    return reg


def ledger_start(kind,name):
    p=DEST/'execution_ledger.json';items=read(p) if p.exists() else []
    assert not any(z['kind']==kind and z['name']==name for z in items),(kind,name)
    i=len(items);items.append({'id':i,'kind':kind,'name':name,'status':'started','started_unix':time.time()});save(p,items)
    return i


def ledger_end(i,status,**facts):
    p=DEST/'execution_ledger.json';a=read(p);assert a[i]['status']=='started'
    a[i].update(status=status,finished_unix=time.time(),**facts);save(p,a)


def evaluate(r,y,fid,old,new,fit,counts=None,masks=None):
    masks={'fit_full':fit,'inner':r.fold.eq(1).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()} if masks is None else masks
    out={};cells=[]
    for role,mask in masks.items():
        cc=raw_counts(fid,y,mask,len(old));out[role]=changes(cc,old,new)
        for (route,cls),indices in r[mask].groupby(['route','label_index']).groups.items():
            ii=np.asarray(indices);a=old[fid[ii]];b=new[fid[ii]];truth=y[ii]
            cells.append({'role':role,'route':route,'class':int(cls),'support':len(ii),
               'old_correct':int((a==truth).sum()),'new_correct':int((b==truth).sum()),
               'repairs':int(((a!=truth)&(b==truth)).sum()),'regressions':int(((a==truth)&(b!=truth)).sum()),
               'repair_inputs':len(np.unique(fid[ii[(a!=truth)&(b==truth)]])),
               'repair_components':int(r.component.iloc[ii[(a!=truth)&(b==truth)]].nunique())})
    return out,pd.DataFrame(cells)
