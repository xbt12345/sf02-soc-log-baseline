"""Registered v91-plan execution: no overwrite of historical sources or models."""
import json,time
import numpy as np
from v89_common import ROOT,OUT,LAST,OLD,DEST as V89,read,save,sha,data,raw_counts
from v85_protection import changes,cm_from_counts

DEST=ROOT/'artifacts/v92_evidence_training_20260928'
SEED=9201

def emit(**facts):print(json.dumps(facts,ensure_ascii=False),flush=True)

def event(stage,state,**facts):
    p=DEST/'execution_ledger.json';items=read(p) if p.exists() else []
    items.append({'stage':stage,'status':state,'unix':time.time(),**facts});save(p,items)

def check():
    reg=read(DEST/'registration.json')
    for p,h in reg['source_bindings'].items():assert sha(ROOT/p)==h,p
    for p,h in reg['input_bindings'].items():assert sha(ROOT/p)==h,p
    return reg

def evaluate(r,y,fid,old,new,role_names=None):
    roles={};cells=[]
    for role,mask in [('fit',~r.fold.isin([0,1,2]).to_numpy()),('inner',r.fold.eq(1).to_numpy()),('C',r.fold.eq(2).to_numpy()),('H',r.fold.eq(0).to_numpy())]:
        if role_names is not None and role not in role_names:continue
        cc=raw_counts(fid,y,mask,len(old));roles[role]=changes(cc,old,new)
        for (route,cls),indices in r[mask].groupby(['route','label_index']).groups.items():
            ix=np.asarray(indices);a=old[fid[ix]];b=new[fid[ix]];truth=y[ix]
            cells.append({'role':role,'route':route,'class':int(cls),'support':len(ix),'old_correct':int((a==truth).sum()),'new_correct':int((b==truth).sum()),
                'repairs':int(((a!=truth)&(b==truth)).sum()),'regressions':int(((a==truth)&(b!=truth)).sum()),
                'repair_components':int(r.component.iloc[ix[(a!=truth)&(b==truth)]].nunique())})
    return roles,cells
