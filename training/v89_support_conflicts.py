"""Distinguish exact-input label conflict from remaining training errors; no fits."""
import json
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from v89_common import *


def main():
    r,y,fid,z,old,sel,fit=data();tr=np.load(DEST/'support_full_rows.npy');x=load_sparse(LAST/'X')
    cc=np.bincount(fid[tr]*3+y[tr],minlength=len(old)*3).reshape(-1,3)
    fs=[json.loads(s) for s in pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts]
    fields=['action','outcome','transport_protocol','src_role','dst_role','dst_port_fixed']
    keys=[tuple(str(f[k]) for k in fields) if all(k in f for k in fields) else None for f in fs]
    chosen={tuple(k) for k in read(DEST/'support_design.json')['chosen_keys']}
    ii=np.array([i for i in tr if keys[int(r.projection_id.iloc[i])] in chosen and y[i]==2])
    m=joblib.load(DEST/'support_full.joblib');pred=(x[fid[ii]]@m['coef']+m['intercept']).argmax(1)
    conflict=(cc[fid[ii],:2].sum(1)>0)
    rec={'source_sha256':sha(__file__),'new_fits':0,'S_fine_training_rows':len(ii),'S_fine_training_correct':int((pred==2).sum()),'S_fine_training_errors':int((pred!=2).sum()),'S_training_errors_same_input_other_class':int(((pred!=2)&conflict).sum()),'S_training_errors_no_same_input_other_class':int(((pred!=2)&~conflict).sum()),'scope':'Fixed complete-support R0; exact input class conflicts count, not raw-context impossibility.'}
    save(DEST/'support_training_conflicts.json',rec);emit(**rec)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
