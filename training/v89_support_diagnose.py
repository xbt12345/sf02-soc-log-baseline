"""Post-fit diagnostic of trained versus held-out behavior; not candidate tuning."""
import json
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from v89_common import ROOT,OUT,DEST,LAST,read,save,sha
from run_v75 import load_sparse


def main():
    r=pd.read_parquet(OUT/'rows.parquet');y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    train=np.load(DEST/'support_full_rows.npy');target=np.load(DEST/'support_target_rows.npy');spec=read(DEST/'support_design.json')
    fields=['action','outcome','transport_protocol','src_role','dst_role','dst_port_fixed']
    fs=[json.loads(s) for s in pd.read_parquet(OUT/'projections.parquet').facts]
    keys=[tuple(str(f[k]) for k in fields) if all(k in f for k in fields) else None for f in fs];rowkey=[keys[int(p)] for p in r.projection_id]
    chosen={tuple(a) for a in spec['chosen_keys']};fine=np.array([rowkey[i] in chosen for i in train]);m=joblib.load(DEST/'support_full.joblib')
    used=np.unique(fid[train]);score=np.asarray(x[used]@m['coef'])+m['intercept'];p=score.argmax(1)[np.searchsorted(used,fid[train])]
    cells=[]
    for key in sorted(chosen):
        for cls in [1,2]:
            ii=np.array([j for j,i in enumerate(train) if rowkey[i]==key and y[i]==cls]);rows=train[ii]
            cells.append({'key':key,'class':cls,'training_rows':len(rows),'training_components':int(r.component.iloc[rows].nunique()),'training_correct':int((p[ii]==cls).sum()),'training_error_inputs':len(np.unique(fid[rows[p[ii]!=cls]]))})
    cc=np.bincount(fid[train]*3+y[train],minlength=x.shape[0]*3).reshape(-1,3)
    byinput=[]
    for group in np.unique(fid[target]):
        ii=target[fid[target]==group];key=rowkey[ii[0]];s=np.asarray(x[group]@m['coef']).ravel()+m['intercept'];truth=np.bincount(y[ii],minlength=3)
        samekey=np.array([rowkey[i]==key for i in train]);counts=np.bincount(y[train[samekey]],minlength=3)
        byinput.append({'input_group':int(group),'target_class_counts':truth.tolist(),'target_components':r.component.iloc[ii].unique().tolist(),'exact_same_input_train_class_counts':cc[group].tolist(),
            'same_fine_key_train_class_counts':counts.tolist(),'full_model_scores':s.tolist(),'full_model_prediction':int(s.argmax()),'M_minus_S_margin':float(s[1]-s[2]),'key':key})
    save(DEST/'support_train_and_holdout_diagnosis.json',{'status':'completed','source_sha256':sha(__file__),'new_fits':0,'fine_training_cells':cells,'target_input_groups':byinput,
        'scope':'Diagnostic after all five fits; no re-selection or refit. Same fine behavior is not identical intent; no ground-truth policy/context recovered. Subject composition residual and six held-out S remain evidence limits.'})
    print(json.dumps({'training_cells':cells,'target_distinct_input_groups':len(byinput)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
