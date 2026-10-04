"""Independent predictions, every format/class and real marker perturbations."""
import json
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from run_v75 import ROOT,OUT,read,save,sha,adapter,matrices,features,load_sparse,metrics,NAMES
from v75_corrective import DEST,stable
from v75_views import view,byte_matrix
from v75_metadata import encode
from verify_v75_raw import rename_markers
from analyze_v75 import ids,floor


def main():
    r=pd.read_parquet(OUT/'rows.parquet');p=pd.read_parquet(DEST/'predictions.parquet')
    np.testing.assert_array_equal(r.row_position,p.row_position)
    rec=read(DEST/'complete.json')
    for n,h in rec['files_sha256'].items():assert sha(DEST/n)==h,n
    base=pd.read_parquet(OUT/'four_arm/predictions.parquet',columns=['D_pred'])
    for a in ['E','F']:
        mask=r.is_validation.to_numpy();cm=np.zeros((3,3),np.int64)
        np.add.at(cm,(r.label_index.to_numpy()[mask],p[a+'_pred'].to_numpy()[mask]),1)
        assert metrics(cm)==rec['curves'][-1]['validation'][a]
    cells=[]
    for route in sorted(r.route.unique()):
        for label in range(3):
            for role in ['fit','validation']:
                mask=(r.route==route)&(r.label_index==label)&(r.is_validation==(role=='validation'))
                cells.append({'route':route,'label':NAMES[label],'role':role,'rows':int(mask.sum()),
                    'D_errors':int(((base.D_pred!=r.label_index)&mask).sum()),
                    **{a+'_errors':int(((p[a+'_pred']!=r.label_index)&mask).sum()) for a in ['E','F']}})
    pd.DataFrame(cells).to_csv(DEST/'all_format_class_roles.csv',index=False)
    wanted=set()
    for _,z in r.groupby(['route','label_index']):
        if len(z)<=50:wanted.update(z.row_position.tolist())
        else:
            for hold in [False,True]:wanted.update(z.loc[z.is_validation.eq(hold),'row_position'].head(5).tolist())
    wanted.update([45738,45750,78222,81090]);wanted=sorted(wanted)
    raw={};ports={};offset=0
    for b in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized','src_port'],use_threads=False):
        for pos in wanted[np.searchsorted(wanted,offset):np.searchsorted(wanted,offset+len(b))]:
            raw[pos]=b.column(0)[pos-offset].as_py();ports[pos]=b.column(1)[pos-offset].as_py()
        offset+=len(b)
    v=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');m=matrices();m['new_text']=load_sparse(DEST/'text')
    def xof(messages):
        parsed=[v.prepare_record({'message_sanitized':s}) for s in messages]
        texts=[stable(view(s)[0]) for s in messages]
        facts=normalize(enc.transform([p['facts'] for p in parsed]).astype(np.float32),norm='l2',copy=False)
        extra,_,_=encode([ports[i] for i in wanted],[p['facts'].get('src_port_fixed',65536) for p in parsed])
        return sparse.hstack([byte_matrix(texts),facts,extra],format='csr',dtype=np.float32)
    messages=[raw[i] or '' for i in wanted];x=xof(messages);changed=xof([rename_markers(s) for s in messages])
    cached=features(r,wanted,m,'new');delta=x-cached
    assert not delta.nnz or abs(delta.data).max()<1e-7
    diff=x-changed;marker={'feature_different_rows':int((diff.getnnz(axis=1)>0).sum()),'models':{}}
    for a in ['E','F']:
        model=joblib.load(DEST/(a+'.joblib'));pr=model.predict_proba(x);pr2=model.predict_proba(changed)
        np.testing.assert_array_equal(pr.argmax(1),p.loc[wanted,a+'_pred'])
        marker['models'][a]={'label_changes':int((pr.argmax(1)!=pr2.argmax(1)).sum()),'max_probability_change':float(abs(pr-pr2).max())}
    fit=~r.is_validation.to_numpy();y=r.label_index.to_numpy()
    fits={}
    for a in ['E','F']:
        cm=np.zeros((3,3),np.int64);np.add.at(cm,(y[fit],p[a+'_pred'].to_numpy()[fit]),1);fits[a]=metrics(cm)
    fact_ids=ids('facts');text_ids=ids('corrective/text')
    meta=np.load(OUT/'metadata_code.npy').astype(np.uint64)
    key_bytes=np.column_stack([fact_ids[r.projection_id],text_ids[r.new_text_id],meta]).astype('<u8')
    keys=pd.factorize(key_bytes.view('S24').reshape(-1),sort=False)[0]
    floors={'all_train':floor(keys,y,np.ones(len(r),bool)),'fit':floor(keys,y,fit)}
    result={'all_raw_feature_replays_match':True,'raw_rows':len(wanted),'formats':int(r.loc[wanted,'route'].nunique()),
        'all_102_format_class_role_cells_reported':len(cells)==102,'marker_invariance':marker,'fit_metrics':fits,'representation_floors':floors,
        'source_sha256':sha(__file__),'files_sha256':{'all_format_class_roles.csv':sha(DEST/'all_format_class_roles.csv')},
        'quality_promoted':False,'answers_read':False}
    save(DEST/'audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
