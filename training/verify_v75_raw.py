"""Real raw-event replay, rare-slice coverage, and deidentification invariance."""
import hashlib
import json
import re
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from run_v75 import ROOT,OUT,read,save,sha,adapter,matrices,features
from v75_views import view,byte_matrix
from v75_metadata import encode as metadata_encode


def rename_markers(raw):
    """Bijective numeric offset of deidentification marker runs, including embedded runs."""
    return re.sub(r'(USER|HOST|CRED|ORG)-([0-9]+(?:-[0-9]+)*)',
        lambda m:m[1]+'-'+'-'.join(str(int(x)+70000000) for x in m[2].split('-')),raw)


def main():
    r=pd.read_parquet(OUT/'rows.parquet');wanted=set()
    for _,z in r.groupby(['route','label_index']):
        if len(z)<=50:wanted.update(z.row_position.tolist())
        else:
            for hold in [False,True]:wanted.update(z.loc[z.is_validation.eq(hold),'row_position'].head(5).tolist())
    wanted.update([45738,45750,78222,81090]);wanted=sorted(wanted)
    raw={};record_ports={};offset=0
    for b in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized','src_port'],use_threads=False):
        for pos in wanted[np.searchsorted(wanted,offset):np.searchsorted(wanted,offset+len(b))]:
            raw[pos]=b.column(0)[pos-offset].as_py();record_ports[pos]=b.column(1)[pos-offset].as_py()
        offset+=len(b)
    assert len(raw)==len(wanted)
    ledger=pd.read_parquet(OUT/'raw_ledger.parquet',filters=[('row_position','in',wanted)]).set_index('row_position')
    dictionary=pd.read_parquet(OUT/'text_dictionary.parquet',filters=[('text_id','in',r.loc[wanted,'new_text_id'].tolist())]).set_index('text_id')
    for pos in wanted:
        text,a=view(raw[pos]);z=ledger.loc[pos]
        assert a['is_null']==z.raw_is_null
        assert a['sha256']==z.raw_sha256
        assert json.loads(z.spans_json)==[list(s) for s in a['spans']]
        assert text==dictionary.loc[int(r.new_text_id.iat[pos]),'text']
    v=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');m=matrices()
    messages=[raw[p] or '' for p in wanted]
    changed=[rename_markers(s) for s in messages]
    def actual_x(messages,kind):
        parsed=[v.prepare_record({'message_sanitized':s}) for s in messages]
        text=[p['text'] for p in parsed] if kind=='old' else [view(s)[0] for s in messages]
        fx=normalize(enc.transform([p['facts'] for p in parsed]).astype(np.float32),norm='l2',copy=False)
        blocks=[byte_matrix(text),fx]
        if kind=='new':
            extra,_,_=metadata_encode([record_ports[i] for i in wanted],[p['facts'].get('src_port_fixed',65536) for p in parsed])
            blocks.append(extra)
        return sparse.hstack(blocks,format='csr',dtype=np.float32)
    smoke={}
    cfg=read(OUT/'configuration.json')
    for kind in ['old','new']:
        x=actual_x(messages,kind);cached=features(r,wanted,m,kind)
        delta=x-cached
        assert not delta.nnz or abs(delta.data).max()<1e-7
        changed_x=actual_x(changed,kind)
        for arm,(k,_) in cfg['arms'].items():
            if k!=kind:continue
            model=joblib.load(OUT/'four_arm'/(arm+'.joblib'))
            a=model.predict_proba(x);b=model.predict_proba(changed_x)
            smoke[arm]={'rows':len(wanted),'changed_labels':int((a.argmax(1)!=b.argmax(1)).sum()),
                'maximum_probability_change':float(abs(a-b).max()),
                'probability_changed_rows_1e_6':int((abs(a-b).max(1)>1e-6).sum())}
    result={'raw_replay_rows':len(wanted),'rare_route_class_cells_up_to_50_all_rows_included':True,
        'raw_span_reconstruction_and_cache_match':True,'exact_frozen_parser_feature_replay_match':True,
        'all_formats_represented':int(r.loc[wanted,'route'].nunique()),'marker_renaming':smoke,
        'interpretation':'Consistent deidentification-marker renaming diagnostic; not all possible identities or behavioral invariance. Nonzero effects must be retained as unresolved, not hidden.',
        'model_quality_accepted':False,'source_sha256':sha(__file__)}
    save(OUT/'raw_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
