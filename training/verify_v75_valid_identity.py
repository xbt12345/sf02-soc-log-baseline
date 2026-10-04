"""Corrected semantic-preserving identity probe; retains month/day suffixes."""
import json
import re
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize
from run_v75 import ROOT,OUT,adapter,save,sha
from v75_corrective import stable
from v75_views import view,byte_matrix
from v75_metadata import encode


def rename(raw):
    # Prefix one digit to the identity component, preserving trailing date
    # month/day and compound-token suffixes verbatim. Injective for digit strings.
    return re.sub(r'(USER|HOST|CRED|ORG)-([0-9]+)',lambda m:m[1]+'-9'+m[2],raw)


def main():
    assert rename('USER-2034-08-21T23:22:41Z')=='USER-92034-08-21T23:22:41Z'
    assert rename('xCRED-004-12')=='xCRED-9004-12'
    r=pd.read_parquet(OUT/'rows.parquet');wanted=set()
    for _,z in r.groupby(['route','label_index']):
        if len(z)<=50:wanted.update(z.row_position.tolist())
        else:
            for hold in [False,True]:wanted.update(z.loc[z.is_validation.eq(hold),'row_position'].head(5).tolist())
    wanted.update([45738,45750,78222,81090]);wanted=sorted(wanted);messages={};ports={};offset=0
    for b in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized','src_port'],use_threads=False):
        for pos in wanted[np.searchsorted(wanted,offset):np.searchsorted(wanted,offset+len(b))]:
            messages[pos]=b.column(0)[pos-offset].as_py() or '';ports[pos]=b.column(1)[pos-offset].as_py()
        offset+=len(b)
    v=adapter();enc=joblib.load(OUT/'facts_encoder.joblib')
    def xof(msgs,corrected):
        parsed=[v.prepare_record({'message_sanitized':s}) for s in msgs]
        tx=[view(s)[0] for s in msgs]
        if corrected:tx=[stable(s) for s in tx]
        facts=normalize(enc.transform([p['facts'] for p in parsed]).astype(np.float32),norm='l2',copy=False)
        extra,_,_=encode([ports[i] for i in wanted],[p['facts'].get('src_port_fixed',65536) for p in parsed])
        return sparse.hstack([byte_matrix(tx),facts,extra],format='csr',dtype=np.float32)
    original=[messages[i] for i in wanted];changed=[rename(s) for s in original];results={}
    groups=[(False,{'D':OUT/'four_arm/D.joblib','D_full':OUT/'full/FULL.joblib'}),
            (True,{'E':OUT/'corrective/E.joblib','F':OUT/'corrective/F.joblib','F_full':OUT/'F_diagnostic/model.joblib'})]
    for corrected,models in groups:
        x=xof(original,corrected);z=xof(changed,corrected);diff=x-z
        for name,path in models.items():
            model=joblib.load(path);a=model.predict_proba(x);b=model.predict_proba(z)
            results[name]={'rows':len(wanted),'feature_changed_rows':int((diff.getnnz(axis=1)>0).sum()),
                'prediction_flips':int((a.argmax(1)!=b.argmax(1)).sum()),'max_probability_delta':float(abs(a-b).max()),
                'model_sha256':sha(path)}
    result={'supersedes':'The prior numeric-run perturbation also modified ISO month/day suffixes, so its flip counts are invalid as pure identity-dependence evidence.',
        'preserves_calendar_and_compound_suffixes':True,'models':results,'source_sha256':sha(__file__),
        'scope':'359 real rows, 17 formats including all <=50 rare route/class rows. Only explicit synthetic identifier renaming, not arbitrary OOD invariance.'}
    save(OUT/'valid_identity_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
