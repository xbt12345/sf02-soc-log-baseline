"""Compare structured-source and body-source windows; descriptive only."""
import json
from collections import Counter

import numpy as np
import pandas as pd

from run_v75 import ROOT, OUT, save, sha
from v106_frozen_wrapper_audit import DEST


def summarize(frame, key):
    rows=[]
    for _, z in frame.groupby(key,sort=False):
        z=z.sort_values('timestamp'); times=z.timestamp.to_numpy(dtype=float)
        ports=z.port.to_numpy(); begin=0; counts=Counter(); maximum=0
        for end,(time,port) in enumerate(zip(times,ports)):
            while times[begin]<time-300:
                old=ports[begin]
                if np.isfinite(old) and 0<=old<65536:
                    counts[old]-=1
                    if not counts[old]: del counts[old]
                begin+=1
            if np.isfinite(port) and 0<=port<65536: counts[port]+=1
            maximum=max(maximum,len(counts))
        m=int((z.truth==1).sum()); s=int((z.truth==2).sum())
        rows.append(('mixed' if m and s else 'M_only' if m else 'S_only',len(z),maximum))
    a=pd.DataFrame(rows,columns=['kind','rows','ports'])
    return {name:{'symbols':len(z),'rows':int(z.rows.sum()),
                  'at_least_10_ports':int((z.ports>=10).sum()),
                  'at_least_50_ports':int((z.ports>=50).sum())} for name,z in a.groupby('kind')}


def main():
    target=DEST/'source_context_correction.json'; assert not target.exists()
    files=[DEST/'paired_OOF_predictions.parquet',ROOT/'data/official/train.parquet',
           OUT/'rows.parquet',OUT/'projections.parquet']
    d=pd.read_parquet(files[0]); positions=d.row_position.to_numpy()
    a=pd.read_parquet(files[1],columns=['message_sanitized','src_ip','timestamp']).iloc[positions].reset_index(drop=True)
    a['body']=a.message_sanitized.str.extract(r'\bsrc\s+[^\s:]+:([^\s/]+)',expand=False)
    a['truth']=d.truth; a['root']=d.root
    pid=pd.read_parquet(files[2],columns=['projection_id']).projection_id.iloc[positions]
    facts=pd.read_parquet(files[3],columns=['facts']).facts
    lookup={int(i):json.loads(facts.iloc[int(i)]).get('dst_port_fixed') for i in pd.unique(pid)}
    a['port']=pd.to_numeric([lookup[int(i)] for i in pid],errors='coerce')
    assert a.body.notna().all() and np.isfinite(a.timestamp).all()
    roots={}
    for root in (637660,2868,27221,3929,46091,2300):
        z=a[a.root==root]
        roots[str(root)]={'rows':len(z),'structured_sources':int(z.src_ip.nunique()),
            'body_sources':int(z.body.nunique()),'largest_body_source_rows':int(z.body.value_counts().max())}
    result={'status':'descriptive_entity_and_window_correction_no_training',
        'source_sha256':sha(__file__),'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'root_identity_counts':roots,'windows_5min':{'structured_src_ip':summarize(a,'src_ip'),'message_body_src':summarize(a,'body')},
        'limitations':['Neither address is certified to be a stable real entity.',
            'Window includes current event for descriptive comparison, not a deployed past-only feature.',
            'Official timestamp semantics and its match to body clock remain unverified.',
            'Counts must not define M/S labels. No new classifier or threshold was fitted.']}
    save(target,result); print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
