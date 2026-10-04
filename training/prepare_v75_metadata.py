"""Label-free record source-port feature addition, registered before any fit."""
import collections
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from run_v75 import ROOT,OUT,SparseWriter,sha,save
from v75_metadata import encode


def main():
    assert OUT.exists() and not (OUT/'four_arm').exists()
    if (OUT/'metadata.json').exists():raise FileExistsError('metadata already prepared')
    old=ROOT/'artifacts/v39_local_r2_20260913/prepared'
    r=pd.read_parquet(old/'rows.parquet',columns=['projection_id','route','label_index'])
    p=pd.read_parquet(old/'projections.parquet',columns=['facts'])
    msg=np.array([json.loads(s).get('src_port_fixed',65536) for s in p.facts],np.int32)
    w=SparseWriter(OUT/'metadata',18);codes=np.empty(len(r),np.uint32);offset=0
    states=collections.Counter();totals=collections.Counter();byroute=collections.defaultdict(collections.Counter)
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=4096,columns=['src_port'],use_threads=False):
        raw=batch.column(0).to_pylist();z=r.iloc[offset:offset+len(batch)]
        x,key,a=encode(raw,msg[z.projection_id]);w.append(x);codes[offset:offset+len(batch)]=key
        states.update(a.pop('states'));totals.update(a)
        observed=key>0;message_missing=msg[z.projection_id]==65536
        for route in z.route.unique():
            mask=z.route.eq(route).to_numpy()
            byroute[route].update({'rows':int(mask.sum()),'record_port_observed':int((mask&observed).sum()),
                'message_unobserved_record_observed':int((mask&observed&message_missing).sum()),
                'conflicting_observations':int((mask&observed&((key-1)%2==1)).sum())})
        offset+=len(batch)
    w.close();np.save(OUT/'metadata_code.npy',codes);assert offset==len(r)
    receipt={'rows':offset,'new_fits':0,'answers_read':False,'field':'independent record src_port, not assumed identical to message source-port role',
        'counts':dict(totals),'states':dict(states),'by_route':dict(byroute),
        'old_message_ports_unchanged_by_v48_native_action_patch':True,
        'source_sha256':{n:sha(ROOT/'training'/n) for n in ['prepare_v75_metadata.py','v75_metadata.py']}}
    save(OUT/'metadata_audit.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
