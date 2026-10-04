"""Check whether discrepant structured/body endpoint symbols admit a safe bijection."""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from train_v65_rank_heads import ROOT,DATA
from v61_common import save,sha


def bijection_mask(frame,left,right):
    pairs=frame.drop_duplicates(['collector',left,right])
    f=pairs.groupby(['collector',left])[right].nunique()
    r=pairs.groupby(['collector',right])[left].nunique()
    a=pd.MultiIndex.from_frame(frame[['collector',left]]).map(f)
    b=pd.MultiIndex.from_frame(frame[['collector',right]]).map(r)
    return (np.asarray(a)==1)&(np.asarray(b)==1),f,r


def main():
    out=ROOT/'artifacts/v66_issue_resolution_20260920'
    p=pd.read_parquet(DATA/'records.parquet',columns=['row_position','role'])
    wanted=set(p.loc[p.role.eq('fit'),'row_position'])
    m=pd.read_parquet(DATA/'private_join_audit.parquet');m.index=p.row_position
    parts=[];offset=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=65536,columns=['src_ip','dst_ip']):
        d=batch.to_pandas();d.index=np.arange(offset,offset+len(d));offset+=len(d);parts.append(d[d.index.isin(wanted)])
    d=pd.concat(parts);d['body_src']=m.loc[d.index,'source'];d['body_dst']=m.loc[d.index,'destination'];d['collector']=m.loc[d.index,'namespace']
    diagnosis=pd.read_parquet(out/'row_diagnosis.parquet').set_index('row_position')
    failed=diagnosis.label.eq(2)&diagnosis.H1_pred.ne(2)&diagnosis.behavior.isin(['deny|blocked|tcp|outside|dmz','deny|blocked|udp|outside|dmz'])
    report={}
    for side in ['src','dst']:
        subset=d[d[side+'_ip'].fillna('').ne('')];safe,forward,reverse=bijection_mask(subset,side+'_ip','body_'+side)
        d[side+'_mapping_eligible']=False;d.loc[subset.index,side+'_mapping_eligible']=safe
        report[side]={'structured_symbols':len(forward),'body_symbols':len(reverse),
            'structured_maps_many':int(forward.gt(1).sum()),'body_maps_many':int(reverse.gt(1).sum()),
            'max_bodies_per_structured':int(forward.max()),'max_structured_per_body':int(reverse.max()),
            'all_bijective':bool(forward.le(1).all() and reverse.le(1).all()),
            'eligible_rows':int(safe.sum()),'eligible_failed_hard_S_rows':int(d.loc[failed.index[failed],side+'_mapping_eligible'].sum())}
    d.rename_axis('row_position').reset_index().to_parquet(out/'endpoint_mapping_private.parquet',index=False)
    save(out/'endpoint_mapping.json',{'result':report,'source_sha256':sha(__file__),
       'scope':'Label-free per-collector symbol mapping audit on current fit records. Many-to-many mappings are quarantined. Even bijection alone does not establish physical entity identity or temporal sessions.'})
    print(report)


if __name__=='__main__':main()
