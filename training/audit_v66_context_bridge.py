"""Challenge locally bijective endpoint mappings on other unlabelled official records."""
import re
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from train_v65_rank_heads import ROOT
from prepare_v61 import ASA,endpoint
from v61_common import save,sha


def main():
    out=ROOT/'artifacts/v66_issue_resolution_20260920'
    mapping=pd.read_parquet(out/'endpoint_mapping_private.parquet').set_index('row_position')
    diagnosis=pd.read_parquet(out/'row_diagnosis.parquet').set_index('row_position')
    hard=diagnosis.label.eq(2)&diagnosis.H1_pred.ne(2)&diagnosis.behavior.isin(['deny|blocked|tcp|outside|dmz','deny|blocked|udp|outside|dmz'])
    selected=mapping.loc[hard[hard].index];selected=selected[selected.src_mapping_eligible]
    pairs=selected[['collector','src_ip','body_src']].drop_duplicates()
    assert pairs.groupby('src_ip').body_src.nunique().max()==1
    lookup=pairs.set_index('src_ip').body_src.to_dict();positions=set(mapping.index)
    candidates=[];broken=set();offset=0
    for b in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=65536,columns=['event_id','src_ip','message_sanitized','product_name']):
        d=b.to_pandas();d['row_position']=np.arange(offset,offset+len(d));offset+=len(d)
        for row in d[d.src_ip.isin(lookup)].itertuples(index=False):
            msg=str(row.message_sanitized);same='USER-0010-0324' in msg;parsed=ASA.fullmatch(msg.strip())
            body=endpoint(parsed['src'])[1] if parsed else None
            consistent=body==lookup[row.src_ip] if parsed else None
            if same and parsed and not consistent:broken.add(row.src_ip)
            if row.row_position not in positions:
                candidates.append({'row_position':int(row.row_position),'event_id':row.event_id,
                   'structured_source':row.src_ip,'candidate_body_source':lookup[row.src_ip],
                   'same_collector_token':same,'ASA_parse_success':parsed is not None,
                   'body_source_agrees':consistent,'product_for_audit_only':row.product_name})
    d=pd.DataFrame(candidates,columns=['row_position','event_id','structured_source','candidate_body_source','same_collector_token','ASA_parse_success','body_source_agrees','product_for_audit_only'])
    d['mapping_invalidated_by_other_ASA']=d.structured_source.isin(broken)
    d.to_parquet(out/'bridge_candidates_private.parquet',index=False)
    eligible=d[d.same_collector_token&~d.ASA_parse_success&~d.mapping_invalidated_by_other_ASA]
    report={'fit_locally_bijective_failed_source_pairs':len(pairs),
      'pairs_invalidated_on_other_same_collector_ASA':len(broken),
      'outside_fit_structured_matches':len(d),'outside_fit_same_collector_rows':int(d.same_collector_token.sum()),
      'same_collector_non_ASA_candidates_after_mapping_challenge':len(eligible),
      'cross_collector_candidates_unqualified':int((~d.same_collector_token).sum()),
      'source_sha256':sha(__file__),
      'scope':'No context labels read, no candidates admitted to training. Local bijection is challenged on other ASA bodies; even survivors require provenance before cross-collector/physical-entity inference.'}
    save(out/'context_bridge.json',report);print(report)


if __name__=='__main__':main()
