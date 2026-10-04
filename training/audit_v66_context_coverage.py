"""Find potential additional official context without using its labels as features."""
import re
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from train_v65_rank_heads import ROOT,DATA,load_fit
from v61_common import save,read,sha


def main():
    out=ROOT/'artifacts/v66_issue_resolution_20260920';frame,_=load_fit()
    positions=pd.read_parquet(DATA/'records.parquet',columns=['row_position']).row_position
    metadata=pd.read_parquet(DATA/'private_join_audit.parquet');metadata.index=positions
    diagnosis=pd.read_parquet(out/'row_diagnosis.parquet')
    failed=diagnosis[diagnosis.label.eq(2)&diagnosis.H1_pred.ne(2)&diagnosis.behavior.isin(['deny|blocked|tcp|outside|dmz','deny|blocked|udp|outside|dmz'])]
    own=metadata.loc[failed.row_position];sources=set(own.source);fit_positions=set(frame.row_position)
    cache_positions=set(positions);joined=[];offset=0;collector='USER-0010-0324';consistency=[]
    ip_pattern=re.compile(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])')
    structured_matches=0
    for b in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=65536,columns=['event_id','src_ip','dst_ip','message_sanitized']):
        d=b.to_pandas();d['row_position']=np.arange(offset,offset+len(d));offset+=len(d)
        structured_matches+=int(d.src_ip.isin(sources).sum())
        for row in d[d.row_position.isin(fit_positions)].itertuples(index=False):
            m=metadata.loc[row.row_position]
            consistency.append({'row_position':int(row.row_position),
              'structured_src_nonempty':bool(row.src_ip),'structured_dst_nonempty':bool(row.dst_ip),
              'structured_src_matches_body':row.src_ip==m.source,'structured_dst_matches_body':row.dst_ip==m.destination})
        # Source columns are not presumed interchangeable with endpoints inside the message.
        for row in d[d.message_sanitized.str.contains(collector,regex=False,na=False)].itertuples(index=False):
            message=str(row.message_sanitized);mentions=set(ip_pattern.findall(message))&sources
            if not mentions:continue
            is_asa=bool(re.search(r'\bDeny\s+(?:tcp|udp|icmp6?)\s+src\s+',message,re.I))
            joined.append({'row_position':int(row.row_position),'event_id':row.event_id,'source':row.src_ip,
              'body_target_source_symbols':sorted(mentions),
              'same_collector_token':collector in message,'asa_deny_grammar':is_asa,
              'in_current_fit':row.row_position in fit_positions,'in_v61_cache':row.row_position in cache_positions,
              'message_sha256':__import__('hashlib').sha256(message.encode()).hexdigest(),
              'new_message_for_inspection':message if not is_asa and collector in message else ''})
    d=pd.DataFrame(joined,columns=['row_position','event_id','source','body_target_source_symbols','same_collector_token','asa_deny_grammar','in_current_fit','in_v61_cache','message_sha256','new_message_for_inspection'])
    d.to_parquet(out/'context_candidates_private.parquet',index=False)
    candidates=d[~d.asa_deny_grammar & d.same_collector_token]
    consistency=pd.DataFrame(consistency).sort_values('row_position')
    assert np.array_equal(consistency.row_position,frame.row_position)
    consistency['label']=frame.label.to_numpy();consistency.to_parquet(out/'structured_body_consistency.parquet',index=False)
    summary={'failed_S_source_symbols':len(sources),'matching_structured_source_rows':structured_matches,
      'matching_same_collector_body_mention_rows':len(d),
      'same_collector_non_ASA_rows':len(candidates),'same_collector_non_ASA_source_symbols':len(set(s for ls in candidates.body_target_source_symbols for s in ls)),
      'structured_body_consistency_by_class':{str(c):{'rows':len(g),**{col:int(g[col].sum()) for col in g if col not in ['row_position','label']}} for c,g in consistency.groupby('label')},
      'same_collector_ASA_outside_current_fit':int((d.asa_deny_grammar&d.same_collector_token&~d.in_current_fit).sum()),
      'scope':'Read-only official train context scan; no new context labels read. Supplied structured-source equality and same-collector raw IP mentions checked separately. A raw IP mention is not necessarily a source endpoint. Collector token presence is a candidate filter, not verified physical host/session linkage. Other-partition rows are not admitted to model context.',
      'source_sha256':sha(__file__)}
    save(out/'context_coverage.json',summary);print(summary)


if __name__=='__main__':main()
