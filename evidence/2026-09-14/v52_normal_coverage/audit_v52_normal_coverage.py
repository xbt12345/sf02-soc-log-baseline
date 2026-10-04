"""Broad lexical inventory ONLY: candidates are not verified flows or labels."""
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/2026-09-14/v52_normal_coverage'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    r=pd.read_parquet(ROOT/'artifacts/v48_information_repair_r2_20260914/rows.parquet')
    r=r[r.label_index==0].set_index('row_position')
    original=ROOT/'data/official/train.parquet';offset=0;records=[];examples={};raw=[]
    pat1=r'(?i)\b(tcp|udp|icmp|icmp6)\b'
    pat2=r'(?i)\b(allow|allowed|deny|denied|accept|accepted|drop|dropped|block|blocked|connection|flow)\b'
    for batch in pq.ParquetFile(original).iter_batches(batch_size=16384,columns=['message_sanitized']):
        col=batch.column(0)
        matches=pc.fill_null(pc.and_(pc.match_substring_regex(col,pat1),pc.match_substring_regex(col,pat2)),False)
        indices=pc.indices_nonzero(matches).to_pylist()
        for i in indices:
            pos=offset+i
            if pos not in r.index:continue
            row=r.loc[pos];s=col[i].as_py();records.append({'row_position':pos,'route':row.route,'body_group':row.body_group,'projection_id':int(row.projection_id)})
            group=(row.route,row.body_group)
            if group not in examples and sum(k[0]==row.route for k in examples)<6:
                spans=[{'start':m.start(),'end':m.end(),'context':s[max(0,m.start()-90):m.end()+140]} for m in re.finditer(pat1,s)][:6]
                examples[group]={'row_position':pos,'route':row.route,'body_group':str(row.body_group),'raw_length':len(s),'protocol_contexts':spans}
                raw.append({'row_position':pos,'route':row.route,'message_sanitized':s})
        offset+=batch.num_rows
    d=pd.DataFrame(records,columns=['row_position','route','body_group','projection_id']);d.to_parquet(OUT/'candidates.parquet',index=False)
    p=pd.read_parquet(ROOT/'artifacts/v48_information_repair_r2_20260914/projections.parquet')
    for ex in examples.values():
        pid=int(r.loc[ex['row_position']].projection_id);ex['retained_facts']=json.loads(p.iloc[pid].facts);ex['retained_text_preview']=p.iloc[pid].text[:700]
    result={'scope':'Only allowed development benign rows. Protocol word AND action/flow word is candidate discovery, not proof of genuine network telemetry or parser failure. No training or evaluation split changes.',
        'allowed_normal_rows':len(r),'candidate_rows':len(d),'by_route':{str(k):{'rows':len(g),'body_groups':int(g.body_group.nunique())} for k,g in d.groupby('route')},'examples':list(examples.values())}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(raw).to_parquet(OUT/'raw_examples.parquet',index=False)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (OUT/'receipt.json').write_text(json.dumps({'source_data_sha256':sha(original),'files':{x.name:sha(x) for x in OUT.iterdir() if x.is_file()}},indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['allowed_normal_rows','candidate_rows','by_route']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
