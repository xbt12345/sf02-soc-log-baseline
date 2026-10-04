"""Locate residual marker-instability; raw snippets stay local in evidence."""
import collections
import difflib
import json
import sys
import pandas as pd
import pyarrow.parquet as pq
from run_v75 import ROOT,OUT,adapter,save
from v75_corrective import stable
from v75_views import view
from verify_v75_raw import rename_markers
if '--valid' in sys.argv:
    from verify_v75_valid_identity import rename as rename_markers

r=pd.read_parquet(OUT/'rows.parquet');wanted=set()
for _,z in r.groupby(['route','label_index']):
    if len(z)<=50:wanted.update(z.row_position.tolist())
    else:
        for hold in [False,True]:wanted.update(z.loc[z.is_validation.eq(hold),'row_position'].head(5).tolist())
wanted.update([45738,45750,78222,81090]);v=adapter();offset=0;result=[];counts=collections.Counter()
for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized'],use_threads=False):
    for pos in sorted(wanted.intersection(range(offset,offset+len(batch)))):
        s=batch.column(0)[pos-offset].as_py() or '';ss=rename_markers(s)
        a=stable(view(s)[0]);b=stable(view(ss)[0]);f=v.prepare_record({'message_sanitized':s})['facts'];g=v.prepare_record({'message_sanitized':ss})['facts']
        keys=[k for k in set(f)|set(g) if f.get(k)!=g.get(k)]
        if a!=b or keys:
            counts[(str(r.route.iat[pos]),'text' if a!=b else 'facts')]+=1
            changes=[]
            for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
                if tag!='equal':changes.append({'before':a[max(0,i-24):j+24],'after':b[max(0,k-24):l+24]})
            result.append({'row_position':pos,'route':r.route.iat[pos],'text_changed':a!=b,'changed_fact_keys':keys,'changes':changes[:4]})
    offset+=len(batch)
save(OUT/('corrective/valid_marker_differences.json' if '--valid' in sys.argv else 'corrective/marker_differences.json'),result)
print(json.dumps({'counts':[{'route':k[0],'kind':k[1],'rows':v} for k,v in counts.items()],'examples':result[:8]},ensure_ascii=False,indent=2))
