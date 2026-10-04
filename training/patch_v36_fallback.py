"""Incremental v1.3 fallback repair, then rebuild shared roles from all rows."""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soc_v3_prepare as base
import v36_representation as rep
from run_v36_prepare import save,protocol


def run(args):
    old=Path(args.previous);out=Path(args.output_dir);started=time.perf_counter()
    before=json.loads((old/'result.json').read_text(encoding='utf-8'))
    assert before['version']=='v36-representation-1.2'
    assert base.file_hash(old/'prepared.parquet')==before['prepared_sha256']
    assert base.file_hash(old/'groups.parquet')==before['groups_sha256']
    assert base.file_hash(args.train)==base.EXPECTED_SHA
    if out.exists():raise FileExistsError(out)
    out.mkdir();reader=pq.ParquetFile(old/'prepared.parquet')
    writer=pq.ParquetWriter(out/'prepared.parquet',reader.schema_arrow,compression='zstd')
    original=pq.ParquetFile(args.train).iter_batches(batch_size=2048,columns=['event_id','message_sanitized'],use_threads=False)
    old_groups=pq.read_table(old/'groups.parquet')['old_group'].to_numpy()
    parent=np.arange(int(old_groups.max())+1,dtype=np.int32);lookup={};offset=0
    labels=[];products=[];templates=[];coverage=collections.Counter();changes=collections.Counter()
    def find(g):
        g=int(g)
        while parent[g]!=g:parent[g]=parent[int(parent[g])];g=int(parent[g])
        return g
    for b in reader.iter_batches(batch_size=2048,use_threads=False):
        rows=b.to_pylist();raw=next(original).to_pylist()
        for j,(r,o) in enumerate(zip(rows,raw)):
            pos=offset+j;assert r['row_position']==pos and r['event_id']==o['event_id']
            if r['route']=='legacy_fallback' and not r['original_empty']:
                new=rep.prepare_record({'message_sanitized':o['message_sanitized']})
                changed=r['b1']!=new['b1'] or json.loads(r['facts'])!=new['facts']
                changes[(r['product'],r['label_index'],new['route'])]+=int(changed)
                r.update(b1=new['b1'],facts=json.dumps(new['facts'],sort_keys=True,ensure_ascii=False,separators=(',',':')),
                         route=new['route'],supported=new['supported'],no_observed_fact=new['no_observed_fact'],
                         evidence=json.dumps(new['evidence'],ensure_ascii=False,separators=(',',':')))
            gid=int(old_groups[pos])
            if not r['original_empty']:
                key=hashlib.sha256(r['b1'].encode()).digest()
                if key not in lookup:lookup[key]=gid
                left,right=find(gid),find(lookup[key]);parent[right]=left
            labels.append(r['label_index']);products.append(r['product']);templates.append(r['asa_template'])
            coverage[(r['route'],r['product'],r['label_index'])]+=1
        writer.write_table(pa.Table.from_pylist(rows,schema=reader.schema_arrow));offset+=len(rows)
        if offset%131072==0:print(json.dumps({'stage':'fallback_patch','rows':offset,'seconds':round(time.perf_counter()-started)}),flush=True)
    writer.close();assert offset==2056871
    roots=np.array([find(i) for i in range(len(parent))],dtype=np.int32);groups=roots[old_groups];y=np.asarray(labels,dtype=np.uint8)
    pq.write_table(pa.table({'row_position':np.arange(offset,dtype=np.int32),'old_group':old_groups,'union_group':groups,'label_index':y}),out/'groups.parquet',compression='zstd')
    protocol(out,y,groups,np.asarray(products,dtype=object),np.asarray(templates,dtype=object))
    save(out/'coverage.json',[{'route':r,'product':p,'label_index':l,'rows':n} for (r,p,l),n in coverage.items()])
    sources={p.name:base.file_hash(p) for p in [Path(__file__),Path(rep.__file__),Path(__file__).with_name('v331_prepare.py')]}
    result=dict(before,version=rep.VERSION,source_hashes=sources,prepared_sha256=base.file_hash(out/'prepared.parquet'),
         groups_sha256=base.file_hash(out/'groups.parquet'),union_components=len(np.unique(roots)),
         B1_and_joint_roles_unchanged=False,patch_input_sha256=before['prepared_sha256'],
         patch_only_columns=['b1','facts','route','supported','no_observed_fact','evidence'],elapsed_seconds=time.perf_counter()-started,
         fallback_changes=[{'source':s,'label_index':y,'route':r,'rows':n} for (s,y,r),n in changes.items() if n])
    save(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--train',required=True)
    run(p.parse_args())
