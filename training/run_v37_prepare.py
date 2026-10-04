"""Full official-row paired B2 preparation; no labels are repaired or removed."""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,struct,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soc_v3_prepare as base
import v331_prepare as prior
import v36_representation as old
import v37_representation as rep
from run_v36_prepare import protocol,save

LABELS=['benign','malicious','suspicious']
CANONICAL=['row_position','event_id','label_index','b0','b0facts','b1','facts','route','supported','original_empty','product']
def packed_json(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(v):return hashlib.sha256(v.encode('utf-8')).digest()
def worker(payload):
    offset,rows,cached=payload;out=[]
    for j,row in enumerate(rows):
        raw=row['message_sanitized'] or '';v=rep.prepare_message(raw);rh=digest(raw);label=LABELS.index(row['label_binary'])
        if cached is not None:
            c=cached[j]
            assert c['event_id']==row['event_id'] and c['label_index']==label and c['raw_hash']==rh
            oldtext,oldfacts,legacy,fmt,template=c['b1'],c['facts'],c['b0'],c['format'],c['asa_template']
        else:
            c=old.prepare_message(raw);o=prior.prepare_record({'message_sanitized':raw})
            oldtext,oldfacts,legacy,fmt,template=c['b1'],packed_json(c['facts']),o['text'],o['format'],o['asa_template'] or ''
        out.append({'row_position':offset+j,'event_id':row['event_id'],'label_index':label,
                    'product':row['product_name'] or '<missing>','format':fmt,'asa_template':template,
                    'b0':oldtext,'b0facts':oldfacts,'b1':v['b1'],'facts':packed_json(v['facts']),
                    'legacy_group_text':legacy,'raw_hash':rh,'route':v['route'],'supported':v['supported'],
                    'original_empty':not raw.strip(),'no_observed_fact':not v['facts'],
                    'evidence':packed_json(v['evidence']),'quality':packed_json(v['quality'])})
    return out

def batches(train,previous,batchsize):
    oldit=pq.ParquetFile(previous/'prepared.parquet').iter_batches(batch_size=batchsize,columns=['event_id','label_index','raw_hash','b0','b1','facts','format','asa_template'],use_threads=False) if previous else None
    offset=0
    for batch in pq.ParquetFile(train).iter_batches(batch_size=batchsize,columns=['event_id','label_binary','product_name','message_sanitized'],use_threads=False):
        cached=next(oldit).to_pylist() if oldit else None
        rows=batch.to_pylist()
        yield offset,rows,cached
        offset+=len(rows)

def run(args):
    train=Path(args.train);out=Path(args.output_dir);previous=Path(args.previous) if args.previous else None
    assert base.file_hash(train)==base.EXPECTED_SHA
    assert not out.exists(),'New preparation directory required'
    if previous:
        receipt=json.loads((previous/'result.json').read_text(encoding='utf-8'))
        assert receipt['version']==old.VERSION and base.file_hash(previous/'prepared.parquet')==receipt['prepared_sha256']
        for name,sha in receipt['source_hashes'].items():assert base.file_hash(Path(__file__).parent/name)==sha
    out.mkdir(parents=True);start=time.perf_counter()
    schema=pa.schema([('row_position',pa.int32()),('event_id',pa.string()),('label_index',pa.uint8()),('product',pa.string()),
       ('format',pa.string()),('asa_template',pa.string()),('b0',pa.large_string()),('b0facts',pa.large_string()),
       ('b1',pa.large_string()),('facts',pa.large_string()),('legacy_group_text',pa.large_string()),('raw_hash',pa.binary(32)),
       ('route',pa.string()),('supported',pa.bool_()),('original_empty',pa.bool_()),('no_observed_fact',pa.bool_()),
       ('evidence',pa.large_string()),('quality',pa.large_string())])
    maps=[{} for _ in range(5)];parent=[];rowgroups=[];labels=[];products=[];templates=[]
    collisions=[collections.defaultdict(lambda:[0,0,0]) for _ in range(4)]
    coverage=collections.Counter();canonical=hashlib.sha256()
    def find(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    def accept(rows):
        for r in rows:
            gid=len(parent);parent.append(gid);rowgroups.append(gid)
            vals=[r['legacy_group_text'],r['b0'],packed_json([r['b0'],json.loads(r['b0facts'])]),r['b1'],packed_json([r['b1'],json.loads(r['facts'])])]
            if not r['original_empty']:
                for i,val in enumerate(vals):
                    key=digest(val) if val else b'empty_projection'
                    if key in maps[i]:
                        left,right=find(gid),find(maps[i][key]);parent[right]=left
                    else:maps[i][key]=gid
                for i,val in enumerate(vals[1:]):collisions[i][digest(val)][r['label_index']]+=1
            labels.append(r['label_index']);products.append(r['product']);templates.append(r['asa_template'])
            coverage[(r['route'],r['product'],r['label_index'],r['supported'])]+=1
            b=json.dumps([r[k] for k in CANONICAL],ensure_ascii=False,separators=(',',':')).encode('utf-8')
            canonical.update(struct.pack('<I',len(b)));canonical.update(b)
        writer.write_table(pa.Table.from_pylist(rows,schema=schema))
    with pq.ParquetWriter(out/'prepared.parquet',schema,compression='zstd') as writer:
        stream=iter(batches(train,previous,512))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            pending=[]
            for _ in range(args.workers*2):
                try:pending.append(ex.submit(worker,next(stream)))
                except StopIteration:break
            while pending:
                accept(pending.pop(0).result())
                try:pending.append(ex.submit(worker,next(stream)))
                except StopIteration:pass
                if len(labels)%65536==0:print(json.dumps({'stage':'prepare_v37','rows':len(labels),'seconds':round(time.perf_counter()-start)}),flush=True)
    n=len(labels);assert n==2056871
    groups=np.asarray([find(g) for g in rowgroups],dtype=np.int32);y=np.asarray(labels,dtype=np.uint8)
    pq.write_table(pa.table({'row_position':np.arange(n,dtype=np.int32),'union_group':groups,'label_index':y}),out/'groups.parquet',compression='zstd')
    protocol(out,y,groups,np.asarray(products,dtype=object),np.asarray(templates,dtype=object))
    decl=json.loads((out/'protocol.json').read_text(encoding='utf-8'));decl['version']='v37-shared-protocol-1.0';save(out/'protocol.json',decl)
    summary={}
    for name,table in zip(['old_text','old_B2','new_text','new_B2'],collisions):
        counts=np.array(list(table.values()),dtype=np.int64);mixed=(counts>0).sum(1)>1
        summary[name]={'unique_nonempty_origin_keys':len(table),'mixed_groups':int(mixed.sum()),'mixed_rows':int(counts[mixed].sum()),'minimum_empirical_errors':int((counts.sum(1)-counts.max(1)).sum())}
    save(out/'projection_collisions.json',summary)
    save(out/'coverage.json',[dict(route=r,product=p,label_index=l,supported=s,rows=n) for (r,p,l,s),n in sorted(coverage.items())])
    source_hashes={p.name:base.file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py')) if p.name in
       ['run_v37_prepare.py','v37_representation.py','v36_representation.py','v351_safeguards.py','v331_prepare.py','soc_v32_prepare.py','soc_v3_prepare.py','run_v36_prepare.py','v32_split.py']}
    roles=pq.read_table(out/'protocol.parquet')
    save(out/'result.json',{'version':rep.VERSION,'rows':n,'full_official_rows':True,'original_input_sha256':base.EXPECTED_SHA,
       'prepared_sha256':base.file_hash(out/'prepared.parquet'),'groups_sha256':base.file_hash(out/'groups.parquet'),
       'source_hashes':source_hashes,'canonical_input_views_sha256':canonical.hexdigest(),
       'canonical_groups_sha256':hashlib.sha256(groups.astype('<i4').tobytes()+y.tobytes()).hexdigest(),
       'canonical_roles_sha256':{name:hashlib.sha256(roles[name].to_numpy().astype('i1').tobytes()).hexdigest() for name in roles.column_names[1:]},
       'union_groups':int(len(np.unique(groups))),'row_class_counts':np.bincount(y,minlength=3).tolist(),
       'model_trained':False,'elapsed_seconds':time.perf_counter()-start})
    print(json.dumps({'stage':'prepared_v37','rows':n,'seconds':round(time.perf_counter()-start),'protocol_eligible':decl['all_tasks_eligible']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--previous');p.add_argument('--workers',type=int,default=2)
    run(p.parse_args())

