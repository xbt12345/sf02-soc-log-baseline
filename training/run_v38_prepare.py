"""Verify full official rows, derive compact v3.8 projections and inner roles."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import v38_representation as rep
from v351_safeguards import assert_group_isolation
from audit_v37_prepared import variants

EXPECTED='6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def save(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def full_source_audit(row):
    # Series.product is a method, so attribute access does not read the column.
    return row['product'] in ('Barracuda WAF','Duo')

def main(a):
    start=time.perf_counter();old=Path(a.previous);out=Path(a.output);train=Path(a.train)
    if out.exists():raise FileExistsError('New output required: '+str(out))
    receipt=json.loads((old/'result.json').read_text(encoding='utf-8'))
    assert sha(train)==EXPECTED
    assert sha(old/'prepared.parquet')==receipt['prepared_sha256']
    assert sha(old/'groups.parquet')==receipt['groups_sha256']
    for name,digest in receipt['source_hashes'].items():assert sha(Path(__file__).parent/name)==digest,name
    d=pq.read_table(old/'prepared.parquet',columns=['row_position','event_id','label_index','product','route','b1','facts','raw_hash','original_empty']).to_pandas()
    proto=pq.read_table(old/'protocol.parquet').to_pandas()
    for k in proto.columns[1:]:assert hashlib.sha256(proto[k].to_numpy(dtype='i1').tobytes()).hexdigest()==receipt['canonical_roles_sha256'][k]
    out.mkdir(parents=True)
    codes,keys=pd.factorize(pd.MultiIndex.from_frame(d[['b1','facts','route']]),sort=False)
    records=[]
    for i,(text,facts,route) in enumerate(keys):
        p=rep.project(text,facts,route)
        rec={'projection_id':i,'text':p['text'],'base_facts':rep.canonical(p['facts']),'ports':rep.canonical(p['ports']),
             'removed_audit':rep.canonical(p['removed_audit'])}
        for view in rep.VIEWS:rec[view]=rep.canonical(rep.view_record(p,view)['facts'])
        records.append(rec)
    pq.write_table(pa.Table.from_pylist(records),out/'projections.parquet',compression='zstd')
    print(json.dumps({'stage':'unique_projection','rows':len(d),'projections':len(records)}),flush=True)
    g=pq.read_table(old/'groups.parquet')['union_group'].to_numpy()
    # Existing joint groups remain conservative identity/legacy-template groups.
    # Shared semantic atoms are intentionally reusable across groups; equality
    # of a coarse feature is not newly claimed to imply one incident.
    rows=d[['row_position','event_id','label_index','product','route','original_empty']].copy()
    rows['union_group']=g;rows['projection_id']=codes.astype(np.int32)
    for k in proto.columns[1:]:rows['outer_'+k]=proto[k].to_numpy(dtype=np.int8)
    roles=proto['known_dev'].to_numpy(dtype=np.int8).copy();roles[roles==3]=-1
    # Internal source stress chosen before fitting; only allowed known_dev rows.
    held=(roles>=0)&d['product'].isin(['Duo','Barracuda WAF']).to_numpy()
    held_groups=np.unique(g[held]);roles[(roles>=0)&np.isin(g,held_groups)]=2
    assert_group_isolation(g,roles)
    assert np.all(roles[proto.known_dev.to_numpy()==3]==-1)
    rows['inner_role']=roles
    support=rows[roles>=0].groupby(['inner_role','product','label_index']).agg(rows=('union_group','size'),groups=('union_group','nunique')).reset_index()
    summary=rows[roles>=0].groupby(['inner_role','label_index']).agg(rows=('union_group','size'),groups=('union_group','nunique')).reset_index()
    for role in (0,1,2):assert set(rows.loc[roles==role,'label_index'])=={0,1,2}
    pq.write_table(pa.Table.from_pandas(rows,preserve_index=False),out/'rows.parquet',compression='zstd')
    save(out/'support.json',{'protocol':'known_dev_allowed_inner_with_Duo_WAF_source_holdout','roles':{'-1':'excluded prior outer evaluation','0':'fit','1':'selection','2':'internal audit plus held source groups'},
        'source_role_support':support.to_dict('records'),'global_support':summary.to_dict('records'),
        'held_source_groups':len(held_groups),'outer_results_already_viewed':True,'independent_external_validation':False,
        'limitation':'Frozen conservative groups retained, no giant group split. Semantic equality across groups is not asserted absent; label conflicts audited separately.'})
    port_counts=collections.Counter();position=0;cases=[];seen=collections.Counter();checks=collections.Counter();failures=[]
    for b in pq.ParquetFile(train).iter_batches(batch_size=2048,columns=['event_id','label_binary','message_sanitized'],use_threads=False):
        for rawrow in b.to_pylist():
            oldrow=d.iloc[position];raw=rawrow['message_sanitized'] or ''
            assert rawrow['event_id']==oldrow.event_id
            assert rawrow['label_binary']==['benign','malicious','suspicious'][oldrow.label_index]
            assert hashlib.sha256(raw.encode('utf-8')).digest()==oldrow.raw_hash
            rid=int(codes[position]);r=records[rid]
            p={'text':r['text'],'facts':json.loads(r['base_facts']),'ports':json.loads(r['ports']),'removed_audit':json.loads(r['removed_audit'])}
            if oldrow.route in ('asa','asa_acl','cef_fields','native_flow','native_firewall','vpc_v2'):
                states=rep.port_audit(raw,oldrow.route,p)
                for key,item in states.items():
                    port_counts[(oldrow.route,key,item['state'])]+=1
                    if item['value'] is not None and p['ports'][key]!=item['value']:failures.append(['port_audit_mismatch',position,key])
            # All WAF, all Duo, and bounded diverse projections per remaining route/class.
            samplekey=(oldrow.route,int(oldrow.label_index))
            if full_source_audit(oldrow) or seen[samplekey]<16:
                direct=rep.prepare_message(raw)
                if direct!=p:failures.append(['cache_direct_mismatch',position])
                for name,changed in variants(raw,oldrow.route):
                    altered=rep.prepare_message(changed)
                    if altered!=direct:failures.append(['invariance',position,name])
                    checks[name]+=1
                # Outer source/time/identity fields may not be read.
                altered=rep.prepare_record({'message_sanitized':raw,'product_name':None,'timestamp':'2099','label_binary':'not_a_feature','src_port':1})
                if altered!=direct:failures.append(['outer_fields',position])
                cases.append({'row_position':position,'raw':raw,'route':oldrow.route,'projection_id':rid})
                seen[samplekey]+=1;checks['direct_cached_projection']+=1
            position+=1
        if position%131072==0:print(json.dumps({'stage':'verify_original_rows','rows':position,'seconds':round(time.perf_counter()-start)}),flush=True)
    assert position==2056871
    save(out/'port_states.json',[{'route':k[0],'port':k[1],'state':k[2],'rows':n} for k,n in sorted(port_counts.items())])
    save(out/'audit_cases.json',cases)
    save(out/'audit.json',{'full_original_rows_verified':position,'sample_checks':dict(checks),'failures':failures,'passed':not failures,
                          'scope':'Full input identity; cached projection transform; selected direct inference and bounded real wrapper tests, not quality'})
    if failures:raise RuntimeError('Input audit failed; see audit.json')
    files={p.name:sha(p) for p in out.glob('*') if p.is_file()}
    save(out/'complete.json',{'version':rep.VERSION,'official_sha256':EXPECTED,'previous_prepared_sha256':receipt['prepared_sha256'],
        'rows':position,'projections':len(records),'files':files,'source_hashes':{n:sha(Path(__file__).parent/n) for n in ['v38_representation.py','v38_learning.py','run_v38_prepare.py']},'seconds':time.perf_counter()-start,'model_trained':False})
    print(json.dumps({'stage':'v38_preparation_completed','seconds':round(time.perf_counter()-start)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
