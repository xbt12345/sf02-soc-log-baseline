"""Independent full-row identity, grouping and real-message invariance audit."""
import argparse,collections,hashlib,json,re,struct
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soc_v3_prepare as base
import v37_representation as rep
import v37_learning as learning
from run_v37_prepare import CANONICAL
from run_v36_prepare import save

def parser_hashes():
    import ast
    root=Path(__file__).parent;pending=['run_v37_prepare'];seen={}
    while pending:
        n=pending.pop();p=root/(n+'.py')
        if p.name in seen:continue
        seen[p.name]=base.file_hash(p)
        for item in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
            names=[a.name.split('.')[0] for a in item.names] if isinstance(item,ast.Import) else ([item.module.split('.')[0]] if isinstance(item,ast.ImportFrom) and item.module else [])
            pending.extend(k for k in names if (root/(k+'.py')).exists() and k+'.py' not in seen)
    return seen

def variants(raw,route):
    """Only bounded collector / encoding changes, not invented security events."""
    out=[]
    if route=='cef_fields':
        start=raw.find('CEF:');body=raw[start:]
        out=[('complete_clock','<190>Jul 26 00:00:00 changedhost '+body),
             ('redacted_clock','<190>Jul 26 USER-9999 changedhost '+body),
             ('missing_clock','<190>Jul 26 - changedhost '+body),
             ('no_wrapper',body)]
        # Native header values are metadata, extension untouched.
        pieces=body.split('|',7)
        if len(pieces)==8:out.append(('new_vendor_product','CEF:0|NEWVENDOR|NEWPRODUCT|99|88|newname|9|'+pieces[7]))
    elif route=='asa':
        parts=rep.prior.asa_parts(raw)
        if parts:out=[('collector_clock','<164>Jul 26 2099 00:00:00: newhost '+parts['body'])]
    elif route=='vpc_v2':
        a=raw.split();b=a[:];b[1]='999999999999';b[2]='eni-new';b[10]='9999999999';b[11]='USER-9999'
        out=[('flow_metadata',' '.join(b))]
    elif route=='native_flow':
        out=[('collector_and_pattern',re.sub(r'^<\d+>Original Address=\S+\s+1\s+\S+\s+\S+',
           '<134>Original Address=203.0.113.1 1 USER-9999 newhost',raw).split('pattern:')[0].rstrip()+' pattern: harmless')]
    elif route=='native_firewall':
        out=[('collector_identity_clock',re.sub(r'^<\d+>Original Address=\S+\s+1\s+\S+\s+\S+',
            '<134>Original Address=203.0.113.1 1 USER-9999 newhost',raw))]
    elif route in ('syslog_body','syslog_auth','audit_fields','asa_acl','asa_protocol','format_payload'):
        found=rep.collector_body(raw)
        if found:
            out=[('native_complete_clock','<134>Jul 26 00:00:00 changedhost '+found[0]),
                 ('native_redacted_clock','<134>Jul 26 USER-9999 changedhost '+found[0]),
                 ('native_missing_clock','<134>Jul 26 - changedhost '+found[0])]
    elif route in ('windows_message','authentication','bounded_payload'):
        changes=[]
        for a,b,path,begin in rep.lexical.members(raw):
            clock=(len(path)==1 and (path[0] in ('timestamp','isotimestamp') or path[0].startswith('@') and 'stamp' in path[0].lower()))
            upstream=(route=='bounded_payload' and path and path[0]=='behaviors' and path[-1] in ('severity','description','tactic','scenario'))
            if clock or upstream:changes.append((begin,b,'"2099-01-01T00:00:00Z"' if clock else '"changed metadata"'))
        if changes:
            text=raw
            for a,b,value in sorted(changes,reverse=True):text=text[:a]+value+text[b:]
            out=[('native_metadata',text)]
    elif route=='windows_rendered':
        changed=re.sub(r'SystemTime="[^"]+"','SystemTime="2099-01-01T00:00:00Z"',raw)
        if changed!=raw:out=[('native_xml_clock',changed)]
    return out

def run(args):
    folder=Path(args.prepared);output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    summary=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    for name,k in [('prepared.parquet','prepared_sha256'),('groups.parquet','groups_sha256')]:assert base.file_hash(folder/name)==summary[k]
    gt=pq.read_table(folder/'groups.parquet');groups=gt['union_group'].to_numpy();y=gt['label_index'].to_numpy()
    prot=pq.read_table(folder/'protocol.parquet');decl=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    assert base.file_hash(folder/'protocol.parquet')==decl['protocol_sha256']
    checks={};maps=[{} for _ in range(5)];canonical=hashlib.sha256()
    for task in prot.column_names[1:]:
        roles=prot[task].to_numpy();active=roles>=0
        lo=np.full(int(groups.max())+1,127,np.int8);hi=np.full(len(lo),-1,np.int8)
        np.minimum.at(lo,groups[active],roles[active]);np.maximum.at(hi,groups[active],roles[active])
        checks[task+'_role_isolation']=bool(np.all(lo[hi>=0]==hi[hi>=0]))
    rawiter=pq.ParquetFile(args.train).iter_batches(batch_size=2048,columns=['event_id','label_binary','message_sanitized','product_name'],use_threads=False)
    offset=0;counts=collections.Counter();selected=[];seen=collections.defaultdict(set);stress=collections.Counter()
    coverage=collections.Counter();asa=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0,0]))
    collisions=[collections.defaultdict(lambda:[0,0,0]) for _ in range(4)]
    for b in pq.ParquetFile(folder/'prepared.parquet').iter_batches(batch_size=2048,use_threads=False):
        originals=next(rawiter).to_pylist();assert len(originals)==len(b)
        for row,rawrow in zip(b.to_pylist(),originals):
            pos=offset;offset+=1;raw=rawrow['message_sanitized'] or ''
            assert row['row_position']==pos and rawrow['event_id']==row['event_id']
            assert ['benign','malicious','suspicious'][int(y[pos])]==rawrow['label_binary'] and y[pos]==row['label_index']
            assert hashlib.sha256(raw.encode()).digest()==row['raw_hash']
            v=rep.prepare_record(rawrow)
            assert v['b1']==row['b1'] and v['facts']==json.loads(row['facts']) and v['route']==row['route']
            assert v['supported']==row['supported']
            assert 'legacy_fallback'!=v['route']
            assert not {'src_port','dst_port'}.intersection(v['facts'])
            d=learning.fact_dictionary(v['facts']);assert not any(k.startswith('observed:') for k in d)
            values=[row['legacy_group_text'],row['b0'],json.dumps([row['b0'],json.loads(row['b0facts'])],sort_keys=True,ensure_ascii=False,separators=(',',':')),
                    row['b1'],v['b2_key']]
            if not row['original_empty']:
                for i,s in enumerate(values):
                    key=hashlib.sha256(s.encode()).digest() if s else b'empty_projection'
                    previous=maps[i].setdefault(key,int(groups[pos]))
                    assert previous==int(groups[pos]),('cross_group',pos,i)
                for i,s in enumerate(values[1:]):collisions[i][hashlib.sha256(s.encode()).digest()][int(y[pos])]+=1
            source=rawrow['product_name'] or '<missing>'
            for task,product in [('source_ad','Windows Active Directory'),('source_duo','Duo'),('source_waf','Barracuda WAF')]:
                if source==product:assert prot[task][pos].as_py()==3
            counts['original_empty']+=row['original_empty']
            counts['information_empty']+=not row['b1'] and not v['facts']
            counts['unsupported']+=not row['supported']
            coverage[(source,v['route'],int(y[pos]),row['supported'])]+=1
            payload=json.dumps([row[k] for k in CANONICAL],ensure_ascii=False,separators=(',',':')).encode()
            canonical.update(struct.pack('<I',len(payload)));canonical.update(payload)
            key=(source,int(y[pos]));samplekey=int(groups[pos]) if not row['original_empty'] else 'empty'
            sample=samplekey not in seen[key] and len(seen[key])<12
            # All WAF rows, and representative independent groups for other types.
            if source=='Barracuda WAF' or sample:
                altered=rep.prepare_record(dict(rawrow,timestamp='2099',product_name=None,vendor_name='NEW',pipeline='NEW',
                                               event_id='different',label_binary='DO_NOT_READ',src_ip='203.0.113.1',src_port=0))
                assert altered['b2_key']==v['b2_key'];stress['outer_columns']+=1
                for name,changed in variants(raw,v['route']):
                    result=rep.prepare_message(changed)
                    assert result['b2_key']==v['b2_key'],('counterfactual',pos,name,v['b2_key'],result['b2_key'])
                    stress[name]+=1
                if sample:
                    seen[key].add(samplekey)
                    selected.append({'row_position':pos,'product':source,'label_index':int(y[pos]),'raw_message':raw,
                                     'b1':v['b1'],'facts':v['facts'],'route':v['route'],'quality':v['quality'],'evidence':v['evidence']})
            if source=='Barracuda WAF':counts['waf_rows']+=1;assert v['route']=='cef_fields'
            if v['route']=='asa':
                for name,s in [('raw',raw),('old',values[2]),('new',v['b2_key'])]:
                    asa[name][hashlib.sha256(s.encode()).digest()][int(y[pos])]+=1
        if offset%131072==0:print(json.dumps({'stage':'audit_v37','rows':offset}),flush=True)
    assert offset==2056871
    def summarize(table):
        c=np.asarray(list(table.values()),dtype=np.int64);mixed=(c>0).sum(1)>1
        return {'groups':len(c),'mixed_groups':int(mixed.sum()),'mixed_rows':int(c[mixed].sum()),'minimum_empirical_errors':int((c.sum(1)-c.max(1)).sum())}
    content=canonical.hexdigest();assert content==summary['canonical_input_views_sha256']
    groups_hash=hashlib.sha256(groups.astype('<i4').tobytes()+y.astype('u1').tobytes()).hexdigest()
    rolehash={n:hashlib.sha256(prot[n].to_numpy().astype('i1').tobytes()).hexdigest() for n in prot.column_names[1:]}
    assert groups_hash==summary['canonical_groups_sha256'] and rolehash==summary['canonical_roles_sha256']
    checks.update(full_identity_labels_and_message_hash=True,full_candidate_reparse=True,all_exact_views_grouped=True,
                  held_sources_never_fit=True,canonical_matches=True,real_waf_rows=counts['waf_rows']==4130,
                  no_raw_fallback=True,no_untyped_numeric_ports=True,outer_metadata_invariance=True,real_collector_invariance=True,
                  official_input_hash=base.file_hash(args.train)==base.EXPECTED_SHA,protocol_eligible=decl['all_tasks_eligible'])
    result={'version':'v37-input-audit-1.0','all_checks_passed':all(checks.values()),'checks':checks,'rows':offset,
        'counts':dict(counts),'invariance_tests':dict(stress),'asa_projection_conflicts':{k:summarize(t) for k,t in asa.items()},
        'view_collisions':{name:summarize(t) for name,t in zip(['old_text','old_B2','new_text','new_B2'],collisions)},
        'canonical_input_views_sha256':content,'canonical_groups_sha256':groups_hash,'canonical_roles_sha256':rolehash,
        'prepared_sha256':base.file_hash(folder/'prepared.parquet'),'groups_sha256':base.file_hash(folder/'groups.parquet'),
        'protocol_sha256':base.file_hash(folder/'protocol.parquet'),'model_quality_accepted':False,
        'probability_invariance_note':'Equal model projections imply equal deterministic encoded scores; actual trained model replay is a separate required stage.',
        'source_hashes':summary['source_hashes'],'transitive_parser_hashes':parser_hashes(),
        'auditor_sha256':base.file_hash(Path(__file__))}
    save(output.with_name('audit_cases.json'),selected)
    save(output.with_name('coverage_support.json'),[dict(product=p,route=r,label_index=l,supported=s,rows=n) for (p,r,l,s),n in sorted(coverage.items())])
    result['audit_cases_sha256']=base.file_hash(output.with_name('audit_cases.json'))
    result['coverage_support_sha256']=base.file_hash(output.with_name('coverage_support.json'))
    save(output,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result['all_checks_passed']:raise RuntimeError('Input gate failed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
