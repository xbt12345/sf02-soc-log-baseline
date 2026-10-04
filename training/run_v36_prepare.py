"""Prepare paired views and rebuild shared development responsibilities, official train only."""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import soc_v3_prepare as base
import v331_prepare as old
import v36_representation as rep
from v32_split import allocate
from v351_safeguards import assert_group_isolation

LABELS = ['benign', 'malicious', 'suspicious']
SOURCE_TASKS = {'source_ad': 'Windows Active Directory', 'source_duo': 'Duo', 'source_waf': 'Barracuda WAF'}
HARD_ASA = 'deny|tcp|outside|dmz|no_icmp_fields'
KNOWN_BASE_SHA = 'fba196159411a30ea31744b3416c71ae58b47a2e6c5a5ff0f7d3130647ca1bce'
KNOWN_BASE_CONTENT = '44c9c9c91a0896534d792c3ab7686c13a126dff35156f4255bd6fcd7696c8c1d'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def worker(payload):
    offset, rows, previous = payload
    result = []
    for j, row in enumerate(rows):
        raw = row['message_sanitized'] or ''
        v = rep.prepare_record({'message_sanitized': raw})
        if previous is not None:
            assert row['event_id'] == previous[j]['event_id']
            assert row['label_binary'] == previous[j]['label']
            b0 = previous[j]['text']
            fmt, template = previous[j]['format'], previous[j]['asa_template']
        else:
            p = old.prepare_record({'message_sanitized': raw})
            b0, fmt, template = p['text'], p['format'], p['asa_template']
        result.append({'row_position': offset+j, 'event_id': row['event_id'],
                       'label_index': LABELS.index(row['label_binary']), 'product': row['product_name'] or '<missing>',
                       'format': fmt, 'asa_template': template or '', 'b0': b0, 'b1': v['b1'],
                       'facts': json.dumps(v['facts'], sort_keys=True, ensure_ascii=False, separators=(',', ':')),
                       'route': v['route'], 'supported': v['supported'], 'no_observed_fact': v['no_observed_fact'],
                       'original_empty': not raw.strip(), 'raw_hash': hashlib.sha256(raw.encode()).digest(),
                       'evidence': json.dumps(v['evidence'], ensure_ascii=False, separators=(',', ':'))})
    return result


def batches(train, previous, size, max_rows):
    offset = 0
    it = None
    if previous:
        it = pq.ParquetFile(previous/'prepared_corpus.parquet').iter_batches(batch_size=size,
                columns=['event_id','label','text','format','asa_template'], use_threads=False)
    for batch in pq.ParquetFile(train).iter_batches(batch_size=size,
            columns=['event_id','label_binary','product_name','message_sanitized'], use_threads=False):
        old_rows = next(it).to_pylist() if it else None
        rows = batch.to_pylist()
        if max_rows and offset+len(rows) > max_rows:
            rows = rows[:max_rows-offset]
            if old_rows is not None:
                old_rows = old_rows[:len(rows)]
        if not rows:
            break
        yield offset, rows, old_rows
        offset += len(rows)
        if max_rows and offset >= max_rows:
            break


def supported_roles(y, groups, products, roles):
    output = {}
    for source in sorted(set(products)):
        mask = products == source
        output[source] = {str(r): {LABELS[c]: {'rows': int((mask&(roles==r)&(y==c)).sum()),
                      'groups': int(len(np.unique(groups[mask&(roles==r)&(y==c)])))} for c in range(3)} for r in range(4)}
    return output


def protocol(out, y, groups, products, templates):
    ng = int(groups.max())+1
    names, source_codes = np.unique(products, return_inverse=True)
    size = len(names)*3
    cross = np.bincount(groups.astype(np.int64)*size+source_codes*3+y, minlength=ng*size).reshape(ng, size)
    counts = cross.reshape(ng, len(names), 3).sum(1)
    ids = np.flatnonzero(counts.sum(1))
    manifest = {'row_position': np.arange(len(y), dtype=np.int32)}
    summaries = {}
    for task in ['known_dev', 'source_ad', 'source_duo', 'source_waf', 'asa_hard']:
        moves = []
        if task == 'known_dev':
            group_roles = allocate(cross, cross, ids, [.6,.1,.1,.2])
            # Known-source support, never model scores: Duo cannot support four
            # independent positive roles. Prioritize fit and evaluation.
            for s in range(len(names)):
                for c in range(3):
                    eligible = np.flatnonzero(cross[:,s*3+c] > 0)
                    if len(eligible) < 2:
                        continue
                    for destination in (0,3):
                        if np.any(group_roles[eligible] == destination):
                            continue
                        candidates = [int(g) for g in eligible if group_roles[g] not in (0,3) or np.sum(group_roles[eligible]==group_roles[g]) > 1]
                        if candidates:
                            g = min(candidates, key=lambda g:(int(cross[g,s*3+c]),g))
                            moves.append({'group':g,'from':int(group_roles[g]),'to':destination,'source':str(names[s]),'class':LABELS[c]})
                            group_roles[g] = destination
            roles = group_roles[groups]
            target = roles == 3
        else:
            target = products == SOURCE_TASKS[task] if task in SOURCE_TASKS else templates == HARD_ASA
            held = np.unique(groups[target])
            available = ids[~np.isin(ids, held)]
            group_roles = allocate(counts, counts, available, [.75,.125,.125])
            roles = group_roles[groups]
            roles[target] = 3
        assert_group_isolation(groups, roles)
        assert not np.any(np.isin(groups[target], groups[(roles>=0)&(roles<3)]))
        if task in SOURCE_TASKS:
            assert not np.any((products == SOURCE_TASKS[task])&(roles>=0)&(roles<3))
        role_counts = {str(r): np.bincount(y[roles==r], minlength=3).tolist() for r in range(4)}
        reasons = []
        for r in (0,1,2):
            if min(role_counts[str(r)]) == 0:
                reasons.append('missing_global_class_in_role_'+str(r))
        if not target.any():
            reasons.append('empty_target')
        support = supported_roles(y,groups,products,roles)
        if task == 'known_dev' and 'Duo' in support:
            if not support['Duo']['0']['suspicious']['groups'] or not support['Duo']['3']['suspicious']['groups']:
                reasons.append('duo_fit_or_evaluation_positive_support_missing')
        summaries[task] = {'role_class_counts':role_counts, 'source_role_support':support,
                           'support_moves':moves, 'eligible':not reasons, 'unsupported_reasons':reasons,
                           'linked_rows_excluded':int((roles<0).sum()), 'role_sha256':hashlib.sha256(roles.tobytes()).hexdigest()}
        manifest[task] = roles.astype(np.int8)
        print(json.dumps({'stage':'protocol','task':task,'eligible':not reasons,'roles':role_counts}),flush=True)
    pq.write_table(pa.table(manifest),out/'protocol.parquet',compression='zstd')
    save(out/'protocol.json',{'version':'v36-protocol-1.0','tasks':summaries,
         'development_not_blind_test':True, 'target_source_calibration':False,
         'all_representations_share_roles':True, 'near_template_incident_isolation_proven':False,
         'empty_raw_rows_keep_event_groups':True, 'all_tasks_eligible':all(s['eligible'] for s in summaries.values()),
         'protocol_sha256':base.file_hash(out/'protocol.parquet')})


def run(args):
    train, out = Path(args.train), Path(args.output_dir)
    previous = Path(args.previous) if args.previous else None
    assert base.file_hash(train) == base.EXPECTED_SHA
    if out.exists():
        raise FileExistsError('New output directory required')
    out.mkdir(parents=True)
    started = time.perf_counter()
    if previous:
        digest = base.file_hash(previous/'prepared_corpus.parquet')
        if digest != KNOWN_BASE_SHA:
            from v331_common import portable_content
            assert portable_content(previous)['corpus_content_sha256'] == KNOWN_BASE_CONTENT
    schema = pa.schema([('row_position',pa.int32()),('event_id',pa.string()),('label_index',pa.uint8()),
        ('product',pa.string()),('format',pa.string()),('asa_template',pa.string()),
        ('b0',pa.large_string()),('b1',pa.large_string()),('facts',pa.large_string()),
        ('route',pa.string()),('supported',pa.bool_()),('no_observed_fact',pa.bool_()),
        ('original_empty',pa.bool_()),('raw_hash',pa.binary(32)),('evidence',pa.large_string())])
    writer = pq.ParquetWriter(out/'prepared.parquet', schema, compression='zstd')
    maps = [{},{},{}]
    parent = []
    b0g,b1g,b2g,yy,products,templates = [],[],[],[],[],[]
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    route_counts, examples = collections.Counter(), collections.defaultdict(list)
    def accept(rows):
        for row in rows:
            keys = [row['b0'],row['b1'],json.dumps([row['b1'],json.loads(row['facts'])],sort_keys=True,separators=(',',':'))]
            if row['original_empty']:
                gid=len(parent);parent.append(gid)
                ids=[gid,gid,gid]
            else:
                old_key=(b'text:' if row['b0'] else b'raw:')+(hashlib.sha256(row['b0'].encode()).digest() if row['b0'] else row['raw_hash'])
                gid=maps[0].get(old_key)
                if gid is None:
                    gid=len(parent);parent.append(gid);maps[0][old_key]=gid
                ids=[gid]
                for view in (1,2):
                    key=hashlib.sha256(keys[view].encode()).digest()
                    item=maps[view].get(key)
                    if item is None:
                        item=(len(maps[view]),gid);maps[view][key]=item
                    ids.append(item[0])
                    left,right=find(gid),find(item[1]);parent[right]=left
            b0g.append(gid);b1g.append(ids[1]);b2g.append(ids[2]);yy.append(row['label_index'])
            products.append(row['product']);templates.append(row['asa_template'])
            key=(row['route'],row['product'],LABELS[row['label_index']]);route_counts[key]+=1
            if len(examples[key])<20:
                examples[key].append({'row_position':row['row_position'],'b1':row['b1'][:2000],
                                      'facts':json.loads(row['facts']),'evidence':json.loads(row['evidence'])})
        writer.write_table(pa.Table.from_pylist(rows,schema=schema))
    stream=iter(batches(train,previous,512,args.max_rows))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending=[]
        for _ in range(args.workers*2):
            try:pending.append(executor.submit(worker,next(stream)))
            except StopIteration:break
        while pending:
            accept(pending.pop(0).result())
            try:pending.append(executor.submit(worker,next(stream)))
            except StopIteration:pass
            if len(yy) % 65536 == 0:
                print(json.dumps({'stage':'prepare','rows':len(yy),'seconds':round(time.perf_counter()-started)}),flush=True)
    writer.close()
    roots=np.array([find(i) for i in range(len(parent))],dtype=np.int32)
    groups=roots[np.asarray(b0g,dtype=np.int32)]
    y=np.asarray(yy,dtype=np.uint8)
    pq.write_table(pa.table({'row_position':np.arange(len(y),dtype=np.int32),'old_group':np.array(b0g,dtype=np.int32),
       'union_group':groups,'label_index':y}),out/'groups.parquet',compression='zstd')
    source_hashes={p.name:base.file_hash(p) for p in [Path(__file__),Path(rep.__file__),Path(old.__file__)]}
    save(out/'coverage.json',[{'route':r,'product':p,'label':l,'rows':n} for (r,p,l),n in sorted(route_counts.items())])
    save(out/'review_examples.json',[{'route':r,'product':p,'label':l,'examples':v} for (r,p,l),v in examples.items()])
    protocol(out,y,groups,np.array(products,dtype=object),np.array(templates,dtype=object))
    save(out/'result.json',{'version':rep.VERSION,'rows':len(y),'full_official_rows':len(y)==2056871,
        'original_input_sha256':base.EXPECTED_SHA,'prepared_sha256':base.file_hash(out/'prepared.parquet'),
        'groups_sha256':base.file_hash(out/'groups.parquet'),'source_hashes':source_hashes,
        'old_groups':len(parent),'union_components':len(np.unique(roots)),
        'preparation_only':True,'model_trained':False,'semantic_quality_not_certified':True,
        'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'stage':'prepared','rows':len(y),'seconds':round(time.perf_counter()-started)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--previous');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-rows',type=int,default=0,help='Engineering subset only; not full preparation')
    run(p.parse_args())
