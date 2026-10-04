"""Independent streaming checks of prepared identities, exact groups and roles."""
import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import soc_v3_prepare as base
import v36_representation as rep


def run(args):
    folder=Path(args.prepared);output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    g=pq.read_table(folder/'groups.parquet')
    groups=g['union_group'].to_numpy();y=g['label_index'].to_numpy()
    roles=pq.read_table(folder/'protocol.parquet')
    checks={};support={};maps=[{}, {}, {}];raw_empty=0;factless=0;no_structured=0
    for task in roles.column_names[1:]:
        r=roles[task].to_numpy();active=r>=0
        # Independent min/max reduction, not preparation's assertion helper.
        lo=np.full(int(groups.max())+1,127,dtype=np.int8);hi=np.full(len(lo),-1,dtype=np.int8)
        np.minimum.at(lo,groups[active],r[active]);np.maximum.at(hi,groups[active],r[active])
        checks[task+'_union_role_isolation']=bool(np.all(lo[hi>=0]==hi[hi>=0]))
    original=pq.ParquetFile(args.train).iter_batches(batch_size=2048,columns=['event_id','label_binary','message_sanitized','product_name'],use_threads=False)
    selected=[];seen=collections.defaultdict(set);offset=0;alignment=True;role_sources=True;canonical=hashlib.sha256()
    for b in pq.ParquetFile(folder/'prepared.parquet').iter_batches(batch_size=2048,
           columns=['row_position','event_id','label_index','b0','b1','facts','original_empty','product','route'],use_threads=False):
        raw=next(original).to_pylist()
        for j,row in enumerate(b.to_pylist()):
            pos=offset+j;original_row=raw[j];label=int(y[pos]);group=int(groups[pos])
            payload=json.dumps([row[k] for k in ['row_position','event_id','label_index','b0','b1','facts','original_empty','product','route']],ensure_ascii=False,separators=(',',':')).encode('utf-8')
            canonical.update(struct.pack('<I',len(payload)));canonical.update(payload)
            alignment &= row['row_position']==pos and row['event_id']==original_row['event_id'] and label==row['label_index'] and ['benign','malicious','suspicious'][label]==original_row['label_binary']
            raw_empty+=row['original_empty'];no_structured+=row['facts']=='{}'
            factless+=(not row['b1'] and row['facts']=='{}')
            if not row['original_empty']:
                keys=[row['b0'] or 'raw:'+hashlib.sha256((original_row['message_sanitized'] or '').encode()).hexdigest(),
                      row['b1'],json.dumps([row['b1'],json.loads(row['facts'])],sort_keys=True,separators=(',',':'))]
                for v,text in enumerate(keys):
                    digest=hashlib.sha256(text.encode()).digest()
                    if digest not in maps[v]:maps[v][digest]=[group,[0,0,0]]
                    item=maps[v][digest]
                    if item[0]!=group:raise AssertionError('Exact input crosses shared union: '+str(pos))
                    item[1][label]+=1
            source=row['product']
            for task,target in [('source_ad','Windows Active Directory'),('source_duo','Duo'),('source_waf','Barracuda WAF')]:
                if source==target and int(roles[task][pos].as_py())!=3:role_sources=False
            # First 20 independent groups per source/class, not first 20 rows.
            key=(source,label)
            sample_key='empty_message' if row['original_empty'] else group
            if sample_key not in seen[key] and len(seen[key])<20:
                seen[key].add(sample_key)
                expected=rep.prepare_record(original_row)
                variant=dict(original_row,timestamp='2099-01-01T00:00:00Z',product_name='UNKNOWN_NEW_PRODUCT',vendor_name=None,
                      pipeline='new_collector',src_ip='203.0.113.199',dst_ip='198.51.100.7',username='different_identity',label_binary='DO_NOT_USE')
                altered=rep.prepare_record(variant)
                if expected['b1']!=row['b1'] or expected['facts']!=json.loads(row['facts']):
                    raise AssertionError('Prepared representation cannot be reproduced: '+str(pos))
                if expected['b1']!=altered['b1'] or expected['facts']!=altered['facts']:
                    raise AssertionError('Outer metadata enters representation')
                selected.append({'row_position':pos,'source':source,'label_index':label,'union_group':group,'route':row['route'],
                                 'raw_message':original_row['message_sanitized'],'b1':row['b1'],'facts':json.loads(row['facts']),
                                 'evidence':expected['evidence'],'review_status':'raw_and_representation_exported; no human_label_certification'})
        offset+=len(b)
    checks.update(all_official_rows=offset==2056871,all_row_identities_and_labels_preserved=bool(alignment),
                  held_source_never_in_fit_selection_calibration=role_sources,
                  all_nonempty_exact_view_groups_inside_union=True,sampled_outer_metadata_invariant=True,
                  official_input_hash=base.file_hash(args.train)==base.EXPECTED_SHA)
    collisions={}
    for v,table in enumerate(maps):
        counts=np.array([x[1] for x in table.values()],dtype=np.int64)
        mixed=(counts>0).sum(1)>1
        collisions['B'+str(v)]={'unique_nonempty_original_input_views':len(table),
            'mixed_label_input_groups':int(mixed.sum()),'mixed_rows':int(counts[mixed].sum()),
            'empirical_minimum_errors_fixed_representation':int((counts.sum(1)-counts.max(1)).sum()),
            'includes_nonempty_originals_projected_to_empty':True}
    result={'checks':checks,'all_checks_passed':all(checks.values()),'rows':offset,'exact_view_collisions':collisions,
       'prepared_sha256':base.file_hash(folder/'prepared.parquet'),'groups_sha256':base.file_hash(folder/'groups.parquet'),
       'protocol_sha256':base.file_hash(folder/'protocol.parquet'),
       'canonical_input_views_sha256':canonical.hexdigest(),
       'canonical_groups_sha256':hashlib.sha256(groups.astype('<i4').tobytes()+y.astype('u1').tobytes()).hexdigest(),
       'canonical_roles_sha256':{task:hashlib.sha256(roles[task].to_numpy().astype('i1').tobytes()).hexdigest() for task in roles.column_names[1:]},
       'original_empty_rows':int(raw_empty),'no_structured_facts_rows':int(no_structured),
       'no_text_or_structured_facts_rows':int(factless),'group_selected_case_samples':len(selected),
       'semantic_quality_certified':False,'near_template_or_incident_isolation_proven':False,
       'scope':'Full identities, labels, exact-view grouping, source holdout and sampled outer-field invariance. Does not certify target labels, all parser semantics, novel sources or genuine threat probabilities.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    output.with_name(output.stem+'_cases.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result['all_checks_passed']:raise AssertionError('Independent audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
