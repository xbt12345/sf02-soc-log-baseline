"""Actual raw-message invariance and canonical value checks before fitting."""
import argparse
import collections
import json
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def main(a):
    root=Path(a.root); run=Path(a.run); runtime=run/'frozen_training_runtime'
    sys.path.insert(0,str(runtime.resolve()))
    import v40_core as core
    from run_v39_prepare import sha,save
    from audit_v37_prepared import variants
    stage=run/'prepared'; prep=root/'artifacts/v39_local_r2_20260913/prepared'
    dest=run/'input_checks.json';assert not dest.exists()
    pr=pq.read_table(prep/'projections.parquet').to_pandas()
    neutral=pq.read_table(stage/'neutral_projection_audit.parquet').to_pandas()
    assert np.array_equal(pr.projection_id,neutral.projection_id)
    cached_facts=[core.canonicalize(v)[0] for v in pr.facts]
    assert [core.prior.canonical(v) for v in cached_facts]==neutral.facts.tolist()
    cases=json.loads((prep/'audit_cases.json').read_text(encoding='utf-8'))
    failures=[];counts=collections.Counter();rawcases=[]
    for item in cases:
        raw=item['raw'];direct=core.prepare_message(raw);pid=item['projection_id']
        if direct['text']!=pr.text.iloc[pid] or direct['facts']!=cached_facts[pid]:failures.append(['cached',item['row_position']])
        changes=variants(raw,item['route'])
        if item['route']=='asa':
            import re
            changes.append(('address_interface',re.sub(r'dmz[-_]\d+','dmz-999',re.sub(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)','203.0.113.99',raw))))
        for name,changed in changes:
            other=core.prepare_message(changed)
            if (other['text'],other['facts'])!=(direct['text'],direct['facts']):failures.append([name,item['row_position']])
            counts[name]+=1
        rawcases.append({'row_position':item['row_position'],'projection_id':pid,'raw':raw,'route':item['route']})
    # Repeat all known absolute-clock affected official records, not only examples.
    allrows=pq.read_table(prep/'rows.parquet',columns=['row_position','projection_id']).to_pandas()
    targeted=set(pr.loc[pr.audit.str.contains('absolute_lexical_clock|absolute_boundary'),'projection_id'])
    positions=dict(zip(allrows.loc[allrows.projection_id.isin(targeted),'row_position'],allrows.loc[allrows.projection_id.isin(targeted),'projection_id']))
    offset=0;clock_cases=[]
    for b in pq.ParquetFile(root/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized'],use_threads=False):
        for pos in sorted(set(positions).intersection(range(offset,offset+len(b)))):
            raw=b.column(0)[pos-offset].as_py() or '';before=core.prepare_message(raw)
            for clock in ['2099-01-01T00:00:00Z','2011-12-31T23:59:59Z']:
                changed=core.prior.ISO_LITERAL.sub(clock,raw);changed=core.prior.TASK_BOUNDARY.sub(lambda m:m[1]+clock+m[3],changed)
                after=core.prepare_message(changed)
                if (before['text'],before['facts'])!=(after['text'],after['facts']):failures.append(['absolute_clock',pos])
                counts['absolute_clock']+=1
            clock_cases.append({'row_position':int(pos),'projection_id':int(positions[pos]),'raw':raw,'route':'clock_probe'})
        offset+=len(b)
    save(run/'raw_inference_cases.json',rawcases+clock_cases)
    save(dest,{'all_checks_passed':not failures,'failures':failures,'raw_cases':len(cases),'clock_records':len(clock_cases),
        'transformations':dict(counts),'canonical_projection_count':len(pr),'prepared_sha256':sha(stage/'complete.json'),
        'near_copy_review_sha256':sha(stage/'near_copy_review.json'),'raw_inference_cases_sha256':sha(run/'raw_inference_cases.json'),
        'script_sha256':sha(__file__),'scope':'Canonical observations and specified real-message transformations, not model quality'})
    print(json.dumps({'passed':not failures,'raw_cases':len(cases),'clock_records':len(clock_cases),'transformations':dict(counts),'failures':failures[:10]}),flush=True)
    assert not failures


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True);main(p.parse_args())
