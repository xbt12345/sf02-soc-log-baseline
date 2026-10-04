"""Read-only raw evidence inventory for the dominant unsupported error slice."""
import argparse
import collections
import json
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


def main(a):
    root=Path(a.root).resolve();run=Path(a.run).resolve();out=run/'missing_context_review.json';assert not out.exists()
    sys.path.insert(0,str(run/'frozen_training_runtime'))
    import v42_core as core
    from run_v39_prepare import sha,save
    df=pd.concat([pq.read_table(run/('primary/fold_%s/R/evaluation.parquet'%f)).to_pandas() for f in range(3)],ignore_index=True)
    pred=df[['p_benign','p_malicious','p_suspicious']].to_numpy().argmax(1)
    targets=df.loc[(df.route=='asa')&(pred!=df.label_index)&(df.context_reason=='missing_or_invalid_src_port_fixed')].copy()
    assert len(targets)==4076
    positions=set(targets.row_position);observations={};pattern=collections.Counter();examples={};off=0
    # Classify explicit literal source-endpoint syntax only. This does not infer
    # an omitted port from identity columns, message ids or class labels.
    import re
    literal=re.compile(r'\bsrc\s+([^\s:]+):((?:\d{1,3}\.){3}\d{1,3})(?:/(\d+))?\b',re.I)
    for batch in pq.ParquetFile(root/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized'],use_threads=False):
        for pos in sorted(positions.intersection(range(off,off+len(batch)))):
            raw=batch.column(0)[pos-off].as_py() or '';p=core.prior.prepare_record({'message_sanitized':raw});f=p['facts']
            assert f.get('src_port_fixed')==65536
            matches=list(literal.finditer(raw))
            if len(matches)==1:category='explicit_source_endpoint_has_port' if matches[0].group(3) else 'explicit_source_endpoint_omits_port'
            else:category='source_endpoint_not_classified_by_literal_probe'
            observations[pos]=category;pattern[category]+=1
            key=(category,f.get('transport_protocol'),f.get('dst_role'),f.get('src_role'))
            if key not in examples:examples[key]={'row_position':pos,'literal_probe':category,'raw':raw,'semantic_facts':f}
        off+=len(batch)
    assert len(observations)==4076
    targets['literal_probe']=targets.row_position.map(observations)
    breakdown=[]
    for key,t in targets.groupby(['literal_probe','label_index']):breakdown.append({'literal_probe':key[0],'label_index':int(key[1]),'rows':len(t),'body_keys':int(t.body_group.nunique())})
    save(out,{'rows':len(targets),'body_keys':int(targets.body_group.nunique()),'literal_probe_counts':dict(pattern),'class_breakdown':breakdown,
        'representative_originals':list(examples.values()),'training_executed':False,'raw_source_sha256':sha(root/'data/official/train.parquet'),'script_sha256':sha(__file__),
        'scope':'Bounded source-endpoint syntax probe, not a complete parser audit or proof that missing ports caused wrong classification. Unclassified syntax needs review.'})
    print(json.dumps({'rows':len(targets),'body_keys':int(targets.body_group.nunique()),'literal_probe_counts':dict(pattern),'representatives':len(examples)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True);main(p.parse_args())
