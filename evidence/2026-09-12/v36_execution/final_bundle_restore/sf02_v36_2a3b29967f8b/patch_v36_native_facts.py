"""Local incremental preparation: B1 unchanged, B2 gains native observations.

    Shared B1 groups already contain every possible B2 equality: B2 includes
    the whole B1 text. Reuse roles only after proving all B1 values unchanged.
    Cloud can prepare identical representations directly from raw instead.
"""
import argparse
import collections
import json
import shutil
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import soc_v3_prepare as base
import v36_representation as rep
from run_v36_prepare import save


def run(args):
    old=Path(args.previous);out=Path(args.output_dir)
    previous=json.loads((old/'result.json').read_text(encoding='utf-8'))
    assert previous['version']=='v36-representation-1.1'
    assert base.file_hash(old/'prepared.parquet')==previous['prepared_sha256']
    assert base.file_hash(old/'groups.parquet')==previous['groups_sha256']
    assert base.file_hash(args.train)==base.EXPECTED_SHA
    if out.exists():raise FileExistsError(out)
    out.mkdir();started=time.perf_counter();changes=collections.Counter();rows=0
    raw=pq.ParquetFile(args.train).iter_batches(batch_size=2048,columns=['event_id','message_sanitized'],use_threads=False)
    reader=pq.ParquetFile(old/'prepared.parquet');writer=pq.ParquetWriter(out/'prepared.parquet',reader.schema_arrow,compression='zstd')
    for b in reader.iter_batches(batch_size=2048,use_threads=False):
        records=b.to_pylist();original=next(raw).to_pylist()
        for row,r in zip(records,original):
            assert row['event_id']==r['event_id']
            if row['route'] in ('windows_message','windows_rendered'):
                observation=rep.native_auth_observation(r['message_sanitized'],xml=row['route']=='windows_rendered')
                if observation:
                    facts=json.loads(row['facts']);evidence=json.loads(row['evidence']);before=dict(facts)
                    rep.add_native_auth(facts,evidence,observation)
                    changes[(row['product'],row['label_index'],facts.get('outcome','conflict'))]+=int(facts!=before)
                    row['facts']=json.dumps(facts,sort_keys=True,ensure_ascii=False,separators=(',',':'))
                    row['evidence']=json.dumps(evidence,ensure_ascii=False,separators=(',',':'))
                    row['no_observed_fact']=not facts
        # Columns other than facts/evidence/structured-fact presence are preserved.
        writer.write_table(pa.Table.from_pylist(records,schema=reader.schema_arrow));rows+=len(records)
        if rows%131072==0:print(json.dumps({'stage':'native_fact_patch','rows':rows,'seconds':round(time.perf_counter()-started)}),flush=True)
    writer.close()
    assert rows==2056871
    for name in ['groups.parquet','protocol.parquet','protocol.json','coverage.json']:
        shutil.copyfile(old/name,out/name)
    source_hashes={p.name:base.file_hash(p) for p in [Path(__file__),Path(rep.__file__),Path(__file__).with_name('v331_prepare.py')]}
    result=dict(previous,version=rep.VERSION,prepared_sha256=base.file_hash(out/'prepared.parquet'),source_hashes=source_hashes,
         elapsed_seconds=time.perf_counter()-started,patch_input_sha256=previous['prepared_sha256'],
         patch_only_columns=['facts','evidence','no_observed_fact'],B1_and_joint_roles_unchanged=True,
         native_observation_changes=[{'source':s,'label_index':y,'outcome':o,'rows':n} for (s,y,o),n in changes.items()])
    save(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--train',required=True)
    run(p.parse_args())
