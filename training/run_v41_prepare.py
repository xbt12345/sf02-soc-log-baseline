"""Freeze all decisions and native preflight before the first classifier fit."""
import argparse
import datetime
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import v41_core as core
from run_v39_prepare import sha,save

CONFIG={'version':core.VERSION,'views':{'P':0.1,'F':0.1,'D':0.1,'A':0.1},'folds':3,
 'main_fields':list(core.MAIN_FIELDS),'pair_fields':[list(v) for v in core.PAIR_FIELDS],
 'pair_minimum_distinct_bodies':3,'weights':'original rows equal; exact multiplicity aggregation',
 'decision':'three_class_argmax','new_primary_fit_limit':12,'mandatory_primary_fits':9,'stress_fit_limit':1,
 'probability_tolerance':1e-10,'recall_drop_limit':0.005,'normal_errors_per_10000_increase_limit':1.0,
 'small_slice_maximum_class_rows':100,'small_slice_maximum_class_body_keys':30,
 'small_slice_rule':'Within every route by class and route by original availability mask by class, consider tails with <=100 rows OR <=30 body keys. Reject any newly entirely lost tail previously containing a correct row. Also protect every previously correct authentication/WAF suspicious/bounded_payload row and the 18 Windows canaries.',
 'selection':'P first if all predefined gates pass; else A if all native-target and non-target protection gates pass. F/D diagnostic only. At most one historical stress fit.',
 'P_asa_unseen_error_limits':{'unseen_input':1096,'unseen_parameter':652,'unseen_joint_value':740},
 'A_target':'Native Windows authentication scoped by raw title/provider; pooled errors decrease with at least two nonworsening folds, all non-target route by class error totals must not increase.',
 'fresh_blind_test':False,'text_encoder':'P/F/D reuse exact same-fit B; A refits fit-only text vocabulary after typed literal replacement'}


def main(a):
    root=Path(a.root).resolve();out=Path(a.output).resolve();assert not out.exists()
    prep=root/'artifacts/v39_local_r2_20260913/prepared';old=root/'artifacts/v40_local_r1_20260913'
    audit_dir=root/'evidence/2026-09-13/v41_native_preflight_r2';audit=json.loads((audit_dir/'audit.json').read_text(encoding='utf-8'))
    for n,h in audit['files'].items():assert sha(audit_dir/n)==h
    for n,h in audit['source_sha256'].items():assert sha(Path(__file__).parent/n)==h
    receipt=json.loads((prep/'complete.json').read_text(encoding='utf-8'))
    assert sha(root/'data/official/train.parquet')==receipt['official_sha256']
    for n,h in receipt['files'].items():assert sha(prep/n)==h
    out.mkdir();stage=out/'prepared';stage.mkdir();runtime=out/'frozen_training_runtime';runtime.mkdir()
    for p in (old/'frozen_training_runtime').glob('*.py'):shutil.copyfile(p,runtime/p.name)
    for n in ['v41_core.py','v41_native_auth.py','test_v41.py','audit_v41_native.py','run_v41_prepare.py','run_v41_train.py','review_v41.py','verify_v41.py']:
        shutil.copyfile(Path(__file__).parent/n,runtime/n)
    shutil.copyfile(root/'docs/V41_NEXT_PLAN.md',stage/'approved_plan.md')
    shutil.copyfile(old/'prepared/near_copy_review.json',stage/'near_copy_review.json')
    for n in ['audit.json','native_updates.parquet','native_observations.parquet','native_scope.parquet']:
        shutil.copyfile(audit_dir/n,stage/n)
    save(stage/'configuration.json',CONFIG)
    save(stage/'native_configuration.json',{'boundary_ready':audit['A_boundary_ready'],
        'scope':'Unique native authentication provider/code plus unique matching native message title; rendered failure reason followed by contiguous status lines before process sections, or bounded winlog fields.',
        'code':'Existing B 33-bit Status/SubStatus encoding; no extra credential outcome mapping.',
        'text':'Matching typed status literal values replaced with native_code; unrelated hexadecimal payload unchanged. Fit-only text vocabulary rebuilt.',
        'duplicates':'Consistent carriers one fact; disagreeing candidate fields veto repair. Unbounded candidate fields are veto-only, never features.',
        'unknown_or_conflict':'Unchanged B fallback, explicitly not claiming old facts are verified or all ambiguity removed.',
        'labels_weights_folds_C':'Unchanged; no parameter/dedup/finite recoding combined; no metadata features.'})
    rows=pq.read_table(prep/'rows.parquet').to_pandas();allowed=rows[rows.fold>=0].copy()
    assert len(allowed)==1378650 and allowed.body_group.nunique()==508823
    assert allowed.groupby('body_group').fold.nunique().max()==1
    meta_path=root/'artifacts/v37_prepared_r13_20260913/prepared.parquet'
    meta=pq.read_table(meta_path,columns=['row_position','raw_hash']).to_pandas()
    assert np.array_equal(meta.row_position,np.arange(len(rows)))
    raw=meta.iloc[allowed.row_position.to_numpy()].copy();raw['fold']=allowed.fold.to_numpy();raw['empty']=allowed.original_empty.to_numpy()
    overlap=int((raw.loc[~raw['empty']].groupby('raw_hash').fold.nunique()>1).sum());assert overlap==0
    del meta,raw
    pr=pq.read_table(prep/'projections.parquet',columns=['text','facts']).to_pandas()
    additions=[];mapping=[];index={}
    for r in pq.read_table(stage/'native_updates.parquet').to_pandas().itertuples(index=False):
        key=(r.old_projection_id,r.text,r.facts)
        if key not in index:
            index[key]=len(pr)+len(additions);additions.append({'text':r.text,'facts':r.facts})
        mapping.append({'row_position':r.row_position,'model_projection_id':index[key]})
    ap=pd.concat([pr,pd.DataFrame(additions)],ignore_index=True)
    pq.write_table(pa.Table.from_pandas(ap,preserve_index=False),stage/'A_projections.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(mapping,schema=pa.schema([('row_position',pa.int64()),('model_projection_id',pa.int64())])),stage/'A_updates.parquet',compression='zstd')
    old_bindings={}
    for rname,views in [('v39_local_r2_20260913',['SEMANTIC']),('v40_local_r1_20260913',['N','I'])]:
        for fold in range(3):
            for view in views:
                d=root/('artifacts/%s/primary/fold_%s/%s'%(rname,fold,view));rec=json.loads((d/'complete.json').read_text(encoding='utf-8'))
                for n,k in [('model.joblib','model_sha256'),('evaluation.parquet','predictions_sha256')]:
                    assert sha(d/n)==rec[k];old_bindings[(d/n).relative_to(root).as_posix()]=rec[k]
    save(stage/'input_audit.json',{'allowed_rows':len(allowed),'body_keys':int(allowed.body_group.nunique()),'exact_nonempty_raw_fold_overlap':overlap,
        'body_fold_overlap':0,'near_copy_review':'Inherited exact same 30-row/15-pair evidence retained; no claim of complete incident independence.',
        'native_changed_rows':len(mapping),'A_added_projection_variants':len(additions),'old_projection_uniformity_not_assumed':True,
        'source_bindings':old_bindings,'A_boundary_ready':audit['A_boundary_ready']})
    files={p.name:sha(p) for p in stage.iterdir() if p.is_file()}
    save(stage/'complete.json',{'version':core.VERSION,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'v39_prepared':str(prep),'v39_prepared_receipt_sha256':sha(prep/'complete.json'),'official_sha256':receipt['official_sha256'],
        'v37_raw_hash_metadata_sha256':sha(meta_path),'files':files,'runtime_sources':{p.name:sha(p) for p in runtime.glob('*.py')},'new_model_trained':False})
    print(json.dumps({'prepared':str(stage),'allowed_rows':len(allowed),'A_added_variants':len(additions),'A_boundary_ready':audit['A_boundary_ready']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);main(p.parse_args())
