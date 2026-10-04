"""Bind inherited data and roles; freeze new code before nine bounded fits."""
import argparse
import collections
import datetime
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import v40_core as core
from run_v39_prepare import sha,save


CONFIG={'version':core.VERSION,'views':{'R':1.0,'N':0.1,'I':0.1},'folds':3,
 'main_fields':list(core.MAIN_FIELDS),'pair_fields':[list(p) for p in core.PAIR_FIELDS],
 'pair_minimum_distinct_bodies':3,'weights':'original rows equal; exact multiplicity aggregation',
 'decision':'three_class_argmax','new_primary_fit_limit':9,'stress_fit_limit':1,
 'probability_tolerance':1e-10,'recall_drop_limit':0.005,'normal_errors_per_10000_increase_limit':1.0,
 'selection':'At least two folds with lower ASA errors and lower pooled ASA errors; review rare slices; choose eligible candidate with fewest ASA errors, then total errors, then simpler R/N/I.',
 'fresh_blind_test':False,'nominal_main_fields_scope':'observed src/dst ports and ICMP type/code',
 'semantic_deduplication':['matching outcome/auth_result','matching response=missing/authentication_interaction=no_response'],
 'text_encoder':'Reuse exact same-fit v39 frozen word encoder; verify full baseline replay',
 'availability_control':'reuse original frozen v39 availability, not a newly trained control'}


def main(a):
    root=Path(a.root); old=root/'artifacts/v39_local_r2_20260913'; prep=old/'prepared'; out=Path(a.output)
    if out.exists():raise FileExistsError('New output required')
    out.mkdir(parents=True); stage=out/'prepared';stage.mkdir()
    receipt=json.loads((prep/'complete.json').read_text(encoding='utf-8'))
    assert sha(root/'data/official/train.parquet')==receipt['official_sha256']
    for n,h in receipt['files'].items():assert sha(prep/n)==h,n
    runtime=out/'frozen_training_runtime';runtime.mkdir()
    for p in (old/'frozen_training_runtime').glob('*.py'):shutil.copyfile(p,runtime/p.name)
    for name in ['v40_core.py','run_v40_prepare.py','run_v40_train.py','test_v40.py']:
        shutil.copyfile(Path(__file__).parent/name,runtime/name)
    # New implementation is bound to the previously trained dependency closure.
    assert sha(runtime/'v39_core.py')==sha(old/'frozen_training_runtime/v39_core.py')
    shutil.copyfile(root/'docs/V40_NEXT_PLAN.md',stage/'approved_plan.md')
    save(stage/'configuration.json',CONFIG)
    rows=pq.read_table(prep/'rows.parquet',columns=['row_position','label_index','route','body_group','projection_id','fold','original_empty','inner_role','union_group']).to_pandas()
    allowed=rows[rows.fold>=0].copy()
    assert len(allowed)==1378650 and allowed.body_group.nunique()==508823
    assert allowed.groupby('body_group').fold.nunique().max()==1
    assert set(allowed.row_position)==set(rows.loc[rows.inner_role>=0,'row_position'])
    # Read frozen raw hashes only for identity, never to select model parameters.
    meta_path=root/'artifacts/v37_prepared_r13_20260913/prepared.parquet'
    meta=pq.read_table(meta_path,columns=['row_position','raw_hash']).to_pandas()
    assert np.array_equal(meta.row_position.to_numpy(),np.arange(len(rows)))
    raw=meta.iloc[allowed.row_position.to_numpy()].copy();raw['fold']=allowed.fold.to_numpy();raw['empty']=allowed.original_empty.to_numpy()
    exact_overlap=int((raw.loc[~raw['empty']].groupby('raw_hash').fold.nunique()>1).sum())
    assert exact_overlap==0
    del meta,raw
    pr=pq.read_table(prep/'projections.parquet').to_pandas()
    data=[];changes=collections.Counter()
    for item in pr.itertuples(index=False):
        f,audit=core.canonicalize(item.facts)
        data.append({'projection_id':item.projection_id,'facts':core.prior.canonical(f),'quality':core.prior.canonical(audit)})
        changes.update(audit['merged_aliases'])
    pq.write_table(pa.Table.from_pylist(data),stage/'neutral_projection_audit.parquet',compression='zstd')
    # Persist compact original examples for manual near-copy and meaning review.
    evals=[]
    for fold in range(3):
        t=pq.read_table(old/('primary/fold_%s/SEMANTIC/evaluation.parquet'%fold),columns=['row_position','projection_id','route','label_index','pred_label']).to_pandas()
        evals.append(t[t.pred_label.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()!=t.label_index.to_numpy()])
    wrong=pd.concat(evals,ignore_index=True)
    near=pq.read_table(prep/'near_copy_candidates.parquet').to_pandas()
    counts=wrong[wrong.projection_id.isin(near.projection_id)].groupby(['route','projection_id']).size().sort_values(ascending=False)
    selected=[];perroute=collections.Counter()
    for (route,pid),n in counts.items():
        if perroute[route] >= (8 if route=='asa' else 2):continue
        perroute[route]+=1;selected.append(int(pid))
    examples={}
    for pid in selected:
        take=allowed[allowed.projection_id==pid].drop_duplicates('body_group').head(2)
        for r in take.itertuples(index=False):examples[int(r.row_position)]={'row_position':int(r.row_position),'projection_id':pid,'body_group':int(r.body_group),'fold':int(r.fold),'route':r.route}
    offset=0
    for batch in pq.ParquetFile(root/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized'],use_threads=False):
        for pos in sorted(set(examples).intersection(range(offset,offset+len(batch)))):
            text=batch.column(0)[pos-offset].as_py() or '';v=core.prior.prepare_message(text);cached=pr.iloc[examples[pos]['projection_id']]
            assert v['text']==cached.text and core.prior.canonical(v['facts'])==cached.facts
            examples[pos]['raw']=text
        offset+=len(batch)
    save(stage/'near_copy_examples.json',list(examples.values()))
    save(stage/'input_audit.json',{'allowed_rows':len(allowed),'body_keys':int(allowed.body_group.nunique()),
        'exact_nonempty_raw_fold_overlap':exact_overlap,'body_fold_overlap':0,'near_copy_candidate_projections':len(near),
        'near_copy_example_rows':len(examples),'semantic_alias_projection_counts':dict(changes),
        'dedicated_active_missing_code_removed':True,'missingness_information_removed':False,
        'missingness_note':'Zero vector still identifies absence; observed-value recoding changes the linear model geometry, not whether absence is recoverable.',
        'v39_provenance_reference':str(prep/'provenance.parquet'),
        'group_identity_not_verified_incident_identity':True,'manual_near_copy_review_required_before_fit':True})
    files={p.name:sha(p) for p in stage.iterdir() if p.is_file()}
    save(stage/'complete.json',{'version':core.VERSION,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'v39_prepared':str(prep),'v39_prepared_receipt_sha256':sha(prep/'complete.json'),
        'official_sha256':receipt['official_sha256'],'v37_raw_hash_metadata_sha256':sha(meta_path),
        'files':files,'runtime_sources':{p.name:sha(p) for p in runtime.glob('*.py')},'new_model_trained':False})
    print(json.dumps({'prepared':str(stage),'allowed_rows':len(allowed),'exact_raw_overlap':exact_overlap,'near_examples':len(examples),'alias_counts':dict(changes)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);main(p.parse_args())
