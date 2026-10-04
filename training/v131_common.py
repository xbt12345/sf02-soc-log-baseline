"""Immutable data and plan interfaces for execution of the V130 learning plan."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from experiment_review import sha, check_bindings, ReviewError
from v130_verify_learning_review import check_plan as check_design

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'training/review_policy/v130_learning_qualification_plan.json'
CONTRACT=ROOT/'training/review_policy/v131_execution_contract.json'
OUT=ROOT/'artifacts/v131_learning_trial_20260930'
REVIEW=ROOT/'artifacts/v130_learning_review_20260930_r2'
INPUT=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
TRACE=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
OFFICIAL=ROOT/'data/official/train.parquet'
MANIFEST=ROOT/'artifacts/v116_nested_selection_20260929/inner_split_manifest.parquet'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def check_plan():
    p=read(PLAN); check_design(p)
    c=read(CONTRACT)
    if c['plan_sha256']!=sha(PLAN) or c['version']!='V131-exec-V130':
        raise ReviewError('Wrong or changed execution plan')
    check_bindings(p['evidence_sha256'])
    if c['run_directory'] != OUT.relative_to(ROOT).as_posix():
        raise ReviewError('Wrong output directory')
    return p


def seal_run(trainer,sources):
    target=OUT/'run_seal.json'
    if target.exists():raise FileExistsError(target)
    check_plan()
    paths={Path(trainer).resolve(),PLAN,CONTRACT,Path(__file__),ROOT/'training/experiment_review.py'}
    paths.update(Path(p).resolve() for p in sources)
    bound={}
    for path in sorted(paths):
        if not path.is_file() or not path.is_relative_to(ROOT):raise ReviewError('Missing dependency '+str(path))
        bound[path.relative_to(ROOT).as_posix()]=sha(path)
    save(target,{'status':'sealed_before_fit','version':'V131-exec-V130',
        'trainer':Path(trainer).resolve().relative_to(ROOT).as_posix(),'plan_sha256':sha(PLAN),
        'source_sha256':bound,'quality_acceptance':False,'model_promoted':False})


def require_run_seal(trainer):
    s=read(OUT/'run_seal.json')
    if s['status']!='sealed_before_fit' or s['version']!='V131-exec-V130' or s['plan_sha256']!=sha(PLAN):
        raise ReviewError('Invalid run seal')
    if (ROOT/s['trainer']).resolve()!=Path(trainer).resolve():raise ReviewError('Wrong trainer')
    check_bindings(s['source_sha256'])
    return check_plan()


def require_checkpoint(meta,kind):
    endpoint=2000 if kind=='probe' else 100
    if meta.get('status')!='fit_executed' or meta.get('kind')!=kind or meta.get('endpoint')!=endpoint:
        raise ReviewError('Incomplete/unregistered endpoint')
    expected=2000 if kind=='probe' else meta['expected_steps']
    if meta.get('optimizer_steps')!=expected:raise ReviewError('Incomplete update budget')


def load_data():
    x=sparse.load_npz(INPUT).astype(np.float32).tocsr()
    d=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    y=pd.read_parquet(OFFICIAL,columns=['label_binary']).label_binary.map(
        {'benign':0,'malicious':1,'suspicious':2}).to_numpy(np.int8)
    if x.shape!=(22546,66287) or len(d)!=112807 or len(y)!=2056871:
        raise ReviewError('Input population changed')
    if not np.array_equal(y[d.row_position],d.truth) or d.row_position.duplicated().any():
        raise ReviewError('Independent official label mismatch')
    keys=pd.read_parquet(REVIEW/'training_role_error_ledger.parquet',columns=['local','canonical_key'])
    if keys.groupby('local').canonical_key.nunique().max()!=1:raise ReviewError('Key identity changed')
    keys=keys.drop_duplicates('local').set_index('local').canonical_key
    d['canonical_key']=d.local.map(keys)
    if d.canonical_key.isna().any():raise ReviewError('Incomplete actual-input map')
    return x,d


def fit_context(d,fold,n=22546):
    fit=d[d.fold!=fold].copy(); held=d[d.fold==fold]
    if set(fit.root)&set(held.root):raise ReviewError('Leaking source roots')
    c=np.bincount(fit.local.to_numpy(np.int64)*3+fit.truth.to_numpy(np.int64),minlength=n*3).reshape(n,3)
    if c[:,0].any() or int(c.sum())!=len(fit):raise ReviewError('Wrong original label mass')
    mix=fit.groupby('canonical_key').truth.nunique()
    pure=np.zeros(n,np.float32)
    for loc,key in fit[['local','canonical_key']].drop_duplicates().itertuples(index=False):
        pure[loc]=float(mix[key]==1)
    totals=(c*pure[:,None]).sum(0).astype(np.float64)
    if not (totals[1:]>0).all():raise ReviewError('Missing pure-input class')
    return fit,c,pure,totals,np.flatnonzero(c.sum(1))


def learning_gates(stats):
    p=read(PLAN)['learning_gate']
    pairs={
        'M_preserved':stats['M_errors']<=p['full_train_M_errors_max'],
        'S_total':stats['S_errors']<=p['full_train_S_errors_max'],
        'S_pure':stats['pure_S_errors']<=p['canonical_nonconflict_S_errors_max'],
        'hard_S':stats['hard_S_errors']<=p['hard_527_S_errors_max'],
        'matched_M':stats['matched_M_errors']<=p['matched_344_M_errors_max'],
        'old_correct_S':stats['old_correct_S_regressions']<=p['new_errors_on_B124_correct_train_S_max'],
        'hard_source_coverage':stats['hard_error_roots']<=p['hard_S_source_groups_with_errors_max']}
    return pairs
