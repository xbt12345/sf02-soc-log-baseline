"""V133 factorial runtime adapter, preserving the sealed historical review module."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_review import ROOT, ReviewError, read, sha, check_bindings, require_checkpoint
from v133_verify_targeted_plan import validate_plan
from v131_common import load_data, fit_context, INPUT, TRACE, OFFICIAL, REVIEW

PLAN=ROOT/'training/review_policy/v133_targeted_learning_plan.json'
CONTRACT=ROOT/'training/review_policy/v135_execution_contract.json'
OUT=ROOT/'artifacts/v135_stable_learning_trial_20260930'
ARMS=['R_const','R_decay','O_const','O_decay']


def save(path,value):
    Path(path).write_text(__import__('json').dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def check_plan():
    p=read(PLAN);validate_plan(p);check_bindings(p['evidence_sha256'])
    c=read(CONTRACT)
    if c['plan_sha256']!=sha(PLAN) or c['version']!='V135-exec-V133':
        raise ReviewError('Invalid execution contract')
    if c['candidate']!='R_decay' or c['output']!=OUT.relative_to(ROOT).as_posix():
        raise ReviewError('Wrong execution population/candidate')
    return p


def seal_run(trainer,sources):
    check_plan();destination=OUT/'run_seal.json'
    if destination.exists():raise FileExistsError('Never overwrite seal')
    paths={PLAN,CONTRACT,Path(trainer).resolve(),Path(__file__),ROOT/'training/experiment_review.py'}|set(sources)
    bound={}
    for p in sorted(paths):
        p=p.resolve()
        if not p.is_file() or not p.is_relative_to(ROOT):raise ReviewError('Unbound source '+str(p))
        bound[p.relative_to(ROOT).as_posix()]=sha(p)
    save(destination,{'status':'sealed_before_fit','version':'V135-exec-V133',
        'trainer_path':Path(trainer).resolve().relative_to(ROOT).as_posix(),
        'plan_sha256':sha(PLAN),'source_sha256':bound,'quality_acceptance':False})


def require_run_seal(trainer):
    p=check_plan();s=read(OUT/'run_seal.json')
    if s['status']!='sealed_before_fit' or s['version']!='V135-exec-V133' or s['plan_sha256']!=sha(PLAN):
        raise ReviewError('Invalid seal')
    if (ROOT/s['trainer_path']).resolve()!=Path(trainer).resolve():raise ReviewError('Trainer identity changed')
    # Reuse the actual execution-binding check; legacy one-factor plan dispatch
    # is not appropriate to a registered 2x2 factorial and stays immutable.
    check_bindings(s['source_sha256'])
    return p


def endpoint(receipt):
    require_checkpoint({'training_epochs':100,'selector':{'epoch':100}},receipt)
    if receipt['optimizer_steps']!=receipt['expected_steps']:raise ReviewError('Incomplete update budget')


def learning_rate(arm,epoch):
    if arm not in ARMS or not 1<=epoch<=100:raise ReviewError('Unregistered schedule')
    if arm.endswith('const') or epoch<=80:return .002
    return .00002+.5*(.002-.00002)*(1+math.cos(math.pi*(epoch-80)/20))


def stats(frame,prob,pure,old):
    loc=frame.local.to_numpy();y=frame.truth.to_numpy();pred=prob[loc].argmax(1)
    pr=pure[loc].astype(bool);wrong=pred!=y
    mix=frame.groupby(['canonical_key','truth']).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
    mixed=mix[(mix>0).sum(1)>1];floor=int((mixed.sum(1)-mixed.max(1)).sum())
    result={'M_errors':int((wrong&(y==1)).sum()),'S_errors':int((wrong&(y==2)).sum()),
        'pure_M_errors':int((wrong&pr&(y==1)).sum()),'pure_S_errors':int((wrong&pr&(y==2)).sum()),
        'empirical_minimum_errors':floor,'pure_errors':int((wrong&pr).sum()),
        'old_correct_pure_regressions':int((wrong&pr&(old==y)).sum()),
        'repaired_vs_V131_R':int(((old!=y)&~wrong).sum()),'new_errors_vs_V131_R':int(((old==y)&wrong).sum()),
        'all_mixed_keys_have_M_majority':bool((mixed[1]>mixed[2]).all()),
        'no_mixed_majority_reversal':bool(np.all(pred[frame.canonical_key.isin(mixed.index).to_numpy()]==1))}
    result['mastered']=result['M_errors']==0 and result['S_errors']==floor and result['pure_errors']==0
    return result


def guard_sources(actual,expected,epoch):
    if actual.epoch.nunique()!=1 or int(actual.epoch.iloc[0])!=epoch or actual.duplicated(['root','truth']).any():
        raise ReviewError('Overwritten/duplicated source epoch')
    a=actual.set_index(['root','truth']).support.sort_index()
    b=expected.sort_index()
    if not a.equals(b):raise ReviewError('Source/class original population changed')


def window_pass(history):
    if [h['epoch'] for h in history]!=list(range(1,101)):raise ReviewError('Incomplete epoch history')
    return all(h['stats']['mastered'] for h in history[95:100])


def confirmation_allowed(learning,quality):
    return bool(learning.get('R_decay_all_roles_mastered') and quality.get('candidate_quality_passed'))
