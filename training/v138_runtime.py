"""Execution adapter for V137; the historical fixed-epoch API stays immutable."""
import sys
from pathlib import Path
import importlib.metadata
import torch
from experiment_review import ROOT, read, sha, check_bindings, ReviewError
from v137_issue_guard import PLAN, validate_plan, adversaries
from v135_runtime import require_run_seal as require_history

OUT = ROOT/'artifacts/v138_single_issue_round1_20260930'
CONTRACT = ROOT/'training/review_policy/v138_execution_contract.json'


def save(path, value):
    import json
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def review_execution():
    p=read(PLAN);validate_plan(p);check_bindings(p['evidence_sha256'])
    r=read(CONTRACT)
    if r['plan_sha256']!=sha(PLAN) or r['candidate']!='H_L' or r['round']!=1:
        raise ReviewError('Changed issue, round or candidate')
    if r['fits_max']!=6 or r['gradient_evaluations_max_per_fit']!=200 or r['accepted_updates_max_per_fit']!=200:
        raise ReviewError('Unregistered optimizer budget')
    if r['output']!=OUT.relative_to(ROOT).as_posix() or r['held_score_stopping'] or r['automatic_round2']:
        raise ReviewError('Unregistered selection or second round')
    if r['torch_lbfgs']!={'max_iter_per_call':1,'max_eval_per_call':200,'tolerance_grad':1e-7,'tolerance_change':1e-9,'history_size':20,'line_search_fn':'strong_wolfe','lr':1}:
        raise ReviewError('Changed numerical method')
    require_history(ROOT/'training/v135_train.py')
    return p


def seal_run(trainer, extras):
    review_execution()
    target=OUT/'run_seal.json'
    if target.exists():raise FileExistsError(target)
    paths={PLAN,CONTRACT,Path(trainer).resolve(),Path(__file__).resolve(),ROOT/'training/experiment_review.py'} | set(extras)
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name:
            q=Path(name).resolve()
            if q.is_file() and q.suffix=='.py' and q.is_relative_to(ROOT) and not q.relative_to(ROOT).parts[0].startswith('.venv'):
                paths.add(q)
    history=read(ROOT/'artifacts/v135_stable_learning_trial_20260930/run_seal.json')
    paths.update(ROOT/x for x in history['source_sha256'])
    for rel in read(PLAN)['evidence_sha256']:paths.add(ROOT/rel)
    registry=read(ROOT/read(PLAN)['retention']['registry_path'])
    paths.add(ROOT/read(PLAN)['retention']['registry_path'])
    paths.update(ROOT/e['evidence'] for e in registry['engineering_contracts'])
    bindings={p.resolve().relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','pyarrow','tabm','torch']}
    save(target,{'status':'sealed_before_fit','version':'V138-exec-V137-round1','trainer_path':Path(trainer).resolve().relative_to(ROOT).as_posix(),
        'plan_sha256':sha(PLAN),'source_sha256':bindings,'packages':packages,'quality_acceptance':False,'model_promoted':False,
        'review_adapter':'Uses experiment_review.check_bindings; epoch-based stopping API is inapplicable to budgeted LBFGS.'})


def require_run_seal(trainer):
    p=review_execution();s=read(OUT/'run_seal.json')
    if s['status']!='sealed_before_fit' or s['plan_sha256']!=sha(PLAN) or (ROOT/s['trainer_path']).resolve()!=Path(trainer).resolve():
        raise ReviewError('Wrong runtime seal')
    check_bindings(s['source_sha256'])
    if s['packages']!={k:importlib.metadata.version(k) for k in s['packages']}:
        raise ReviewError('Runtime package versions changed')
    return p


def endpoint(receipt):
    if receipt['status']!='fit_executed' or receipt['arm'] not in ['H_A','H_L'] or receipt['candidate_selected_by_score']:
        raise ReviewError('Unregistered endpoint')
    n=receipt['full_gradient_evaluations'];u=receipt['accepted_updates']
    if not 0<=n<=200 or not 0<=u<=200 or u>n:
        raise ReviewError('Exceeded budget')
    if receipt['arm']=='H_A' and (n!=200 or u!=200 or receipt['termination']!='fixed_update_budget'):
        raise ReviewError('Adam terminated early')
    if receipt['arm']=='H_L' and receipt['termination'] not in ['gradient_budget','gradient_budget_trial_rolled_back','accepted_update_budget','numerical_no_change']:
        raise ReviewError('Unregistered LBFGS stopping')
    return True
