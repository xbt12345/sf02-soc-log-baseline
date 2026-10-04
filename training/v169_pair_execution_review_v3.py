"""Full inherited registry identity and physical budget review."""
from pathlib import Path
from experiment_review import ROOT,read,sha
from v169_pair_execution_review import review_plan as base_review
from v169_pair_execution_review_v2 import require_run_seal as resource_require

def registry_chain():
    path='training/review_policy/v168_observed_runtime_boundaries.json';seen=set();result=[]
    while path:
        if path in seen:raise ValueError('Historical registry cycle')
        seen.add(path);value=read(ROOT/path)
        result.append(dict(path=path,sha256=sha(ROOT/path),actions={c['id']:c['required_action'] for c in value['cases']}))
        path=value.get('inherit_applicable_constraints')
    return result

def review_plan(plan):
    result=base_review(plan)
    if plan['applicable_failure_registry_chain']!=registry_chain():raise ValueError('Complete historical action chain missing or changed')
    if plan['resources']!=read(ROOT/plan['resource_review_path'])['resources']:raise ValueError('Actual worst resource budget identity changed')
    if plan['batch_global_fault_stops_remaining_fits'] is not True or plan['same_point_final_replay_required'] is not True:raise ValueError('Required failure/replay guard missing')
    return result

def require_run_seal(path,current_trainer,phase='running'):
    plan=resource_require(path,current_trainer,phase);review_plan(plan);return plan
