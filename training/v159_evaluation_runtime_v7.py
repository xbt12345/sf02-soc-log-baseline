"""Sealed evaluation-only recovery; old training contract/seal remain unchanged."""
import json,sys,importlib.metadata
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings,ReviewError
from v159_boundary_runtime_v4 import OUT,PREP,save,endpoint,require as require_training
REPAIR=ROOT/'artifacts/v159_evaluation_cached_continuation_20261002'
PLAN=ROOT/'training/review_policy/v159_evaluation_cached_continuation_contract.json'

def require(entry):
    seal=read(REPAIR/'run_seal.json');actor=Path(entry).resolve().relative_to(ROOT).as_posix()
    if actor!='training/v159_boundary_evaluate_v7.py' or seal['allowed_entries']!=[actor] or seal['protocol']!='V159-evaluation-only-cached-source-CE-continuation-v7':raise ReviewError('Wrong evaluation-only recovery actor')
    if seal['plan_sha256']!=sha(PLAN) or seal['source_sha256'].get(actor)!=sha(entry):raise ReviewError('Changed recovery plan or actual source')
    check_bindings(seal['source_sha256']);q=read(PLAN);check_bindings(q['source_sha256'])
    if q['fresh_replay_head_cap']!=280 or q['failed_evaluation_head_calls']!=110 or q['stage_cumulative_evaluation_head_cap']!=750 or q['net_stage_technical_increment']!=30 or q['new_fit_cap']!=0 or q['new_gradient_cap']!=0 or q['new_update_cap']!=0:raise ReviewError('Changed recovery cost or training authority')
    if q['cached_actual_v5_head_calls']!=360 or q['cached_scope_keys']!=['0_A','0_B','1_A'] or sum(q['fresh_replay_role_caps'].values())!=280:raise ReviewError('Changed cached scopes or remaining calls')
    if seal['python_version']!=sys.version or seal['package_versions']!={n:importlib.metadata.version(n) for n in seal['package_versions']}:raise ReviewError('Recovery runtime changed')
    p=require_training(ROOT/'training/v159_boundary_train_v4.py')
    p['evaluation_repair_role_caps']=q['fresh_replay_role_caps'];p['failed_evaluation_head_calls']=110
    p['original_cumulative_caps_before_evaluation_repair']=dict(p['total_cumulative_caps'])
    p['total_cumulative_caps']=dict(p['total_cumulative_caps']);p['total_cumulative_caps']['classifier_forward_chunks']=p['total_cumulative_caps']['opinion_feature_blocks']=78226;p['total_cumulative_caps']['evaluation_classifier_forward_chunks']=750
    p['evaluation_repair_contract_path']=PLAN.relative_to(ROOT).as_posix();return p
