"""Separate activation of six fixed fusion fits after all 54 legal teachers."""
import importlib.metadata,sys
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v158_nested_runtime_v2 import PLAN,physical,review as base_review

OUT=ROOT/'artifacts/v158_fusion_trial_20261001'
BANK=ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'
CONTRACT=ROOT/'training/review_policy/v158_fusion_execution_contract.json'

def save(p,v):Path(p).write_text(__import__('json').dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def review():
    p=base_review();c=read(CONTRACT)
    expected=dict(version='V158-six-fixed-probability-fusion',fits=6,folds=[0,1,2],arms=['A','B'],candidate='B',
        plan_sha256=sha(PLAN),expert_count=17,condition_width=527,full_gradients_per_fit=200,
        proposals_per_fit=600,accepted_updates_per_fit=200,preflight_gradients=0,
        activation_entries=['training/v158_fusion_train.py','training/v158_fusion_evaluate.py'],
        endpoint=p['fusion_endpoint'],solver=p['fusion_solver'],OOF_query_labels_used_for_base_fit=False,
        deployment_FIT_guard_labels_are_legal=True,outer_labels_used_for_selection=False,
        automatic_repeat=False,automatic_confirmation=False,quality_acceptance=False)
    for k,v in expected.items():
        if c.get(k)!=v:raise ValueError('Changed registered fusion execution '+k)
    check_bindings(c['source_sha256']);q=read(BANK/'qualification.json')
    if q['base_fits']!=45 or q['legacy_fits']!=9 or q['fusion_fits']!=0 or not q['init_joint_TRAIN_retention']['passed']:raise ValueError('Incomplete legal bank')
    check_bindings(q['source_sha256']);check_bindings(read(BANK/'pre_saved_array_bindings.json')['source_sha256'])
    expected_budgets=[]
    for f in range(3):
        import pandas as pd,math
        frame=pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet');k=math.ceil(frame.local.nunique()/2048)
        expected_budgets.append(dict(fold=f,fit_original_rows=len(frame),fit_locals=frame.local.nunique(),fit_chunks=k,
            preflight_classifier_forward_cap_per_arm=2*k+24,fit_classifier_forward_cap_per_arm=1404*k+12,
            evaluation_classifier_forward_cap_per_arm=6*12+k,global_gradient_cap_per_arm=200))
    if c['role_call_budgets']!=expected_budgets:raise ValueError('Wrong original-population fusion call budgets')
    return p,c

def seal(entry,extras):
    _,c=review();assert not (OUT/'run_seal.json').exists()
    files=physical()|set(extras)|{Path(entry).resolve(),Path(__file__).resolve(),PLAN,CONTRACT}
    files|={ROOT/k for k in c['source_sha256']}
    # Include the actual saved probability sources/teacher endpoints, not just
    # the array manifest's bytes. New arrays are frozen before classifier calls.
    files|={ROOT/k for k in read(BANK/'pre_saved_array_bindings.json')['source_sha256']}
    files|={ROOT/k for k in read(BANK/'qualification.json')['source_sha256']}
    save(OUT/'run_seal.json',dict(status='sealed_before_any_fusion_classifier_forward_gradient_or_update',
        plan_sha256=sha(PLAN),contract_sha256=sha(CONTRACT),allowed_entries=c['activation_entries'],
        python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},
        source_sha256={p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p):sha(p) for p in sorted(z.resolve() for z in files)},
        driver_scope='OS and GPU driver not physically snapshotted',quality_acceptance=False))

def require(entry):
    p,c=review();s=read(OUT/'run_seal.json');actor=Path(entry).resolve().relative_to(ROOT).as_posix()
    if s['status']!='sealed_before_any_fusion_classifier_forward_gradient_or_update' or s['plan_sha256']!=sha(PLAN) or s['contract_sha256']!=sha(CONTRACT) or s['allowed_entries']!=c['activation_entries'] or actor not in s['allowed_entries'] or s['source_sha256'].get(actor)!=sha(entry):raise ValueError('Invalid actual fusion entry/seal')
    if s['python_version']!=sys.version or s['package_versions']!={n:importlib.metadata.version(n) for n in s['package_versions']}:raise ValueError('Fusion runtime changed')
    check_bindings(s['source_sha256']);return p,c

def endpoint(r):
    if r['status']!='fusion_fit_executed' or r['fold'] not in [0,1,2] or r['arm'] not in ['A','B'] or r['selected_by_score']:raise ValueError('Invalid registered fusion endpoint')
    if not 0<=r['accepted_updates']<=r['full_gradients']<=200 or not r['accepted_updates']<=r['proposal_evaluations']<=600:raise ValueError('Fusion resource budget exceeded')
    if r['termination'] not in ['gradient_budget','proposal_budget','accepted_update_budget','zero_gradient','no_feasible_step']:raise ValueError('Unregistered stop')
    return True
