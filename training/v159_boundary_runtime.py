"""Prospective source/runtime seal for exactly six fixed V159 fits."""
import importlib.metadata,json,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v158_nested_runtime_v2 import physical

OUT=ROOT/'artifacts/v159_class_boundary_trial_20261002'
BANK=ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'
PLAN=ROOT/'training/review_policy/v159_boundary_execution_contract.json'
PREP=ROOT/'artifacts/v159_boundary_input_preparation_20261002'
INPUT=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz'

def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def review():
    p=read(PLAN)
    assert p['candidate_module']=='training/v159_current_input_boundary_v3.py' and p['instrumentation_only_refinement_of_v2']
    assert p['version']=='V159-six-fixed-current-input-class-boundary-execution' and p['arms']==['A','B']
    assert p['total_future_caps']['classifier_forward_chunks']==77832 and p['total_future_caps']['full_class_gradients']==2412
    assert p['activation_entries']==['training/v159_boundary_train.py','training/v159_boundary_evaluate.py']
    assert p['shape']['parameters']==1060832 and p['probability_base']['floor']==1e-12 and p['probability_base']['origin_probability_tolerance']==3e-12
    assert p['solver']['max_step']==1 and p['solver']['max_backtracks']==40 and p['seed']==15901
    check_bindings(p['source_sha256']);return p

def seal(entry,extras):
    p=review();q=read(PREP/'qualification.json')
    assert q['input_qualification_passed'] and q['official_classifier_calls']==q['official_gradients']==0
    check_bindings(q['source_sha256'])
    files=physical()|set(extras)|{Path(entry).resolve(),PLAN,PREP/'qualification.json'}|{ROOT/k for k in p['source_sha256']}|{ROOT/k for k in q['source_sha256']}
    # Transitive saved-teacher/input identities, not only their manifest bytes.
    files|={ROOT/k for k in read(BANK/'pre_saved_array_bindings.json')['source_sha256']}
    files|={ROOT/k for k in read(ROOT/'artifacts/v158_fusion_trial_20261001/run_seal.json')['source_sha256'] if not Path(k).is_absolute()}
    for f in range(3):files|=set((PREP/f'fold{f}').glob('*'))
    assert not (OUT/'run_seal.json').exists()
    save(OUT/'run_seal.json',dict(status='sealed_before_any_official_boundary_classifier_feature_gradient_or_update',
        allowed_entries=p['activation_entries'],plan_sha256=sha(PLAN),python_version=sys.version,
        package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},
        source_sha256={z.relative_to(ROOT).as_posix() if z.is_relative_to(ROOT) else str(z):sha(z) for z in sorted(q.resolve() for q in files)},
        driver_scope='OS and GPU driver not physically snapshotted',quality_acceptance=False))

def require(entry):
    p=review();s=read(OUT/'run_seal.json');actor=Path(entry).resolve().relative_to(ROOT).as_posix()
    assert s['status']=='sealed_before_any_official_boundary_classifier_feature_gradient_or_update'
    assert s['allowed_entries']==p['activation_entries'] and actor in s['allowed_entries'] and s['plan_sha256']==sha(PLAN) and s['source_sha256'][actor]==sha(entry)
    assert s['python_version']==sys.version and s['package_versions']=={n:importlib.metadata.version(n) for n in s['package_versions']}
    check_bindings(s['source_sha256']);return p

def endpoint(r):
    assert r['status']=='V159_boundary_fit_executed' and r['fold'] in [0,1,2] and r['arm'] in ['A','B'] and not r['selected_by_score']
    assert 0<=r['accepted_updates']<=r['gradient_iterations']<=200 and r['full_class_gradients']==2*r['gradient_iterations']
    assert r['accepted_updates']<=r['proposal_evaluations']<=600
    assert r['termination'] in ['gradient_iteration_budget','accepted_update_budget','proposal_budget','zero_direction_no_automatic_fallback','no_strict_common_descent_no_automatic_fallback','no_feasible_step']
    return True
