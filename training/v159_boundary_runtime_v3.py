"""Prospective source/runtime seal for exactly six fixed V159 fits."""
import importlib.metadata,json,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v158_nested_runtime_v2 import physical
from experiment_review_v159_v2 import review_plan,seal_run,require_run_seal,require_checkpoint

OUT=ROOT/'artifacts/v159_class_boundary_trial_20261002'
BANK=ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'
PLAN=ROOT/'training/review_policy/v159_boundary_execution_contract_v3.json'
PREP=ROOT/'artifacts/v159_boundary_input_preparation_20261002'
INPUT=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz'

def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def review():
    p=read(PLAN)
    assert p['candidate_module']=='training/v159_current_input_boundary_v3.py' and p['instrumentation_only_refinement_of_v2']
    assert p['version']=='V159-six-fixed-current-input-class-boundary-execution' and p['arms']==['A','B']
    assert p['total_future_caps']['classifier_forward_chunks']==78072 and p['total_future_caps']['full_class_gradients']==2412
    assert p['activation_entries']==['training/v159_boundary_train_v3.py','training/v159_boundary_evaluate_v3.py']
    assert p['shape']['parameters']==1060832 and p['probability_base']['floor']==1e-12 and p['probability_base']['origin_probability_tolerance']==3e-12
    assert p['solver']['max_step']==1 and p['solver']['max_backtracks']==40 and p['seed']==15901
    assert review_plan(p)['plan_review_passed'];return p

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
    seal_run(PLAN,entry,files,OUT/'run_seal.json')

def require(entry):
    p=require_run_seal(OUT/'run_seal.json',entry)
    assert p['version']=='V159-six-fixed-current-input-class-boundary-execution'
    return p

def endpoint(r):
    require_checkpoint(read(PLAN),r)
    return True
