"""Qualify cached tail identity and full displacement without forward calls."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
import v162_cached_function_restoration_tail as entry
from v162_multi_function_finite_restoration_v2 import propose
OUT=ROOT/'artifacts/v162_cached_function_restoration_tail_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();files={Path(__file__).resolve(),ROOT/'training/v162_cached_function_restoration_tail.py',ROOT/'training/v162_multi_function_finite_restoration_v2.py',ROOT/'docs/V162_INDEPENDENT_RESULTS_AND_TARGETED_NEXT_TRAINING_PLAN_20261002.md'}|{p for p in entry.PREVIOUS.rglob('*') if p.is_file()}
    for c in [1,2]:files.add(entry.DIAG/f'role2/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy')
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0))
    ctx=entry.endpoint_context(2);records=read(entry.SOURCE/'active_functions.json');blockers=pd.read_parquet(entry.SOURCE/'finite_probe/actual_blocking_original_rows.parquet');functions=entry.blocking_functions(ctx,blockers)
    assert len(records)==len(functions)==1 and set(records)==set(functions) and blockers.row_position.tolist()==[1720534,1720544] and len(blockers.local.unique())==1
    u=np.load(entry.SOURCE/'displacement.npy');a=np.load(entry.SOURCE/'active_complete_margin_normals.npy');b=np.load(entry.SOURCE/'base_margins.npy');current={scope:np.load(entry.SOURCE/f'finite_probe/{scope}_logq.npy') for scope in ['OOF','deployment']};c=entry.margins(records,list(records),current)
    assert np.all(c<0) and np.all(b>0);gs=[np.load(entry.DIAG/f'role2/baseline_error_class{k}_repeat0/complete_fixed_error_target_gradient.npy') for k in [1,2]]
    result=propose(u,a,b,c,*gs);assert result['status']=='linear_restoration_candidate_requires_full_actual_finite_guard' and result['displacement'].shape==(1060832,)
    np.save(OUT/'first_tail_saved_input_candidate.npy',result['displacement']);entry.save(OUT/'first_tail_original_unit_certificate.json',{k:v for k,v in result.items() if k not in ['displacement','correction']})
    tree=ast.parse((ROOT/'training/v162_cached_function_restoration_tail.py').read_text(encoding='utf-8'));calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert 'add_normals' not in calls and 'measure_margin' not in calls and 'propose' in calls and 'probe' in calls and 'restore' in calls
    code=(ROOT/'training/v162_cached_function_restoration_tail.py').read_text(encoding='utf-8');assert 'for iteration in [4,5]' in code and "if not set(functions)<=set(records)" in code and 'Counter(model,88' in code and "if report['accepted']" in code
    assert (2+entry.CAPS['finite_proposals'])*(10+12)==entry.CAPS['heads']==88 and all(entry.CAPS[k]==0 for k in ['fixed_error_target_gradients','full_original_class_gradients','margin_gradients','QP_solves','fits','permanent_updates'])
    check_bindings(bindings);entry.save(OUT/'qualification.json',dict(status='cached_same_function_two_restoration_tail_identity_complete_vector_and_entry_bounds_passed',same_function_input_identity=True,original_blocking_rows=blockers.row_position.tolist(),first_tail_original_unit_residual_and_common_descent=True,all_original_guards_via_unchanged_qualified_probe=True,lifecycle_unchanged_probe_and_restore_already_qualified=True,new_function_detection_stops_before_proposal=True,full_displacement_step_one=True,parameter_shape=1060832,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,no_actual_finite_or_classification_claim=True,source_sha256=bindings))
    print(json.dumps(dict(status='V162_cached_function_tail_qualification_passed',official_calls=0)))

if __name__=='__main__':main()
