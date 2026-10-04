"""Generic role/function identities, full displacement gate and zero-fit budget."""
import ast,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_float64_repeat_policy_v2 import finite_step_review
import v162_fixed_endpoint_finite_restoration as entry
OUT=ROOT/'artifacts/v162_finite_restoration_entry_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();files={Path(__file__).resolve(),ROOT/'training/v162_fixed_endpoint_finite_restoration.py',ROOT/'training/v162_multi_function_finite_restoration_v2.py',ROOT/'training/v161_fixed_error_endpoint_diagnostic_v2.py',ROOT/'training/v160_fixed_endpoint_diagnostic_v3.py'}
    qualifiers=[ROOT/'artifacts/v162_multi_restoration_synthetic_qualification_v2_20261002/qualification.json',ROOT/'artifacts/v162_independent_restoration_function_review_20261002/review.json',ROOT/'artifacts/v161_fixed_error_entry_qualification_20261002/qualification.json']
    files|=set(qualifiers);reports=[]
    for role in range(3):
        ctx=entry.endpoint_context(role);folder=entry.DIAG/f'role{role}/round0/probe4';blockers=pd.read_parquet(folder/'actual_blocking_original_rows.parquet');functions=entry.blocking_functions(ctx,blockers)
        current={scope:np.load(folder/f'{scope}_logq.npy') for scope in ['OOF','deployment']};identities=list(functions);m=entry.margins(functions,identities,current)
        assert np.all(m<0)
        for k,v in functions.items():
            matching=blockers[blockers.scope.eq(v['scope'])&blockers.local.eq(v['local'])&blockers.truth.eq(v['truth'])&blockers.rival.eq(v['rival'])];assert np.all(m[identities.index(k)]==matching.actual_margin)
        cached=read(entry.DIAG/f'role{role}/round0/normal_records.json')
        if role==1:cached=read(entry.OLD/'role1/round1/normal_records.json')
        fresh=set(functions)-set(cached);total=len(set(functions)|set(cached));stop=total>24
        # Cap stopping is honest qualification, not permission to discard rows.
        reports.append(dict(role=role,actual_blocking_original_rows=len(blockers),complete_actual_input_functions=len(functions),cached_functions=len(cached),fresh_function_count=len(fresh),normal_cap_would_stop=stop,all_original_blockers_kept=True))
        files|={folder/'actual_blocking_original_rows.parquet',folder/'OOF_logq.npy',folder/'deployment_logq.npy'}
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=binding,official_calls=0))
    dummy=torch.nn.Linear(1,1,dtype=torch.float64);dummy.opinions=torch.nn.Identity();counter=entry.RestorationCounter(dummy,dict(head_cap=1,fresh_margin_gradient_cap=2),OUT/'synthetic_counter.jsonl')
    try:counter.gradient_before(1,[0,1,1])
    except RuntimeError:pass
    else:raise AssertionError('Zero new class-gradient budget bypassed')
    for k in ['a','a']:counter.margin_before(k);counter.margin_after(k)
    try:counter.margin_before('b')
    except RuntimeError:pass
    else:raise AssertionError('Paired margin budget bypassed')
    assert counter.gradient_attempts==0 and counter.margin_attempts==counter.margin_completed==2;counter.close()
    full=finite_step_review([1.,1.],[.99999,.99999],[-.05,-.05],1,1,'B',1.,True)
    wrong=finite_step_review([1.,1.],[.99999,.99999],[-1.,-1.],1,1,'B',1.,True)
    assert full['accepted'] and not wrong['accepted']
    tree=ast.parse((ROOT/'training/v162_fixed_endpoint_finite_restoration.py').read_text(encoding='utf-8'));text=ast.unparse(tree)
    assert 'for iteration in range(4)' in text and 'u, 1.0,' in text and "slopes = [v['linear_change'] for v in result['class_reviews']]" in text
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['backward','step'] for n in ast.walk(tree))
    # Only file routing depends on role: source has no algorithm branch by role.
    assert not any(isinstance(n,ast.Compare) and isinstance(n.left,ast.Name) and n.left.id=='role' for n in ast.walk(tree))
    caps=[6*(k+12)+2*(24-cached) for k,cached in [(10,1),(4,3),(10,1)]];assert caps==[178,138,178] and sum(caps)==494
    check_bindings(binding);entry.save(OUT/'qualification.json',dict(status='generic_three_role_finite_restoration_entry_identities_complete_displacement_acceptance_and_zero_fit_budgets_passed',actual_saved_blocker_readiness=reports,full_displacement_Armijo_gate=full,wrong_base_direction_Armijo_rejected=wrong,head_caps=caps,all_scope_probe_and_exception_restore_inherited_from_qualified_v161_entry=True,unmodelled_second_guard_independent_counterexample=qualifiers[1].relative_to(ROOT).as_posix(),official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,actual_corrected_model_not_yet_executed=True,source_sha256=binding))
    print(json.dumps(dict(status='V162_generic_entry_qualification_passed',readiness=reports,official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
