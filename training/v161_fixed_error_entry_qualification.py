"""Qualify fixed target identities, entry acceptance and restore, no official calls."""
import ast,json,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
import v161_fixed_error_endpoint_diagnostic_v2 as entry
OUT=ROOT/'artifacts/v161_fixed_error_entry_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir()
    files={Path(__file__).resolve(),ROOT/'training/v161_fixed_error_endpoint_diagnostic_v2.py',ROOT/'training/v161_fixed_pure_error_risk.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v159_boundary_train_v4.py'}
    files|={p for p in entry.COHORT.rglob('*') if p.is_file()}
    files|={ROOT/'artifacts/v161_error_risk_synthetic_qualification_20261002/qualification.json',ROOT/'artifacts/v161_independent_actual_objective_counterexample_replay_20261002/review.json',ROOT/'artifacts/v161_synthetic_cumulative_correct_guard_review_20261002/review.json'}
    files|={entry.OLD/f'role{f}/baseline_class1/OOF_original_rows.parquet' for f in range(3)}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0))
    cohorts=[]
    for role in range(3):
        ctx=entry.endpoint_context(role);r=ctx['OOF_rows'];assert np.all(r.protected_correct<=r.pure_current_input)
        cohorts.append(dict(role=role,fixed_targets=ctx['target_counts'].sum(0).tolist(),protected_rows=int(r.protected_correct.sum()),complete_original_mass=ctx['mass'].tolist()))
    assert sum(sum(r['fixed_targets']) for r in cohorts)==6824
    assert [r['protected_rows'] for r in cohorts]==[88931,38919,90528]
    old=read(ROOT/'artifacts/v161_independent_actual_objective_counterexample_replay_20261002/review.json')['cases'][2]
    gate=entry.finite_step_review(old['focused_risk_before'],old['focused_risk_after'],[-1.,-1.],59770,32550,'B',.0625,True)
    assert not gate['accepted'] and np.all(np.array(old['whole_risk_drop'])>0)
    # Exercise the actual entry's acceptance wiring and finally restoration.
    model=torch.nn.Linear(1,1,bias=False,dtype=torch.float64);base=tuple(p.detach().clone() for p in model.parameters());direction=np.array([1.])
    toy=dict(fold=0,ids=np.array([0]),mass=np.array([0,101,101]),baseline_stats=dict(M_errors=1,S_errors=1))
    q=np.array([[.1,.45,.45]]);lp=np.log(q);empty=pd.DataFrame({'initial_correct':pd.Series(dtype=bool)})
    risk=dict(fixed_pure_error_contribution=np.array([.005,.005]),full_original_class_CE=np.array([.5,.5]))
    def stat(ctx,q,scope):return dict(M_errors=0,S_errors=0,protected_regressions=0,mastered=True,new_errors_vs_initial=0)
    common=[patch.object(entry,'error_risk',return_value=(risk,q,lp,None)),patch.object(entry,'probabilities',return_value=(q,lp)),patch.object(entry,'store_scope'),patch.object(entry,'actual_blockers',return_value=empty),patch.object(entry,'joint_check',return_value=dict(passed=True)),patch.object(entry,'record_endpoint_progress')]
    from contextlib import ExitStack
    with ExitStack() as stack:
        for mock in common:stack.enter_context(mock)
        stack.enter_context(patch.object(entry,'stats',side_effect=stat))
        folder=OUT/'whole_CE_increase_classification_repair';folder.mkdir()
        report,_=entry.probe(model,base,toy,folder,direction,1.,[.013,.013],[-.007,-.007])
        assert report['accepted'] and report['full_CE_not_acceptance_gate'] and torch.equal(next(model.parameters()),base[0])
    with ExitStack() as stack:
        for mock in common:stack.enter_context(mock)
        stack.enter_context(patch.object(entry,'stats',side_effect=lambda c,q,s:dict(stat(c,q,s),M_errors=2 if s=='OOF' else 0)))
        folder=OUT/'full_original_classification_regression';folder.mkdir()
        report,_=entry.probe(model,base,toy,folder,direction,1.,[.013,.013],[-.007,-.007])
        assert not report['accepted'] and not report['full_original_M_S_error_count_guard'] and torch.equal(next(model.parameters()),base[0])
    folder=OUT/'forced_failure_restore';folder.mkdir()
    with patch.object(entry,'error_risk',side_effect=RuntimeError('synthetic injected forward failure')):
        try:entry.probe(model,base,toy,folder,direction,1.,[.013,.013],[-.007,-.007])
        except RuntimeError:pass
        else:raise AssertionError('Injected failure swallowed')
    assert torch.equal(next(model.parameters()),base[0])
    model.opinions=torch.nn.Identity();counter=entry.ErrorCounter(model,dict(head_cap=1,fresh_margin_gradient_cap=2),OUT/'synthetic_counter.jsonl',np.array([[0,1,1]]))
    for cls in [1,1,2,2]:counter.gradient_before(cls,[0,101,101]);counter.gradient_after(cls,[0,101,101])
    try:counter.gradient_before(2,[0,101,101])
    except RuntimeError:pass
    else:raise AssertionError('Fixed error gradient budget bypassed')
    assert counter.gradient_attempts==counter.gradient_completed==4;counter.close()
    tree=ast.parse((ROOT/'training/v161_fixed_error_endpoint_diagnostic_v2.py').read_text(encoding='utf-8'))
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    text=ast.unparse(functions['run']);assert 'repeat_gradient(first[3], second[3])' in text and 'for repetition in range(2)' in text
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in ast.walk(tree))
    check_bindings(bindings)
    entry.save(OUT/'qualification.json',dict(status='fixed_error_entry_cohort_denominator_real_repeat_wiring_acceptance_and_exception_restore_passed',cohorts=cohorts,old_actual_misleading_CE_candidate_rejected=gate,toy_entry_acceptance_cases=True,complete_head_gradient_qualification='artifacts/v161_error_risk_synthetic_qualification_20261002/qualification.json',official_heads=0,official_features=0,official_error_gradients=0,official_margin_gradients=0,official_fits=0,permanent_updates=0,source_sha256=bindings))
    print(json.dumps(dict(status='V161_entry_qualification_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as err:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(err).__name__,error=str(err),traceback=traceback.format_exc(),official_calls=0))
        raise
