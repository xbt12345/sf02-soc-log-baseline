"""Bounded-entry protocol tests with synthetic records and mocked model IO."""
import ast,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,sha,check_bindings
import v160_fixed_endpoint_diagnostic_v2 as entry
OUT=ROOT/'artifacts/v160_diagnostic_entry_synthetic_qualification_v2_20261002'

def main():
    assert not OUT.exists();OUT.mkdir()
    paths=[Path(__file__),ROOT/'training/v160_fixed_endpoint_diagnostic_v2.py',ROOT/'training/v160_margin_normal.py',ROOT/'training/v160_active_margin_direction_v4.py']
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths};entry.save(OUT/'pre_synthetic_bindings.json',dict(source_sha256=bindings,official_calls=0))
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.1,.2],dtype=torch.float64));self.opinions=torch.nn.Identity()
        def forward(self,x):return self.opinions(x)+self.weight
    model=Dummy();counter=entry.DiagnosticCounter(model,2,OUT/'synthetic_counter_events.jsonl')
    model(torch.zeros(2));model(torch.zeros(2));rejected=[]
    try:model(torch.zeros(2))
    except RuntimeError:rejected.append('head_cap')
    for cls in [1,2]:counter.gradient_before(cls,[0,2,2]);counter.gradient_after(cls,[0,2,2])
    try:counter.gradient_before(1,[0,2,2])
    except RuntimeError:rejected.append('class_gradient_cap')
    for j in range(48):counter.margin_before(str(j));counter.margin_after(str(j))
    try:counter.margin_before('49')
    except RuntimeError:rejected.append('margin_gradient_cap')
    counts=counter.counts();counter.close();assert counts['head_attempts']==counts['head_completed']==counts['feature_attempts']==counts['feature_completed']==2 and counts['gradient_attempts']==counts['gradient_completed']==2 and counts['margin_attempts']==counts['margin_completed']==48 and len(rejected)==3
    rr=pd.DataFrame(dict(row_position=[10,11,12,13],root=[7,7,8,9],local=[0,0,1,2],truth=[1,1,2,2],initial_correct=[True,True,True,False],protected_correct=[True,True,False,False]))
    ctx=dict(fold=0,OOF_rows=rr,deployment_rows=rr,ids=np.arange(3))
    q=np.array([[.05,.45,.5],[.02,.9,.08],[.1,.2,.7]])
    lp=np.log(q);ob=entry.actual_blockers(ctx,q,lp,'OOF');db=entry.actual_blockers(ctx,q,lp,'deployment')
    assert ob.row_position.tolist()==[10,11] and db.row_position.tolist()==[10,11,12] and ob.rival.tolist()==[2,2] and db.rival.tolist()==[2,2,1]
    assert ob.protection_kind.eq('initial_pure_OOF_correct').all() and db.protection_kind.eq('accepted_deployment_initial_correct').all()
    # A failure after assigning a probe must restore every original tensor.
    base=tuple(p.detach().clone() for p in model.parameters());saved_risk=entry.risk
    def deliberate_failure(*a,**k):raise RuntimeError('synthetic_after_assignment_failure')
    entry.risk=deliberate_failure;target=OUT/'restoration_case';target.mkdir()
    try:entry.probe(model,base,ctx,target,np.array([1.,-1.]),.5,np.array([1.,1.]),[-1.,-1.],None)
    except RuntimeError as e:assert str(e)=='synthetic_after_assignment_failure'
    else:raise AssertionError('Probe exception was swallowed')
    finally:entry.risk=saved_risk
    assert all(torch.equal(p,b) for p,b in zip(model.parameters(),base))
    tree=ast.parse((ROOT/'training/v160_fixed_endpoint_diagnostic_v2.py').read_text(encoding='utf-8'))
    functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert 'diagnose' in functions and 'probe' in functions and isinstance(next(n for n in ast.walk(functions['probe']) if isinstance(n,ast.Try)),ast.Try)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['save','step'] and isinstance(n.func.value,ast.Name) and n.func.value.id in ['torch','optimizer'] for n in ast.walk(tree))
    check_bindings(bindings)
    result=dict(status='synthetic_entry_budget_blocker_identity_and_exception_restoration_passed',budget_caps_rejected=rejected,mock_counter_counts=counts,duplicate_original_blocker_rows_preserved=True,scope_protection_distinct=True,probe_exception_restores_original_parameters=True,no_model_checkpoint_or_optimizer_update_call=True,actual_synthetic_dummy_forwards=2,synthetic_dummy_attempted_over_cap=1,synthetic_counter_only_gradient_events=True,official_heads=0,official_features=0,official_class_gradients=0,official_margin_gradients=0,official_fits=0,official_updates=0,source_sha256=bindings)
    entry.save(OUT/'qualification.json',result);print(json.dumps(dict(status=result['status'],official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),official_calls=0))
        raise
