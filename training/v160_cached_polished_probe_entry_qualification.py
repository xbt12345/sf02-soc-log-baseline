"""Saved-row protection/progress and zero-gradient entry budget checks."""
import ast,json,shutil,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,sha,check_bindings
import v160_cached_polished_direction_finite_probe as entry
OUT=ROOT/'artifacts/v160_cached_polished_probe_entry_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir()
    original=entry.DIAG/'role0/round0/probe8';files={Path(__file__).resolve(),ROOT/'training/v160_cached_polished_direction_finite_probe.py',ROOT/'training/v160_fixed_endpoint_diagnostic_v3.py',ROOT/'training/v159_boundary_train_v4.py'}
    files|={original/f'{scope}_original_rows.parquet' for scope in ['OOF','deployment']}
    files|={entry.DIAG/f'role{role}/baseline_class1/OOF_original_rows.parquet' for role in range(3)}|{entry.DIAG/'role0/baseline_deployment/deployment_original_rows.parquet'}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};entry.save(OUT/'pre_saved_entry_bindings.json',dict(source_sha256=bindings,official_calls=0))
    protections=[]
    for role in [1,2]:
        r=entry.read_endpoint_rows(role);old=r.protected_correct.to_numpy();current=(r.pred.eq(r.truth)&r.pure_current_input).to_numpy();union=old|current
        assert not np.any(old&~union) and not np.any(union&~r.pure_current_input.to_numpy())
        repairs=r.pred.eq(r.truth)&~r.initial_correct&r.pure_current_input;assert np.all(union[repairs.to_numpy()])
        protections.append(dict(role=role,old_protected_original_rows=int(old.sum()),new_protected_original_rows=int(union.sum()),fixed_endpoint_pure_repairs_protected=int(repairs.sum()),mixed_correct_not_newly_frozen=True))
    assert protections[1]['fixed_endpoint_pure_repairs_protected']==14
    target=OUT/'saved_progress_case';target.mkdir()
    for scope in ['OOF','deployment']:shutil.copyfile(original/f'{scope}_original_rows.parquet',target/f'{scope}_original_rows.parquet')
    entry.record_endpoint_progress(target,0);progress=json.loads((target/'fixed_endpoint_progress.json').read_text(encoding='utf-8'))
    for scope in ['OOF','deployment']:assert progress[scope]['classification_changes_vs_fixed_endpoint']==progress[scope]['repairs_vs_fixed_endpoint']==progress[scope]['new_errors_vs_fixed_endpoint']==0
    assert progress['OOF']['old_error_margin_improved_rows']>0 and progress['OOF']['median_old_error_margin_change']>0
    # This is a synthetic counter attempt, not a model or gradient measurement.
    dummy=torch.nn.Linear(1,1,dtype=torch.float64);dummy.opinions=torch.nn.Identity();counter=entry.Counter(dummy,1,OUT/'synthetic_zero_gradient_counter.jsonl',0)
    try:counter.gradient_before(1,[0,1,1])
    except RuntimeError:pass
    else:raise AssertionError('Zero gradient cap was bypassed')
    assert counter.counts()['gradient_attempts']==0;counter.close()
    tree=ast.parse((ROOT/'training/v160_cached_polished_direction_finite_probe.py').read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['backward','step'] for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['measure','polish','solve_direction'] for n in ast.walk(tree))
    check_bindings(bindings)
    entry.save(OUT/'qualification.json',dict(status='cached_entry_fixed_endpoint_pure_repair_protection_saved_margin_progress_and_zero_gradient_budget_passed',protections=protections,actual_saved_safe_probe_progress=progress,no_new_inference_gradient_or_model_updates=True,no_new_solver_call_in_entry=True,official_heads=0,official_features=0,official_class_gradients=0,official_margin_gradients=0,official_fits=0,permanent_updates=0,source_sha256=bindings))
    print(json.dumps(dict(status='cached_polished_probe_entry_qualification_passed',official_calls=0,protected_repairs=protections)))

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),official_calls=0))
        raise
