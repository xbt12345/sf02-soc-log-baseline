"""Supplement source closure before first call, never reopen registered caps."""
import ast,json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v162_run_qualified_finite_restoration import PLAN,SEAL,require_activation
import v162_fixed_endpoint_finite_restoration as entry

def main():
    assert not PLAN.exists() and not SEAL.exists() and not any((entry.OUT/f'role{r}').exists() for r in range(3))
    original=read(entry.OUT/'run_seal.json');check_bindings(original['source_sha256'])
    qualifier=ROOT/'artifacts/v162_finite_restoration_lifecycle_qualification_20261002';q=read(qualifier/'qualification.json');assert 'passed' in q['status'] and q['official_heads']==q['official_gradients']==q['official_fits']==0 and q['all_forward_results_mocked_from_cached_tables'];check_bindings(q['source_sha256'])
    files={ROOT/k for k in original['source_sha256']}|{ROOT/k for k in q['source_sha256']}|{p for p in qualifier.rglob('*') if p.is_file()}|{Path(__file__).resolve(),ROOT/'training/v162_run_qualified_finite_restoration.py',entry.OUT/'run_seal.json',entry.OUT/'registration.json'}
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    plan=dict(protocol='V162-post-seal-lifecycle-qualified-activation-v2',allowed_activation_entries=['training/v162_run_qualified_finite_restoration.py'],called_registered_entry='training/v162_fixed_endpoint_finite_restoration.py',registered_run_seal_sha256=sha(entry.OUT/'run_seal.json'),registered_plan_sha256=sha(entry.PLAN),registered_caps=read(entry.PLAN)['new_caps'],original_registration_has_zero_calls=True,original_registered_budget_not_reset_or_multiplied=True,additional_budget=0,lifecycle_same_function_second_restoration_and_actual_assignment_exception_restore_qualified=True,all_lifecycle_forward_tables_mocked_not_official_safety=True,new_fits=0,permanent_updates=0,source_sha256=bindings)
    entry.save(PLAN,plan);bindings[PLAN.relative_to(ROOT).as_posix()]=sha(PLAN)
    entry.save(SEAL,dict(status='registered_finite_diagnostic_qualified_for_first_actual_calls',allowed_activation_entries=plan['allowed_activation_entries'],activation_plan_sha256=sha(PLAN),source_sha256=bindings))
    require_activation();entry.save(entry.OUT/'activation_registration_v2.json',dict(status='lifecycle_supplement_bound_before_first_official_call',physical_sources=len(bindings),additional_budget=0,registered_caps=plan['registered_caps'],official_heads=0,official_gradients=0,official_fits=0,permanent_updates=0,activation_seal_sha256=sha(SEAL)))
    print(json.dumps(dict(status='V162_lifecycle_qualified_activation_sealed',physical_sources=len(bindings),additional_budget=0,registered_caps=plan['registered_caps'])))

if __name__=='__main__':main()
