"""Prospectively bind the closed diagnostic's measured direction and tail cost."""
import ast,importlib.metadata,json,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v161_cached_direction_backtrack_tail import OUT,PLAN,DIAG,PROTOCOL,save,require

def main():
    assert not OUT.exists() and not PLAN.exists()
    prior=read(DIAG/'run_seal.json');check_bindings(prior['source_sha256'])
    qualifier=ROOT/'artifacts/v161_saved_curvature_and_tail_qualification_20261002';q=read(qualifier/'qualification.json');assert 'qualified' in q['status'] and q['official_heads']==q['official_gradients']==q['official_fits']==0;check_bindings(q['source_sha256'])
    audit=ROOT/'artifacts/v161_saved_fixed_error_result_audit_20261002/audit.json';a=read(audit);check_bindings(a['source_sha256']);assert a['actual_new_counts']['head_attempts']==1114 and a['actual_new_counts']['gradient_attempts']==12 and not a['all_roles_finite_error_target_pass']
    files={Path(__file__).resolve(),ROOT/'training/v161_cached_direction_backtrack_tail.py',ROOT/'docs/V161_ACTUAL_FIXED_ERROR_DIAGNOSTIC_AND_BACKTRACK_DOMAIN_20261002.md',audit}|{ROOT/k for k in prior['source_sha256']}|{ROOT/k for k in q['source_sha256']}|{p for p in DIAG.rglob('*') if p.is_file()}|{p for p in qualifier.rglob('*') if p.is_file()}
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    diag=read(DIAG/'role2/diagnostic.json');assert not diag['finite_error_target_pass'] and diag['exception'] is None
    plan=dict(protocol=PROTOCOL,activation_entries=['training/v161_cached_direction_backtrack_tail.py'],role=2,endpoint_parameter_sha256=diag['initial_parameter_sha256'],same_direction='artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role2/round0/polished_direction.npy',new_caps=dict(heads=484,features=484,fixed_error_target_gradients=0,full_original_class_gradients=0,margin_gradients=0,finite_proposals=20,QP_solves=0,fits=0,permanent_updates=0),prior_actual_costs=dict(heads=16842,features=16842,full_original_class_gradients=368,fixed_error_target_gradients=12,margin_gradients=10),future_cumulative_actual_caps=dict(heads=17326,features=17326,full_original_class_gradients=368,fixed_error_target_gradients=12,margin_gradients=10),preserved_technical_head_cap=82174,original_closed_20_depth_results_preserved=True,step_sequence='2**(-j), j=20..39',backtrack_tail_is_only_new_factor=True,no_new_direction_QP_gradient_or_normal=True,full_original_classification_and_error_target_gate_unchanged=True,all_original_mixed_rows_scored=True,fixed_endpoint_repairs_protected=True,every_candidate_and_exit_restore=True,quality_acceptance=False,source_sha256=bindings)
    OUT.mkdir();save(OUT/'pre_registration_bindings.json',dict(source_sha256=bindings,official_calls=0));save(PLAN,plan);bindings[PLAN.relative_to(ROOT).as_posix()]=sha(PLAN)
    save(OUT/'run_seal.json',dict(status='cached_same_direction_original40_tail_sealed_before_calls',protocol=PROTOCOL,allowed_entries=plan['activation_entries'],plan_sha256=sha(PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings))
    require();save(OUT/'registration.json',dict(status='V161_cached_same_direction_tail_registered',physical_sources=len(bindings),new_caps=plan['new_caps'],official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,run_seal_sha256=sha(OUT/'run_seal.json')))
    print(json.dumps(dict(status='V161_cached_tail_sealed',physical_sources=len(bindings),new_caps=plan['new_caps'])))

if __name__=='__main__':main()
