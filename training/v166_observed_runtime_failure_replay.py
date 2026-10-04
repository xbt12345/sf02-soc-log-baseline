"""Replay actual V166 runtime failure boundaries, no model or optimizer calls."""
import json,traceback
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v166_observed_runtime_failure_replay_20261002'
REGISTRY=ROOT/'training/review_policy/v166_observed_runtime_boundaries.json'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists();OUT.mkdir();registry=read(REGISTRY);assert registry['actual_diagnostic']=='V166' and registry['no_new_execution_authority'];sources={Path(__file__).resolve(),REGISTRY,ROOT/'training/experiment_review.py'}|{ROOT/r['evidence'] for r in registry['cases']};reviews={p:read(p) for p in sources if p.suffix=='.json'}
    inherited=ROOT/registry['inherit_applicable_constraints'];sources.add(inherited);sources.add(ROOT/read(inherited)['inherit_applicable_constraints'])
    for p,value in reviews.items():
        if 'source_sha256' in value:check_bindings(value['source_sha256'])
    independent=ROOT/'artifacts/v166_independent_actual_coverage_review_20261002/pre_review_bindings.json';check_bindings(read(independent)['source_sha256']);sources.add(independent)
    for role in range(3):sources.update(ROOT/f'artifacts/v166_coverage_first_diagnostic_20261002/role{role}/{name}' for name in ['diagnostic.json','restoration_review.json','treatment/probe.json','paired_treatment_vs_V164_endpoint.json','joint_restoration/original_unit_restoration_review.json'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));cases={};trial=ROOT/'artifacts/v166_coverage_first_diagnostic_20261002';geometry=read(ROOT/registry['cases'][0]['evidence']);coverage=read(ROOT/registry['cases'][1]['evidence']);quality=read(ROOT/registry['cases'][2]['evidence']);root=read(ROOT/registry['cases'][3]['evidence']);actual=read(ROOT/registry['cases'][4]['evidence'])
    failures=geometry['roles'][1]['known_failure_rows'];assert len(failures)==5 and all(r['original_unit_certificate_passed'] and r['actual_margin_is_negative'] and r['finite_negative_resolved_beyond_arithmetic'] for r in failures);assert all(r['predicted_control_anchored_margin']>=-max(r['original_unit_dot_errors']['correction'],r['original_scale_resolution']) for r in failures);probe=read(trial/'role1/treatment/probe.json');assert not probe['accepted'] and not probe['classification_guard'];assert all(r['passed'] for r in read(trial/'role1/joint_restoration/original_unit_restoration_review.json')['inequality_reviews']);cases[registry['cases'][0]['id']]=dict(passed=True,actual_measured_counterexamples=5,actual_finite_candidate_rejected=True)
    scope=coverage['roles'][1];assert scope['previously_measured_blocking_functions']==5 and scope['fresh_unmeasured_blocking_functions']==0 and scope['observed_union_if_added']==scope['registered_capacity']==25;assert {r['identity'] for r in failures}==set(scope['known_bad_input_identities']);cases[registry['cases'][1]['id']]=dict(passed=True,no_missing_function_claim=True,no_capacity_extension_permission=True)
    paired=read(trial/'role1/paired_treatment_vs_V164_endpoint.json');assert paired['M']['repairs_vs_V164_endpoint']==16 and paired['S']['new_errors_vs_V164_endpoint_all_original_rows']==paired['S']['registered_cumulative_protection_regressions']==10;assert quality['roles'][1]['pure_errors_before']==1952 and quality['roles'][1]['pure_errors_after']==1946 and not quality['roles'][1]['actual_candidate_accepted'];assert probe['finite_error_target_review']['accepted'];cases[registry['cases'][2]['id']]=dict(passed=True,total_error_drop_cannot_override_protected_class_failure=True)
    accepted=[read(trial/f'role{r}/treatment/probe.json')['accepted'] for r in range(3)];assert accepted==[True,False,True] and not root['all_three_actual_finite_guards_passed'] and not root['supports_new_short_training_registration'];assert root['any_actual_pure_classification_gain'];cases[registry['cases'][3]['id']]=dict(passed=True,real_positive_controls_retained=[0,2],failed_role1_retained=True)
    for role in range(3):
        diag=read(trial/f'role{role}/diagnostic.json');restored=read(trial/f'role{role}/restoration_review.json');assert diag['exception'] is None and diag['initial_parameter_sha256']==diag['restored_parameter_sha256'] and restored['all_parameter_tensors_exact'];assert diag['new_fits']==diag['permanent_updates']==0
    assert actual['cumulative_heads']==18784 and actual['cumulative_all_complete_derivatives']==618 and actual['latest_actual_training']=='V164' and actual['latest_complete_quality_delivery']=='V159' and actual['all_restored_full_tensors_exact'];cases[registry['cases'][4]['id']]=dict(passed=True,all_actual_endpoints_restored=True,temporary_gains_not_submitted=True,cost_not_reset=True)
    assert set(cases)=={r['id'] for r in registry['cases']};check_bindings(bindings);save(OUT/'replay.json',dict(status='V166_five_actual_observed_runtime_constraints_replayed',cases=cases,official_heads=0,official_features=0,official_derivatives=0,optimizer_calls=0,fits=0,permanent_updates=0,no_new_execution_authority=True,classification_mastery=False,root_goal_complete=False,source_sha256=bindings));print(json.dumps(dict(status='V166_actual_runtime_failure_constraints_replayed',cases=len(cases),official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
