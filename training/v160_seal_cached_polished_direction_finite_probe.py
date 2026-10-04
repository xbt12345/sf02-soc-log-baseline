"""Bind cached direction, complete original function and prospective cost."""
import ast,importlib.metadata,json,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v160_cached_polished_direction_finite_probe import OUT,PLAN,PRIOR,DIAG,QUAL,PROTOCOL,save,require

def main():
    assert not OUT.exists() and not PLAN.exists();qualification=read(QUAL/'qualification.json');assert 'passed' in qualification['status'] and qualification['official_heads']==qualification['official_class_gradients']==qualification['official_margin_gradients']==qualification['official_fits']==0
    check_bindings(qualification['source_sha256']);prior=read(DIAG/'run_seal.json');check_bindings(prior['source_sha256'])
    entry_qualification=ROOT/'artifacts/v160_cached_polished_probe_entry_qualification_20261002/qualification.json'
    eq=read(entry_qualification);assert 'passed' in eq['status'] and eq['official_heads']==eq['official_class_gradients']==eq['official_fits']==0;check_bindings(eq['source_sha256'])
    files={Path(__file__).resolve(),ROOT/'training/v160_cached_polished_direction_finite_probe.py',ROOT/'docs/V160_SAVED_VECTOR_NUMERIC_POLISH_QUALIFICATION_AND_NEXT_PROBE.md',QUAL/'qualification.json',ROOT/'artifacts/v160_complete_fixed_endpoint_result_records_20261002/actual_result_summary.json'}|{ROOT/k for k in qualification['source_sha256']}|{ROOT/k for k in prior['source_sha256']}|{p for p in DIAG.rglob('*') if p.is_file()}|{p for p in QUAL.rglob('*') if p.is_file()}
    roles=[]
    files.add(entry_qualification);files|={ROOT/k for k in eq['source_sha256']}
    for role,k,round_number in [(1,4,1),(2,10,0)]:
        target=QUAL/f'role{role}_round{round_number}';cert=read(target/'certificate.json');assert cert['status']=='numeric_polish_certified_for_finite_probe_only'
        fit=read(PRIOR/f'fold{role}_B/fit.json');roles.append(dict(role=role,OOF_chunks=k,deployment_chunks=12,head_cap=22*(k+12),finite_proposal_cap=20,endpoint_parameter_sha256=fit['endpoint_parameter_sha256'],qualified_direction=(target/'direction.npy').relative_to(ROOT).as_posix(),qualified_certificate=(target/'certificate.json').relative_to(ROOT).as_posix(),qualified_raw=(target/'raw.npy').relative_to(ROOT).as_posix(),failed_original_round=round_number))
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    summary=read(ROOT/'artifacts/v160_complete_fixed_endpoint_result_records_20261002/actual_result_summary.json');assert summary['cumulative_V159_V160_heads']==15318 and summary['cumulative_V159_V160_class_gradients']==368 and summary['cumulative_V159_V160_margin_gradients']==10
    plan=dict(protocol=PROTOCOL,activation_entries=['training/v160_cached_polished_direction_finite_probe.py'],roles=roles,new_caps=dict(heads=836,features=836,class_gradients=0,margin_gradients=0,finite_proposals=40,fits=0,permanent_updates=0),prior_actual_costs=dict(heads=15318,features=15318,class_gradients=368,margin_gradients=10),future_cumulative_actual_caps=dict(heads=16154,features=16154,class_gradients=368,margin_gradients=10),preserved_cumulative_technical_head_cap=82174,additional_actual_probes_within_remaining_technical_cap=True,old_diagnostic_entry_closed_not_direct_unused_budget_authority=True,single_factor='bounded original-unit arithmetic polish of the same saved active-cone directions',preserved_numeric_policy='training/v159_float64_repeat_policy_v2.py',full_deployment22546=True,full_original_OOF=True,joint_retention_unchanged=True,no_role0_replay=True,no_new_gradient_or_normal_collection=True,all_probe_outputs_saved_before_guard=True,per_probe_finally_restore=True,no_new_checkpoint_or_fit=True,runtime_package_match_before_every_entry=True,backtracks=20,step_sequence='2**(-j), j=0..19',quality_acceptance=False,source_sha256=bindings)
    assert sum(r['head_cap'] for r in roles)==836 and 16154<=plan['preserved_cumulative_technical_head_cap']
    plan['additional_fixed_endpoint_correct_OOF_protection']='Fixed B endpoint correct pure/non-conflicting legal OOF original rows including its existing repairs; retain old masks, score all mixed original rows, no candidate-based selection or freezing mixed-correct labels.'
    plan['candidate_progress']='Every candidate records exact repair/new/classification change versus the fixed endpoint and every original old-error truth-versus-old-rival margin ledger, separate from finite risk acceptance.'
    OUT.mkdir();save(OUT/'pre_registration_bindings.json',dict(status='before_any_new_real_finite_probe',source_sha256=bindings));save(PLAN,plan);bindings[PLAN.relative_to(ROOT).as_posix()]=sha(PLAN)
    save(OUT/'run_seal.json',dict(status='sealed_cached_polished_direction_finite_probe_before_calls',protocol=PROTOCOL,allowed_entries=plan['activation_entries'],plan_sha256=sha(PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings))
    require();save(OUT/'registration.json',dict(status='two_fixed_cached_polished_direction_probes_sealed',physical_sources=len(bindings),new_caps=plan['new_caps'],official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,quality_acceptance=False,run_seal_sha256=sha(OUT/'run_seal.json')))
    print(json.dumps(dict(status='cached_polished_direction_probe_sealed_before_calls',physical_sources=len(bindings),new_caps=plan['new_caps'])))

if __name__=='__main__':main()
