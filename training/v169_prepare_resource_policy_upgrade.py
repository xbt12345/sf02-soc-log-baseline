"""Resource-only entry/review/candidate; retains all prior scientific fields."""
import copy,json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v169_pair_execution_review_v5 import review_plan,BASIS,BASIS_SHA256,RESOURCE,ENTRY

def save(path,value):
    assert not path.exists()
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    resource_path=ROOT/RESOURCE;candidate_path=ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v4.json'
    out=ROOT/'artifacts/v169_execution_contract_candidate_v4_20261002'
    assert not resource_path.parent.exists() and not candidate_path.exists() and not out.exists()
    assert not (ROOT/'artifacts/v169_prior_pair_training').exists() and not (ROOT/'training/review_policy/v169_prior_pair_execution_contract.json').exists()
    basis=read(ROOT/BASIS);check_bindings(basis['source_sha256']);assert sha(ROOT/BASIS)==BASIS_SHA256
    old_entry=ROOT/'training/v169_prior_pair_training_entry_v13.py';entry=ROOT/ENTRY
    assert entry.read_bytes()==old_entry.read_bytes().replace(b'v169_pair_execution_review_v4',b'v169_pair_execution_review_v5')
    report=copy.deepcopy(basis)
    additions=[Path(__file__).resolve(),ROOT/BASIS,old_entry,entry,ROOT/'training/v169_pair_execution_review_v5.py',ROOT/'training/v169_seal_prior_pair_training_v5.py']
    bindings={**basis['source_sha256'],**{p.relative_to(ROOT).as_posix():sha(p) for p in additions}}
    for arm in ['A','B']:
        qualification=read(ROOT/f'artifacts/v169_full_live_new_solver_resource_qualification_v2_20261002/arm{arm}/qualification.json')
        bindings.update(qualification['source_sha256'])
    report.update(status='V169_v14_resource_only_policy_rebase_of_original_BLAS24_full_touch_v9_pending_root_review',entry_path=ENTRY,entry_sha256=sha(entry),previous_resource_review_path=BASIS,previous_resource_review_sha256=BASIS_SHA256,only_entry_resource_policy_import_changed=True,scientific_entry_before_path=old_entry.relative_to(ROOT).as_posix(),scientific_entry_before_sha256=sha(old_entry),entry_exact_byte_transformation='Replace exactly one v169_pair_execution_review_v4 import with v169_pair_execution_review_v5; all other bytes identical',existing_policy_v4_still_requires6GiB_and_cannot_execute_this_candidate=True,current_gate_not_changed=True,execution_authority=False,new_fit_permission=False,supports_physical_seal=False,source_sha256=bindings)
    resource_path.parent.mkdir();save(resource_path,report)
    prior_path=ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v3.json'
    prior=read(prior_path);check_bindings(prior['source_sha256'])
    plan=copy.deepcopy(prior)
    plan.update(entry=ENTRY,resource_review_path=RESOURCE,resources=report['resources'])
    plan['source_sha256'].update({**bindings,prior_path.relative_to(ROOT).as_posix():sha(prior_path),RESOURCE:sha(resource_path)})
    allowed={'entry','resource_review_path','resources','source_sha256'}
    assert {k for k in prior if prior[k]!=plan[k]}==allowed
    assert {k for k in prior['resources'] if prior['resources'][k]!=plan['resources'][k]}=={'minimum_free_RAM_bytes'}
    review=review_plan(plan);negative=[]
    for name,update in [
        ('old_mean_objective',dict(objective_policy='mean')),
        ('missing_beta_parameter',dict(complete_parameter_widths=dict(A=1060832,B=1060832))),
        ('wrong_counts',dict(prior_actual_costs=dict(heads=0))),
        ('longer_schedule',dict(schedule=dict(accepted_updates=21,corrections=2,working_functions=64,backtracks=8))),
        ('missing_history',dict(applicable_failure_registry_chain=plan['applicable_failure_registry_chain'][:1])),
        ('unmatched_endpoint_best_selection',dict(unmatched_terminal_result='best')),
        ('no_frequency',dict(all_original_frequency_and_denominators_preserved=False)),
        ('full_quality_claim',dict(full_task_quality_acceptance=True)),
        ('missing_global_stop',dict(batch_global_fault_stops_remaining_fits=False)),
        ('no_final_same_point',dict(same_point_final_replay_required=False)),
        ('wrong_resource_review_path',dict(resource_review_path=BASIS)),
        ('old_execution_entry',dict(entry=old_entry.relative_to(ROOT).as_posix())),
        ('unregistered_RAM_threshold',dict(resources={**plan['resources'],'minimum_free_RAM_bytes':plan['resources']['minimum_free_RAM_bytes']-1}))
    ]:
        bad=copy.deepcopy(plan);bad.update(update)
        try:review_plan(bad)
        except (ValueError,KeyError):negative.append(name)
        else:raise AssertionError('Expected refusal '+name)
    from v169_prior_pair_training_entry_v14 import require
    try:require('initial')
    except RuntimeError as e:assert 'not sealed' in str(e)
    else:raise AssertionError('Unsealed entry must refuse')
    save(candidate_path,plan);out.mkdir()
    qualification=dict(status='V169_resource_only_candidate_v4_all_scientific_fields_unchanged_and_negative_cases_refused',candidate_path=candidate_path.relative_to(ROOT).as_posix(),candidate_sha256=sha(candidate_path),mechanical_review=review,negative_cases_refused=negative,only_prior_candidate_keys_changed=sorted(allowed),only_prior_resource_key_changed='minimum_free_RAM_bytes',all_remaining_scientific_contract_fields_literal_identical=True,entry_all_bytes_except_policy_version_literal_identical=True,unsealed_actual_entry_refused=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,new_fit_permission=False,source_sha256={**plan['source_sha256'],candidate_path.relative_to(ROOT).as_posix():sha(candidate_path)})
    save(out/'qualification.json',qualification)
    print(json.dumps(dict(status=qualification['status'],resource=RESOURCE,resource_sha256=sha(resource_path),candidate_sha256=sha(candidate_path),negative_cases=len(negative),official_calls=0)))

if __name__=='__main__':main()
