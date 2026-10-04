"""Prospective evaluation-only v6 seal, conserve failed110+360 plus fresh640."""
import ast,json,sys,importlib.metadata
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v159_evaluation_runtime_v6 import OUT,REPAIR,PLAN,save,require
def main():
    assert not REPAIR.exists() and not PLAN.exists();REPAIR.mkdir()
    evidence=ROOT/'artifacts/v159_saved_endpoint_source_pairing_audit_20261002/audit.json';q=read(evidence);check_bindings(q['source_sha256']);assert q['total_failed_evaluation_head_calls']==470 and q['fresh_complete_model_replay_cap']==640 and all(z['matching_accepted_row_pair_passed'] for z in q['reviews'])
    previous=ROOT/'artifacts/v159_evaluation_source_sum_repair_20261002'
    files={Path(__file__).resolve(),evidence,ROOT/'training/v159_boundary_evaluate_v6.py',ROOT/'training/v159_evaluation_runtime_v6.py',ROOT/'training/v159_build_source_pairing_evaluation_revision.py',ROOT/'training/v159_source_sum_repeat_policy.py',previous/'run_seal.json',OUT/'evaluation_failure_v5.json',OUT/'evaluation_v5_original_console.txt'}|{ROOT/k for k in read(previous/'run_seal.json')['source_sha256']}|{ROOT/k for k in q['source_sha256']}
    for p in previous.rglob('*'):
        if p.is_file():files.add(p)
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    binding={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    caps=read(ROOT/'artifacts/v159_saved_source_sum_qualification_20261002/qualification.json')['fresh_complete_replay_caps']
    plan=dict(protocol='V159-evaluation-only-propagated-source-CE-pair-v6',activation_entries=['training/v159_boundary_evaluate_v6.py'],fixed_trained_candidate='V159-v4',fresh_replay_role_caps=caps,fresh_replay_head_cap=640,fresh_replay_opinion_feature_cap=640,failed_evaluation_head_calls=470,failed_evaluation_components=dict(v4=110,v5=360),stage_cumulative_evaluation_head_cap=1110,net_stage_technical_increment=390,previous_original_cumulative_head_cap=78196,new_total_cumulative_head_cap=78586,total_cumulative_gradient_cap=2420,new_fit_cap=0,new_gradient_cap=0,new_update_cap=0,
      numeric_policy_unchanged=True,fixed_per_row_eps=8,source_sum_rule='fixed per-row stableCE and clippedCE envelopes propagated separately; each actual and stored sum independently matches its own exact row ledger within2eps',endpoint_source_table_reference='accepted-last-state rows that originally generated that saved source table; independently compare endpoint rows/probability with same frozen model',all_full_q_logq_rows_and_sources_saved_before_assert=True,old_training_contract_seal_models_and_failed_logs_unchanged=True,source_sha256=binding,quality_acceptance=False)
    save(REPAIR/'pre_registration_bindings.json',dict(status='before_new_evaluation_only_forward',source_sha256=binding));save(PLAN,plan);binding[PLAN.relative_to(ROOT).as_posix()]=sha(PLAN)
    seal=dict(status='sealed_evaluation_only_before_any_new_head_or_feature_call',protocol=plan['protocol'],allowed_entries=plan['activation_entries'],plan_path=PLAN.relative_to(ROOT).as_posix(),plan_sha256=sha(PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=binding)
    save(REPAIR/'run_seal.json',seal);require(ROOT/'training/v159_boundary_evaluate_v6.py');save(REPAIR/'registration.json',dict(status='evaluation_only_endpoint_source_pairing_repair_sealed',physical_sources=len(binding),fresh_head_cap=640,failed_prior_heads=470,net_total_technical_increment=390,new_heads=0,new_features=0,new_gradients=0,new_fits=0,new_updates=0,seal_sha256=sha(REPAIR/'run_seal.json'),quality_acceptance=False));print(json.dumps(dict(status='evaluation_only_v6_sealed_before_calls',fresh_cap=640,prior_failed=470,physical_sources=len(binding))))
if __name__=='__main__':main()
