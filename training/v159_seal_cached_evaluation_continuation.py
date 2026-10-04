"""Seal cached360 + remaining280 evaluation, zero new training authority."""
import ast,json,sys,importlib.metadata
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v159_evaluation_runtime_v7 import OUT,REPAIR,PLAN,save,require
def main():
    assert not REPAIR.exists() and not PLAN.exists();REPAIR.mkdir()
    evidence=ROOT/'artifacts/v159_cached_evaluation_review_20261002/qualification.json';q=read(evidence);check_bindings(q['source_sha256']);assert q['cached_actual_head_calls']==360 and q['remaining_actual_replay_head_cap']==280 and q['net_increment_over_original_stage720']==30
    previous=ROOT/'artifacts/v159_evaluation_source_sum_repair_20261002';unused=ROOT/'artifacts/v159_evaluation_source_pairing_repair_20261002'
    assert not (unused/'replayed_outputs').exists()
    files={Path(__file__).resolve(),evidence,ROOT/'training/v159_boundary_evaluate_v7.py',ROOT/'training/v159_evaluation_runtime_v7.py',ROOT/'training/v159_build_cached_evaluation_continuation_source.py',ROOT/'training/v159_cached_evaluation_review.py',previous/'run_seal.json',unused/'run_seal.json',unused/'registration.json',OUT/'evaluation_input_bindings_v5.json',OUT/'evaluation_failure_v5.json'}|{ROOT/k for k in read(previous/'run_seal.json')['source_sha256']}|{ROOT/k for k in q['source_sha256']}
    files|={p for p in (ROOT/'artifacts/v159_cached_evaluation_review_20261002').rglob('*') if p.is_file()}
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    plan=dict(protocol='V159-evaluation-only-cached-source-CE-continuation-v7',activation_entries=['training/v159_boundary_evaluate_v7.py'],cached_scope_keys=['0_A','0_B','1_A'],cached_actual_v5_head_calls=360,cached_actual_v5_opinion_feature_calls=360,cached_actual_v5_gradients=0,fresh_replay_role_caps=q['remaining_actual_replay_caps'],fresh_replay_head_cap=280,fresh_replay_opinion_feature_cap=280,failed_evaluation_head_calls=110,stage_cumulative_evaluation_head_cap=750,net_stage_technical_increment=30,new_total_cumulative_head_cap=78226,total_cumulative_gradient_cap=2420,new_fit_cap=0,new_gradient_cap=0,new_update_cap=0,
      cache_boundary='Full actual v5 q not persisted; use frozen training full q for argmax only with prior real full-q assertion control-flow evidence. Saved actual FIT rows/logq/source and their complete model replay costs verified, not claimed as full-q cache.',unused_v6_seal_no_model_calls=True,numeric_policy_unchanged=True,fixed_per_row_eps=8,exact_source_table_to_original_row_ledger=True,remaining_full_q_logq_rows_sources_saved_before_assert=True,old_training_contract_models_seals_failed_logs_unchanged=True,source_sha256=bindings,quality_acceptance=False)
    save(REPAIR/'pre_registration_bindings.json',dict(status='before_remaining_evaluation_forward_only',source_sha256=bindings));save(PLAN,plan);bindings[PLAN.relative_to(ROOT).as_posix()]=sha(PLAN)
    seal=dict(status='sealed_evaluation_only_before_any_new_head_or_feature_call',protocol=plan['protocol'],allowed_entries=plan['activation_entries'],plan_path=PLAN.relative_to(ROOT).as_posix(),plan_sha256=sha(PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings)
    save(REPAIR/'run_seal.json',seal);require(ROOT/'training/v159_boundary_evaluate_v7.py');save(REPAIR/'registration.json',dict(status='cached_evaluation_continuation_sealed',physical_sources=len(bindings),cached_real_heads=360,fresh_head_cap=280,initial_failed_eval_heads=110,stage_cumulative_head_cap=750,net_technical_increment=30,new_heads=0,new_features=0,new_gradients=0,new_fits=0,new_updates=0,seal_sha256=sha(REPAIR/'run_seal.json'),quality_acceptance=False));print(json.dumps(dict(status='cached_remaining_evaluation_sealed_before_calls',cached=360,remaining=280,physical_sources=len(bindings))))
if __name__=='__main__':main()
