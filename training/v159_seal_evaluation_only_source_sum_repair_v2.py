"""Seal all trained states and new evaluation-only policy before new forwards."""
import ast,json,sys,importlib.metadata
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v159_evaluation_runtime_v5 import OUT,REPAIR,PLAN,save,require
def main():
    assert not PLAN.exists();REPAIR.mkdir(exist_ok=True);assert not list(REPAIR.iterdir())
    evidence=ROOT/'artifacts/v159_saved_source_sum_qualification_20261002/qualification.json';q=read(evidence);check_bindings(q['source_sha256']);assert q['failed_evaluation_actual_counts']['head']==110 and q['fresh_complete_replay_head_cap']==640 and all(q['cases'].values())
    files={Path(__file__).resolve(),Path(__file__).with_name('v159_build_evaluation_repair_sealer_v2.py'),ROOT/'training/v159_seal_evaluation_only_source_sum_repair.py',ROOT/'artifacts/v159_evaluation_source_sum_repair_registration_failure_20261002/failure.json',ROOT/'artifacts/v159_evaluation_only_repair_registration_original_console_20261002.txt',ROOT/'artifacts/v159_independent_group_CE_failure_scope_audit_20261002/audit.json',evidence,ROOT/'training/v159_boundary_evaluate_v5.py',ROOT/'training/v159_evaluation_runtime_v5.py',ROOT/'training/v159_source_sum_repeat_policy.py',ROOT/'training/v159_build_source_sum_evaluation_source.py'}|{ROOT/k for k in read(OUT/'run_seal.json')['source_sha256']}|{OUT/'run_seal.json',OUT/'evaluation_failure.json',OUT/'evaluation_original_console.txt',OUT/'evaluation_input_bindings.json'}|{ROOT/k for k in q['source_sha256']}
    for f in range(3):
        for arm in ['A','B']:files|={p for p in (OUT/f'fold{f}_{arm}').glob('*') if p.is_file()}
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    binding={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    plan=dict(protocol='V159-evaluation-only-propagated-source-CE-v5',activation_entries=['training/v159_boundary_evaluate_v5.py'],fixed_trained_candidate='V159-v4',fresh_replay_role_caps=q['fresh_complete_replay_caps'],fresh_replay_head_cap=640,fresh_replay_opinion_feature_cap=640,failed_evaluation_head_calls=110,stage_cumulative_evaluation_head_cap=750,net_stage_technical_increment=30,previous_total_cumulative_head_cap=78196,new_total_cumulative_head_cap=78226,total_cumulative_gradient_cap=2420,new_fit_cap=0,new_gradient_cap=0,new_update_cap=0,clip_CE_policy='independently require each row clipped probability CE within fixed8eps; propagate its own envelope separately from stable logp CE',fixed_per_row_eps=8,source_sum_rule='propagate each row fixed8eps CE envelope; independently certify each stored/replayed sum against math.fsum using2eps; integer identities/support/errors exact',actual_replay_rows_and_sources_saved_before_assert=True,old_run_seal_unchanged=True,source_sha256=binding,quality_acceptance=False)
    save(REPAIR/'pre_registration_bindings.json',dict(status='before_new_evaluation_only_forward',source_sha256=binding));save(PLAN,plan)
    binding[PLAN.relative_to(ROOT).as_posix()]=sha(PLAN)
    seal=dict(status='sealed_evaluation_only_before_any_new_head_or_feature_call',protocol=plan['protocol'],allowed_entries=plan['activation_entries'],plan_path=PLAN.relative_to(ROOT).as_posix(),plan_sha256=sha(PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=binding)
    save(REPAIR/'run_seal.json',seal);require(ROOT/'training/v159_boundary_evaluate_v5.py')
    save(REPAIR/'registration.json',dict(status='evaluation_only_source_sum_repair_sealed',physical_sources=len(binding),fresh_head_cap=640,failed_prior_heads=110,net_technical_increment=30,new_heads=0,new_features=0,new_gradients=0,new_fits=0,new_updates=0,seal_sha256=sha(REPAIR/'run_seal.json'),quality_acceptance=False))
    print(json.dumps(dict(status='evaluation_only_repair_sealed_before_calls',fresh_cap=640,prior_failed=110,net_increment=30,physical_sources=len(binding))))
if __name__=='__main__':main()
