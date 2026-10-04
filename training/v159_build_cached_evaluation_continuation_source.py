"""Continue only unevaluated scopes; retain verified actual v5 cached scopes."""
from experiment_review import ROOT
def main():
    old=ROOT/'training/v159_evaluation_runtime_v5.py';new=old.with_name('v159_evaluation_runtime_v7.py');assert not new.exists()
    text=old.read_text(encoding='utf-8').replace('v159_evaluation_source_sum_repair_20261002','v159_evaluation_cached_continuation_20261002').replace('v159_evaluation_source_sum_repair_contract','v159_evaluation_cached_continuation_contract').replace('v159_boundary_evaluate_v5','v159_boundary_evaluate_v7').replace('propagated-source-CE-v5','cached-source-CE-continuation-v7').replace("q['fresh_replay_head_cap']!=640","q['fresh_replay_head_cap']!=280")
    text=text.replace("    if seal['python_version']", "    if q['cached_actual_v5_head_calls']!=360 or q['cached_scope_keys']!=['0_A','0_B','1_A'] or sum(q['fresh_replay_role_caps'].values())!=280:raise ReviewError('Changed cached scopes or remaining calls')\n    if seal['python_version']")
    new.write_text(text,encoding='utf-8')
    old=ROOT/'training/v159_boundary_evaluate_v6.py';new=old.with_name('v159_boundary_evaluate_v7.py');assert not new.exists()
    text=old.read_text(encoding='utf-8').replace('v159_evaluation_runtime_v6','v159_evaluation_runtime_v7').replace('evaluation_input_bindings_v6','evaluation_input_bindings_v7').replace('evaluation_failure_v6','evaluation_failure_v7')
    text=text.replace('from v159_source_sum_repeat_policy import source_sum_repeat_review','from v159_source_sum_repeat_policy import source_sum_repeat_review\nfrom v159_cached_evaluation_review import cached_scope,CACHED')
    needle="            replay=REPAIR/'replayed_outputs'/f'fold{f}_{arm}'"
    insert="""            if (f,arm) in CACHED:
                q,fitrows,audit,cached_counts=cached_scope(ctx,r,REPAIR/'cached_scope_reviews'/f'fold{f}_{arm}')
                all_fit_rows[arm].append(fitrows);predictions[arm][f]=q;audits.append(audit);evaluation_counts.append(cached_counts)
                continue
"""
    assert needle in text;text=text.replace(needle,insert+needle)
    text=text.replace('failed_evaluation_head_calls=470,failed_evaluation_opinion_feature_calls=470,','failed_evaluation_head_calls=110,failed_evaluation_opinion_feature_calls=110,cached_v5_actual_evaluation_heads=360,new_v7_actual_evaluation_heads=280,cached_full_q_provenance_boundary="v5 actual full q not persisted; frozen complete training q has actual v5 full argmax/repeat assertion control-flow evidence; actual FIT rows/source arrays saved and independently verified",')
    text=text.replace('cumulative_actual_head_calls=124+470+','cumulative_actual_head_calls=124+110+')
    new.write_text(text,encoding='utf-8');print('cached continuation v7 source created, remaining scopes only; no calls')
if __name__=='__main__':main()
