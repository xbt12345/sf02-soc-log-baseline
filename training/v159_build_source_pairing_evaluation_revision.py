"""Only repair endpoint/source ledger pairing; freeze all prior costs/sources."""
from experiment_review import ROOT
def main():
    old=ROOT/'training/v159_evaluation_runtime_v5.py';new=old.with_name('v159_evaluation_runtime_v6.py');assert not new.exists()
    runtime=old.read_text(encoding='utf-8').replace('v159_evaluation_source_sum_repair_20261002','v159_evaluation_source_pairing_repair_20261002').replace('v159_evaluation_source_sum_repair_contract','v159_evaluation_source_pairing_repair_contract').replace('v159_boundary_evaluate_v5','v159_boundary_evaluate_v6').replace('propagated-source-CE-v5','propagated-source-CE-pair-v6')
    runtime=runtime.replace("q['failed_evaluation_head_calls']!=110","q['failed_evaluation_head_calls']!=470").replace("q['stage_cumulative_evaluation_head_cap']!=750","q['stage_cumulative_evaluation_head_cap']!=1110").replace("q['net_stage_technical_increment']!=30","q['net_stage_technical_increment']!=390").replace("p['failed_evaluation_head_calls']=110","p['failed_evaluation_head_calls']=470").replace('=78226;','=78586;').replace("['evaluation_classifier_forward_chunks']=750","['evaluation_classifier_forward_chunks']=1110")
    new.write_text(runtime,encoding='utf-8')
    old=ROOT/'training/v159_boundary_evaluate_v5.py';new=old.with_name('v159_boundary_evaluate_v6.py');assert not new.exists()
    text=old.read_text(encoding='utf-8').replace('v159_evaluation_runtime_v5','v159_evaluation_runtime_v6').replace('evaluation_input_bindings_v5','evaluation_input_bindings_v6').replace('evaluation_failure_v5','evaluation_failure_v6')
    text=text.replace("stored_endpoint=pd.read_parquet(folder/f'endpoint_{scope}_rows.parquet');stored_sources=", "stored_endpoint=pd.read_parquet(folder/f\"accepted{r['accepted_updates']}_{scope}_rows.parquet\");stored_sources=")
    text=text.replace("failed_evaluation_head_calls=110,failed_evaluation_opinion_feature_calls=110,","failed_evaluation_head_calls=470,failed_evaluation_opinion_feature_calls=470,").replace('cumulative_actual_head_calls=124+110+','cumulative_actual_head_calls=124+470+')
    needle="                actual.to_parquet(replay/f\"accepted{item['update']}_deployment_rows.parquet\",index=False)"
    assert needle in text;text=text.replace(needle,"                np.save(replay/f\"accepted{item['update']}_full_deployment_probability.npy\",q);np.save(replay/f\"accepted{item['update']}_full_deployment_log_probability.npy\",lp)\n"+needle)
    needle="                actual_oo.to_parquet(replay/f\"accepted{item['update']}_OOF_rows.parquet\",index=False)"
    assert needle in text;text=text.replace(needle,"                np.save(replay/f\"accepted{item['update']}_full_OOF_probability.npy\",oq);np.save(replay/f\"accepted{item['update']}_full_OOF_log_probability.npy\",olp)\n"+needle)
    needle="            terminal_deployment=rows(ctx,q,lp,'deployment');"
    assert needle in text;text=text.replace(needle,"            np.save(replay/'endpoint_full_deployment_probability.npy',q);np.save(replay/'endpoint_full_deployment_log_probability.npy',lp)\n"+needle)
    needle="            terminal_OOF=rows(ctx,qo,lpo,'OOF');"
    assert needle in text;text=text.replace(needle,"            np.save(replay/'endpoint_full_OOF_probability.npy',qo);np.save(replay/'endpoint_full_OOF_log_probability.npy',lpo)\n"+needle)
    new.write_text(text,encoding='utf-8')
    print('v6 endpoint paired with source table original ledger; all full replay arrays now saved, no model calls')
if __name__=='__main__':main()
