"""New immutable evaluation-only entry; preserve failed evaluation outputs."""
from experiment_review import ROOT
def main():
    old=ROOT/'training/v159_boundary_evaluate_v4.py';new=old.with_name('v159_boundary_evaluate_v5.py');assert not new.exists()
    text=old.read_text(encoding='utf-8').replace('from v159_boundary_runtime_v4 import','from v159_evaluation_runtime_v5 import').replace('ROOT,OUT,PREP,read,save,sha,require,endpoint','ROOT,OUT,PREP,REPAIR,read,save,sha,require,endpoint')
    text=text.replace('from v159_float64_repeat_policy_v2 import finite_armijo,repeat_values','from v159_float64_repeat_policy_v2 import finite_armijo,repeat_values\nfrom v159_source_sum_repeat_policy import source_sum_repeat_review')
    text=text.replace("OUT/'evaluation_input_bindings.json'","OUT/'evaluation_input_bindings_v5.json'")
    needle="            model=model_for();counter=Counter(model,plan['role_call_budgets'][f]['evaluation_classifier_cap_per_arm'],folder/'evaluation_calls.jsonl',0);states=[]"
    replacement="""            replay=REPAIR/'replayed_outputs'/f'fold{f}_{arm}';assert not replay.exists();replay.mkdir(parents=True)
            model=model_for();counter=Counter(model,plan['evaluation_repair_role_caps'][f'{f}_{arm}'],replay/'calls.jsonl',0);states=[]"""
    assert needle in text;text=text.replace(needle,replacement)
    needle="                assert np.array_equal(actual.row_position,stored.row_position)"
    insert="""                actual.to_parquet(replay/f"accepted{item['update']}_deployment_rows.parquet",index=False)
                deployment_source=actual.assign(wrong=actual.pred.ne(actual.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()
                deployment_source.to_parquet(replay/f"accepted{item['update']}_deployment_sources.parquet",index=False)
                stored_deployment_source=pd.read_parquet(folder/f"accepted{item['update']}_deployment_sources.parquet")
                deployment_source_review=source_sum_repeat_review(actual,stored,deployment_source,stored_deployment_source)
                save(replay/f"accepted{item['update']}_deployment_source_repeat_review.json",deployment_source_review);assert deployment_source_review['passed']
"""
    assert needle in text;text=text.replace(needle,insert+needle)
    needle="                actual_oo=rows(ctx,oq,olp,'OOF')"
    assert needle in text;text=text.replace(needle,needle+"\n                actual_oo.to_parquet(replay/f\"accepted{item['update']}_OOF_rows.parquet\",index=False)")
    needle="                saved_source=pd.read_parquet(folder/f\"accepted{item['update']}_OOF_sources.parquet\");assert np.array_equal"
    assert needle in text;text=text.replace(needle,"                source.to_parquet(replay/f\"accepted{item['update']}_OOF_sources.parquet\",index=False)\n                saved_source=pd.read_parquet(folder/f\"accepted{item['update']}_OOF_sources.parquet\");assert np.array_equal")
    needle="                assert repeat_values(source[['stable_CE_sum','probability_clip_CE_sum']].to_numpy(),saved_source[['stable_CE_sum','probability_clip_CE_sum']].to_numpy(),'source_CE_sum')['passed']"
    assert needle in text;text=text.replace(needle,"                source_review=source_sum_repeat_review(actual_oo,oo,source,saved_source);save(replay/f\"accepted{item['update']}_OOF_source_repeat_review.json\",source_review);assert source_review['passed']")
    needle="            q,lp=probabilities(model,ctx,'deployment',np.arange(22546),True);assert repeat_values"
    assert needle in text;text=text.replace(needle,"            q,lp=probabilities(model,ctx,'deployment',np.arange(22546),True)\n            terminal_deployment=rows(ctx,q,lp,'deployment');terminal_deployment.to_parquet(replay/'endpoint_deployment_rows.parquet',index=False)\n            assert repeat_values")
    needle="            value,qo,lpo,_=risk(model,ctx,'OOF',ctx['ids']);assert repeat_values"
    assert needle in text;text=text.replace(needle,"            value,qo,lpo,_=risk(model,ctx,'OOF',ctx['ids'])\n            terminal_OOF=rows(ctx,qo,lpo,'OOF');terminal_OOF.to_parquet(replay/'endpoint_OOF_rows.parquet',index=False)\n            for scope,actual_endpoint in [('OOF',terminal_OOF),('deployment',terminal_deployment)]:\n                actual_sources=actual_endpoint.assign(wrong=actual_endpoint.pred.ne(actual_endpoint.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()\n                actual_sources.to_parquet(replay/f'endpoint_{scope}_sources.parquet',index=False)\n                stored_endpoint=pd.read_parquet(folder/f'endpoint_{scope}_rows.parquet');stored_sources=pd.read_parquet(folder/f\"accepted{r['accepted_updates']}_{scope}_sources.parquet\")\n                terminal_source_review=source_sum_repeat_review(actual_endpoint,stored_endpoint,actual_sources,stored_sources);save(replay/f'endpoint_{scope}_source_repeat_review.json',terminal_source_review);assert terminal_source_review['passed']\n            assert repeat_values")
    text=text.replace("cumulative_actual_head_calls=124+pre['actual_head_forward_calls']","cumulative_actual_head_calls=124+110+pre['actual_head_forward_calls']")
    text=text.replace("historical_technical_cost=plan['historical_technical_cost'],","historical_technical_cost=plan['historical_technical_cost'],failed_evaluation_head_calls=110,failed_evaluation_opinion_feature_calls=110,evaluation_repair_contract=plan['evaluation_repair_contract_path'],evaluation_repair_seal_sha256=sha(REPAIR/'run_seal.json'),")
    text=text.replace("OUT/'evaluation_failure.json'","OUT/'evaluation_failure_v5.json'")
    new.write_text(text,encoding='utf-8');print('evaluation v5 source created; no model calls')
if __name__=='__main__':main()
