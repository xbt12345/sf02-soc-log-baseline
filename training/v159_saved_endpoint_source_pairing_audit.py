"""Diagnose saved actual endpoint CE/table provenance, zero official calls."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v159_source_sum_repeat_policy import source_sum_repeat_review
from v159_float64_repeat_policy_v2 import repeat_values
TRIAL=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
REPAIR=ROOT/'artifacts/v159_evaluation_source_sum_repair_20261002'
OUT=ROOT/'artifacts/v159_saved_endpoint_source_pairing_audit_20261002'
def main():
    assert not OUT.exists();OUT.mkdir()
    paths=[Path(__file__).resolve(),ROOT/'training/v159_source_sum_repeat_policy.py',ROOT/'training/v159_float64_repeat_policy_v2.py',TRIAL/'evaluation_failure_v5.json',REPAIR/'run_seal.json',REPAIR/'registration.json']
    cases=[]
    for f in range(3):
        for arm in ['A','B']:
            folder=TRIAL/f'fold{f}_{arm}';r=read(folder/'fit.json')
            replay=REPAIR/'replayed_outputs'/folder.name
            if replay.exists():paths.append(replay/'calls.jsonl')
            for scope in ['OOF','deployment']:
                actual_path=replay/f'endpoint_{scope}_rows.parquet'
                if actual_path.exists():
                    matched=folder/f"accepted{r['accepted_updates']}_{scope}_rows.parquet";src=folder/f"accepted{r['accepted_updates']}_{scope}_sources.parquet";endpoint=folder/f'endpoint_{scope}_rows.parquet'
                    cases.append((f,arm,scope,actual_path,matched,src,endpoint));paths += [actual_path,matched,src,endpoint]
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in paths};(OUT/'pre_saved_bindings.json').write_text(json.dumps(dict(source_sha256=binding),indent=2)+'\n',encoding='utf-8')
    reviews=[]
    for f,arm,scope,actual_path,matched,src,endpoint in cases:
        a,b,e=[pd.read_parquet(p) for p in [actual_path,matched,endpoint]];s=pd.read_parquet(src)
        actual=a.assign(wrong=a.pred.ne(a.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()
        wrong_pair=source_sum_repeat_review(a,e,actual,s);correct_pair=source_sum_repeat_review(a,b,actual,s)
        assert correct_pair['passed'] and np.array_equal(a[['row_position','truth','pred']],e[['row_position','truth','pred']]) and repeat_values(a[['stable_CE','probability_clip_CE']],e[['stable_CE','probability_clip_CE']])['passed']
        failed=[]
        if not wrong_pair['passed']:
            groups=e.groupby(['root','truth'],sort=True).indices
            for j,(key,ids) in enumerate(groups.items()):
                for row,quantity in [('stable_CE','stable_CE_sum'),('probability_clip_CE','probability_clip_CE_sum')]:
                    end_sum=math.fsum(e[row].to_numpy()[ids]);accepted_sum=math.fsum(b[row].to_numpy()[ids]);stored=float(s.iloc[j][quantity]);lim=2*np.finfo(np.float64).eps*max(1.,abs(end_sum))
                    if abs(stored-end_sum)>lim:failed.append(dict(root=int(key[0]),truth=int(key[1]),quantity=quantity,source_table_original_row_ledger_sum=accepted_sum,other_endpoint_row_ledger_sum=end_sum,stored_source_sum=stored,unmatched_pair_gap=abs(stored-end_sum),same_reduction_limit=lim))
        reviews.append(dict(fold=f,arm=arm,scope=scope,wrong_endpoint_row_pair_passed=wrong_pair['passed'],matching_accepted_row_pair_passed=correct_pair['passed'],failed_wrong_pair_reductions=failed,actual_row_CE=correct_pair['row_reviews']))
    events=[]
    for path in REPAIR.glob('replayed_outputs/*/calls.jsonl'):events += [json.loads(z) for z in path.read_text().splitlines()]
    counts={kind:sum(e['kind']==kind and e['event']=='attempt' for e in events) for kind in ['head','feature','full_class_gradient']};assert counts==dict(head=360,feature=360,full_class_gradient=0)
    assert all(sum(e['kind']==kind and e['event']=='completed' for e in events)==n for kind,n in counts.items())
    check_bindings(binding)
    result=dict(status='actual_saved_endpoint_failure_is_mismatched_row_source_pair_not_changed_numeric_policy',reviews=reviews,failed_v5_evaluation_counts=counts,old_v4_failed_counts=dict(head=110,feature=110,full_class_gradient=0),total_failed_evaluation_head_calls=470,fresh_complete_model_replay_cap=640,stage_cumulative_evaluation_cap=1110,stage_net_increment_over_original720=390,total_cumulative_head_cap=78586,full_class_gradient_cumulative_cap=2420,cumulative_actual_before_next_eval=dict(head_calls=14306,opinion_feature_calls=14306,full_class_gradients=362,fits=6,updates=165),numeric_policy_unchanged=True,new_official_heads=0,new_features=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,source_sha256=binding)
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],actual_failed_v5=counts,failed_pairs=[dict(fold=z['fold'],arm=z['arm'],scope=z['scope'],mismatched_reductions=len(z['failed_wrong_pair_reductions'])) for z in reviews if not z['wrong_endpoint_row_pair_passed']]),ensure_ascii=False))
if __name__=='__main__':main()
