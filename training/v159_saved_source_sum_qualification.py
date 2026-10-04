"""Qualify source propagation from saved actual rows and fixed synthetic errors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v159_source_sum_repeat_policy import source_sum_repeat_review
from v159_float64_repeat_policy_v2 import EPS,repeat_values
OUT=ROOT/'artifacts/v159_saved_source_sum_qualification_20261002'
TRIAL=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
def source(rows):
    return rows.assign(wrong=rows.pred.ne(rows.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()
def main():
    assert not OUT.exists();OUT.mkdir();pairs=[];paths=[Path(__file__).resolve(),ROOT/'training/v159_source_sum_repeat_policy.py',ROOT/'training/v159_float64_repeat_policy_v2.py',TRIAL/'evaluation_failure.json',TRIAL/'evaluation_original_console.txt',TRIAL/'evaluation_input_bindings.json']
    receipts=[]
    for f in range(3):
        for arm in ['A','B']:
            folder=TRIAL/f'fold{f}_{arm}';r=read(folder/'fit.json');receipts.append(r);paths += [folder/'fit.json',folder/'calls.jsonl']
            for item in r['last5']:
                row=folder/f"accepted{item['update']}_OOF_rows.parquet";src=folder/f"accepted{item['update']}_OOF_sources.parquet";pairs.append((row,src));paths += [row,src]
            if (folder/'evaluation_calls.jsonl').exists():paths.append(folder/'evaluation_calls.jsonl')
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in paths};(OUT/'pre_saved_table_bindings.json').write_text(json.dumps(dict(status='before_saved_rows_and_tables_decode_no_model_calls',source_sha256=binding),indent=2)+'\n',encoding='utf-8')
    reviewed=[]
    for rows_path,src_path in pairs:
        rr=pd.read_parquet(rows_path);ss=pd.read_parquet(src_path);actual=source(rr);proof=source_sum_repeat_review(rr,rr,actual,ss);assert proof['passed'],proof
        reviewed.append(dict(rows=rows_path.relative_to(ROOT).as_posix(),original_rows=len(rr),source_groups=len(actual),self_consistent=True,maximum_reduction_gap=max(z['actual_difference'] for z in proof['groups'])))
    n=1000;rr=pd.DataFrame(dict(row_position=np.arange(n),root=7,truth=1,pred=1,stable_CE=1e-16,probability_clip_CE=1e-16));ss=source(rr)
    shifted=rr.copy();shifted['stable_CE']+=4*EPS;shifted['probability_clip_CE']+=4*EPS;as_=source(shifted)
    valid=source_sum_repeat_review(shifted,rr,as_,ss);assert valid['passed']
    assert not repeat_values(as_[['stable_CE_sum','probability_clip_CE_sum']].to_numpy(),ss[['stable_CE_sum','probability_clip_CE_sum']].to_numpy(),'old_source_sum')['passed']
    cases=dict(admissible_per_row_errors_propagate_in_source_sum=True,old_point_envelope_rejects_admissible_sum=True)
    bad=shifted.copy();bad.loc[0,'stable_CE']+=64*EPS;cases['one_out_of_envelope_row_rejected']=not source_sum_repeat_review(bad,rr,source(bad),ss)['passed']
    corrupt=as_.copy();corrupt.loc[0,'stable_CE_sum']+=1e-8;cases['corrupt_aggregate_rejected']=not source_sum_repeat_review(shifted,rr,corrupt,ss)['passed']
    corrupt=as_.copy();corrupt.loc[0,'errors']=1;cases['source_error_count_exact']=not source_sum_repeat_review(shifted,rr,corrupt,ss)['passed']
    bad=shifted.copy();bad.loc[0,'pred']=2;cases['row_prediction_exact']=not source_sum_repeat_review(bad,rr,source(bad),ss)['passed']
    cases['missing_row_rejected']=not source_sum_repeat_review(shifted.iloc[:-1],rr,source(shifted.iloc[:-1]),ss)['passed'];assert all(cases.values())
    events=[]
    for path in TRIAL.glob('fold*_*/*evaluation_calls.jsonl'):events += [json.loads(z) for z in path.read_text().splitlines()]
    counts={kind:sum(e['kind']==kind and e['event']=='attempt' for e in events) for kind in ['head','feature','full_class_gradient']};assert counts==dict(head=110,feature=110,full_class_gradient=0)
    assert all(sum(e['kind']==kind and e['event']=='completed' for e in events)==count for kind,count in counts.items())
    caps={f"{r['fold']}_{r['arm']}":(len(r['last5'])+1)*(12+(10 if r['fold'] in [0,2] else 4)) for r in receipts};assert sum(caps.values())==640
    check_bindings(binding)
    result=dict(status='saved_actual_source_tables_and_fixed_row_error_propagation_cases_passed',actual_saved_tables=reviewed,cases=cases,valid_synthetic_sum_example=valid,failed_evaluation_actual_counts=counts,failed_actual_source_arrays_not_saved_magnitude_unknown=True,
      completed_training=dict(fits=6,head_calls=13376,full_class_gradients=342,proposals=574,accepted_updates=165),cumulative_actual_before_repair=dict(head_calls=13946,opinion_feature_calls=13946,full_class_gradients=362,fits=6,updates=165),fresh_complete_replay_caps=caps,fresh_complete_replay_head_cap=640,new_total_stage_evaluation_head_cap=750,previous_stage_evaluation_cap=720,net_technical_stage_increment=30,new_total_cumulative_head_cap=78226,full_class_gradient_cumulative_cap=2420,official_new_heads=0,official_new_features=0,official_new_gradients=0,official_new_fits=0,official_new_updates=0,quality_acceptance=False,source_sha256=binding)
    (OUT/'qualification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],saved_tables=len(reviewed),failed_evaluation_counts=counts,fresh_replay_cap=640,net_technical_increment=30),ensure_ascii=False))
if __name__=='__main__':main()
