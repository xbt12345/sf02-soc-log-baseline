"""Verify persisted actual v5 states without re-forward; explicit full-q boundary."""
import math,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v159_float64_repeat_policy_v2 import repeat_values
from v159_source_sum_repeat_policy import source_sum_repeat_review
from v159_boundary_train_v4 import context,stats
TRIAL=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
PAST=ROOT/'artifacts/v159_evaluation_source_sum_repair_20261002/replayed_outputs'
OUT=ROOT/'artifacts/v159_cached_evaluation_review_20261002'
CACHED=[(0,'A'),(0,'B'),(1,'A')]

def source(frame):
    return frame.assign(wrong=frame.pred.ne(frame.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()

def q_from_actual_rows(frame):
    one=frame.drop_duplicates('local');q=np.full((22546,3),np.nan);q[one.local]=one[['p0','p1','p2']].to_numpy()
    assert np.array_equal(q[frame.local],frame[['p0','p1','p2']].to_numpy());return q

def risk_from_actual_rows(frame):
    return np.array([math.fsum(frame.loc[frame.truth.eq(c),'stable_CE'])/int(frame.truth.eq(c).sum()) for c in [1,2]])

def cached_scope(ctx,r,output):
    f,arm=r['fold'],r['arm'];folder=TRIAL/f'fold{f}_{arm}';past=PAST/folder.name;output=Path(output);output.mkdir(parents=True,exist_ok=False)
    journal=[json.loads(z) for z in (past/'calls.jsonl').read_text().splitlines()];count=(len(r['last5'])+1)*(12+(10 if f==0 else 4))
    counts={}
    for kind,key in [('head','head'),('feature','feature'),('full_class_gradient','gradient')]:
        for event,suffix in [('attempt','attempts'),('completed','completed')]:counts[key+'_'+suffix]=sum(e['kind']==kind and e['event']==event for e in journal)
    assert counts==dict(head_attempts=count,head_completed=count,feature_attempts=count,feature_completed=count,gradient_attempts=0,gradient_completed=0)
    windows=[]
    for item in r['last5']:
        values={}
        for scope in ['OOF','deployment']:
            actual=pd.read_parquet(past/f"accepted{item['update']}_{scope}_rows.parquet");saved=pd.read_parquet(folder/f"accepted{item['update']}_{scope}_rows.parquet");saved_source=pd.read_parquet(folder/f"accepted{item['update']}_{scope}_sources.parquet")
            actual_source=source(actual);review=source_sum_repeat_review(actual,saved,actual_source,saved_source)
            (output/f"accepted{item['update']}_{scope}_source_review.json").write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8');assert review['passed']
            assert repeat_values(actual[['p0','p1','p2']],saved[['p0','p1','p2']],'probability')['passed'] and repeat_values(actual[['logp0','logp1','logp2']],saved[['logp0','logp1','logp2']],'log_probability')['passed']
            actual_stats=stats(ctx,q_from_actual_rows(actual),scope);assert actual_stats==item[scope+'_stats'];values[scope]=actual_stats
            if scope=='OOF':
                risk=risk_from_actual_rows(actual);history=read(folder/'progress.json')[item['update']-1];assert repeat_values(risk,history['stable_class_risks'],'risk')['passed']
        windows.append(dict(update=item['update'],parameter_sha256=item['parameter_sha256'],OOF_stats=values['OOF'],deployment_stats=values['deployment'],actual_OOF_classifier_replay=True,replay_scope='actual v5 saved state rows, not fresh v7 model call',actual_stable_class_risks=risk.tolist()))
    terminals={}
    for scope in ['OOF','deployment']:
        actual=pd.read_parquet(past/f'endpoint_{scope}_rows.parquet');saved_endpoint=pd.read_parquet(folder/f'endpoint_{scope}_rows.parquet');saved_row_basis=pd.read_parquet(folder/f"accepted{r['accepted_updates']}_{scope}_rows.parquet");stored_source=pd.read_parquet(folder/f"accepted{r['accepted_updates']}_{scope}_sources.parquet")
        assert repeat_values(actual[['p0','p1','p2']],saved_endpoint[['p0','p1','p2']],'probability')['passed'] and repeat_values(actual[['logp0','logp1','logp2']],saved_endpoint[['logp0','logp1','logp2']],'log_probability')['passed']
        review=source_sum_repeat_review(actual,saved_row_basis,source(actual),stored_source);(output/f'endpoint_{scope}_source_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8');assert review['passed']
        actual_stats=stats(ctx,q_from_actual_rows(actual),scope);assert actual_stats==r[scope+'_stats'];terminals[scope]=(actual,actual_stats)
    risk=risk_from_actual_rows(terminals['OOF'][0]);assert repeat_values(risk,r['final_stable_class_risks'],'risk')['passed']
    # Full 22546 q was compared to this frozen array before the later v5
    # failure; actual full v5 q itself was not persisted. This is a control
    # flow inference plus frozen-array identity, not a claimed full-q cache.
    q=np.load(folder/'endpoint_deployment_probability.npy');assert q.shape==(22546,3) and np.isfinite(q).all() and np.array_equal(q[terminals['deployment'][0].local].argmax(1),terminals['deployment'][0].pred)
    audit=dict(fold=f,arm=arm,window=windows,distinct_last5=len(windows)==5 and len({e['parameter_sha256'] for e in windows})==5,OOF_classification_mastered=terminals['OOF'][1]['mastered'],deployment_mastered=terminals['deployment'][1]['mastered'],actual_evaluation_counts=counts,cached_actual_v5_scope=True,
      full_deployment_probability_provenance='frozen training endpoint22546 array, actual v5 full-q repeat_values probability/argmax assertion ran before later fold1A source-pair failure; actual full v5 q not saved',new_model_calls=0,new_gradients=0)
    return q,terminals['deployment'][0],audit,counts

def main():
    assert not OUT.exists();OUT.mkdir()
    paths=[Path(__file__).resolve(),ROOT/'training/v159_boundary_evaluate_v5.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v159_source_sum_repeat_policy.py',TRIAL/'evaluation_failure_v5.json',ROOT/'artifacts/v159_saved_endpoint_source_pairing_audit_20261002/audit.json',ROOT/'artifacts/v159_evaluation_source_sum_repair_20261002/run_seal.json',ROOT/'artifacts/v159_evaluation_source_pairing_repair_20261002/registration.json']
    for f,arm in CACHED:
        paths += [p for p in (TRIAL/f'fold{f}_{arm}').glob('*') if p.is_file()]+[p for p in (PAST/f'fold{f}_{arm}').glob('*') if p.is_file()]
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths};(OUT/'pre_cache_bindings.json').write_text(json.dumps(dict(status='before_cached_actual_array_decode_no_model_calls',source_sha256=bindings),indent=2)+'\n',encoding='utf-8')
    failure=read(TRIAL/'evaluation_failure_v5.json');assert failure['source_sha256']==sha(ROOT/'training/v159_boundary_evaluate_v5.py') and 'line 89' in failure['traceback']
    reports=[]
    for f,arm in CACHED:
        _,_,audit,_=cached_scope(context(f),read(TRIAL/f'fold{f}_{arm}/fit.json'),OUT/f'fold{f}_{arm}');reports.append(audit)
    assert sum(r['actual_evaluation_counts']['head_attempts'] for r in reports)==360
    check_bindings(bindings)
    result=dict(status='three_completed_actual_v5_scopes_cached_verified_source_pairs_corrected_no_new_model_calls',reports=reports,cached_actual_head_calls=360,cached_actual_opinion_feature_calls=360,failed_initial_v4_head_calls=110,remaining_actual_replay_caps={'1_B':16,'2_A':132,'2_B':132},remaining_actual_replay_head_cap=280,total_recovery_actual_head_cap=640,stage_cumulative_evaluation_head_cap=750,net_increment_over_original_stage720=30,total_cumulative_head_cap=78226,total_cumulative_gradient_cap=2420,unused_v6_fresh_replay_registered_but_no_calls=True,new_official_heads=0,new_features=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,source_sha256=bindings)
    (OUT/'qualification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],cached=360,remaining=280,total_recovery=640,net_increment=30),ensure_ascii=False))
if __name__=='__main__':main()
