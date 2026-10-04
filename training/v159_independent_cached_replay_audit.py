"""Audit saved actual replay rows and correct source pairing; no classifier calls."""
import json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from v159_float64_repeat_policy_v2 import repeat_values
from v159_source_sum_repeat_policy import source_sum_repeat_review

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
PAST=ROOT/'artifacts/v159_evaluation_source_sum_repair_20261002/replayed_outputs'
CACHE=ROOT/'artifacts/v159_cached_evaluation_review_20261002'
OUT=ROOT/'artifacts/v159_independent_cached_replay_audit_20261002'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def sources(frame):
    return frame.assign(wrong=frame.pred.ne(frame.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()
def identity(frame,gold):
    assert frame.row_position.is_unique
    assert np.array_equal(frame.truth,gold[frame.row_position])
    assert frame.pred.isin([0,1,2]).all()
    assert np.array_equal(frame[['p0','p1','p2']].to_numpy().argmax(1),frame.pred)
def reduction(frame,table):
    keyed=table.set_index(['root','truth']); assert keyed.index.is_unique
    assert len(keyed)==frame.groupby(['root','truth']).ngroups
    for key,g in frame.groupby(['root','truth']):
        r=keyed.loc[key]; assert r.support==len(g) and r.errors==int(g.pred.ne(g.truth).sum())
        for column in ['stable_CE','probability_clip_CE']:
            total=math.fsum(g[column]);value=float(r[column+'_sum'])
            assert abs(value-total)<=2*np.finfo(np.float64).eps*max(1,abs(total))

def main():
    assert not OUT.exists()
    qualification=read(CACHE/'qualification.json')
    assert qualification['cached_actual_head_calls']==360 and qualification['new_official_heads']==0
    goldfile=ROOT/'data/official/train.parquet'
    assert sha(goldfile)=='6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
    gold=pd.read_parquet(goldfile,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    reports=[];counts=0
    for f,arm in [(0,'A'),(0,'B'),(1,'A')]:
        name=f'fold{f}_{arm}';fit=read(RUN/name/'fit.json')
        events=[json.loads(s) for s in (PAST/name/'calls.jsonl').read_text().splitlines()]
        n=sum(e['kind']=='head' and e['event']=='completed' for e in events)
        assert n==(len(fit['last5'])+1)*(12+(10 if f==0 else 4))
        assert sum(e['kind']=='feature' and e['event']=='completed' for e in events)==n
        assert not any(e['kind']=='full_class_gradient' for e in events)
        pairs=0
        for label,update in [(f"accepted{item['update']}",item['update']) for item in fit['last5']]+[('endpoint',fit['accepted_updates'])]:
            for scope in ['OOF','deployment']:
                actual=pd.read_parquet(PAST/name/f'{label}_{scope}_rows.parquet')
                frozen=pd.read_parquet(RUN/name/f'{label}_{scope}_rows.parquet')
                matching=pd.read_parquet(RUN/name/f'accepted{update}_{scope}_rows.parquet')
                stored_source=pd.read_parquet(RUN/name/f'accepted{update}_{scope}_sources.parquet')
                identity(actual,gold);identity(frozen,gold);identity(matching,gold)
                for cols,key in [(['p0','p1','p2'],'probability'),(['logp0','logp1','logp2'],'log_probability')]:
                    assert repeat_values(actual[cols],frozen[cols],key)['passed']
                reduction(matching,stored_source)
                review=source_sum_repeat_review(actual,matching,sources(actual),stored_source)
                assert review['passed']
                persisted=read(CACHE/name/f'{label}_{scope}_source_review.json')
                assert persisted==review
                pairs+=1
        q=np.load(RUN/name/'endpoint_deployment_probability.npy')
        assert q.shape==(22546,3) and np.isfinite(q).all()
        reports.append(dict(fold=f,arm=arm,actual_cached_head_calls=n,row_source_pairs_checked=pairs,
          frozen_full_q_sha256=sha(RUN/name/'endpoint_deployment_probability.npy'),
          full_replayed_q_saved=False,full_q_scope='Only v5 executed repeat assertion and original frozen q; actual replay cache contains FIT rows only.'))
        counts+=n
    assert counts==360
    failure=read(RUN/'evaluation_failure_v5.json');assert 'line 89' in failure['traceback']
    for relative,digest in qualification['source_sha256'].items():assert sha(ROOT/relative)==digest
    report=dict(status='three_saved_actual_replays_and_matching_source_tables_independently_passed',reports=reports,
      cached_completed_heads=counts,new_official_heads=0,new_features=0,new_gradients=0,new_fits=0,
      quality_acceptance=False,full_task_quality_evaluated=False,
      source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),goldfile,CACHE/'qualification.json',RUN/'evaluation_failure_v5.json']})
    OUT.mkdir();(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','cached_completed_heads','new_official_heads','quality_acceptance']},ensure_ascii=False))
if __name__=='__main__':main()
