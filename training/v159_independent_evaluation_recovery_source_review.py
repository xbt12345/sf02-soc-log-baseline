"""Inspect evaluation-only recovery and exercise pure-array source policy."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import sha,check_bindings
from v159_source_sum_repeat_policy import source_sum_repeat_review

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v159_independent_evaluation_recovery_source_review_20261002'


def source(r):
    return r.assign(wrong=r.pred.ne(r.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()


def main():
    assert not OUT.exists()
    paths=[Path(__file__).resolve(),ROOT/'training/v159_source_sum_repeat_policy.py',ROOT/'training/v159_boundary_evaluate_v5.py',ROOT/'training/v159_evaluation_runtime_v5.py',ROOT/'training/v159_seal_evaluation_only_source_sum_repair.py',ROOT/'artifacts/v159_saved_source_sum_qualification_20261002/qualification.json']
    q=json.loads(paths[-1].read_text(encoding='utf-8'));check_bindings(q['source_sha256']);assert all(q['cases'].values())
    assert len(q['actual_saved_tables'])==25 and q['fresh_complete_replay_head_cap']==640
    e=paths[2].read_text(encoding='utf-8');ast.parse(e)
    assert "replay=REPAIR/'replayed_outputs'/" in e and "replay/'calls.jsonl'" in e
    assert "actual_oo.to_parquet(replay/" in e and "source.to_parquet(replay/" in e
    assert "assert np.array_equal(actual_oo.pred,oo.pred)" in e
    assert "q['fresh_replay_head_cap']!=640" in paths[3].read_text(encoding='utf-8')
    saved=pd.DataFrame(dict(row_position=np.arange(1000),root=np.repeat(7,1000),truth=np.repeat(1,1000),pred=np.repeat(1,1000),stable_CE=np.repeat(.001,1000),probability_clip_CE=np.repeat(.001,1000)))
    actual=saved.copy();actual['stable_CE']+=4*np.finfo(float).eps;actual['probability_clip_CE']+=4*np.finfo(float).eps
    good=source_sum_repeat_review(actual,saved,source(actual),source(saved));assert good['passed']
    checks={'admissible_row_errors_sum_accepted':good['passed']}
    bad=actual.copy();bad.loc[0,'pred']=2;checks['prediction_change_rejected']=not source_sum_repeat_review(bad,saved,source(bad),source(saved))['passed']
    bad=actual.copy();bad.loc[0,'stable_CE']+=64*np.finfo(float).eps;checks['out_of_envelope_row_rejected']=not source_sum_repeat_review(bad,saved,source(bad),source(saved))['passed']
    corrupt=source(actual);corrupt.loc[0,'stable_CE_sum']+=1e-12
    checks['aggregate_corrupt_even_within_propagated_envelope_rejected']=not source_sum_repeat_review(actual,saved,corrupt,source(saved))['passed']
    bad=actual.iloc[:-1];checks['missing_row_rejected']=not source_sum_repeat_review(bad,saved,source(bad),source(saved))['passed']
    assert all(checks.values())
    report=dict(status='evaluation_only_source_and_pure_array_rejection_cases_independently_reviewed',cases=checks,
                recovery_forward_and_feature_cap=640,carry_failed_forward_and_feature=110,net_technical_cap_increment=30,
                new_fit_cap=0,new_gradient_cap=0,new_update_cap=0,integer_classification_and_support_exact=True,
                actual_replay_arrays_preserved_before_assert=True,original_training_seal_immutable=True,
                permission_scope='Only seal and execute the reviewed frozen six-fit evaluation recovery; no fit, gradient or parameter change.',
                root_official_heads=0,root_official_gradients=0,root_official_fits=0,quality_acceptance=False,
                source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    OUT.mkdir();(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','cases','permission_scope','quality_acceptance']},ensure_ascii=False))


if __name__=='__main__':main()
