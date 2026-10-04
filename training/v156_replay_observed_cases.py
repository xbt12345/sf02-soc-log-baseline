"""Executable actual geometry limits; no new features or classifier calls."""
import json
from pathlib import Path
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v156_conditional_neighborhood import OUT,save

def main():
    path=ROOT/'training/review_policy/v156_observed_neighborhood_cases.json';case=read(path);check_bindings(case['source_sha256'])
    q=pd.read_parquet(OUT/'fixed_cohort_relation_summary.parquet');q=q[q.role.eq('outer_HELD')]
    def count(space,cohort,relation,truth=None,errors=False):
        x=q[q.space.eq(space)&q.cohort.eq(cohort)&q.relation.eq(relation)]
        if truth is not None:x=x[x.truth.eq(truth)]
        return int(x['baseline_errors' if errors else 'original_rows'].sum())
    spaces=['actual_CSR_input','all16_H1','all16_V146_A_H2']
    assert [count(s,'hard578','same_class_closer') for s in spaces]==case['hard578_same_class_closer_by_space']==[48,112,16]
    assert [count(s,'strict51','other_class_closer') for s in spaces]==case['strict51_other_class_closer_by_space']==[41,28,23]
    assert all(count(s,'strict51','other_class_closer',errors=True)==0 for s in spaces)
    assert count(spaces[2],'all_ASA','same_class_closer',truth=2)>count(spaces[1],'all_ASA','same_class_closer',truth=2)
    assert (count(spaces[2],'all_ASA','no_same_class',truth=2),count(spaces[2],'all_ASA','no_same_class',truth=2,errors=True))==(238,8)
    w=pd.read_parquet(OUT/'selected_factual_witnesses.parquet');hard=w[w.cohort.eq('hard578')&w.neighbor_kind.eq('same_class')]
    assert not hard.whole_fine_behavior_identical.any()
    counter=read(ROOT/'artifacts/v156_independent_geometry_counterexample_20261001/audit.json')
    assert counter['source_sha256']==sha(ROOT/'training/v156_independent_geometry_counterexample.py')
    assert counter['complete_predictions_identical'] and [z['nearest_class'] for z in counter['cases']]==[2,1]
    assert all(z['max_logit_change']==0 for z in counter['cases'])
    result=dict(status='actual_global_local_support_and_equivalent_classifier_geometry_counterexamples_replayed',
        hard_same_class_closer=[48,112,16],correct_S_other_class_closer=[41,28,23],
        no_same_class_S=238,no_same_class_S_errors=8,fine_identical_hard_same_witnesses=0,
        feature_function_calls=0,model_forward_calls=0,gradients=0,classifier_fits=0,updates=0,
        issue_solved=False,new_training_qualified=False,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),path,OUT/'fixed_cohort_relation_summary.parquet',OUT/'selected_factual_witnesses.parquet']})
    save(OUT/'observed_cases_replay.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False))

if __name__=='__main__':main()
