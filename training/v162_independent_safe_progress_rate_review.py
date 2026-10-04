"""Compare old/new actual safe candidates on identical original targets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v160_independent_fixed_diagnostic_review import read,sha
from v161_independent_all_finite_results_review import target_risk,save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v162_independent_safe_progress_rate_review_20261002'

def main():
    assert not OUT.exists();OUT.mkdir()
    cases=[(1,ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role1/round0/probe14',
              ROOT/'artifacts/v162_fixed_endpoint_finite_restoration_20261002/role1/restoration1/finite_probe'),
           (2,ROOT/'artifacts/v161_cached_direction_backtrack_tail_20261002/role2/probe20',
              ROOT/'artifacts/v162_cached_function_restoration_tail_20261002/role2/restoration5/finite_probe')]
    sources={Path(__file__).resolve(),ROOT/'data/official/train.parquet'};reports=[]
    gold=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary']).label_binary.map(
        {'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    for role,old,new in cases:
        baseline=ROOT/f'artifacts/v162_fixed_endpoint_finite_restoration_20261002/role{role}/baseline/OOF_original_rows.parquet'
        targets=ROOT/f'artifacts/v161_independent_frozen_error_cohort_review_20261002/role{role}/fixed_pure_error_targets.parquet'
        sources|={baseline,targets}
        b=pd.read_parquet(baseline);t=pd.read_parquet(targets).row_position
        risks0=target_risk(b,t);values=[]
        for path in [old,new]:
            source=path/'OOF_original_rows.parquet';sources|={source,path/'probe.json'}
            f=pd.read_parquet(source);p=read(path/'probe.json')
            assert p['accepted'] and p['classification_guard']
            assert np.array_equal(f.row_position,b.row_position) and np.array_equal(f.truth,gold[f.row_position])
            drop=risks0-target_risk(f,t);assert np.all(drop>0)
            before_wrong=b.pred.ne(b.truth);after_wrong=f.pred.ne(f.truth)
            values.append(dict(actual_original_target_drop=drop.tolist(),
                               original_row_repairs=int((before_wrong&~after_wrong).sum()),
                               correct_row_regressions=int((~before_wrong&after_wrong).sum()),
                               source=path.relative_to(ROOT).as_posix()))
        ratio=np.array(values[1]['actual_original_target_drop'])/np.array(values[0]['actual_original_target_drop'])
        reports.append(dict(role=role,old=values[0],new=values[1],M_S_actual_drop_ratio=ratio.tolist()))
    save(OUT/'source_bindings.json',dict(source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)},official_calls=0))
    report=dict(status='same_original_targets_actual_safe_progress_rate_compared',reports=reports,official_calls=0,new_fits=0,
                permanent_updates=0,quality_acceptance=False,
                limitations='One-step safe-target loss progress, not future step prediction or classification mastery. Both roles retain original class denominators and source populations.')
    save(OUT/'review.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
