"""Executable real counterexamples: loss, cohort repair and aggregate gains."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiment_review import read,check_bindings
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/v154_directional_neighborhood_v2_20261001'

def main():
    case=read(ROOT/'training/review_policy/v154_observed_neighborhood_cases.json');check_bindings(case['source_sha256'])
    a=read(BASE/'audit.json');cohort=pd.read_parquet(ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet')
    counts=[]
    for arm in ['A','B']:
        base=pd.read_parquet(BASE/f'{arm}_base_outer_original_rows.parquet')
        plus=pd.read_parquet(BASE/f'{arm}_plus_TRAIN_gradient_outer_original_rows.parquet')
        minus=pd.read_parquet(BASE/f'{arm}_minus_TRAIN_gradient_outer_original_rows.parquet')
        random=pd.read_parquet(BASE/f'{arm}_fixed_random_outer_original_rows.parquet')
        for z in [plus,minus,random]:assert np.array_equal(z.row_position,base.row_position) and np.array_equal(z.truth,base.truth)
        hard=cohort.known_578_cohort.to_numpy();strict=cohort.same_family_and_outer_fold_control_S.to_numpy()
        hard_repair=int((hard&base.pred.ne(base.truth)&plus.pred.eq(plus.truth)).sum())
        plus_M_deterioration=int((plus.truth.eq(1)&plus.pred.ne(plus.truth)).sum()-(base.truth.eq(1)&base.pred.ne(base.truth)).sum())
        strict_minus_new=int((strict&base.pred.eq(base.truth)&minus.pred.ne(minus.truth)).sum())
        net_minus=int(base.pred.ne(base.truth).sum()-minus.pred.ne(minus.truth).sum())
        assert (hard_repair,plus_M_deterioration,strict_minus_new,net_minus)==((2,148,6,80) if arm=='A' else (4,94,6,50))
        assert np.array_equal(random.pred,base.pred)
        for endpoint in [r for r in a['endpoints'] if r['arm']==arm]:
            for point in endpoint['points'][1:]:
                assert point['TRAIN_member_CE']>endpoint['points'][0]['TRAIN_member_CE']
                assert point['pure_TRAIN_errors']==point['all_base_correct_TRAIN_regressions']==0
        counts.append(dict(arm=arm,hard_plus_repairs=hard_repair,plus_M_deterioration=plus_M_deterioration,
            strict_minus_new_S_errors=strict_minus_new,minus_net_ASA_repair=net_minus))
    assert not a['automatic_training_qualified'] and not a['quality_acceptance'] and not a['model_selection']
    failure=read(ROOT/'artifacts/v154_directional_neighborhood_20261001/failure.json')
    assert failure['classifier_gradients']==failure['classifier_forward_calls']==0
    print(json.dumps(dict(status='real_directional_counterexamples_replayed',counts=counts,
        classifier_fits=0,new_updates=0,loss_or_aggregate_or_hard_repair_not_training_authority=True),ensure_ascii=False))

if __name__=='__main__':main()
