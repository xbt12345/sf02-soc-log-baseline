"""Read-only replay of V125 failure cases from saved original-row decisions."""
import json

import numpy as np
import pandas as pd

from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v125_order_trial_20260929'


def main():
    d=pd.read_parquet(OUT/'expert_ASA_predictions.parquet')
    trace=pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',
                          columns=['row_position','root','fold','truth','raw_message'])
    body=pd.read_parquet(OUT/'body_span_ledger.parquet',columns=['row_position','normalized_body_sha256'])
    if not (d.row_position.equals(trace.row_position) and d.row_position.equals(body.row_position)):
        raise ValueError('V125 original-row evidence no longer aligned')
    y=d.truth.to_numpy();fold=d.fold.to_numpy();root=d.root.to_numpy()
    a=d.expert_pred_A.to_numpy();b=d.expert_pred_B.to_numpy();c=d.expert_pred_C.to_numpy()
    miss={arm:{str(cl):int(((y==cl)&(p!=cl)).sum()) for cl in (1,2)}
          for arm,p in [('A',a),('B',b),('C',c)]}
    fold_errors=[{'fold':k,'A':int(((fold==k)&(a!=y)).sum()),
                  'B':int(((fold==k)&(b!=y)).sum())} for k in range(3)]
    target=(root==29)&(y==2)&(a==2)&(b==1)
    base=np.load(OUT/'postmortem_B_base_only_pred.npy')
    same_wrong=int((target&(base==1)).sum())
    raw=trace.loc[root==29].copy()
    day=raw.raw_message.str.extract(r'^<\d+>([A-Za-z]+\s+\d+)')[0]
    date_s=int((raw.truth.eq(2)&day.eq('Jul 26')).sum())
    date_m=int((raw.truth.eq(1)&day.eq('Jul 26')).sum())
    other=body.loc[fold!=1,'normalized_body_sha256'].unique()
    unseen=int((~body.loc[target,'normalized_body_sha256'].isin(other)).sum())
    actual={'ASA_errors':miss,'fold_errors':fold_errors,'root29_new_S_misses':int(target.sum()),
            'root29_new_S_misses_base_only_still_wrong':same_wrong,
            'root29_Jul26_S':date_s,'root29_Jul26_M':date_m,
            'root29_new_S_misses_exact_body_unseen_elsewhere':unseen}
    expected={'ASA_errors':{'A':{'1':318,'2':2074},'B':{'1':326,'2':4154},'C':{'1':326,'2':4002}},
              'fold_errors':[{'fold':0,'A':278,'B':280},{'fold':1,'A':1172,'B':3242},{'fold':2,'A':942,'B':958}],
              'root29_new_S_misses':2046,'root29_new_S_misses_base_only_still_wrong':1884,
              'root29_Jul26_S':4017,'root29_Jul26_M':32,
              'root29_new_S_misses_exact_body_unseen_elsewhere':2044}
    if actual!=expected:raise ValueError('V125 historical counterexample no longer replays: '+json.dumps(actual))
    print(json.dumps({'status':'V125_counterexamples_replayed','model_updates':0,**actual},ensure_ascii=False))


if __name__=='__main__':main()
