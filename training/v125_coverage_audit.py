"""Post-fit source-29 coverage and apparent clock association; no model updates."""
import json

import numpy as np
import pandas as pd

from v125_experiment_review import ROOT,sha,require_run_seal

OUT=ROOT/'artifacts/v125_order_trial_20260929'
TRACE=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'


def main():
    require_run_seal(OUT/'run_seal.json',ROOT/'training/v125_train.py')
    t=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth','raw_message'])
    p=pd.read_parquet(OUT/'expert_ASA_predictions.parquet')
    spans=pd.read_parquet(OUT/'body_span_ledger.parquet',columns=['row_position','normalized_body_sha256'])
    if not (t.row_position.equals(p.row_position) and t.row_position.equals(spans.row_position)):
        raise ValueError('Original-row alignment changed')
    d=t[['row_position','local','root','fold','truth']].copy()
    d['body']=spans.normalized_body_sha256
    for arm in 'ABC':d[arm]=p['expert_pred_'+arm]
    regressed=(d.root.eq(29)&d.truth.eq(2)&d.A.eq(2)&d.B.eq(1))
    train=d[d.fold.ne(1)]
    body_support=train.groupby('body').agg(
        M=('truth',lambda s:int(s.eq(1).sum())),
        S=('truth',lambda s:int(s.eq(2).sum())),
        roots=('root','nunique'))
    q=d.loc[regressed,['body']].join(body_support,on='body')
    raw=t[t.root.eq(29)].copy()
    raw['day']=raw.raw_message.str.extract(r'^<\d+>([A-Za-z]+\s+\d+)')[0]
    if raw.day.isna().any():raise ValueError('Unrecognized root-29 day header')
    sub=d[d.root.eq(29)]
    s=sub.truth.eq(2)
    report={'status':'postfit_coverage_audit_no_training',
        'source_sha256':sha(ROOT/'training/v125_coverage_audit.py'),
        'source29':{'rows':len(raw),'folds':sorted(raw.fold.unique().tolist()),
            'M_rows':int(raw.truth.eq(1).sum()),'S_rows':int(raw.truth.eq(2).sum()),
            'S_day_Jul26':int((raw.truth.eq(2)&raw.day.eq('Jul 26')).sum()),
            'M_day_Jul26':int((raw.truth.eq(1)&raw.day.eq('Jul 26')).sum()),
            'other_M_distinct_days':int(raw.loc[raw.truth.eq(1),'day'].nunique()),
            'A_S_errors':int((sub.loc[s,'A']!=2).sum()),
            'B_S_errors':int((sub.loc[s,'B']!=2).sum()),
            'C_S_errors':int((sub.loc[s,'C']!=2).sum())},
        'A_correct_S_to_B_M':{'original_rows':int(regressed.sum()),
            'distinct_old_local':int(d.loc[regressed,'local'].nunique()),
            'distinct_normalized_body_hashes':int(d.loc[regressed,'body'].nunique()),
            'exact_body_seen_in_other_fold_rows':int(q.M.notna().sum()),
            'exact_body_unseen_in_other_fold_rows':int(q.M.isna().sum()),
            'exact_body_with_other_fold_S_support_rows':int(q.S.fillna(0).gt(0).sum())},
        'limitations':['Calendar day is associated with labels within one repeatedly inspected source; this does not prove a legitimate threat rule or the old model\'s exact attribution.',
                       'Exact normalized-body absence does not mean no shared n-gram or semantic behavior in training.',
                       'Source groups are graph components, not confirmed independent organizations.'],
        'classifier_fits':0,'optimizer_steps':0,'model_promoted':False}
    path=OUT/'postmortem_coverage.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
