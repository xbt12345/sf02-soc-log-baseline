"""Bounded no-fit qualification for the 8 visible-parameter error behaviors.

Descriptive labels/errors come from inspected development data. This audit
does not create training weights, pseudo-labels, or inference-time routes.
"""
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v124_header_trial_20260929'
LADDER=ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet'
TRACE=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'


def main():
    target=OUT/'capability_qualification.json'
    if target.exists():raise FileExistsError(target)
    d=pd.read_parquet(LADDER)
    raw=pd.read_parquet(TRACE,columns=['row_position','raw_message'])
    assert d.row_position.equals(raw.row_position)
    q=d[d.diagnostic_bucket.eq('known_parameter_two_roots_per_class')]
    assert len(q)==62 and q.destination_key.nunique()==8
    observations=[];seen=set()
    for (fold,key),targets in q.groupby(['fold','destination_key'],sort=True):
        fit=d[d.fold.ne(fold)&d.destination_key.eq(key)]
        assert set(fit.truth.unique())=={1,2}
        assert all(fit[fit.truth.eq(cls)].root.nunique()>=2 for cls in (1,2))
        assert fit.root.isin(targets.root).sum()==0
        cls={str(c):{'rows':int(fit.truth.eq(c).sum()),
                     'roots':int(fit[fit.truth.eq(c)].root.nunique()),
                     'unique_N1_locals':int(fit[fit.truth.eq(c)].local.nunique())} for c in (1,2)}
        examples=[]
        for c in (1,2):
            for _,r in fit[fit.truth.eq(c)].drop_duplicates('root').head(3).iterrows():
                examples.append({'truth':c,'root':int(r.root),'row_position':int(r.row_position),
                                 'raw_message':raw.raw_message.iloc[r.name]})
        observations.append({'fold':int(fold),'destination_key':key,'target_S_errors':len(targets),
            'target_roots':int(targets.root.nunique()),'fit':cls,
            'fit_rows_with_same_old_N1_local_both_labels':int(fit.groupby('local').truth.nunique().gt(1).sum()),
            'fit_examples_from_distinct_roots':examples,
            'mechanism_status':'unresolved',
            'reason':'Multiple labeled roots do not establish which raw field defines a portable M/S distinction; no new raw-to-matrix defect or class-conditional cross-root rule qualified by this count alone.'})
        seen.add(key)
    assert len(seen)==8 and sum(x['target_S_errors'] for x in observations)==62
    result={'status':'bounded_qualification_complete_no_capacity_factor_approved',
       'eligible_capacity_factors':0,'new_classifier_fits':0,'visible_priority_S_errors':62,
       'distinct_protocol_parameter_behaviors':8,'query_fold_behavior_cards':len(observations),
       'cards':observations,
       'decision':'Run registered header input trial. Do not run conditional parser/branch experiment without a separately evidenced field mechanism and frozen controls.',
       'limitations':['Cards are purposefully targeted after inspecting V121 errors; no blind transfer claim.',
          'Only a small set of raw examples per class/root is rendered; numerical fit support uses every legal fit row.',
          'Ineligible means not qualified by this bounded audit, not unlearnable in principle.']}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cards'},ensure_ascii=False))


if __name__=='__main__':main()
