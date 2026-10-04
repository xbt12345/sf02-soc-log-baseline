"""Fixed-terminal fit/heldout support comparison for 62 priority S errors."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v124_header_trial_20260929'
LADDER=ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet'


def main():
    target=OUT/'fit_support_probe.json'
    if target.exists():raise FileExistsError(target)
    q=json.loads((OUT/'capability_qualification.json').read_text(encoding='utf-8'))
    d=pd.read_parquet(LADDER)
    pred=pd.read_parquet(OUT/'expert_ASA_predictions.parquet')
    assert d.row_position.equals(pred.row_position)
    obs=[]
    for fold in range(3):
        p={arm:np.load(OUT/f'fold{fold}_{arm}/epoch25_prob.npy').argmax(1)
           for arm in ('A','B')}
        for card in q['cards']:
            if card['fold']!=fold:continue
            k=card['destination_key']
            fit=d[d.fold.ne(fold)&d.destination_key.eq(k)]
            held=d[d.fold.eq(fold)&d.destination_key.eq(k)&
                d.diagnostic_bucket.eq('known_parameter_two_roots_per_class')]
            assert len(held)==card['target_S_errors']
            result={'fold':fold,'destination_key':k,'target_S_rows':len(held),'fit_by_class':{}}
            for c in (1,2):
                rows=fit[fit.truth.eq(c)]
                result['fit_by_class'][str(c)]={'rows':len(rows),'roots':int(rows.root.nunique()),
                    **{f'{arm}_correct':int((p[arm][rows.local.to_numpy()]==c).sum()) for arm in ('A','B')}}
            for arm in ('A','B'):
                result[f'held_{arm}_correct']=int((pred.loc[held.index,f'expert_pred_{arm}']==2).sum())
            obs.append(result)
    assert len(obs)==len(q['cards'])==15 and sum(z['target_S_rows'] for z in obs)==62
    result={'status':'fixed_epoch_fit_support_diagnosis_no_new_training',
        'new_fits':0,'cards':obs,
        'fit_S_correct_over_fifteen_roles':{arm:sum(z['fit_by_class']['2'][f'{arm}_correct'] for z in obs)
            for arm in ('A','B')},
        'fit_S_support_over_fifteen_roles':sum(z['fit_by_class']['2']['rows'] for z in obs),
        'held_priority_S_correct':{arm:sum(z[f'held_{arm}_correct'] for z in obs) for arm in ('A','B')},
        'limitations':['Fifteen fold-by-behavior fit roles reuse some records; totals are not independent population totals.',
            'Fit-side correctness plus heldout failure does not establish which missing field or label rule would fix transfer.',
            'Only fixed epoch25 is used; no checkpoint selection.']}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cards'},ensure_ascii=False))


if __name__=='__main__':main()
