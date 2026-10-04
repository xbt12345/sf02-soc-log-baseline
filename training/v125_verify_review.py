"""Independent row-identity/count verification for the V125 no-fit review."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v125_review_20260929'
PREV=ROOT/'artifacts/v124_header_trial_20260929'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    dest=OUT/'verification.json'
    if dest.exists():raise FileExistsError(dest)
    delivery=read(PREV/'delivery.json')
    for rel,h in delivery['artifact_sha256'].items():assert sha(ROOT/rel)==h,rel
    report=read(OUT/'diagnosis.json')
    for rel,h in report['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    official=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary'])
    y=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    full=pd.read_parquet(PREV/'full_prediction_ledger.parquet')
    expert=pd.read_parquet(PREV/'expert_ASA_predictions.parquet')
    assert len(y)==len(full)==2056871 and np.array_equal(full.row_position,np.arange(len(y)))
    assert np.array_equal(y[expert.row_position],expert.truth)
    counts={}
    for arm in ('A','B'):
        p=full['final_pred_'+arm].to_numpy()
        counts[arm]={str(c):{'rows':int((y==c).sum()),'correct':int(((y==c)&(p==c)).sum())} for c in (0,1,2)}
        for c in (0,1,2):
            old=read(PREV/'independent_postfit_audit.json')['full_per_class'][arm][str(c)]
            assert counts[arm][str(c)]=={'rows':old['support'],'correct':old['correct']}
    cross=pd.read_parquet(OUT/'crossed_frozen_predictions.parquet')
    assert len(cross)==4*682 and not cross.duplicated(['model','input','row_position']).any()
    assert np.array_equal(y[cross.row_position],cross.truth)
    for (model,inp),part in cross.groupby(['model','input']):
        expect=report['crossed_header_682']['model'+model+'_input'+inp]
        for c in (1,2):
            assert expect[str(c)]=={'rows':int(part.truth.eq(c).sum()),'errors':int((part.truth.eq(c)&part.prediction.ne(c)).sum())}
        if model==inp:
            lookup=expert.set_index('row_position')
            assert np.array_equal(part.prediction,lookup.loc[part.row_position,'expert_pred_'+model])
    support=pd.read_parquet(OUT/'priority_fit_roles.parquet')
    ref=expert.set_index('row_position')
    assert not support.duplicated(['outer_fold','arm','row_position']).any()
    assert np.array_equal(y[support.row_position],support.truth)
    assert np.array_equal(ref.loc[support.row_position,'root'],support.root)
    assert (ref.loc[support.row_position,'fold'].to_numpy()!=support.outer_fold.to_numpy()).all()
    for (fold,arm),part in support.groupby(['outer_fold','arm']):
        p=np.load(PREV/f'fold{fold}_{arm}/epoch25_prob.npy')[part.local]
        assert np.array_equal(p.argmax(1),part.prediction)
        assert np.allclose(p[:,2]-p[:,1],part.p_S_minus_p_M,atol=1e-8,rtol=0)
    for arm,part in support.groupby('arm'):
        s=part[part.truth.eq(2)];e=s[s.prediction.ne(2)];expect=report['priority_support'][arm]
        assert len(s)==expect['S_role_rows']==206
        assert s.row_position.nunique()==expect['S_unique_original_rows']==136
        assert len(e)==expect['S_error_role_rows']
        assert e.row_position.nunique()==expect['S_error_unique_original_rows']
    chain=pd.read_parquet(OUT/'parameter_chain.parquet')
    assert chain.row_position.equals(expert.row_position) and chain.parameter_chain_matches.all()
    assert chain.numeric_parameters_checked.sum()==report['input_chain']['raw_numeric_parameter_comparisons']==173142
    assert chain.unavailable_ports_not_reconstructed.sum()==52472
    plan=read(ROOT/'training/review_policy/v125_next_training_plan.json')
    assert plan['latest_actual_training']=='V124' and not plan['model_promoted']
    assert plan['training']['primary_fits']==9 and len(plan['training']['order'])==9
    assert set(map(tuple,plan['training']['order']))=={(f,a) for f in range(3) for a in ('A','B','C')}
    assert sum(plan['training']['expected_steps_per_arm'])*3==plan['training']['expected_primary_steps']==13350
    assert plan['classifier_fits_this_review']==report['classifier_fits_new']==0
    assert plan['confirmation']['only_after_all_primary_gates']
    evidence=list(OUT.glob('*.json'))+list(OUT.glob('*.parquet'))
    evidence += [Path(__file__),ROOT/'training/v125_frozen_diagnosis.py',
        ROOT/'docs/V125_ROOT_CAUSE_REVIEW_AND_NEXT_PLAN.md',
        ROOT/'training/review_policy/v125_next_training_plan.json',ROOT/'training/review_policy/v125_risk_actions.json']
    result={'status':'v125_frozen_review_counts_and_identity_verified','new_classifier_fits':0,
        'latest_actual_training':'V124','model_promoted':False,
        'parent_bound_files_unchanged':len(delivery['artifact_sha256']),
        'full_official_label_recount':counts,'crossed_header_rows':682,
        'priority_fit_unique_S':136,'priority_fit_S_roles':206,
        'scope':'Independent official-label counts and saved-model-probability identities; numeric chain was executed by the audit, not reimplemented here. No new training or blind test.',
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(evidence)}}
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
