"""Independent row expansion and nested-role review; no training or tuning."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v128_mechanism_review_20260929'
PARENT = ROOT/'artifacts/v127_frozen_branch_trial_20260929'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def require(ok, msg):
    if not ok:raise ValueError(msg)


def role_check(m, d):
    require(not m.duplicated(['outer_fold','row_position']).any(),'Duplicate legal role')
    for f in range(3):
        q=m[m.outer_fold==f]
        legal=d[d.fold!=f]
        require(np.array_equal(q.row_position.to_numpy(),legal.row_position.to_numpy()),'Missing or invalid outer fit rows')
        require(not set(q.root)&set(d.loc[d.fold==f,'root']),'Outer source contamination')
        require(q.groupby('root').inner_fold.nunique().max()==1,'Group split')
        for j in range(3):
            t=q[q.inner_fold!=j]
            c=q[q.crossfit_teacher==j]
            i=q[q.matched_infit_teacher==j]
            require(not set(t.root)&set(c.root),'CF teacher has seen predicted root')
            require(set(i.root)<=set(t.root),'IS role not actually in teacher fit')


def main():
    target=OUT/'verification.json'
    if target.exists():raise FileExistsError(target)
    delivery=read(PARENT/'delivery.json')
    binding={**delivery['artifact_sha256'],**delivery['source_sha256'],
             delivery['report']:delivery['report_sha256'],delivery['risk_addendum']:delivery['risk_addendum_sha256']}
    for p,h in binding.items():require(sha(ROOT/p)==h,'V127 historical file changed: '+p)
    audit=read(OUT/'audit.json');score=read(OUT/'score_role_audit.json');prep=read(OUT/'nested_role_preparation.json')
    for v,source in [(audit,'v128_no_fit_review.py'),(score,'v128_score_role_audit.py'),(prep,'v128_prepare_roles.py')]:
        require(v['source_sha256']==sha(ROOT/'training'/source),'Review source changed')
        for p,h in v.get('inputs_sha256',{}).items():require(sha(ROOT/p)==h,'Review input changed')
    d=pd.read_parquet(OUT/'score_role_and_error_ledger.parquet')
    official=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary'])
    labels=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    require(len(labels)==2056871 and len(d)==112807 and d.row_position.is_unique,'Wrong population')
    require(np.array_equal(labels[d.row_position],d.truth),'Official labels differ')
    require(np.array_equal(d.row_position,pd.read_parquet(PARENT/'expert_ASA_predictions.parquet',columns=['row_position']).row_position),'Row order changed')
    for item in audit['constant_replacement_units_correction']:
        f,a=item['fold'],item['arm'];role=d.fold.to_numpy()==f
        loc=d.local.to_numpy();z=np.load(PARENT/f'fold{f}_base_member_logits.npy').astype(np.float64)
        rr=np.load(PARENT/f'fold{f}_{a}/epoch50_residual.npy').astype(np.float64)
        mean=rr[loc[~role]].mean(0)
        actual=softmax(z[loc[role]]+rr[loc[role],None,:],axis=-1).mean(1).argmax(1)
        constant=softmax(z[loc[role]]+mean,axis=-1).mean(1).argmax(1)
        require(int((actual!=constant).sum())==item['changed_original_rows'],'Original-row flip count wrong')
        require(np.unique(loc[role][actual!=constant]).size==item['changed_unique_old_local_inputs'],'Input count wrong')
        require(int((actual!=d.truth.to_numpy()[role]).sum())==item['real_total_errors'],'Real total error wrong')
        require(int((constant!=d.truth.to_numpy()[role]).sum())==item['constant_total_errors'],'Constant total error wrong')
    for c in (1,2):
        m=(d.truth==c)&(d.pred_P!=c)
        n=d.loc[m,'in_fit_base_models_correct_count_of_2'].value_counts()
        match=next(x for x in score['score_exposure'] if x['slice']=='P_errors' and x['truth']==c)
        require(len(d[m])==match['rows'] and int(n.get(2,0))==match['both_in_fit_models_correct']
                and int(n.get(0,0))==match['neither_in_fit_model_correct'],'Exposure counts wrong')
    m=pd.read_parquet(OUT/'nested_score_roles.parquet')
    role_check(m,d)
    require(sha(OUT/'nested_score_roles.parquet')==prep['manifest_sha256'],'Prepared roles changed')
    require(len(m)==225614 and m.row_position.nunique()==112807,'Role denominator wrong')
    # Actual population negative control: using in-fit teachers for CF must fail.
    bad=m.copy();bad['crossfit_teacher']=bad['matched_infit_teacher']
    try:role_check(bad,d)
    except ValueError as e:negative=str(e)
    else:raise ValueError('Leaking teacher-role negative control accepted')
    steps=[]
    for f in range(3):
        q=m[m.outer_fold==f]
        for j in range(3):steps.append(math.ceil(q.loc[q.inner_fold!=j,'local'].nunique()/256)*25)
    plan=read(ROOT/'training/review_policy/v128_next_training_plan.json')
    require(steps==plan['teacher_bank']['expected_steps'] and sum(steps)==8900,'Teacher budget not from actual roles')
    require(plan['data']['new_inner_manifest_sha256']==sha(OUT/'nested_score_roles.parquet'),'Plan role binding wrong')
    require(plan['matched_primary']['primary_fits_including_teachers']==9+sum(x['fits'] for x in plan['matched_primary']['arms'].values())==21,'Fit budget wrong')
    require(plan['matched_primary']['all_neural_steps']==8900+17800,'Step budget wrong')
    require(not plan['quality_acceptance'] and not plan['model_promoted'] and not plan['confirmation']['automatic_in_this_plan'],'Plan exceeds research authority')
    files=[p for p in OUT.iterdir() if p.is_file()]
    files += [ROOT/'training'/n for n in ('v128_no_fit_review.py','v128_score_role_audit.py','v128_prepare_roles.py','v128_verify_review.py')]
    files += [ROOT/'docs/V128_SCORE_ROLE_REVIEW_AND_NEXT_PLAN.md',ROOT/'training/review_policy/v128_next_training_plan.json',ROOT/'training/review_policy/v128_risk_actions.json']
    result={'status':'V128_no_fit_evidence_and_role_preparation_verified','latest_actual_training':'V127',
            'classifier_fits':0,'optimizer_steps':0,'model_promoted':False,'quality_acceptance':False,
            'unchanged_V127_bound_files':len(binding),'original_rows':112807,'full_official_rows':len(labels),
            'nested_roles':len(m),'source_contamination_negative_control':negative,
            'P_mean_replacement_original_row_flips':[x['changed_original_rows'] for x in audit['constant_replacement_units_correction'] if x['arm']=='P'],
            'checks_scope':'Original-row diagnostic recomputation, historical hashes and legal nested role schedule only; no runtime/training quality acceptance.',
            'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
