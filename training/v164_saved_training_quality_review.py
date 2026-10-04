"""All actual accepted rows, including unprotected mixed-origin changes."""
import json,math,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v161_independent_all_finite_results_review import parameter_hash
TRIAL=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
OUT=ROOT/'artifacts/v164_saved_training_quality_review_20261002'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def compare(frame,before,fixed):
    assert np.array_equal(frame[['row_position','local','truth','root','pure_current_input']].to_numpy(),fixed[['row_position','local','truth','root','pure_current_input']].to_numpy())
    wrong=frame.pred.ne(frame.truth);priorwrong=before.pred.ne(before.truth);fixedwrong=fixed.pred.ne(fixed.truth);pure=frame.pure_current_input;reports={}
    for cls,label in [(1,'M'),(2,'S')]:
        subset=frame.truth.eq(cls);repair=subset&fixedwrong&~wrong;new=subset&~fixedwrong&wrong
        old_margin=fixed[f'logp{cls}']-fixed[[f'logp{r}' for r in range(3) if r!=cls]].max(axis=1);margin=frame[f'logp{cls}']-frame[[f'logp{r}' for r in range(3) if r!=cls]].max(axis=1);remaining=subset&fixedwrong&wrong
        groups=frame.loc[subset].assign(correct=~wrong[subset]).groupby('root').agg(support=('truth','size'),correct=('correct','sum'));recall=groups.correct/groups.support
        roots=frame.loc[repair].groupby('root').size().sort_values(ascending=False)
        reports[label]=dict(original_class_mass=int(subset.sum()),errors=int((subset&wrong).sum()),pure_errors=int((subset&wrong&pure).sum()),mixed_errors=int((subset&wrong&~pure).sum()),repairs_vs_previous_accepted=int((subset&priorwrong&~wrong).sum()),new_errors_vs_previous_accepted=int((subset&~priorwrong&wrong).sum()),repairs_vs_fixed_endpoint=int(repair.sum()),new_errors_vs_fixed_endpoint_all_original_rows=int(new.sum()),new_mixed_errors_vs_fixed_endpoint=int((new&~pure).sum()),new_pure_errors_vs_fixed_endpoint=int((new&pure).sum()),protected_regressions=int((subset&wrong&frame.protected_correct).sum()),stable_CE_mean=math.fsum(frame.loc[subset,'stable_CE'])/int(subset.sum()),fixed_endpoint_repair_distinct_locals=int(frame.loc[repair,'local'].nunique()),fixed_endpoint_repair_distinct_roots=int(frame.loc[repair,'root'].nunique()),largest_root_share_of_fixed_endpoint_repairs=float(roots.iloc[0]/roots.sum()) if len(roots) else None,root_proxy_support_groups=len(groups),zero_recall_root_proxy_groups=int((groups.correct==0).sum()),equal_root_proxy_recall=float(recall.mean()),remaining_fixed_errors=int(remaining.sum()),remaining_fixed_error_margin_quantiles={str(q):float(margin[remaining].quantile(q)) for q in [0.,.25,.5,.75,1.]} if remaining.any() else {},remaining_fixed_error_margin_change_quantiles={str(q):float((margin-old_margin)[remaining].quantile(q)) for q in [0.,.25,.5,.75,1.]} if remaining.any() else {},root_is_source_proxy_not_independent_attack_behavior=True)
    return reports

def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/fit.json').exists() for r in range(3));OUT.mkdir();goldpath=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldpath,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    files={p for p in TRIAL.rglob('*') if p.is_file()}|{Path(__file__).resolve(),goldpath,ROOT/'training/v161_independent_all_finite_results_review.py'};bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));roles=[]
    for role in range(3):
        folder=TRIAL/f'role{role}';review_folder=OUT/f'role{role}';review_folder.mkdir();fit=read(folder/'fit.json');fixed=pd.read_parquet(folder/'baseline/OOF_original_rows.parquet');previous=fixed;mask=fixed.protected_correct.to_numpy(bool);states=[];oldmixcorrect=int((~fixed.pure_current_input&fixed.pred.eq(fixed.truth)&~fixed.protected_correct).sum())
        for index in range(1,fit['permanent_updates']+1):
            state=folder/f'accepted{index}';frame=pd.read_parquet(state/'OOF_original_rows.parquet');receipt=read(state/'commit.json');checkpoint=torch.load(state/'checkpoint.pt',map_location='cpu',weights_only=True)['state'];assert parameter_hash(checkpoint)==receipt['parameter_sha256']
            assert np.array_equal(frame.truth,gold[frame.row_position]);wrong=frame.pred.ne(frame.truth).to_numpy();beforewrong=previous.pred.ne(previous.truth).to_numpy();assert not np.any(mask&wrong);newrepairs=beforewrong&~wrong;mask=mask|newrepairs
            assert np.array_equal(frame.protected_correct,mask);protected=pd.read_parquet(state/'cumulative_correct_protection.parquet');assert np.array_equal(protected.row_position,frame.loc[mask,'row_position']);assert receipt['newly_repaired_original_rows']==int(newrepairs.sum()) and receipt['newly_repaired_mixed_original_rows']==int((newrepairs&~frame.pure_current_input.to_numpy()).sum())
            stats=compare(frame,previous,fixed);save(review_folder/f'accepted{index}_all_original_classification_quality_cpu_review.json',dict(state_index=index,classes=stats,initial_mixed_correct_outside_registered_pure_guard=oldmixcorrect,supervised_development_training_not_external_validation=True,official_calls=0));states.append(dict(state_index=index,classes=stats,parameter_sha256=receipt['parameter_sha256']));previous=frame
        endpoint=pd.read_parquet(folder/'endpoint/OOF_original_rows.parquet');assert np.array_equal(endpoint.truth,gold[endpoint.row_position]) and np.array_equal(endpoint.pred,previous.pred);endpointstate=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)['state'];assert parameter_hash(endpointstate)==fit['endpoint_parameter_sha256']
        if states:assert states[-1]['parameter_sha256']==fit['endpoint_parameter_sha256']
        classes=compare(endpoint,previous,fixed);roles.append(dict(role=role,status=fit['status'],accepted_updates=fit['permanent_updates'],exception=fit['exception'],initial_mixed_correct_outside_registered_pure_guard=oldmixcorrect,all_actual_state_reviews=states,endpoint_classes=classes,last_five_actual_distinct_states_mastered=fit['last_five_actual_distinct_states_mastered'],training_side_mastery=fit['last_five_actual_distinct_states_mastered'] and fit['endpoint_OOF_stats']['mastered']))
    # New CPU review files are not original training inputs and are deliberately
    # outside the pre-review binding set. Original recorded inputs remain exact.
    check_bindings(bindings);save(OUT/'review.json',dict(status='all_actual_short_training_states_original_gold_cumulative_repair_protection_and_full_classification_quality_reviewed',roles=roles,training_side_all_roles_mastered=all(r['training_side_mastery'] for r in roles),official_review_calls=0,supervised_development_training_not_external_validation=True,root_goal_complete=False,full_task_or_source_transfer_acceptance=False,source_sha256=bindings));print(json.dumps(dict(status='V164_saved_training_quality_review_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
