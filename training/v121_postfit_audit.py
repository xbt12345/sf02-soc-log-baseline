"""Recount registered V121 outcomes independently from raw prediction ledgers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_review import ROOT, read, sha, check_bindings
from v121_train import DEST, PLAN, OFFICIAL


def main():
    target=DEST/'postfit_audit.json'
    if target.exists():raise FileExistsError('Preserve existing postfit audit.')
    plan=read(PLAN)
    check_bindings(read(DEST/'run_seal.json')['source_sha256'])
    evaluation=read(DEST/'primary_evaluation.json')
    expert=pd.read_parquet(DEST/'expert_ASA_predictions.parquet')
    full=pd.read_parquet(DEST/'full_prediction_ledger.parquet')
    support=read(DEST/'support_results.json')
    if len(expert)!=112807 or len(full)!=2056871 or expert.row_position.duplicated().any() or full.row_position.duplicated().any():
        raise ValueError('Prediction ledger population or identity failure.')
    if not np.array_equal(full.row_position.to_numpy(),np.arange(len(full))):
        raise ValueError('Full prediction ledger order changed.')
    original=pd.read_parquet(OFFICIAL,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2})
    if original.isna().any() or not np.array_equal(original.iloc[expert.row_position.to_numpy()].to_numpy(),expert.truth.to_numpy()):
        raise ValueError('Original official labels do not agree with prediction population.')
    byfold=[];byclass={}
    for fold in range(3):
        for label in (1,2):
            q=expert[expert.fold.eq(fold) & expert.truth.eq(label)]
            byfold.append({'fold':fold,'truth':label,'support':len(q),
                           'A_errors':int(q.expert_pred_A.ne(label).sum()),
                           'B_errors':int(q.expert_pred_B.ne(label).sum())})
    for label in (1,2):
        q=expert[expert.truth.eq(label)]
        byclass[str(label)]={'support':len(q),
             'A_errors':int(q.expert_pred_A.ne(label).sum()),
             'B_errors':int(q.expert_pred_B.ne(label).sum()),
             'repaired':int((q.expert_pred_A.ne(label) & q.expert_pred_B.eq(label)).sum()),
             'regressed':int((q.expert_pred_A.eq(label) & q.expert_pred_B.ne(label)).sum())}
        for field,key in [('A_errors','A'),('B_errors','B')]:
            if byclass[str(label)][field]!=evaluation['raw_expert_metrics'][key][str(label)]['missed']:
                raise ValueError('Independent class recount differs from evaluation.')
    grouped=expert.assign(A_error=expert.expert_pred_A.ne(expert.truth),
                          B_error=expert.expert_pred_B.ne(expert.truth)).groupby(['fold','root','truth']).agg(
                              rows=('row_position','size'),A_errors=('A_error','sum'),B_errors=('B_error','sum')).reset_index()
    grouped['delta_errors']=grouped.B_errors-grouped.A_errors
    roots=grouped.sort_values('delta_errors',ascending=False).head(8).to_dict('records')
    support_totals={}
    for q in support:
        key=(q['truth'],q['support_bin'])
        v=support_totals.setdefault(str(key),{'truth':q['truth'],'support_bin':q['support_bin'],'rows':0,'A_errors':0,'B_errors':0})
        for field in ('rows','A_errors','B_errors'):v[field]+=q[field]
    if sum(v['rows'] for v in support_totals.values())!=len(expert):
        raise ValueError('Support slices omitted some original ASA rows.')
    for label in (1,2):
        q=[v for v in support_totals.values() if v['truth']==label]
        if sum(v['A_errors'] for v in q)!=byclass[str(label)]['A_errors'] or sum(v['B_errors'] for v in q)!=byclass[str(label)]['B_errors']:
            raise ValueError('Support errors do not reconcile to class errors.')
    other_mask=~full.row_position.isin(expert.row_position)
    other_wrong=int((full.loc[other_mask,'teacher_prediction'].to_numpy()!=original[other_mask].to_numpy()).sum())
    if other_wrong!=107:raise ValueError('Frozen non-ASA path changed.')
    if not np.array_equal(full.loc[other_mask,'final_pred_A'],full.loc[other_mask,'final_pred_B']):
        raise ValueError('Non-ASA predictions differ across arms.')
    fit=[]
    for fold in range(3):
        for arm in ('A','B'):
            folder=DEST/f'fold{fold}_{arm}';r=read(folder/'fit.json')
            progress=read(folder/'progress.json')
            assert r['completed_epochs']==25 and r['prediction_epoch']==25
            assert len(progress)==25 and progress[-1]['optimizer_steps_cumulative']==r['optimizer_steps']
            assert all(q['mass_reconstruction_max_error']<=1e-8 for q in progress)
            assert all(q['logical_batches']==plan['expected_primary_schedule'][fold]['logical_batches_per_epoch'] for q in progress)
            fit.append({'fold':fold,'arm':arm,'steps':r['optimizer_steps'],'fit_seconds':r['seconds'],
                        'initial_state_sha256':r['initial_state_sha256'],
                        'fit_S_correct':read(folder/'checkpoints.json')[-1]['fit_by_class']['2']['correct']})
        if fit[-1]['initial_state_sha256']!=fit[-2]['initial_state_sha256']:
            raise ValueError('A/B initialization mismatch.')
    if sum(x['steps'] for x in fit)!=8900 or evaluation['confirmation_allowed']:
        raise ValueError('Six-fit bound or registered stop rule violated.')
    result={'status':'V121_six_fits_verified_failed_quality','latest_actual_training':'V121',
       'classifier_fits_new':6,'optimizer_steps':8900,'confirmation_fits':0,
       'quality_acceptance':False,'model_promoted':False,
       'ASA_by_class':byclass,'ASA_by_fold_class':byfold,'largest_group_error_deltas':roots,
       'support_totals':list(support_totals.values()),'frozen_non_ASA_errors':other_wrong,
       'fits':fit,'primary_quality_gates':evaluation['full_primary_recount']['gates'],
       'paired_A_B_init_identical_all_folds':True,'every_epoch_mass_and_step_check_passed':True,
       'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [
           Path(__file__),PLAN,DEST/'run_seal.json',DEST/'registration.json',
           DEST/'primary_evaluation.json',DEST/'expert_ASA_predictions.parquet',
           DEST/'full_prediction_ledger.parquet',DEST/'support_results.json',
           DEST/'source_group_changes.csv',OFFICIAL]},
       'scope':'Previously inspected development folds. Classification failure is actual; no external test was run.'}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'class_errors':byclass,
                      'confirmation_fits':0},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
