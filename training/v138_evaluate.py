"""Actual cache/model replays and independent full official-row scoring."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc
import numpy as np
import pandas as pd
import torch
from v138_runtime import ROOT,OUT,read,save,sha,require_run_seal,endpoint
from v138_train import configure,cache_backbone,cache_tensors,quick_stats,historical
from v138_readout import probabilities
from v135_model import tensor_hash
from v135_runtime import load_data,fit_context
from v135_evaluate import quality_for
from v131_evaluate import load_reference,full_folds,teacher_predictions
from experiment_review import evaluate_primary
from v137_issue_guard import protect_original_rows


def main():
    require_run_seal(ROOT/'training/v138_train.py');configure()
    if (OUT/'delivery.json').exists():raise FileExistsError('Do not overwrite final evidence')
    x,d=load_data();predictions={a:{} for a in ['H_A','H_L']};learning={a:[] for a in predictions};audits=[]
    for f in range(3):
        frame,c,pure,_,ids=fit_context(d,f);oldq=np.load(OUT/f'fold{f}_zero_probability.npy');oldpred=oldq[frame.local].argmax(1)
        state,h,backboneq=cache_backbone(x,f)
        if not np.array_equal(h,np.load(OUT/f'fold{f}_hidden.npy')):raise ValueError('Actual frozen backbone cache changed')
        del state,h,backboneq;gc.collect();torch.cuda.empty_cache()
        hidden,facts,model=cache_tensors(f)
        for arm in ['H_A','H_L']:
            folder=OUT/f'fold{f}_{arm}';r=read(folder/'fit.json');endpoint(r)
            for name,key in [('endpoint_readout.pt','model_sha256'),('sealed_all_prob.npy','probability_sha256'),('closures.jsonl','closures_sha256'),('progress.json','progress_sha256'),('endpoint_original_rows.parquet','rows_sha256')]:
                if sha(folder/name)!=r[key]:raise ValueError('Fit journal changed '+name)
            logs=[__import__('json').loads(l) for l in (folder/'closures.jsonl').read_text(encoding='utf-8').splitlines()]
            if [v['full_gradient_evaluation'] for v in logs]!=list(range(1,r['full_gradient_evaluations']+1)):raise ValueError('Gradient count changed')
            if any(v['original_class_mass_seen']!=c.sum(0).tolist() or not v['finite'] for v in logs):raise ValueError('Exposure or numerical risk changed')
            history=read(folder/'progress.json')
            if len(history)!=r['accepted_updates'] or [v['accepted_update'] for v in history]!=list(range(1,len(history)+1)):raise ValueError('Accepted state journal changed')
            window=[]
            for item in history[-5:]:
                st=torch.load(folder/f"accepted{item['accepted_update']}_readout.pt",map_location='cpu',weights_only=True)
                if tensor_hash(st['readout'])!=item['parameter_sha256']:raise ValueError('Last states changed')
                model.load_state_dict(st['readout']);q=probabilities(model,hidden,facts,np.arange(len(c)))
                actual=quick_stats(frame,q,pure,oldpred,[22,6,28][f])
                if actual!=item['stats']:raise ValueError('Actual state vs reported TRAIN classification mismatch')
                window.append({'accepted_update':item['accepted_update'],'stats':actual,'parameter_sha256':item['parameter_sha256']})
            distinct=len(window)==5 and len({v['parameter_sha256'] for v in window})==5
            mastered=distinct and all(z['stats']['mastered'] for z in window)
            if mastered!=r['last_five_distinct_accepted_states_mastered']:raise ValueError('False stable learning')
            st=torch.load(folder/'endpoint_readout.pt',map_location='cpu',weights_only=True)
            if st['seal_sha256']!=sha(OUT/'run_seal.json') or st['source_model_sha256']!=sha(historical(f)/'epoch100_model.pt'):raise ValueError('Endpoint binding mismatch')
            model.load_state_dict(st['readout']);q=probabilities(model,hidden,facts,np.arange(len(c)));saved=np.load(folder/'sealed_all_prob.npy')
            if not np.array_equal(q,saved):raise ValueError('Actual endpoint probability replay failed')
            actual=quick_stats(frame,q,pure,oldpred,[22,6,28][f])
            if actual!=r['endpoint_stats']:raise ValueError('Endpoint TRAIN stats changed')
            ledger=pd.read_parquet(folder/'endpoint_original_rows.parquet')
            if not np.array_equal(ledger.row_position,frame.row_position) or not np.array_equal(ledger.truth,frame.truth) or not np.array_equal(ledger.pred,q[frame.local].argmax(1)):raise ValueError('Original-row ledger mismatch')
            guard=protect_original_rows(frame[['row_position','truth']],pd.DataFrame({'row_position':frame.row_position,'pred':oldpred}),ledger[['row_position','pred']],
                frame.loc[pure[frame.local].astype(bool)&(oldpred==frame.truth),['row_position']])
            if guard!=r['all_old_correct_pure_input_guard']:raise ValueError('Original-correct protection mismatch')
            predictions[arm][f]=q;learning[arm].append({'fold':f,'endpoint':actual,'stable_window_mastered':mastered,'accepted_states':window,
                'original_correct_pure_guard':guard,'full_gradient_evaluations':r['full_gradient_evaluations'],'accepted_updates':r['accepted_updates'],'termination':r['termination']})
            audits.append({'fold':f,'arm':arm,'all_cached_input_probabilities_replayed':len(q),'original_train_rows':len(frame),'last_actual_states':len(window),'exact_probability_replay':True})
        del hidden,facts,model;gc.collect();torch.cuda.empty_cache()
    qualify={'status':'TRAIN_review_frozen_before_new_HELD_truth_join','candidate':'H_L','learning':learning,
        'all_roles_mastered':{a:all(z['stable_window_mastered'] and z['original_correct_pure_guard']['repair_protection_passed'] for z in v) for a,v in learning.items()},
        'no_new_HELD_join_before_this_receipt':True,'issue_round':1}
    save(OUT/'learning_qualification.json',qualify)
    ref=load_reference(d);teacher=teacher_predictions(len(ref),full_folds())
    oldledger=pd.read_parquet(ROOT/'artifacts/v135_stable_learning_trial_20260930/full_prediction_ledger.parquet')
    if not np.array_equal(oldledger.row_position,ref.row_position):raise ValueError('Old reference rows changed')
    base=oldledger.pred_A0.to_numpy();ladder=pd.read_parquet(ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet')
    if not np.array_equal(ladder.row_position,d.row_position):raise ValueError('Behavior support ledger misaligned')
    headers=set(pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet').row_position)
    ledger=pd.DataFrame({'row_position':ref.row_position,'pred_A0':base,'pred_R_decay':oldledger.pred_R_decay})
    asa=d[['row_position','local','root','fold','truth','canonical_key']].copy();asa['pred_A0']=base[d.row_position];asa['pred_R_decay']=oldledger.pred_R_decay.to_numpy()[d.row_position]
    quality={};sources=[];prediction={};retention={}
    for arm in ['H_A','H_L']:
        pred=teacher.copy();v=np.empty(len(d),np.int8)
        for f,q in predictions[arm].items():
            mask=d.fold.eq(f).to_numpy();v[mask]=q[d.loc[mask,'local']].argmax(1)
        pred[d.row_position]=v;prediction[arm]=pred;ledger['pred_'+arm]=pred;asa['pred_'+arm]=v
        quality[arm]=quality_for(ref,d,pred,base,ladder,headers)
        retention[arm]=protect_original_rows(ref[['row_position','truth']],pd.DataFrame({'row_position':ref.row_position,'pred':base}),
            pd.DataFrame({'row_position':ref.row_position,'pred':pred}),pd.DataFrame({'row_position':np.array([],dtype=np.int64)}))
        src=d.assign(wrong=v!=d.truth).groupby(['root','truth']).wrong.agg(['size','sum']).reset_index();src['arm']=arm;sources.append(src)
    matched=evaluate_primary(ref,pd.DataFrame({'row_position':ref.row_position,'pred_A':prediction['H_A'],'pred_B':prediction['H_L']}),
        {'expected_full_rows':2056871,'asa_error_limits':{'M':318,'S':2074,'total':2170},'minimum_improved_folds':2,'protected_S_roots':[21702,20849,29],'required_full_classes':[0,1,2]})
    a,b=matched['ASA']['A'],matched['ASA']['B']
    gates={'M_errors_do_not_increase':b['1']['missed']<=a['1']['missed'],'S_errors_do_not_increase':b['2']['missed']<=a['2']['missed'],
        'at_least_one_class_strictly_improves':any(b[str(c)]['missed']<a[str(c)]['missed'] for c in (1,2)),
        'at_least_two_folds_improve':sum(z['B_errors']<z['A_errors'] for z in matched['folds'])>=2}
    passed=qualify['all_roles_mastered']['H_L'] and quality['H_L']['quality_passed'] and all(gates.values())
    save(OUT/'quality.json',{'candidate':'H_L','candidate_quality_passed':bool(passed),'arms':quality,'matched_control':matched,'matched_control_gates':gates,'historical_correct_regression_reports':retention})
    ledger.to_parquet(OUT/'full_prediction_ledger.parquet',index=False);asa.to_parquet(OUT/'ASA_prediction_ledger.parquet',index=False)
    pd.concat(sources).to_csv(OUT/'source_errors.csv',index=False)
    save(OUT/'verification.json',{'status':'actual_models_and_independent_official_original_rows_replayed','replays':audits,'official_original_rows':len(ref),'ASA_rows':len(d),
        'new_primary_fits':6,'new_primary_full_gradient_evaluations':sum(z['full_gradient_evaluations'] for v in learning.values() for z in v),
        'new_primary_accepted_updates':sum(z['accepted_updates'] for v in learning.values() for z in v),'preflight_diagnostic_full_gradients':6,
        'scope':'Previously inspected development folds, not blind, official submission, or external environment.'})
    delivery={'status':'round1_completed_quality_passed_not_promoted' if passed else 'round1_completed_not_task_accepted','execution_version':'V138-exec-V137-round1',
        'issue':'TRAIN-PURE-READOUT','round':1,'candidate':'H_L','classifier_fits':6,'full_gradient_evaluations':sum(z['full_gradient_evaluations'] for v in learning.values() for z in v),
        'accepted_updates':sum(z['accepted_updates'] for v in learning.values() for z in v),'quality_acceptance':bool(passed),'model_promoted':False,
        'learning':qualify,'ASA_errors':{a:{cl:q['ASA']['B'][cl]['missed'] for cl in ['1','2']} for a,q in quality.items()},
        'probe_receipts_pending':True,'automatic_second_round':False,'plan_sha256':sha(ROOT/'training/review_policy/v137_single_issue_plan.json'),
        'seal_sha256':sha(OUT/'run_seal.json'),'verification_sha256':sha(OUT/'verification.json'),'quality_sha256':sha(OUT/'quality.json')}
    save(OUT/'delivery.json',delivery);print(delivery,flush=True)


if __name__=='__main__':main()
