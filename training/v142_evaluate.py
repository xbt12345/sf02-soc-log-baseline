"""Actual second-layer state replays, immutable guards and complete scoring."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc
import numpy as np
import pandas as pd
import torch
from v142_runtime import ROOT,OUT,BASE,PROBE,read,save,sha,require_run_seal,endpoint
from v142_train import tensors
from v138_train import configure,historical,quick_stats
from v135_model import Classifier,batch,tensor_hash
from v135_runtime import load_data,fit_context
from v140_retention_check import check as retained_check
from v138_readout import probabilities
from v131_evaluate import load_reference,full_folds,teacher_predictions
from v135_evaluate import quality_for
from v138_closeout import pair
from experiment_review import evaluate_primary
from v142_infer import infer_numeric_input


def main():
    require_run_seal(ROOT/'training/v142_train.py');configure()
    if (OUT/'final_delivery.json').exists():raise FileExistsError('Preserve completed result')
    x,d=load_data();folds=[];predictions={};allrows=[];windows=[[] for _ in range(5)];audits=[]
    for f in range(3):
        folder=OUT/f'fold{f}_S2';r=read(folder/'fit.json');endpoint(r)
        for name,key in [('endpoint.pt','model_sha256'),('sealed_all_prob.npy','probability_sha256'),('endpoint_original_rows.parquet','rows_sha256'),('closures.jsonl','closures_sha256'),('progress.json','progress_sha256')]:
            if sha(folder/name)!=r[key]:raise ValueError('Fit evidence changed '+name)
        # Recompute the actual frozen feature function from registered x.
        body=Classifier(128).to('cuda');body.load_state_dict(torch.load(historical(f)/'epoch100_model.pt',map_location='cpu',weights_only=True)['model']);body.eval();h1s=[];h2s=[]
        with torch.no_grad():
            for start in range(0,x.shape[0],256):
                _,h1,h2=batch(body,x,np.arange(start,min(start+256,x.shape[0])),True);h1s.append(h1.cpu().numpy());h2s.append(h2.cpu().numpy())
        actualh1=np.concatenate(h1s);actualh2=np.concatenate(h2s)
        if not np.array_equal(actualh1,np.load(PROBE/f'fold{f}_h1.npy')):raise ValueError('First-layer actual cache changed')
        from v138_runtime import OUT as CACHE
        if not np.array_equal(actualh2,np.load(CACHE/f'fold{f}_hidden.npy')):raise ValueError('Reference second-layer cache changed')
        del body,actualh1,actualh2;gc.collect();torch.cuda.empty_cache()
        frame,c,pure,_,ids=fit_context(d,f);h,facts,m=tensors(f);initial={k:v.clone() for k,v in m.state_dict().items()}
        old=np.load(OUT/f'fold{f}_zero_probability.npy');oldpred=old[frame.local].argmax(1)
        if not np.array_equal(probabilities(m,h,facts,np.arange(len(c))),old):raise ValueError('Actual zero-control changed')
        log=[__import__('json').loads(z) for z in (folder/'closures.jsonl').read_text(encoding='utf-8').splitlines()]
        if [z['full_gradient_evaluation'] for z in log]!=list(range(1,r['full_gradient_evaluations']+1)):raise ValueError('Gradient count mismatch')
        if any(z['original_class_mass_seen']!=c.sum(0).tolist() or not z['finite'] for z in log):raise ValueError('Original mass or numeric status mismatch')
        history=read(folder/'progress.json')
        if len(history)!=r['accepted_updates'] or [z['accepted_update'] for z in history]!=list(range(1,len(history)+1)):raise ValueError('Update journal changed')
        state_reports=[]
        for k,item in enumerate(history[-5:]):
            st=torch.load(folder/f"accepted{item['accepted_update']}.pt",map_location='cpu',weights_only=True)
            if tensor_hash(st['state'])!=item['parameter_sha256']:raise ValueError('State hash changed')
            m.load_state_dict(st['state']);q=probabilities(m,h,facts,np.arange(len(c)));actual=quick_stats(frame,q,pure,oldpred,[22,6,28][f])
            if actual!=item['stats']:raise ValueError('State classification mismatch')
            rr=frame[['row_position','truth']].copy();rr['training_role']=f;rr['pred']=q[frame.local].argmax(1);windows[k].append(rr)
            state_reports.append({'accepted_update':item['accepted_update'],'stats':actual,'parameter_sha256':item['parameter_sha256']})
        st=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)
        if st['seal_sha256']!=sha(OUT/'run_seal.json'):raise ValueError('Model not registered')
        m.load_state_dict(st['state']);q=probabilities(m,h,facts,np.arange(len(c)));saved=np.load(folder/'sealed_all_prob.npy')
        if not np.array_equal(q,saved):raise ValueError('Actual model replay failed')
        original=torch.load(historical(f)/'epoch100_model.pt',map_location='cpu',weights_only=True)['model']
        readout=torch.load(BASE/f'fold{f}_C/endpoint_readout.pt',map_location='cpu',weights_only=True)['readout']
        fresh=infer_numeric_input(x,original,readout,st['state'])
        fresh_gap=float(np.abs(fresh-q).max())
        if fresh_gap>2e-6 or not np.array_equal(fresh.argmax(1),q.argmax(1)):raise ValueError('Cache-free registered input inference differs')
        for key in initial:
            if key.startswith(('reference_','head_')) and not torch.equal(initial[key],m.state_dict()[key]):raise ValueError('Frozen state altered')
        rows=pd.read_parquet(folder/'endpoint_original_rows.parquet')
        if not np.array_equal(rows.row_position,frame.row_position) or not np.array_equal(rows.truth,frame.truth) or not np.array_equal(rows.pred,q[frame.local].argmax(1)):raise ValueError('Independent original-row predictions mismatch')
        actual=quick_stats(frame,q,pure,oldpred,[22,6,28][f])
        if actual!=r['endpoint_stats']:raise ValueError('Endpoint stats mismatch')
        distinct=len(state_reports)==5 and len({z['parameter_sha256'] for z in state_reports})==5
        mastered=distinct and all(z['stats']['mastered'] and z['stats']['old_correct_pure_regressions']==0 for z in state_reports)
        if mastered!=r['last_five_distinct_accepted_states_mastered']:raise ValueError('False window mastery')
        allrows.append(rows);predictions[f]=q;folds.append({'fold':f,'endpoint':actual,'stable_window_mastered':mastered,'last_states':state_reports,
                                                        'full_gradient_evaluations':r['full_gradient_evaluations'],'accepted_updates':r['accepted_updates'],'parameter_delta_L2':r['parameter_delta_L2'],'termination':r['termination']})
        audits.append({'fold':f,'exact_zero_and_endpoint_replay':True,'actual_first_and_reference_second_replay':True,'frozen_head_reference_exact':True,
                       'actual_state_replays':len(state_reports),'cache_free_numeric_input_replay_count':len(q),'cache_free_probability_max_gap':fresh_gap,
                       'raw_new_message_preprocessor_external_validation':False})
        del h,facts,m;gc.collect();torch.cuda.empty_cache()
    roles=pd.concat(allrows);roles.to_parquet(OUT/'all_training_role_ledger.parquet',index=False)
    retained=retained_check(roles);window_ret=[retained_check(pd.concat(w)) for w in windows if len(w)==3]
    pure=roles[roles.pure_TRAIN_input];bad=pure[pure.pred.ne(pure.truth)];new=pure[pure.pred_start.eq(pure.truth)&pure.pred.ne(pure.truth)];rep=pure[pure.pred_start.ne(pure.truth)&pure.pred.eq(pure.truth)]
    solved=all(z['stable_window_mastered'] for z in folds) and retained['passed'] and len(window_ret)==5 and all(z['passed'] for z in window_ret) and len(bad)==len(new)==0
    learning={'status':'TRAIN_verified_before_HELD_scoring','issue_solved':bool(solved),'folds':folds,'retention':retained,'last_five_retention':window_ret,
              'before_pure_errors':int(pure.pred_start.ne(pure.truth).sum()),'after_pure_errors':len(bad),'repairs':len(rep),'new_errors':len(new),
              'independent_remaining_rows':int(bad.row_position.nunique()),'remaining_inputs':int(bad.canonical_key.nunique()),'TRAIN_pairs':pair(pure,'pred_start','pred')}
    save(OUT/'learning_qualification.json',learning)
    for name,rr in [('remaining_pure_training_errors',bad),('repaired_pure_training_errors',rep),('new_pure_training_errors',new)]:rr.to_parquet(OUT/(name+'.parquet'),index=False)
    ref=load_reference(d);teacher=teacher_predictions(len(ref),full_folds());basefull=pd.read_parquet(BASE/'full_prediction_ledger.parquet')
    if not np.array_equal(basefull.row_position,ref.row_position):raise ValueError('Full row identities changed')
    baseline=basefull.pred_C.to_numpy();A0=basefull.pred_A0.to_numpy();pred=teacher.copy();asa_pred=np.empty(len(d),np.int8)
    for f,q in predictions.items():
        mask=d.fold.eq(f).to_numpy();asa_pred[mask]=q[d.loc[mask,'local']].argmax(1)
    pred[d.row_position]=asa_pred
    ladder=pd.read_parquet(ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet');headers=set(pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet').row_position)
    task=quality_for(ref,d,pred,A0,ladder,headers)
    profile={'expected_full_rows':2056871,'asa_error_limits':{'M':318,'S':2074,'total':2170},'minimum_improved_folds':2,'protected_S_roots':[21702,20849,29],'required_full_classes':[0,1,2]}
    matched=evaluate_primary(ref,pd.DataFrame({'row_position':ref.row_position,'pred_A':baseline,'pred_B':pred}),profile)
    a,b=matched['ASA']['A'],matched['ASA']['B'];matched_gates={'M_protected':b['1']['missed']<=a['1']['missed'],'S_protected':b['2']['missed']<=a['2']['missed'],
                                                         'one_class_improves':any(b[str(c)]['missed']<a[str(c)]['missed'] for c in [1,2]),
                                                         'two_folds_improve':sum(z['B_errors']<z['A_errors'] for z in matched['folds'])>=2}
    quality_passed=bool(solved and task['quality_passed'] and all(matched_gates.values()))
    full=pd.DataFrame({'row_position':ref.row_position,'pred_A0':A0,'pred_C':baseline,'pred_S2':pred});full.to_parquet(OUT/'full_prediction_ledger.parquet',index=False)
    asa=d[['row_position','local','root','fold','truth','canonical_key']].copy();asa['pred_A0']=A0[d.row_position];asa['pred_C']=baseline[d.row_position];asa['pred_S2']=asa_pred
    asa.to_parquet(OUT/'ASA_prediction_ledger.parquet',index=False)
    asa.assign(wrong=asa.pred_S2!=asa.truth).groupby(['root','truth']).wrong.agg(['size','sum']).reset_index().to_csv(OUT/'source_errors.csv',index=False)
    pairs={'ASA_vs_C':pair(asa,'pred_C','pred_S2'),'ASA_vs_A0':pair(asa,'pred_A0','pred_S2'),'full_vs_A0':pair(full.assign(truth=ref.truth.to_numpy()),'pred_A0','pred_S2')}
    save(OUT/'quality.json',{'task':task,'matched_frozen_control':matched,'matched_gates':matched_gates,'candidate_quality_passed':quality_passed,'model_promoted':False})
    save(OUT/'verification.json',{'status':'actual_models_and_independent_complete_official_rows_replayed','audits':audits,'official_original_rows':len(ref),'ASA_rows':len(d),
                                 'fits':3,'full_gradient_evaluations':sum(z['full_gradient_evaluations'] for z in folds),'accepted_updates':sum(z['accepted_updates'] for z in folds),
                                 'scope':'Previously inspected source development folds; not blind, external acceptance or official submission.'})
    save(OUT/'scoped_training_capabilities.json',{'global_issue_closed':bool(solved),'scopes':[],'deployable':False})
    delivery={'status':'new_second_scope_completed_TRAIN_mastered_not_promoted' if solved else 'new_second_scope_completed_issue_not_solved','latest_actual':'V142',
              'issue':'TRAIN-PURE-READOUT','issue_solved':bool(solved),'classifier_fits':3,'full_gradient_evaluations':sum(z['full_gradient_evaluations'] for z in folds),
              'accepted_updates':sum(z['accepted_updates'] for z in folds),'quality_acceptance':quality_passed,'model_promoted':False,'TRAIN':learning,
              'ASA_errors':{'M':task['ASA']['B']['1']['missed'],'S':task['ASA']['B']['2']['missed']},'original_row_pairs':pairs,
              'old_LP_chain_fits':12,'old_LP_additional_fits':0,'new_second_scope_fits_max':3,'automatic_extra_fits':False,
              'validation_scope':read(OUT/'verification.json')['scope'],'plan_sha256':sha(ROOT/'training/review_policy/v142_second_layer_plan.json'),
              'seal_sha256':sha(OUT/'run_seal.json'),'quality_sha256':sha(OUT/'quality.json'),'verification_sha256':sha(OUT/'verification.json')}
    save(OUT/'final_delivery.json',delivery);print({'status':delivery['status'],'TRAIN_errors':len(bad),'TRAIN_new_errors':len(new),'ASA_errors':delivery['ASA_errors'],'quality_passed':quality_passed},flush=True)


if __name__=='__main__':main()
