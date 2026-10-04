"""Readonly original-row attribution, scoped retention, and result publication."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc
import numpy as np
import pandas as pd
import torch
from v138_runtime import ROOT,OUT,read,save,sha,require_run_seal
from v138_train import configure,cache_tensors
from v135_runtime import load_data,fit_context
from v135_model import tensor_hash
from v131_evaluate import load_reference


def pair(rows,before,after):
    y=rows.truth.to_numpy();a=rows[before].to_numpy();b=rows[after].to_numpy()
    return {str(cl):{'support':int((y==cl).sum()),'before_correct':int(((y==cl)&(a==y)).sum()),'after_correct':int(((y==cl)&(b==y)).sum()),
        'before_errors':int(((y==cl)&(a!=y)).sum()),'after_errors':int(((y==cl)&(b!=y)).sum()),
        'repairs':int(((y==cl)&(a!=y)&(b==y)).sum()),'new_errors':int(((y==cl)&(a==y)&(b!=y)).sum())} for cl in [0,1,2]}


def main():
    require_run_seal(ROOT/'training/v138_train.py');configure()
    if (OUT/'final_delivery.json').exists():raise FileExistsError('Preserve final result')
    delivery=read(OUT/'delivery.json');quality=read(OUT/'quality.json');watchdog=read(OUT/'probe_watchdog_receipt.json')
    _,d=load_data();ref=load_reference(d);rows=pd.read_parquet(OUT/'ASA_prediction_ledger.parquet')
    if not np.array_equal(rows.row_position,d.row_position) or not np.array_equal(rows.truth,d.truth):raise ValueError('Independent ASA identity mismatch')
    full=pd.read_parquet(OUT/'full_prediction_ledger.parquet')
    if not np.array_equal(full.row_position,ref.row_position):raise ValueError('Full population changed')
    full=full.merge(ref[['row_position','truth']],on='row_position',validate='one_to_one')
    originalpairs={'ASA_vs_A0':pair(rows,'pred_A0','pred_H_L'),'ASA_vs_V135_R_decay':pair(rows,'pred_R_decay','pred_H_L'),
        'ASA_vs_matched_H_A':pair(rows,'pred_H_A','pred_H_L'),'full_vs_A0':pair(full,'pred_A0','pred_H_L')}
    train=[];replay=[];roles=[]
    for f in range(3):
        frame,c,pure,_,ids=fit_context(d,f);h,facts,model=cache_tensors(f)
        for arm in ['H_A','H_L']:
            folder=OUT/f'fold{f}_{arm}';st=torch.load(folder/'endpoint_readout.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(st['readout']);loss=0.
            with torch.no_grad():
                for start in range(0,len(ids),2048):
                    ii=ids[start:start+2048];lp=torch.log_softmax(model(h[ii],facts[ii]),-1).mean(1)
                    loss-=float((lp*torch.as_tensor(c[ii],device='cuda',dtype=torch.float64)).sum())/len(frame)
            logs=[__import__('json').loads(l) for l in (folder/'closures.jsonl').read_text(encoding='utf-8').splitlines()]
            r=read(folder/'fit.json');rr=pd.read_parquet(folder/'endpoint_original_rows.parquet');rr['arm']=arm;roles.append(rr)
            train.append({'fold':f,'arm':arm,'endpoint_original_member_CE_actual_no_grad':loss,'initial_original_member_CE':logs[0]['original_member_CE'],
                'initial_gradient_norm':logs[0]['gradient_norm'],'last_evaluated_gradient_norm':logs[-1]['gradient_norm'],
                'last_evaluation_is_endpoint':logs[-1]['parameter_sha256']==tensor_hash(st['readout']),
                'last_trial_loss_not_claimed_endpoint':True,'fit_seconds':r['seconds'],'accepted_updates':r['accepted_updates'],'full_gradient_evaluations':r['full_gradient_evaluations']})
        del h,facts,model;gc.collect();torch.cuda.empty_cache()
    allroles=pd.concat(roles);allroles.to_parquet(OUT/'all_training_role_ledger.parquet',index=False)
    candidate=allroles[allroles.arm.eq('H_L')].copy();pure=candidate[candidate.pure_TRAIN_input];bad=pure[pure.pred.ne(pure.truth)];repaired=pure[pure.pred_start.ne(pure.truth)&pure.pred.eq(pure.truth)]
    bad.to_parquet(OUT/'remaining_pure_training_errors.parquet',index=False)
    repaired.to_parquet(OUT/'repaired_pure_training_errors.parquet',index=False)
    training_attribution={'before_pure_role_errors':int(pure.pred_start.ne(pure.truth).sum()),'after_pure_role_errors':len(bad),
        'remaining_independent_official_rows':int(bad.row_position.nunique()),'remaining_canonical_inputs':int(bad.canonical_key.nunique()),
        'repaired_role_rows':len(repaired),'new_wrong_pure_role_rows':int((pure.pred_start.eq(pure.truth)&pure.pred.ne(pure.truth)).sum()),
        'paired_classes':pair(pure,'pred_start','pred'),'loss_and_actual_closure_audit':train}
    probes=[]
    for f in [0,2]:
        p=read(OUT/f'probe_fold{f}/receipt.json');probes.append(p)
    # Only a scoped capability is frozen; the issue as a whole remains open.
    scoped=[]
    for fold in range(3):
        info=delivery['learning']['learning']['H_L'][fold]
        if info['stable_window_mastered'] and info['original_correct_pure_guard']['repair_protection_passed']:
            guard=candidate[candidate.training_role.eq(fold)&candidate.pure_TRAIN_input][['row_position','training_role','truth','pred']]
            path=OUT/f'fold{fold}_verified_TRAIN_guard.parquet';guard.to_parquet(path,index=False)
            scoped.append({'id':f'TRAIN-PURE-READOUT-fold{fold}','scope':'This registered TRAIN role and frozen input; not held/generalization or whole issue success',
                'pure_role_original_rows':len(guard),'guard':path.relative_to(ROOT).as_posix(),'guard_sha256':sha(path),'new_errors_allowed':0,
                'checkpoint':f'artifacts/v138_single_issue_round1_20260930/fold{fold}_H_L/endpoint_readout.pt','checkpoint_sha256':sha(OUT/f'fold{fold}_H_L/endpoint_readout.pt'),
                'legal_TRAIN_guard_only':True,'all_original_mixed_rows_still_scored':True,'global_issue_closed':False,'deployable':False})
    save(OUT/'scoped_training_capabilities.json',{'global_issue':'TRAIN-PURE-READOUT','global_issue_closed':False,'verified_training_scopes':scoped,
        'next_trial_must_replay_same_original_rows':True,'current_reference_not_replaced':True})
    # A pure-only/incomplete witness does not satisfy the full mixed-population gate.
    decision={'issue':'TRAIN-PURE-READOUT','round1_completed':True,'issue_solved':False,'candidate_promoted':False,
        'round2_eligible_now':False,'pure_feasibility_certificates':{str(p['fold']):p['certificate'] for p in probes},
        'whole_TRAIN_majority_feasibility_verified':False,'round2_executed':False,
        'reason':'No complete full TRAIN feasible-margin certificate or bound round2 runtime. No automatic expansion.',
        'no_more_unconstrained_same_solver_fits':True,'max_issue_rounds':2,'rounds_consumed':1,
        'cross_source_is_next_separate_issue_after_training_classification_closed':True}
    save(OUT/'issue_decision.json',decision)
    save(OUT/'attribution.json',{'training':training_attribution,'original_row_pairs':originalpairs,'probe_receipts':probes,
        'limitations':['TRAIN role counts repeat some original records across folds; independent count reported separately.',
            'Six preflight full gradients are diagnostics, outside primary optimization; 1200 primary closures logged.',
            'Margin probe may find extreme unstable coefficients; any partial LP status is not a complete population certificate.',
            'Previously inspected development folds; no new blind data, official submission or external environment.']})
    save(OUT/'final_delivery.json',{**delivery,'probe_receipts_pending':False,'probe_receipts':probes,'decision':decision,'training_attribution':training_attribution,
        'original_row_pairs':originalpairs,'scoped_training_capabilities':scoped,'primary_fit_seconds_total':sum(z['fit_seconds'] for z in train),
        'supervised_probe_folds':watchdog['supervised_diagnostic_folds'],'supervised_lp_calls':sum(p.get('solver_runs',0) for p in probes),
        'closure_counts_not_optimizer_updates':True,'optimizer_steps':delivery['accepted_updates'],
        'validation_scope':'Previously inspected development source folds; all 2,056,871 official rows scored. TRAIN-only scope is not blind generalization, official submission, or production acceptance.',
        'source_sha256':sha(__file__),
        'attribution_sha256':sha(OUT/'attribution.json'),'decision_sha256':sha(OUT/'issue_decision.json')})
    print({'TRAIN_before':80,'TRAIN_after':len(bad),'independent_remaining_rows':int(bad.row_position.nunique()),'new_wrong_pure':0,
        'ASA_M_errors':delivery['ASA_errors']['H_L']['1'],'ASA_S_errors':delivery['ASA_errors']['H_L']['2'],
        'quality_passed':False,'round2_eligible':False,'scoped_capabilities':len(scoped)},flush=True)


if __name__=='__main__':main()
