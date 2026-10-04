"""Actual six endpoints/windows, frozen labels, and complete original-row quality."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from v158_fusion_runtime import ROOT,OUT,BANK,read,sha,save,require,endpoint
from v158_fusion_train import arrays,model_for,Counter,probabilities,risk,rows
from v138_train import configure,quick_stats
from v135_model import tensor_hash
from v142_retention_check import check as retention
from v131_common import load_data
from v131_evaluate import load_reference
from v135_evaluate import quality_for
from v138_closeout import pair
from experiment_review import evaluate_primary,check_bindings

def journal(path):return [json.loads(z) for z in path.read_text(encoding='utf-8').splitlines()]

def main():
    p,c=require(__file__);configure();assert not (OUT/'final_delivery.json').exists()
    dependencies=[]
    for f in range(3):
        for arm in ['A','B']:
            folder=OUT/f'fold{f}_{arm}';r=read(folder/'fit.json');endpoint(r)
            dependencies.extend(folder/name for name in ['fit.json','endpoint.pt','endpoint_deployment_probability.npy','endpoint_OOF_probability.npy',
                'endpoint_original_FIT_rows.parquet','endpoint_original_OOF_rows.parquet','gradients.jsonl','proposals.jsonl','classifier_calls.jsonl'])
            if (folder/'progress.json').exists():dependencies.append(folder/'progress.json')
            for item in r['last5']:dependencies.extend(folder/name for name in [f"accepted{item['update']}.pt",f"accepted{item['update']}_deployment_FIT_rows.parquet",f"accepted{item['update']}_OOF_rows.parquet"])
    eval_bindings={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies}
    assert not (OUT/'evaluation_input_bindings.json').exists()
    save(OUT/'evaluation_input_bindings.json',dict(status='all_six_actual_fit_models_and_outputs_bound_before_any_evaluation_forward',source_sha256=eval_bindings,new_gradients=0))
    _,d=load_data();predictions={a:{} for a in ['A','B']};all_rows={a:[] for a in ['A','B']};windows={a:[[] for _ in range(5)] for a in ['A','B']};audits=[];receipts=[];total_calls=0
    for f in range(3):
        frame,ids,count,pure,oof,deploy,cond,prior,mass,old,protected=arrays(f)
        for arm in ['A','B']:
            folder=OUT/f'fold{f}_{arm}';r=read(folder/'fit.json');endpoint(r);receipts.append(r)
            assert r['model_sha256']==sha(folder/'endpoint.pt') and r['deployment_probability_sha256']==sha(folder/'endpoint_deployment_probability.npy') and r['OOF_probability_sha256']==sha(folder/'endpoint_OOF_probability.npy')
            gradients=journal(folder/'gradients.jsonl');completed=[z for z in gradients if z['event']=='completed'];assert len(completed)==sum(z['event']=='attempt' for z in gradients)==r['full_gradients']
            assert all(z['original_class_mass']==count.sum(0).tolist() for z in completed)
            proposals=journal(folder/'proposals.jsonl');accepted=[z for z in proposals if z['accepted']];history=read(folder/'progress.json') if (folder/'progress.json').exists() else []
            assert len(proposals)==r['proposal_evaluations'] and len(accepted)==len(history)==r['accepted_updates']
            assert all(z['classification_guard'] and z['trial_CE']<=z['Armijo_bound'] for z in accepted)
            assert [z['parameter_sha256'] for z in accepted]==[z['parameter_sha256'] for z in history]
            training_calls=journal(folder/'classifier_calls.jsonl');assert sum(z['event']=='attempt' for z in training_calls)==sum(z['event']=='completed' for z in training_calls)==r['classifier_forward_calls']
            model=model_for(arm);counter=Counter(model,c['role_call_budgets'][f]['evaluation_classifier_forward_cap_per_arm'],folder/'evaluation_classifier_calls.jsonl');states=[]
            for k,item in enumerate(r['last5']):
                state=torch.load(folder/f"accepted{item['update']}.pt",map_location='cpu',weights_only=True)['state'];assert tensor_hash(state)==item['parameter_sha256']
                model.load_state_dict(state);q=probabilities(model,deploy,cond,prior,np.arange(22546));stats=quick_stats(frame,q,pure,old,[22,6,28][f])
                assert stats==item['stats'] and stats['mastered'] and stats['new_errors_vs_start']==0
                actual=rows(frame,q,f);stored=pd.read_parquet(folder/f"accepted{item['update']}_deployment_FIT_rows.parquet")
                assert np.array_equal(actual.row_position,stored.row_position) and np.array_equal(actual.truth,stored.truth) and np.array_equal(actual.pred,stored.pred)
                gap=float(np.abs(actual[['p0','p1','p2']].to_numpy()-stored[['p0','p1','p2']].to_numpy()).max());assert gap<=2e-12
                windows[arm][k].append(actual);states.append(dict(update=item['update'],stats=stats,probability_gap=gap,parameter_sha256=item['parameter_sha256']))
            terminal=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True);assert (terminal['fold'],terminal['arm'],terminal['seal_sha256'])==(f,arm,sha(OUT/'run_seal.json'))
            model.load_state_dict(terminal['state']);assert tensor_hash(model.state_dict())==r['endpoint_parameter_sha256']
            q=probabilities(model,deploy,cond,prior,np.arange(22546));assert np.array_equal(q,np.load(folder/'endpoint_deployment_probability.npy'))
            value,qo,seen=risk(model,oof,cond,prior,mass,ids);assert abs(value-r['final_OOF_CE'])<=1e-12 and np.array_equal(qo[ids],np.load(folder/'endpoint_OOF_probability.npy')[ids])
            actual=rows(frame,q,f);stored=pd.read_parquet(folder/'endpoint_original_FIT_rows.parquet')
            assert np.array_equal(actual.row_position,stored.row_position) and np.array_equal(actual.truth,stored.truth) and np.array_equal(actual.pred,stored.pred)
            assert np.all(q[frame.local].argmax(1)[protected]==frame.truth.to_numpy()[protected])
            all_rows[arm].append(actual);predictions[arm][f]=q
            audits.append(dict(fold=f,arm=arm,window_states=states,stable_last5=len(states)==5 and len({z['parameter_sha256'] for z in states})==5,
                endpoint_FIT_stats=quick_stats(frame,q,pure,old,[22,6,28][f]),actual_OOF_CE=value,actual_evaluation_classifier_calls=counter.attempted,
                frozen_bank_sources_preserved=True,no_gradient_in_evaluation=True))
            total_calls+=counter.attempted;counter.close();del model;gc.collect();torch.cuda.empty_cache()
    learning={}
    for arm in ['A','B']:
        ledger=pd.concat(all_rows[arm],ignore_index=True);ledger.to_parquet(OUT/f'{arm}_all_training_role_ledger.parquet',index=False)
        joint=retention(ledger);wg=[retention(pd.concat(w,ignore_index=True)) for w in windows[arm] if len(w)==3]
        learning[arm]=dict(joint_TRAIN_retention=joint,window_TRAIN_retention=wg,
            all_roles_mastered=joint['passed'] and len(wg)==5 and all(z['passed'] for z in wg) and all(z['stable_last5'] for z in audits if z['arm']==arm))
    save(OUT/'learning_qualification.json',dict(status='actual_TRAIN_guard_and_window_replays_before_new_outer_quality',arms=learning,selected_by_outer=False))
    ref=load_reference(d);old=pd.read_parquet(ROOT/'artifacts/v146_guarded_pair_training_20261001/full_prediction_ledger.parquet')
    assert np.array_equal(old.row_position,ref.row_position);a0=old.pred_A0.to_numpy();base=old.pred_A.to_numpy()
    full=ref.copy();full['pred_A0']=a0;full['pred_V146_A']=base;asa=d.copy();asa['pred_A0']=a0[d.row_position];asa['pred_V146_A']=base[d.row_position]
    ladder=pd.read_parquet(ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet');assert np.array_equal(ladder.row_position,d.row_position)
    headers=set(pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet').row_position);quality={}
    for arm in ['A','B']:
        pred=base.copy();labels=np.empty(len(d),np.int8)
        for f,q in predictions[arm].items():
            mask=d.fold.eq(f).to_numpy();labels[mask]=q[d.loc[mask,'local']].argmax(1)
        pred[d.row_position]=labels;full['pred_'+arm]=pred;asa['pred_'+arm]=labels
        for cl in range(3):
            probabilities_by_row=np.empty(len(d),np.float64)
            for f,q in predictions[arm].items():
                mask=d.fold.eq(f).to_numpy();probabilities_by_row[mask]=q[d.loc[mask,'local'],cl]
            asa[f'{arm}_p{cl}']=probabilities_by_row
        quality[arm]=quality_for(ref,d,pred,a0,ladder,headers)
    matched=evaluate_primary(ref,full[['row_position','pred_A','pred_B']],dict(expected_full_rows=2056871,asa_error_limits=dict(M=318,S=2074,total=2170),minimum_improved_folds=2,protected_S_roots=[21702,20849,29],required_full_classes=[0,1,2]))
    ma,mb=matched['ASA']['A'],matched['ASA']['B'];outside=asa.truth.eq(2)&~asa.root.isin([21702,20849,29])
    gates=dict(M_no_increase=mb['1']['missed']<=ma['1']['missed'],S_no_increase=mb['2']['missed']<=ma['2']['missed'],
        one_class_improves=any(mb[str(cl)]['missed']<ma[str(cl)]['missed'] for cl in [1,2]),
        two_folds_improve=sum(z['B_errors']<z['A_errors'] for z in matched['folds'])>=2,
        outside_top3_S_no_increase=int(asa.loc[outside,'pred_B'].ne(2).sum())<=int(asa.loc[outside,'pred_A'].ne(2).sum()))
    matched_pass=all(gates.values()) and all(learning[a]['all_roles_mastered'] for a in ['A','B']);task=matched_pass and quality['B']['quality_passed']
    full.to_parquet(OUT/'full_prediction_ledger.parquet',index=False);asa.to_parquet(OUT/'ASA_prediction_ledger.parquet',index=False)
    quality.update(matched=matched,matched_gates=gates,matched_effect_passed=matched_pass,task_acceptance=task,model_promoted=False);save(OUT/'quality.json',quality)
    # Reuse the actual fixed cohorts, not labels selected from these outcomes.
    fixed=pd.read_parquet(ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet')
    assert np.array_equal(fixed.row_position,asa.row_position)
    facts=fixed.facts_json.map(json.loads);icmp=facts.map(lambda z:str(z.get('transport_protocol','')).upper()=='ICMP').to_numpy();assert int(icmp.sum())==2373
    known_new=set(pd.read_parquet(ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001/new_M_regressions_vs_V146_A.parquet').row_position)
    masks=dict(all_ASA=np.ones(len(asa),bool),hard578=fixed.known_578_cohort.to_numpy(),strict51=fixed.same_family_and_outer_fold_control_S.to_numpy(),
        unknown_or_missing_ports=~fixed.both_ports_observed.to_numpy(),ICMP=icmp,V155_new_M16=asa.row_position.isin(known_new).to_numpy())
    mixed=d.groupby('canonical_key').truth.nunique();masks['mixed_current_numeric_input']=d.canonical_key.map(mixed).gt(1).to_numpy()
    assert int(masks['hard578'].sum())==578 and int(masks['strict51'].sum())==51 and int(masks['V155_new_M16'].sum())==16
    blocked=np.zeros(len(asa),bool)
    for f in range(3):
        bank=np.load(BANK/f'fold{f}/deployment_probabilities.npy');mask=d.fold.eq(f).to_numpy()
        blocked[mask]=masks['hard578'][mask]&((bank[d.loc[mask,'local'],:,2]-bank[d.loc[mask,'local'],:,1]).max(1)<0)
    assert int(blocked.sum())>=188;masks['all17_strict_S_less_M']=blocked
    cohort=[]
    for name,mask in masks.items():
        part=asa[mask];part.to_parquet(OUT/f'fixed_cohort_{name}_all_original_rows.parquet',index=False)
        cohort.append(dict(cohort=name,original_rows=len(part),paired=pair(part,'pred_A','pred_B'),vs_V146_A=pair(part,'pred_V146_A','pred_B')))
    assert not asa.loc[blocked,['pred_A','pred_B']].eq(2).any().any()
    save(OUT/'all_fixed_cohort_quality.json',dict(status='all_fixed_hard_and_correct_control_rows_retained',cohorts=cohort,
        convex_bank_unrepairable_S_rows=int(blocked.sum()),quality_authority='descriptive_subgroups_do_not_replace_full_population_or_per_class_gates'))
    sources=asa.assign(A_wrong=asa.pred_A.ne(asa.truth),B_wrong=asa.pred_B.ne(asa.truth)).groupby(['root','truth']).agg(support=('truth','size'),A_errors=('A_wrong','sum'),B_errors=('B_wrong','sum')).reset_index()
    sources.to_parquet(OUT/'all_source_class_quality.parquet',index=False)
    base_cost=read(ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001/base_phase_completion.json');legacy_cost=read(ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001/legacy_phase_completion.json')
    delivery=dict(status='V158_nested_probability_fusion_completed_not_promoted',latest_actual='V158',issue='LEGAL_SOURCE_EXCLUDED_PROBABILITY_FUSION',issue_solved=False,
        classifier_fits=60,current_pipeline_fits=45,legacy_full_population_fits=9,fusion_fits=6,
        current_batch_gradients=base_cost['batch_gradients'],current_dense_full_gradients=base_cost['full_gradients'],legacy_full_gradients=legacy_cost['full_gradients'],
        fusion_full_gradients=sum(z['full_gradients'] for z in receipts),fusion_proposals=sum(z['proposal_evaluations'] for z in receipts),fusion_updates=sum(z['accepted_updates'] for z in receipts),
        actual_evaluation_classifier_forward_calls=total_calls,TRAIN=learning,matched_effect_passed=matched_pass,quality_acceptance=task,model_promoted=False,
        ASA_errors={arm:{'M':quality[arm]['ASA']['B']['1']['missed'],'S':quality[arm]['ASA']['B']['2']['missed']} for arm in ['A','B']},
        original_pairs=dict(B_vs_A=pair(asa,'pred_A','pred_B'),B_vs_A0=pair(asa,'pred_A0','pred_B'),B_vs_V146_A=pair(asa,'pred_V146_A','pred_B')),
        validation_scope='Previously inspected development folds; frozen bank provenance and new actual scorer replay. No independent external environment acceptance.',
        full_population_scope='All2056871 original labels/rows scored; non_ASA frozen registered predictions retained.',
        automatic_extra_fits=False,second_third_and_full_goal_closed=False,seal_sha256=sha(OUT/'run_seal.json'),quality_sha256=sha(OUT/'quality.json'))
    save(OUT/'verification.json',dict(status='actual_six_endpoints_windows_and_full_gold_verified',official_rows=len(ref),ASA_rows=len(d),audits=audits,new_gradients=0,actual_evaluation_classifier_calls=total_calls))
    require(__file__);check_bindings(eval_bindings);save(OUT/'final_delivery.json',delivery);print(dict(status=delivery['status'],ASA_errors=delivery['ASA_errors'],matched_effect=matched_pass,task_acceptance=task),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save(OUT/'evaluation_failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),source_sha256=sha(Path(__file__))))
        raise
