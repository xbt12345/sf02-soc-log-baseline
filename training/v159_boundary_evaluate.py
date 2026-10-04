"""Replay six fixed endpoints/windows and score every official original row."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json,traceback
import numpy as np
import pandas as pd
import torch
from v159_boundary_runtime import ROOT,OUT,PREP,read,save,sha,require,endpoint
from v159_boundary_train import context,model_for,Counter,probabilities,risk,stats,rows
from v159_class_direction import finite_armijo
from v138_train import configure
from v135_model import tensor_hash
from v142_retention_check import check as retention
from v131_common import load_data
from v131_evaluate import load_reference
from v135_evaluate import quality_for
from v138_closeout import pair
from experiment_review import check_bindings,evaluate_primary

def journal(path):return [json.loads(z) for z in path.read_text(encoding='utf-8').splitlines()]

def main():
    plan=require(__file__);configure();assert not (OUT/'final_delivery.json').exists()
    sources={}
    for f in range(3):
        for arm in ['A','B']:
            folder=OUT/f'fold{f}_{arm}';endpoint(read(folder/'fit.json'))
            sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in folder.glob('*') if p.is_file()})
    save(OUT/'evaluation_input_bindings.json',dict(status='all_six_fit_models_journals_rows_bound_before_evaluation_forward',source_sha256=sources,official_new_gradients=0))
    predictions={a:{} for a in ['A','B']};all_fit_rows={a:[] for a in ['A','B']};audits=[];receipts=[];evaluation_counts=[]
    for f in range(3):
        ctx=context(f)
        for arm in ['A','B']:
            folder=OUT/f'fold{f}_{arm}';r=read(folder/'fit.json');endpoint(r);receipts.append(r)
            assert r['model_sha256']==sha(folder/'endpoint.pt') and r['OOF_probability_sha256']==sha(folder/'endpoint_OOF_probability.npy') and r['deployment_probability_sha256']==sha(folder/'endpoint_deployment_probability.npy')
            events=journal(folder/'calls.jsonl');counts=r['counts']
            for kind,key in [('head','head'),('feature','feature'),('full_class_gradient','gradient')]:
                assert sum(e['event']=='attempt' and e['kind']==kind for e in events)==counts[key+'_attempts']
                assert sum(e['event']=='completed' and e['kind']==kind for e in events)==counts[key+'_completed']==counts[key+'_attempts']
            gradients=[e for e in events if e['event']=='completed' and e['kind']=='full_class_gradient']
            assert len(gradients)==r['full_class_gradients'] and [e['class_id'] for e in gradients]==[1,2]*r['gradient_iterations']
            assert all(e['original_class_mass']==ctx['mass'].tolist() for e in gradients)
            prop=journal(folder/'proposals.jsonl');accepted=[e for e in prop if e['accepted']];history=read(folder/'progress.json') if (folder/'progress.json').exists() else []
            assert len(prop)==r['proposal_evaluations'] and len(accepted)==len(history)==r['accepted_updates']
            for e in accepted:
                assert finite_armijo(e['base_risks'],e['trial_risks'],e['class_slopes'],*ctx['mass'][1:],arm,e['step'],e['classification_guard'])
                assert e['deployment_stats']['mastered'] and (arm=='A' or e['OOF_stats']['protected_regressions']==0)
            assert [e['parameter_sha256'] for e in accepted]==[e['parameter_sha256'] for e in history]
            directions=journal(folder/'directions.jsonl');assert len(directions)==r['gradient_iterations']
            model=model_for();counter=Counter(model,plan['role_call_budgets'][f]['evaluation_classifier_cap_per_arm'],folder/'evaluation_calls.jsonl',0);states=[]
            for item in r['last5']:
                state=torch.load(folder/f"accepted{item['update']}.pt",map_location='cpu',weights_only=True)['state'];assert tensor_hash(state)==item['parameter_sha256'];model.load_state_dict(state)
                q,lp=probabilities(model,ctx,'deployment',np.arange(22546),True);actual=rows(ctx,q,lp,'deployment');stored=pd.read_parquet(folder/f"accepted{item['update']}_deployment_rows.parquet")
                assert np.array_equal(actual.row_position,stored.row_position) and np.array_equal(actual.truth,stored.truth) and np.array_equal(actual.pred,stored.pred)
                gap=float(np.abs(actual[['p0','p1','p2']].to_numpy()-stored[['p0','p1','p2']].to_numpy()).max());assert gap<=2e-12
                ds=stats(ctx,q,'deployment');assert ds==item['deployment_stats'] and ds['mastered']
                oo=pd.read_parquet(folder/f"accepted{item['update']}_OOF_rows.parquet");assert np.array_equal(oo.row_position,ctx['frame'].row_position) and np.array_equal(oo.truth,ctx['frame'].truth)
                oq=np.full((22546,3),np.nan);unique=oo.drop_duplicates('local');oq[unique.local]=unique[['p0','p1','p2']].to_numpy()
                assert np.array_equal(oq[oo.local].argmax(1),oo.pred)
                os_=stats(ctx,oq,'OOF');assert os_==item['OOF_stats']
                source=oo.assign(wrong=oo.pred.ne(oo.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()
                saved_source=pd.read_parquet(folder/f"accepted{item['update']}_OOF_sources.parquet");pd.testing.assert_frame_equal(source,saved_source)
                states.append(dict(update=item['update'],parameter_sha256=item['parameter_sha256'],OOF_stats=os_,deployment_stats=ds,probability_gap=gap))
            terminal=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True);assert (terminal['fold'],terminal['arm'],terminal['seal_sha256'])==(f,arm,sha(OUT/'run_seal.json'));model.load_state_dict(terminal['state']);assert tensor_hash(model.state_dict())==r['endpoint_parameter_sha256']
            q,lp=probabilities(model,ctx,'deployment',np.arange(22546),True);assert np.array_equal(q,np.load(folder/'endpoint_deployment_probability.npy'))
            value,qo,lpo,_=risk(model,ctx,'OOF',ctx['ids']);assert np.array_equal(qo[ctx['ids']],np.load(folder/'endpoint_OOF_probability.npy')[ctx['ids']]) and np.allclose(value,r['final_stable_class_risks'],atol=1e-10,rtol=0)
            fitrows=rows(ctx,q,lp,'deployment');all_fit_rows[arm].append(fitrows);predictions[arm][f]=q
            assert stats(ctx,qo,'OOF')==r['OOF_stats'] and stats(ctx,q,'deployment')==r['deployment_stats']
            counts=counter.counts();assert counts['gradient_attempts']==counts['gradient_completed']==0 and counts['head_attempts']==counts['head_completed']==counts['feature_attempts']==counts['feature_completed']
            evaluation_counts.append(counts);counter.close()
            audits.append(dict(fold=f,arm=arm,window=states,distinct_last5=len(states)==5 and len({e['parameter_sha256'] for e in states})==5,
                OOF_classification_mastered=stats(ctx,qo,'OOF')['mastered'],deployment_mastered=stats(ctx,q,'deployment')['mastered'],actual_evaluation_counts=counts))
            del model;gc.collect();torch.cuda.empty_cache()
    learning={}
    for arm in ['A','B']:
        ledger=pd.concat(all_fit_rows[arm],ignore_index=True);ledger.to_parquet(OUT/f'{arm}_all_deployment_FIT_rows.parquet',index=False);joint=retention(ledger);own=[e for e in audits if e['arm']==arm]
        learning[arm]=dict(joint_TRAIN_retention=joint,all_roles_OOF_endpoint_mastered=all(e['OOF_classification_mastered'] for e in own),
            all_roles_last5_distinct_and_mastered=all(e['distinct_last5'] and all(s['OOF_stats']['mastered'] and s['deployment_stats']['mastered'] for s in e['window']) for e in own),
            protected_TRAIN_passed=joint['passed'] and all(e['deployment_mastered'] for e in own))
    first=learning['B']['all_roles_OOF_endpoint_mastered'] and learning['B']['all_roles_last5_distinct_and_mastered'] and all(learning[a]['protected_TRAIN_passed'] for a in ['A','B'])
    save(OUT/'learning_qualification.json',dict(status='all_training_role_classification_and_original_guard_review_before_outer_join',arms=learning,first_issue_training_qualification=first,
        raw_input_conflicts='All mixed rows retained, current-input pure rows zero errors and mixed total at or below fixed22/6/28 required; tuple floor and forced-old-OOF constraints separately reported.',quality_acceptance=False))
    _,d=load_data();ref=load_reference(d);old=pd.read_parquet(ROOT/'artifacts/v146_guarded_pair_training_20261001/full_prediction_ledger.parquet');assert np.array_equal(old.row_position,ref.row_position)
    a0=old.pred_A0.to_numpy();base=old.pred_A.to_numpy();full=ref.copy();full['pred_A0']=a0;full['pred_V146_A']=base;asa=d.copy();asa['pred_A0']=a0[d.row_position];asa['pred_V146_A']=base[d.row_position]
    ladder=pd.read_parquet(ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet');assert np.array_equal(ladder.row_position,d.row_position)
    headers=set(pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet').row_position);quality={}
    for arm in ['A','B']:
        pred=base.copy();labels=np.empty(len(d),np.int8)
        for f,q in predictions[arm].items():
            mask=d.fold.eq(f).to_numpy();labels[mask]=q[d.loc[mask,'local']].argmax(1)
            for c in range(3):asa.loc[mask,f'{arm}_p{c}']=q[d.loc[mask,'local'],c]
        pred[d.row_position]=labels;full['pred_'+arm]=pred;asa['pred_'+arm]=labels;quality[arm]=quality_for(ref,d,pred,a0,ladder,headers)
    matched=evaluate_primary(ref,full[['row_position','pred_A','pred_B']],dict(expected_full_rows=2056871,asa_error_limits=dict(M=318,S=2074,total=2170),minimum_improved_folds=2,protected_S_roots=[21702,20849,29],required_full_classes=[0,1,2]))
    ma,mb=matched['ASA']['A'],matched['ASA']['B'];outside=asa.truth.eq(2)&~asa.root.isin([21702,20849,29])
    gates=dict(M_no_increase=mb['1']['missed']<=ma['1']['missed'],S_no_increase=mb['2']['missed']<=ma['2']['missed'],one_class_improves=any(mb[str(c)]['missed']<ma[str(c)]['missed'] for c in [1,2]),two_folds_improve=sum(e['B_errors']<e['A_errors'] for e in matched['folds'])>=2,outside_top3_S_no_increase=int(asa.loc[outside,'pred_B'].ne(2).sum())<=int(asa.loc[outside,'pred_A'].ne(2).sum()))
    matched_pass=all(gates.values()) and first;task=matched_pass and quality['B']['quality_passed']
    quality.update(matched=matched,matched_gates=gates,matched_effect_passed=matched_pass,task_acceptance=task,model_promoted=False);save(OUT/'quality.json',quality)
    full.to_parquet(OUT/'full_prediction_ledger.parquet',index=False);asa.to_parquet(OUT/'ASA_prediction_ledger.parquet',index=False)
    fixed=pd.read_parquet(ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet');assert np.array_equal(fixed.row_position,asa.row_position)
    facts=fixed.facts_json.map(json.loads);icmp=facts.map(lambda x:str(x.get('transport_protocol','')).upper()=='ICMP').to_numpy();assert icmp.sum()==2373
    newm=set(pd.read_parquet(ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001/new_M_regressions_vs_V146_A.parquet').row_position)
    mixed=asa.groupby('canonical_key').truth.nunique()
    masks=dict(all_ASA=np.ones(len(asa),bool),hard578=fixed.known_578_cohort.to_numpy(),strict51=fixed.same_family_and_outer_fold_control_S.to_numpy(),unknown_or_missing_ports=~fixed.both_ports_observed.to_numpy(),ICMP=icmp,V155_new_M16=asa.row_position.isin(newm).to_numpy(),mixed_current_numeric_input=asa.canonical_key.map(mixed).gt(1).to_numpy())
    assert masks['hard578'].sum()==578 and masks['strict51'].sum()==51 and masks['V155_new_M16'].sum()==16
    cohorts=[]
    for name,mask in masks.items():
        part=asa[mask];part.to_parquet(OUT/f'fixed_cohort_{name}_all_original_rows.parquet',index=False);cohorts.append(dict(cohort=name,original_rows=len(part),paired=pair(part,'pred_A','pred_B'),vs_V146_A=pair(part,'pred_V146_A','pred_B')))
    save(OUT/'all_fixed_cohort_quality.json',dict(status='every_original_fixed_cohort_and_correct_control_scored',cohorts=cohorts))
    sources_table=asa.assign(A_wrong=asa.pred_A.ne(asa.truth),B_wrong=asa.pred_B.ne(asa.truth)).groupby(['root','truth']).agg(support=('truth','size'),A_errors=('A_wrong','sum'),B_errors=('B_wrong','sum')).reset_index();sources_table.to_parquet(OUT/'all_source_class_quality.parquet',index=False)
    pre=read(OUT/'preflight.json');delivery=dict(status='V159_boundary_completed_quality_passed_not_promoted' if task else 'V159_boundary_completed_quality_failed_not_promoted',latest_actual='V159',classifier_fits=6,new_base_fits=0,
        full_class_gradients=sum(r['full_class_gradients'] for r in receipts),preflight_full_class_gradients=pre['actual_full_class_gradients'],gradient_iterations=sum(r['gradient_iterations'] for r in receipts),proposals=sum(r['proposal_evaluations'] for r in receipts),updates=sum(r['accepted_updates'] for r in receipts),
        training_head_calls=sum(r['counts']['head_attempts'] for r in receipts),training_opinion_feature_calls=sum(r['counts']['feature_attempts'] for r in receipts),evaluation_head_calls=sum(c['head_attempts'] for c in evaluation_counts),evaluation_opinion_feature_calls=sum(c['feature_attempts'] for c in evaluation_counts),
        first_issue_training_qualification=first,second_third_and_full_goal_closed=False,TRAIN=learning,matched_effect_passed=matched_pass,quality_acceptance=task,model_promoted=False,
        ASA_errors={a:dict(M=quality[a]['ASA']['B']['1']['missed'],S=quality[a]['ASA']['B']['2']['missed']) for a in ['A','B']},original_pairs=dict(B_vs_A=pair(asa,'pred_A','pred_B'),B_vs_A0=pair(asa,'pred_A0','pred_B'),B_vs_V146_A=pair(asa,'pred_V146_A','pred_B')),
        validation_scope='Previously inspected development folds, not blind head validation; all full2056871 labels scored, non_ASA frozen.',automatic_extra_fits=False,seal_sha256=sha(OUT/'run_seal.json'))
    save(OUT/'verification.json',dict(status='all_six_fixed_models_actual_replayed_and_all_full_gold_scored',audits=audits,official_rows=len(ref),ASA_rows=len(asa),new_gradients=0))
    require(__file__);check_bindings(sources);save(OUT/'final_delivery.json',delivery);print(json.dumps(dict(status=delivery['status'],first_issue_training_qualification=first,ASA=delivery['ASA_errors'],task_acceptance=task),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save(OUT/'evaluation_failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),source_sha256=sha(__file__)))
        raise
