"""Independent saved-result/cost/gold audit; never run an official classifier."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from experiment_review_v159_v3 import require_run_seal,check_bindings,sha
from v159_float64_repeat_policy_v2 import finite_step_review

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
REPAIR=ROOT/'artifacts/v159_evaluation_cached_continuation_20261002'
OLD_REPLAY=ROOT/'artifacts/v159_evaluation_source_sum_repair_20261002'
CACHED={(0,'A'),(0,'B'),(1,'A')}
OUT=ROOT/'artifacts/v159_independent_boundary_numeric_result_audit_20261002'
SEGMENTS=[('observation_weight',0,1060592),('opinion_weight',1060592,1060768),('bias',1060768,1060784),('output_weight',1060784,1060832)]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def events(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def classes(truth,pred):
    result={}
    for c in [0,1,2]:
        support=int((truth==c).sum());tp=int(((truth==c)&(pred==c)).sum());pp=int((pred==c).sum())
        result[str(c)]=dict(support=support,correct=tp,missed=support-tp,predicted=pp,
                            precision=tp/pp if pp else None,recall=tp/support if support else None,
                            f1=2*tp/(support+pp) if support+pp else None)
    return result


def paired(frame,old,new):
    result={}
    for c in [0,1,2]:
        a=frame[frame.truth==c];oc=a[old].to_numpy()==c;nc=a[new].to_numpy()==c
        result[str(c)]=dict(support=len(a),old_errors=int((~oc).sum()),new_errors=int((~nc).sum()),
                            repaired=int((~oc&nc).sum()),regressed=int((oc&~nc).sum()))
    return result


def scope(rows,gold):
    assert rows.row_position.is_unique and np.array_equal(rows.truth,gold[rows.row_position])
    assert rows.pred.isin([0,1,2]).all()
    wrong=rows.pred.to_numpy()!=rows.truth.to_numpy()
    q=rows[['p0','p1','p2']].to_numpy()
    assert np.isfinite(q).all() and np.array_equal(q.argmax(1),rows.pred)
    return dict(classes=classes(rows.truth.to_numpy(),rows.pred.to_numpy()),original_rows=len(rows),
                pure_errors=int((wrong&rows.pure_current_input).sum()),
                protected_regressions=int((wrong&rows.protected_correct).sum()),
                initial_correct_regressions=int((wrong&rows.initial_correct).sum()),
                repairs=int((~wrong&~rows.initial_correct).sum()))


def main():
    assert not OUT.exists()
    plan=require_run_seal(RUN/'run_seal.json',ROOT/'training/v159_boundary_evaluate_v4.py')
    delivery=read(RUN/'final_delivery.json');verification=read(RUN/'verification.json')
    assert delivery['classifier_fits']==6 and delivery['model_promoted']==False
    assert verification['official_rows']==2056871 and verification['new_gradients']==0
    check_bindings(read(RUN/'evaluation_input_bindings_v7.json')['source_sha256'])
    repair_seal=read(REPAIR/'run_seal.json');check_bindings(repair_seal['source_sha256'])
    gold=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert len(gold)==2056871 and np.isfinite(gold).all()
    initial=torch.load(RUN/'initial.pt',map_location='cpu',weights_only=True)['state']
    init=np.concatenate([t.numpy().ravel() for t in initial.values()])
    fits=[];totals=dict(head=0,feature=0,gradients=0,proposals=0,updates=0,evaluation_head=0,evaluation_feature=0)
    for f in range(3):
        mass=plan['role_call_budgets'][f]['original_class_mass']
        for arm in ['A','B']:
            folder=RUN/f'fold{f}_{arm}';r=read(folder/'fit.json')
            for key,file in [('model_sha256','endpoint.pt'),('OOF_probability_sha256','endpoint_OOF_probability.npy'),('deployment_probability_sha256','endpoint_deployment_probability.npy')]:assert sha(folder/file)==r[key]
            log=events(folder/'calls.jsonl');counts={}
            for kind in ['head','feature','full_class_gradient']:
                attempted=[e for e in log if e['kind']==kind and e['event']=='attempt'];completed=[e for e in log if e['kind']==kind and e['event']=='completed']
                assert len(attempted)==len(completed) and [e['ordinal'] for e in attempted]==list(range(1,len(attempted)+1))
                counts[kind]=len(attempted)
                if kind=='full_class_gradient':assert [e['class_id'] for e in attempted]==[1,2]*r['gradient_iterations'] and all(e['original_class_mass']==mass for e in attempted)
            assert counts['head']==counts['feature']==r['counts']['head_attempts'] and counts['full_class_gradient']==r['full_class_gradients']==2*r['gradient_iterations']
            prop=events(folder/'proposals.jsonl');accepted=[e for e in prop if e['accepted']]
            assert len(prop)==r['proposal_evaluations'] and len(accepted)==r['accepted_updates']
            for e in accepted:
                assert finite_step_review(e['base_risks'],e['trial_risks'],e['class_slopes'],*mass[1:],arm,e['step'],e['classification_guard'])['accepted']
                assert e['deployment_stats']['mastered'] and e['deployment_stats']['new_errors_vs_initial']==0
                if arm=='B':assert e['OOF_stats']['protected_regressions']==0
            scopes={s:scope(pd.read_parquet(folder/f'endpoint_{s}_rows.parquet'),gold) for s in ['OOF','deployment']}
            assert scopes['deployment']['pure_errors']==scopes['deployment']['protected_regressions']==scopes['deployment']['initial_correct_regressions']==0
            if arm=='B':assert scopes['OOF']['protected_regressions']==0
            for s in ['OOF','deployment']:
                assert scopes[s]['classes']['1']['missed']==r[s+'_stats']['M_errors'] and scopes[s]['classes']['2']['missed']==r[s+'_stats']['S_errors']
            endpoint=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)['state']
            end=np.concatenate([t.numpy().ravel() for t in endpoint.values()]);delta=end-init
            shifts=[dict(segment=name,max_abs=float(np.abs(delta[a:b]).max()),L2=float(np.linalg.norm(delta[a:b])),changed_components=int(np.count_nonzero(delta[a:b]))) for name,a,b in SEGMENTS]
            rejected=[e for e in prop if not e['accepted']]
            causes=Counter(e['finite_numeric_review']['reason'] for e in rejected)
            terminal=[e for e in prop if e['iteration']==r['gradient_iterations']]
            # Scope-cause counts are from sealed classifier journal summaries,
            # not independent replay of unsaved rejected probabilities.
            terminal_protection=dict(deployment_not_mastered=sum(not e['deployment_stats']['mastered'] for e in terminal),
                                     deployment_initial_correct_new_errors=sum(e['deployment_stats']['new_errors_vs_initial']>0 for e in terminal),
                                     OOF_protected_regressions=sum(e['OOF_stats']['protected_regressions']>0 for e in terminal))
            window=[]
            for item in r['last5']:
                for s in ['OOF','deployment']:
                    row=pd.read_parquet(folder/f"accepted{item['update']}_{s}_rows.parquet")
                    z=scope(row,gold)
                    if s=='deployment':assert z['pure_errors']==z['protected_regressions']==0
                window.append(item['parameter_sha256'])
            assert len(window)==min(5,r['accepted_updates']) and len(set(window))==len(window)
            if window:assert window[-1]==r['endpoint_parameter_sha256']
            replay_root=OLD_REPLAY if (f,arm) in CACHED else REPAIR
            elog=events(replay_root/'replayed_outputs'/folder.name/'calls.jsonl');ec={k:sum(e['kind']==k and e['event']=='completed' for e in elog) for k in ['head','feature','full_class_gradient']}
            assert ec['head']==ec['feature'] and ec['full_class_gradient']==0
            totals['head']+=counts['head'];totals['feature']+=counts['feature'];totals['gradients']+=counts['full_class_gradient'];totals['proposals']+=len(prop);totals['updates']+=len(accepted)
            totals['evaluation_head']+=ec['head'];totals['evaluation_feature']+=ec['feature']
            fits.append(dict(fold=f,arm=arm,scopes=scopes,counts=counts,gradient_iterations=r['gradient_iterations'],accepted_updates=r['accepted_updates'],proposals=len(prop),termination=r['termination'],
                             class_risks_before=r['initial_stable_class_risks'],class_risks_after=r['final_stable_class_risks'],
                             rejected_reasons=dict(causes),terminal_protection_summary=terminal_protection,parameter_shifts=shifts,last_window_states=len(window)))
    assert totals['gradients']==delivery['full_class_gradients'] and totals['updates']==delivery['updates'] and totals['proposals']==delivery['proposals']
    assert totals['head']==totals['feature']==delivery['training_head_calls']
    full=pd.read_parquet(RUN/'full_prediction_ledger.parquet');asa=pd.read_parquet(RUN/'ASA_prediction_ledger.parquet')
    assert np.array_equal(full.row_position,np.arange(2056871)) and np.array_equal(full.truth,gold)
    assert len(asa)==112807 and asa.row_position.is_unique and np.array_equal(asa.truth,gold[asa.row_position])
    result_full={};result_asa={}
    for col in ['pred_A0','pred_V146_A','pred_A','pred_B']:
        assert full[col].isin([0,1,2]).all() and np.array_equal(asa[col],full.loc[asa.row_position,col])
        result_full[col]=classes(gold,full[col].to_numpy());result_asa[col]=classes(asa.truth.to_numpy(),asa[col].to_numpy())
    pairs={col:paired(asa,col,'pred_B') for col in ['pred_A0','pred_V146_A','pred_A']}
    for arm in ['A','B']:
        assert result_asa['pred_'+arm]['1']['missed']==delivery['ASA_errors'][arm]['M'] and result_asa['pred_'+arm]['2']['missed']==delivery['ASA_errors'][arm]['S']
    mask=np.ones(len(full),bool);mask[asa.row_position]=False
    assert np.array_equal(full.loc[mask,'pred_A'],full.loc[mask,'pred_V146_A']) and np.array_equal(full.loc[mask,'pred_B'],full.loc[mask,'pred_V146_A'])
    non_asa_errors=int(full.loc[mask,'pred_B'].ne(full.loc[mask,'truth']).sum());assert non_asa_errors==107
    rootrows=[]
    for root,g in asa[asa.truth==2].groupby('root'):
        rootrows.append(dict(root=int(root),support=len(g),A_errors=int(g.pred_A.ne(2).sum()),B_errors=int(g.pred_B.ne(2).sum())))
    rtab=pd.DataFrame(rootrows)
    sources=dict(S_roots=len(rtab),A_zero_recall=int(rtab.A_errors.eq(rtab.support).sum()),B_zero_recall=int(rtab.B_errors.eq(rtab.support).sum()),
                 A_mean_recall=float((1-rtab.A_errors/rtab.support).mean()),B_mean_recall=float((1-rtab.B_errors/rtab.support).mean()))
    assert totals['evaluation_head']==totals['evaluation_feature']==640 and delivery['failed_evaluation_head_calls']==110
    assert delivery['cached_v5_actual_evaluation_heads']==360 and delivery['new_v7_actual_evaluation_heads']==280
    assert read(ROOT/'artifacts/v159_independent_cached_replay_audit_20261002/audit.json')['cached_completed_heads']==360
    cumulative=dict(head_calls=460+110+totals['head']+totals['evaluation_head'],class_gradients=20+totals['gradients'],fits=6,updates=totals['updates'])
    assert cumulative['head_calls']==delivery['cumulative_actual_head_calls'] and cumulative['class_gradients']==delivery['cumulative_actual_full_class_gradients']
    assert delivery['total_cumulative_caps']['classifier_forward_chunks']==78226
    assert cumulative['head_calls']<=78226 and cumulative['class_gradients']<=2420
    paths=[Path(__file__).resolve(),RUN/'final_delivery.json',RUN/'verification.json',RUN/'full_prediction_ledger.parquet',RUN/'ASA_prediction_ledger.parquet',ROOT/'data/official/train.parquet']
    report=dict(status='all_six_saved_fit_results_and_full_gold_independently_recomputed',fits=fits,actual_costs=totals,cumulative_actual_costs=cumulative,
                full_class_quality=result_full,ASA_class_quality=result_asa,ASA_paired_repairs_and_regressions=pairs,S_source_quality=sources,non_ASA_errors=non_asa_errors,
                worker_first_issue_training_qualification=delivery['first_issue_training_qualification'],worker_quality_acceptance=delivery['quality_acceptance'],
                old_deployment_correct_retained=True,root_official_heads=0,root_official_features=0,root_official_gradients=0,root_official_fits=0,
                classifier_replay_scope='Worker actual360 cached v5 +280 fresh v7 replay verified separately. First3 actual full q not persisted: source control-flow repeat assertion plus frozen full q; all actual FIT row arrays saved. Root only saved arrays/rows/parameters/gold/journals, no new replay.',
                rejected_protection_scope='Aggregate journal summaries only; rejected per-row probabilities were not saved.',
                all_three_goal_issues_closed=False,model_promoted=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    OUT.mkdir();(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','cumulative_actual_costs','ASA_class_quality','ASA_paired_repairs_and_regressions','S_source_quality','worker_first_issue_training_qualification','worker_quality_acceptance']},ensure_ascii=False))


if __name__=='__main__':main()
