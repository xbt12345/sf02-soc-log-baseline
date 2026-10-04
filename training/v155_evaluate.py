"""Actual resource endpoints/window/input replays and full independent truth."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json
from collections import Counter
import numpy as np
import pandas as pd
import torch
from v155_runtime import ROOT,OUT,OLD,read,save,sha,require,endpoint
from v155_train import initialize,role_reference
from v155_sam_objective import values_at
from v138_train import configure,quick_stats,historical
from v135_runtime import load_data
from v135_model import tensor_hash
from v138_readout import probabilities
from v142_retention_check import check
from v142_infer import infer_numeric_input
from v140_runtime import OUT as READOUTS
from v131_evaluate import load_reference
from v135_evaluate import quality_for
from v138_closeout import pair
from experiment_review import evaluate_primary

def main():
    require(ROOT/'training/v155_train.py');configure();assert not (OUT/'final_delivery.json').exists()
    x,d=load_data();predictions={a:{} for a in ['A','B']};role_rows={a:[] for a in ['A','B']}
    windows={a:[[] for _ in range(5)] for a in ['A','B']};audits=[];fit_receipts=[];learning={};forward_calls=Counter()
    def hook(module,*args):
        name=type(module).__name__
        if name in ['Classifier','SecondRepresentation']:forward_calls[name]+=1
    handle=torch.nn.modules.module.register_module_forward_hook(hook)
    for f in range(3):
        frame,c,pure,ids,refrows=role_reference(d,f)
        for arm in ['A','B']:
            folder=OUT/f'fold{f}_{arm}';r=read(folder/'fit.json');endpoint(r);fit_receipts.append(r)
            for n,k in [('endpoint.pt','model_sha256'),('sealed_all_prob.npy','probability_sha256'),('endpoint_original_rows.parquet','rows_sha256'),
                ('gradients.jsonl','gradients_sha256'),('proposals.jsonl','proposals_sha256'),('progress.json','progress_sha256')]:assert sha(folder/n)==r[k]
            h,ff,m=initialize(f);initial={k:v.detach().clone() for k,v in m.state_dict().items()};old=np.load(OUT/f'fold{f}_zero_probability.npy');oldpred=old[frame.local].argmax(1)
            logs=[json.loads(s) for s in (folder/'gradients.jsonl').read_text().splitlines()]
            trials=[json.loads(s) for s in (folder/'proposals.jsonl').read_text().splitlines()];history=read(folder/'progress.json')
            assert len(logs)==r['full_gradient_evaluations'] and len(trials)==r['proposal_evaluations'] and len(history)==r['accepted_updates']
            assert [z['full_gradient_evaluation'] for z in logs]==list(range(1,len(logs)+1))
            assert [z['proposal'] for z in trials]==list(range(1,len(trials)+1))
            assert all(z['original_class_mass_seen']==c.sum(0).tolist() and z['finite'] for z in logs)
            locations=['base'] if arm=='A' else ['base','adversarial']
            assert [z['location'] for z in logs]==locations*r['outer_gradient_attempts']
            assert all(z['radius_L2']==r['radius_L2'] for z in logs)
            accepted=[z for z in trials if z['accepted']]
            assert len(accepted)==len(history) and [z['parameter_sha256'] for z in accepted]==[z['parameter_sha256'] for z in history]
            assert all(z['classification_guard'] and z['frozen_epsilon_proxy_CE']<z['frozen_epsilon_base_proxy_CE'] and z['frozen_epsilon_proxy_CE']<=z['proxy_Armijo_bound'] and not z['moving_epsilon_proxy_monotonic_claimed'] for z in accepted)
            assert all(z['stats']['mastered'] and z['stats']['new_errors_vs_start']==0 for z in history)
            mass=torch.as_tensor(c,device='cuda',dtype=torch.float64);states=[]
            for k,item in enumerate(history[-5:]):
                ck=torch.load(folder/f"accepted{item['accepted_update']}.pt",map_location='cpu',weights_only=True);state=ck['state'];epsilon={n:v.to('cuda') for n,v in ck['epsilon'].items()}
                assert tensor_hash(state)==item['parameter_sha256']
                m.load_state_dict(state);q=probabilities(m,h,ff,np.arange(len(c)));stats=quick_stats(frame,q,pure,oldpred,[22,6,28][f]);assert stats==item['stats']
                proxy,_=values_at(m,h,ff,mass,ids,pure,epsilon,False);plain,_=values_at(m,h,ff,mass,ids,pure,None,False)
                assert abs(proxy['member_CE']-item['frozen_epsilon_proxy_CE'])<1e-12 and abs(plain['member_CE']-item['ordinary_member_CE'])<1e-12
                accepted_record=accepted[item['accepted_update']-1];assert tensor_hash(epsilon)==accepted_record['epsilon_sha256']
                windows[arm][k].append(frame[['row_position','truth']].assign(training_role=f,pred=q[frame.local].argmax(1)))
                states.append(dict(accepted_update=item['accepted_update'],stats=stats,parameter_sha256=item['parameter_sha256'],frozen_proxy_replayed=True))
            st=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)
            assert (st['arm'],st['fold'],st['seal_sha256'])==(arm,f,sha(OUT/'run_seal.json'))
            m.load_state_dict(st['state']);q=probabilities(m,h,ff,np.arange(len(c)));assert np.array_equal(q,np.load(folder/'sealed_all_prob.npy'))
            for n,v in initial.items():
                if n.startswith(('reference_','head_')):assert torch.equal(v,m.state_dict()[n])
            original=torch.load(historical(f)/'epoch100_model.pt',map_location='cpu',weights_only=True)['model']
            readout=torch.load(READOUTS/f'fold{f}_C/endpoint_readout.pt',map_location='cpu',weights_only=True)['readout']
            fresh=infer_numeric_input(x,original,readout,st['state']);gap=float(np.abs(q-fresh).max())
            assert gap<=2e-6 and np.array_equal(q.argmax(1),fresh.argmax(1))
            rows=pd.read_parquet(folder/'endpoint_original_rows.parquet')
            assert np.array_equal(rows.row_position,frame.row_position) and np.array_equal(rows.truth,frame.truth) and np.array_equal(rows.pred,q[frame.local].argmax(1))
            for cl in range(3):assert np.array_equal(rows[f'p{cl}'],q[frame.local,cl])
            stable=len(states)==5 and len({z['parameter_sha256'] for z in states})==5 and all(z['stats']['mastered'] and z['stats']['new_errors_vs_start']==0 for z in states)
            assert stable==r['stable_last_five'];role_rows[arm].append(rows);predictions[arm][f]=q
            plain,_=values_at(m,h,ff,mass,ids,pure,None,False)
            audits.append(dict(arm=arm,fold=f,endpoint_stats=r['endpoint_stats'],stable_window=stable,window_states=states,
                cache_free_actual_numeric_inputs=len(q),cache_free_max_gap=gap,frozen_buffers_exact=True,endpoint_ordinary_risk=plain,
                gradient_evaluations=len(logs),proposal_evaluations=len(trials),accepted_updates=len(history),termination=r['termination']))
            print(dict(stage='endpoint_and_window_actual_replay',arm=arm,fold=f,stable=stable,cache_free_gap=gap),flush=True)
            del m,h,ff,mass;gc.collect();torch.cuda.empty_cache()
    handle.remove()
    for arm in ['A','B']:
        rows=pd.concat(role_rows[arm],ignore_index=True);rows.to_parquet(OUT/f'{arm}_training_role_ledger.parquet',index=False)
        guard=check(rows);wg=[check(pd.concat(z,ignore_index=True)) for z in windows[arm] if len(z)==3]
        learning[arm]=dict(retention=guard,window_retention=wg,all_roles_mastered=guard['passed'] and len(wg)==5 and all(z['passed'] for z in wg) and all(z['stable_window'] for z in audits if z['arm']==arm),
            pure_errors=int((rows.pred.ne(rows.truth)&rows.pure_TRAIN_input).sum()),all_original_correct_new_errors=int((rows.pred_start.eq(rows.truth)&rows.pred.ne(rows.truth)).sum()))
    save(OUT/'learning_qualification.json',dict(status='TRAIN_joint_guards_frozen_before_outer_quality',arms=learning,candidate='B',selected_by_outer=False))
    ref=load_reference(d);base=pd.read_parquet(OLD/'full_prediction_ledger.parquet');a0=base.pred_A0.to_numpy();previous=base.pred_A.to_numpy()
    assert np.array_equal(ref.row_position,base.row_position)
    ladder=pd.read_parquet(ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet')
    headers=set(pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet').row_position)
    full=pd.DataFrame(dict(row_position=ref.row_position,pred_A0=a0,pred_V146_A=previous))
    asa=d[['row_position','local','root','fold','truth','canonical_key']].copy();asa['pred_A0']=a0[d.row_position];asa['pred_V146_A']=previous[d.row_position];quality={}
    for arm in ['A','B']:
        pred=previous.copy();qpred=np.empty(len(d),np.int8)
        for f,q in predictions[arm].items():
            mask=d.fold.eq(f).to_numpy();qpred[mask]=q[d.loc[mask,'local']].argmax(1)
        pred[d.row_position]=qpred;full['pred_'+arm]=pred;asa['pred_'+arm]=qpred;quality[arm]=quality_for(ref,d,pred,a0,ladder,headers)
    matched=evaluate_primary(ref,pd.DataFrame(dict(row_position=ref.row_position,pred_A=full.pred_A,pred_B=full.pred_B)),
        dict(expected_full_rows=2056871,asa_error_limits=dict(M=318,S=2074,total=2170),minimum_improved_folds=2,protected_S_roots=[21702,20849,29],required_full_classes=[0,1,2]))
    ma,mb=matched['ASA']['A'],matched['ASA']['B'];out=asa.truth.eq(2)&~asa.root.isin([21702,20849,29])
    gates=dict(M_no_increase=mb['1']['missed']<=ma['1']['missed'],S_no_increase=mb['2']['missed']<=ma['2']['missed'],one_class_improves=any(mb[str(c)]['missed']<ma[str(c)]['missed'] for c in [1,2]),
        two_folds_improve=sum(z['B_errors']<z['A_errors'] for z in matched['folds'])>=2,outside_top3_S_no_increase=int(asa.loc[out,'pred_B'].ne(2).sum())<=int(asa.loc[out,'pred_A'].ne(2).sum()))
    effect=all(gates.values()) and learning['A']['all_roles_mastered'] and learning['B']['all_roles_mastered'];accepted=effect and quality['B']['quality_passed']
    full.to_parquet(OUT/'full_prediction_ledger.parquet',index=False);asa.to_parquet(OUT/'ASA_prediction_ledger.parquet',index=False)
    quality.update(matched=matched,matched_gates=gates,matched_effect_passed=effect,task_acceptance=accepted,model_promoted=False);save(OUT/'quality.json',quality)
    save(OUT/'verification.json',dict(status='actual_endpoints_windows_frozen_proxy_and_full_official_rows_verified',audits=audits,
        official_rows=len(ref),ASA_rows=len(d),classifier_fits=6,full_gradient_evaluations=sum(z['full_gradient_evaluations'] for z in fit_receipts),
        proposal_evaluations=sum(z['proposal_evaluations'] for z in fit_receipts),accepted_updates=sum(z['accepted_updates'] for z in fit_receipts),
        actual_evaluation_model_forward_chunk_calls=dict(forward_calls),new_evaluation_gradients=0,
        full_population_scope='ASA actual model/function replays; non-ASA immutable V146 registered predictions retained and truth independently scored.',
        validation_scope='Previously inspected development folds, not blind external or official submission.'))
    delivery=dict(status='guarded_full_gradient_neighborhood_trial_completed_not_promoted',latest_actual='V155',issue='PARAMETER_NEIGHBORHOOD_TRANSFER',issue_solved=False,
        matched_effect_passed=effect,quality_acceptance=accepted,model_promoted=False,classifier_fits=6,
        full_gradient_evaluations=sum(z['full_gradient_evaluations'] for z in fit_receipts),proposal_evaluations=sum(z['proposal_evaluations'] for z in fit_receipts),
        accepted_updates=sum(z['accepted_updates'] for z in fit_receipts),preflight_gradients=9,TRAIN=learning,
        ASA_errors={a:{'M':quality[a]['ASA']['B']['1']['missed'],'S':quality[a]['ASA']['B']['2']['missed']} for a in ['A','B']},
        original_pairs=dict(B_vs_A=pair(asa,'pred_A','pred_B'),B_vs_V146_A=pair(asa,'pred_V146_A','pred_B'),B_vs_A0=pair(asa,'pred_A0','pred_B')),
        new_independent_support_created=False,automatic_extra_fits=False,full_classifier_transferred_to_new_environment=False,
        seal_sha256=sha(OUT/'run_seal.json'),quality_sha256=sha(OUT/'quality.json'),verification_sha256=sha(OUT/'verification.json'))
    save(OUT/'final_delivery.json',delivery);print(dict(status=delivery['status'],ASA_errors=delivery['ASA_errors'],matched_effect=effect,task_acceptance=accepted),flush=True)

if __name__=='__main__':main()
