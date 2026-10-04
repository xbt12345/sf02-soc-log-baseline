"""Independent fixed endpoint/window/guard replays and complete official scoring."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json
import numpy as np
import pandas as pd
import torch
from v146_runtime import ROOT,OUT,BASE,read,save,sha,require,endpoint
from v146_train import initialize,values
from v138_train import configure,quick_stats,historical
from v135_runtime import load_data
from v135_model import tensor_hash
from v144_pair_objective import build_pairs
from v138_readout import probabilities
from v142_retention_check import check
from v142_infer import infer_numeric_input
from v140_runtime import OUT as READOUTS
from v131_evaluate import load_reference,full_folds
from v135_evaluate import quality_for
from v138_closeout import pair
from experiment_review import evaluate_primary


def main():
    require(ROOT/'training/v146_train.py');configure()
    if (OUT/'final_delivery.json').exists():raise FileExistsError('Preserve actual delivery')
    x,d=load_data();facts=pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',columns=['facts_json']).facts_json
    predictions={a:{} for a in ['A','B']};role_rows={a:[] for a in ['A','B']};windows={a:[[] for _ in range(5)] for a in ['A','B']};audits=[];fit_receipts=[];learning={}
    for f in range(3):
        frame,c,pure,ids,p,capacity=build_pairs(d,facts,f)
        for arm in ['A','B']:
            folder=OUT/f'fold{f}_{arm}';r=read(folder/'fit.json');endpoint(r);fit_receipts.append(r)
            for n,k in [('endpoint.pt','model_sha256'),('sealed_all_prob.npy','probability_sha256'),('endpoint_original_rows.parquet','rows_sha256'),('gradients.jsonl','gradients_sha256'),('proposals.jsonl','proposals_sha256'),('progress.json','progress_sha256')]:
                if sha(folder/n)!=r[k]:raise ValueError('Fit identity changed '+n)
            h,ff,m=initialize(f);initial={k:v.detach().clone() for k,v in m.state_dict().items()};old=np.load(OUT/f'fold{f}_zero_probability.npy');oldpred=old[frame.local].argmax(1)
            if not np.array_equal(probabilities(m,h,ff,np.arange(len(c))),old):raise ValueError('Zero replay changed')
            logs=[json.loads(s) for s in (folder/'gradients.jsonl').read_text().splitlines()];trials=[json.loads(s) for s in (folder/'proposals.jsonl').read_text().splitlines()];history=read(folder/'progress.json')
            if len(logs)!=r['full_gradient_evaluations'] or len(trials)!=r['proposal_evaluations'] or len(history)!=r['accepted_updates']:raise ValueError('Missing compute journal')
            if any(z['original_class_mass_seen']!=c.sum(0).tolist() or not z['finite'] for z in logs+[]):raise ValueError('Classification mass/finite mismatch')
            if [z['proposal'] for z in trials]!=list(range(1,len(trials)+1)) or [z['full_gradient_evaluation'] for z in logs]!=list(range(1,len(logs)+1)):raise ValueError('Nonsequential trials/gradients')
            accepted=[z for z in trials if z['accepted']]
            if len(accepted)!=len(history) or any(not z['classification_guard'] or z['objective']>z['Armijo_bound'] for z in accepted):raise ValueError('False guarded acceptance')
            if any(z['stats']['new_errors_vs_start'] or not z['stats']['mastered'] for z in history):raise ValueError('Accepted classification regressed')
            if [z['parameter_sha256'] for z in accepted]!=[z['parameter_sha256'] for z in history]:raise ValueError('Accepted history mismatch')
            states=[]
            for k,item in enumerate(history[-5:]):
                state=torch.load(folder/f"accepted{item['accepted_update']}.pt",map_location='cpu',weights_only=True)['state']
                if tensor_hash(state)!=item['parameter_sha256']:raise ValueError('Window state changed')
                m.load_state_dict(state);q=probabilities(m,h,ff,np.arange(len(c)));s=quick_stats(frame,q,pure,oldpred,[22,6,28][f])
                if s!=item['stats']:raise ValueError('Actual state versus journal classification mismatch')
                rows=frame[['row_position','truth']].assign(training_role=f,pred=q[frame.local].argmax(1));windows[arm][k].append(rows)
                states.append(dict(accepted_update=item['accepted_update'],stats=s,parameter_sha256=item['parameter_sha256']))
            st=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)
            if (st['arm'],st['fold'],st['seal_sha256'])!=(arm,f,sha(OUT/'run_seal.json')):raise ValueError('Wrong registered model')
            m.load_state_dict(st['state']);q=probabilities(m,h,ff,np.arange(len(c)))
            if not np.array_equal(q,np.load(folder/'sealed_all_prob.npy')):raise ValueError('Endpoint model replay differs')
            val,_=values(m,h,ff,c,ids,p,arm,r['lambda_fixed'])
            if history and max(abs(val[k]-history[-1]['values'][k]) for k in ['objective','original_member_CE','auxiliary_loss'])>1e-12:raise ValueError('Endpoint actual objective differs')
            for n,v in initial.items():
                if n.startswith(('reference_','head_')) and not torch.equal(v,m.state_dict()[n]):raise ValueError('Frozen buffers changed')
            original=torch.load(historical(f)/'epoch100_model.pt',map_location='cpu',weights_only=True)['model'];readout=torch.load(READOUTS/f'fold{f}_C/endpoint_readout.pt',map_location='cpu',weights_only=True)['readout']
            fresh=infer_numeric_input(x,original,readout,st['state']);gap=float(np.abs(q-fresh).max())
            if gap>2e-6 or not np.array_equal(q.argmax(1),fresh.argmax(1)):raise ValueError('Cache-free numeric function mismatch')
            rows=pd.read_parquet(folder/'endpoint_original_rows.parquet')
            if not np.array_equal(rows.row_position,frame.row_position) or not np.array_equal(rows.truth,frame.truth) or not np.array_equal(rows.pred,q[frame.local].argmax(1)):raise ValueError('Official role truth/endpoint mismatch')
            stable=len(states)==5 and len({z['parameter_sha256'] for z in states})==5 and all(z['stats']['mastered'] and z['stats']['new_errors_vs_start']==0 for z in states)
            if stable!=r['stable_last_five']:raise ValueError('False stable window')
            role_rows[arm].append(rows);predictions[arm][f]=q
            audits.append(dict(arm=arm,fold=f,endpoint_stats=r['endpoint_stats'],stable_window=stable,window_states=states,actual_cache_free_numeric_inputs=len(q),cache_free_max_gap=gap,
                original_classification_rows=len(rows),endpoint_values=val,frozen_buffers_exact=True,actual_parameter_delta_L2=r['parameter_delta_L2'],gradient_evaluations=len(logs),proposal_evaluations=len(trials),accepted_updates=len(history),termination=r['termination']))
            print(dict(stage='actual_endpoint_replayed',arm=arm,fold=f,stable=stable,cache_free_gap=gap),flush=True)
            del m,h,ff;gc.collect();torch.cuda.empty_cache()
    for arm in ['A','B']:
        rows=pd.concat(role_rows[arm],ignore_index=True);rows.to_parquet(OUT/f'{arm}_training_role_ledger.parquet',index=False)
        guard=check(rows);window_guards=[check(pd.concat(z)) for z in windows[arm] if len(z)==3]
        learning[arm]=dict(retention=guard,window_retention=window_guards,all_roles_mastered=guard['passed'] and len(window_guards)==5 and all(z['passed'] for z in window_guards) and all(z['stable_window'] for z in audits if z['arm']==arm),
            pure_errors=int((rows.pred.ne(rows.truth)&rows.pure_TRAIN_input).sum()),all_original_correct_new_errors=int((rows.pred_start.eq(rows.truth)&rows.pred.ne(rows.truth)).sum()))
    save(OUT/'learning_qualification.json',dict(status='TRAIN_and_all_guards_frozen_before_HELD_scoring',arms=learning,candidate='B',selected_by_HELD=False))
    ref=load_reference(d);base=pd.read_parquet(BASE/'full_prediction_ledger.parquet');a0=base.pred_A0.to_numpy();previous=base.pred_S2.to_numpy()
    if not np.array_equal(ref.row_position,base.row_position):raise ValueError('Full population identity differs')
    ladder=pd.read_parquet(ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet');headers=set(pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet').row_position)
    full=pd.DataFrame(dict(row_position=ref.row_position,pred_A0=a0,pred_V142=previous));asa=d[['row_position','local','root','fold','truth','canonical_key']].copy();asa['pred_A0']=a0[d.row_position];asa['pred_V142']=previous[d.row_position];quality={}
    for arm in ['A','B']:
        pred=previous.copy();qpred=np.empty(len(d),np.int8)
        for f,q in predictions[arm].items():
            mask=d.fold.eq(f).to_numpy();qpred[mask]=q[d.loc[mask,'local']].argmax(1)
        pred[d.row_position]=qpred;full['pred_'+arm]=pred;asa['pred_'+arm]=qpred;quality[arm]=quality_for(ref,d,pred,a0,ladder,headers)
    matched=evaluate_primary(ref,pd.DataFrame(dict(row_position=ref.row_position,pred_A=full.pred_A,pred_B=full.pred_B)),dict(expected_full_rows=2056871,asa_error_limits=dict(M=318,S=2074,total=2170),minimum_improved_folds=2,protected_S_roots=[21702,20849,29],required_full_classes=[0,1,2]))
    ma,mb=matched['ASA']['A'],matched['ASA']['B'];out=asa.truth.eq(2)&~asa.root.isin([21702,20849,29])
    gates=dict(M_no_increase=mb['1']['missed']<=ma['1']['missed'],S_no_increase=mb['2']['missed']<=ma['2']['missed'],one_class_improves=any(mb[str(c)]['missed']<ma[str(c)]['missed'] for c in [1,2]),
        two_folds_improve=sum(z['B_errors']<z['A_errors'] for z in matched['folds'])>=2,outside_top3_S_no_increase=int(asa.loc[out,'pred_B'].ne(2).sum())<=int(asa.loc[out,'pred_A'].ne(2).sum()))
    effect=all(gates.values()) and learning['A']['all_roles_mastered'] and learning['B']['all_roles_mastered'];accepted=effect and quality['B']['quality_passed']
    full.to_parquet(OUT/'full_prediction_ledger.parquet',index=False);asa.to_parquet(OUT/'ASA_prediction_ledger.parquet',index=False)
    quality.update(matched=matched,matched_gates=gates,matched_effect_passed=effect,task_acceptance=accepted,model_promoted=False);save(OUT/'quality.json',quality)
    save(OUT/'verification.json',dict(status='actual_models_windows_and_complete_independent_official_rows_verified',audits=audits,official_rows=len(ref),ASA_rows=len(d),classifier_fits=6,
        full_gradient_evaluations=sum(z['full_gradient_evaluations'] for z in fit_receipts),proposal_evaluations=sum(z['proposal_evaluations'] for z in fit_receipts),accepted_updates=sum(z['accepted_updates'] for z in fit_receipts),
        auxiliary_gradient_evaluations=sum(z['auxiliary_gradient_evaluations'] for z in fit_receipts),validation_scope='Previously inspected development folds; not blind, external or official submission.'))
    delivery=dict(status='guarded_pair_trial_completed_not_promoted',latest_actual='V146',issue='FINE-SUPPORT-SUPERVISION',issue_solved=False,matched_effect_passed=effect,
        quality_acceptance=accepted,model_promoted=False,classifier_fits=6,full_gradient_evaluations=sum(z['full_gradient_evaluations'] for z in fit_receipts),
        proposal_evaluations=sum(z['proposal_evaluations'] for z in fit_receipts),accepted_updates=sum(z['accepted_updates'] for z in fit_receipts),TRAIN=learning,
        ASA_errors={a:{'M':quality[a]['ASA']['B']['1']['missed'],'S':quality[a]['ASA']['B']['2']['missed']} for a in ['A','B']},
        original_pairs=dict(B_vs_A=pair(asa,'pred_A','pred_B'),B_vs_V142=pair(asa,'pred_V142','pred_B'),B_vs_A0=pair(asa,'pred_A0','pred_B')),
        known_578_S_errors_direct_support_not_created=True,automatic_extra_fits=False,new_head_fits=0,
        validation_scope=read(OUT/'verification.json')['validation_scope'],seal_sha256=sha(OUT/'run_seal.json'),quality_sha256=sha(OUT/'quality.json'),verification_sha256=sha(OUT/'verification.json'))
    save(OUT/'final_delivery.json',delivery);print(dict(status=delivery['status'],ASA_errors=delivery['ASA_errors'],matched_effect=effect,task_acceptance=accepted),flush=True)


if __name__=='__main__':main()
