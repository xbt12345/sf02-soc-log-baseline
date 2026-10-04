"""V125 independent full-population classification, controls and failure gates."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from v125_experiment_review import ROOT,sha,read,require_run_seal,require_checkpoint
from v125_model import Expert,probabilities
from v107_matched_training import FOLDS,ROWS,FID,DEST as TEACHERS
from v116_preflight import VIEW
from v124_experiment_review import evaluate_primary,class_counts

PLAN=ROOT/'training/review_policy/v125_next_training_plan.json'
OUT=ROOT/'artifacts/v125_order_trial_20260929'
TRACE=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
LADDER=ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet'
MANIFEST=ROOT/'artifacts/v116_nested_selection_20260929/inner_split_manifest.parquet'
OFFICIAL=ROOT/'data/official/train.parquet'
PARENT=ROOT/'artifacts/v124_header_trial_20260929'


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def load_model(fold,arm,seed,folder):
    fit=read(folder/'fit.json');state=torch.load(folder/'epoch25_model.pt',map_location='cpu',weights_only=True)
    if (fit['fold'],fit['arm'],fit['seed'],fit['completed_epochs'],fit['prediction_epoch'])!=(fold,arm,seed,25,25):
        raise ValueError('Wrong/incomplete fit receipt')
    if (state['fold'],state['arm'],state['seed'],state['epoch'],state['seal_sha256'])!=(fold,arm,seed,25,sha(OUT/'run_seal.json')):
        raise ValueError('Model state provenance changed')
    if fit['model_sha256']!=sha(folder/'epoch25_model.pt') or fit['prob_sha256']!=sha(folder/'epoch25_prob.npy'):
        raise ValueError('Endpoint artifact changed')
    if fit['checkpoints_sha256']!=sha(folder/'checkpoints.json') or fit['progress_sha256']!=sha(folder/'progress.json'):
        raise ValueError('Progress/checkpoint receipt changed')
    checks=read(folder/'checkpoints.json');progress=read(folder/'progress.json')
    if [q['epoch'] for q in checks]!=[1,2,5,10,15,20,25] or len(progress)!=25:
        raise ValueError('Incomplete diagnostic trajectory')
    if any(q['model_sha256']!=sha(folder/f"epoch{q['epoch']}_model.pt") or q['prob_sha256']!=sha(folder/f"epoch{q['epoch']}_prob.npy") for q in checks):
        raise ValueError('Intermediate checkpoint identity changed')
    model=Expert(arm,seed).to('cuda')
    model.base.load_state_dict(state['base'])
    if arm!='A':model.branch.load_state_dict(state['branch'])
    return fit,checks,model


def teacher_predictions(n,f):
    fid=np.load(FID,mmap_mode='r')
    if len(fid)!=n:raise ValueError('Teacher mapping incomplete')
    teacher=np.empty(n,dtype=np.int8)
    for fold in range(3):
        folder=TEACHERS/f'fold{fold}_N1_teacher'
        scores_file=folder/'scores_all_input_ids.npy'
        if sha(scores_file)!=read(folder/'fit.json')['scores_sha256']:raise ValueError('Frozen teacher changed')
        scores=np.load(scores_file,mmap_mode='r')
        ids=np.flatnonzero(f.proposed_fold.eq(fold))
        teacher[ids]=scores[np.asarray(fid[ids],np.int64)].argmax(1)
    return teacher


def root_stats(trace,p):
    s=trace.truth.eq(2).to_numpy()
    df=pd.DataFrame({'root':trace.root.to_numpy()[s],'correct':(p[s]==2).astype(np.int8)})
    a=df.groupby('root').correct.agg(['mean','sum'])
    return {'roots':len(a),'mean_recall':float(a['mean'].mean()),'zero_recall_roots':int(a['sum'].eq(0).sum())}


def comparison_slices(trace,ladder,pred):
    y=trace.truth.to_numpy();fold=trace.fold.to_numpy();root=trace.root.to_numpy()
    a,b,c=[pred[z] for z in 'ABC']
    gap=trace.row_position.isin(pd.read_parquet(PARENT/'header_span_ledger.parquet',columns=['row_position']).row_position).to_numpy()
    slices={'header_682':gap,'remaining_ASA':~gap,
        'historical_persistent_S':ladder.A_all_seven_wrong.to_numpy()&(y==2),
        'S_outside_largest3_roots':(y==2)&~np.isin(root,[21702,20849,29])}
    for name in ('known_parameter_two_roots_per_class','unknown_parameter_pooled_support',
                 'same_class_support_but_insufficient_dual_root_coverage',
                 'no_same_class_at_destination_resolution'):
        slices[name]=ladder.diagnostic_bucket.eq(name).to_numpy()
    for k in range(3):
        for cl in (1,2):slices[f'fold{k}_class{cl}']=(fold==k)&(y==cl)
    out={}
    for name,mask in slices.items():
        out[name]={'rows':int(mask.sum()),**{arm+'_errors':int(((pred[arm]!=y)&mask).sum()) for arm in 'ABC'},
            'A_to_B_repaired':int(((a!=y)&(b==y)&mask).sum()),
            'A_to_B_regressed':int(((a==y)&(b!=y)&mask).sum()),
            'A_to_C_repaired':int(((a!=y)&(c==y)&mask).sum()),
            'A_to_C_regressed':int(((a==y)&(c!=y)&mask).sum())}
    return out


def priority_fit_diagnosis(trace,ladder,prob):
    cards=read(PARENT/'capability_qualification.json')['cards']
    output={}
    for arm in 'ABC':
        roles=[]
        for fold in range(3):
            p=prob[(fold,arm)]
            for card in cards:
                if card['fold']!=fold:continue
                q=(trace.fold.ne(fold)&ladder.destination_key.eq(card['destination_key'])&(trace.truth==2))
                for r in trace.loc[q,['row_position','local','root','truth']].itertuples(index=False):
                    roles.append((int(r.row_position),int(r.root),int(p[int(r.local)].argmax())!=2))
        frame=pd.DataFrame(roles,columns=['row_position','root','error'])
        if len(frame)!=206 or frame.row_position.nunique()!=136:raise ValueError('Priority fit support count changed')
        output[arm]={'S_fit_roles':len(frame),'S_unique_original_records':frame.row_position.nunique(),
                     'S_error_roles':int(frame.error.sum()),
                     'S_error_unique_original_records':frame.loc[frame.error,'row_position'].nunique(),
                     'S_error_roots':frame.loc[frame.error,'root'].nunique()}
    return output


def main():
    plan=require_run_seal(OUT/'run_seal.json',ROOT/'training/v125_train.py')
    if plan!=read(PLAN) or (OUT/'primary_evaluation.json').exists():raise FileExistsError('V125 evaluation existing or plan changed')
    reg=read(OUT/'registration.json');x=sparse.load_npz(VIEW);lengths=np.load(OUT/'body_lengths.npy')
    bodies={'A':None,'B':np.load(OUT/'ordered_body_bytes.npy',mmap_mode='r'),
            'C':np.load(OUT/'shuffled_body_bytes.npy',mmap_mode='r')}
    trace=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    ladder=pd.read_parquet(LADDER,columns=['row_position','diagnostic_bucket','A_all_seven_wrong','destination_key'])
    assert len(trace)==112807 and trace.row_position.equals(ladder.row_position)
    pred={a:np.empty(len(trace),dtype=np.int8) for a in 'ABC'}
    prob_s={a:np.empty(len(trace),dtype=np.float32) for a in 'ABC'}
    fit_receipts=[];replays=[];all_prob={};fit_summary={}
    for fold in range(3):
        states=[]
        for arm in 'ABC':
            folder=OUT/f'fold{fold}_{arm}'
            fit,checks,model=load_model(fold,arm,10201,folder)
            require_checkpoint(plan,fit)
            if fit['seal_sha256']!=sha(OUT/'run_seal.json') or fit['optimizer_steps']!=reg['schedule'][fold]['steps_per_arm']:
                raise ValueError('Fit seal or optimizer steps changed')
            p=probabilities(model,x,bodies[arm],lengths,'cuda')
            stored=np.load(folder/'epoch25_prob.npy')
            err=float(np.max(np.abs(p-stored)))
            if not np.allclose(p,stored,atol=2e-6,rtol=2e-6) or not np.array_equal(p.argmax(1),stored.argmax(1)):
                raise ValueError('Endpoint model/probability replay failed')
            all_prob[(fold,arm)]=p
            q=trace.fold.eq(fold).to_numpy();ids=trace.local.to_numpy()[q]
            pred[arm][q]=p[ids].argmax(1);prob_s[arm][q]=p[ids,2]
            replays.append({'fold':fold,'arm':arm,'max_abs_probability_diff':err})
            fit_receipts.append({'fold':fold,'arm':arm,'fit_sha256':sha(folder/'fit.json'),
                                 'steps':fit['optimizer_steps'],'seconds':fit['seconds']})
            fit_summary[f'fold{fold}_{arm}']=checks[-1]['fit_by_class']
            states.append(model)
        # Genuine registered benign-header probe: body and facts stay fixed,
        # while the historical base input is replaced by the verified header view.
        if fold==0:
            pass
        del states
    if len(replays)!=9 or sum(z['steps'] for z in fit_receipts)!=13350:raise ValueError('Not all nine fits completed')
    expert=trace.copy()
    for arm in 'ABC':expert[f'expert_pred_{arm}']=pred[arm];expert[f'expert_S_probability_{arm}']=prob_s[arm]
    expert.to_parquet(OUT/'expert_ASA_predictions.parquet',index=False)
    official=pd.read_parquet(OFFICIAL,columns=['event_id','label_binary'])
    rows=pd.read_parquet(ROWS,columns=['row_position','event_id','label_index','route'])
    f=pd.read_parquet(FOLDS,columns=['row_position','root','proposed_fold'])
    n=len(rows)
    if len(official)!=len(f) or n!=2056871 or not np.array_equal(rows.row_position,np.arange(n)) or not np.array_equal(f.row_position,rows.row_position):raise ValueError('Full reference count/order changed')
    if not np.array_equal(official.event_id,rows.event_id):raise ValueError('Official event identity changed')
    truth=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(dtype=np.int8)
    if not np.array_equal(truth,rows.label_index.to_numpy(dtype=np.int8)):raise ValueError('Official labels changed')
    asa=np.flatnonzero(rows.route.eq('asa'))
    if not np.array_equal(asa,trace.row_position) or not np.array_equal(truth[asa],trace.truth):raise ValueError('ASA official role changed')
    if not np.array_equal(f.proposed_fold.to_numpy()[asa],trace.fold) or not np.array_equal(f.root.to_numpy()[asa],trace.root):raise ValueError('Source closure changed')
    teacher=teacher_predictions(n,f)
    if (teacher[asa]==0).sum()!=0:raise ValueError('Registered ASA teacher gate exclusions changed')
    final={arm:teacher.copy() for arm in 'ABC'}
    for arm in 'ABC':final[arm][asa]=pred[arm]
    other=~rows.route.eq('asa').to_numpy()
    if int((teacher[other]!=truth[other]).sum())!=107:raise ValueError('Non-ASA frozen errors changed')
    ledger=pd.DataFrame({'row_position':rows.row_position,'teacher_prediction':teacher,
                         'gate_open':rows.route.eq('asa').to_numpy(),
                         **{f'final_pred_{arm}':final[arm] for arm in 'ABC'}})
    ledger.to_parquet(OUT/'full_prediction_ledger.parquet',index=False)
    ref=pd.DataFrame({'row_position':rows.row_position,'truth':truth,'route':rows.route,
                      'root':f.root,'fold':f.proposed_fold})
    profile={'expected_full_rows':2056871,'asa_error_limits':{'M':318,'S':2094,'total':2170},
             'minimum_improved_folds':2,'protected_S_roots':[21702,20849,29],
             'required_full_classes':[0,1,2]}
    quality=evaluate_primary(ref,ledger.rename(columns={'final_pred_A':'pred_A','final_pred_B':'pred_B'}),profile)
    measured={'full_task':{arm:class_counts(ref.assign(pred=final[arm]),'pred') for arm in 'ABC'},
              'ASA_expert':{arm:class_counts(trace.assign(pred=pred[arm]),'pred') for arm in 'ABC'}}
    # Feature mechanism is a stricter comparison than B vs A alone.
    b=measured['ASA_expert']['B'];c=measured['ASA_expert']['C']
    byroot={arm:root_stats(trace,pred[arm]) for arm in 'ABC'}
    quality['gates'].update({
        'paired_M_S_F1_protected':all(measured['ASA_expert']['B'][str(cl)]['f1']>=measured['ASA_expert']['A'][str(cl)]['f1'] for cl in (1,2)),
        'S_group_macro_no_decrease':byroot['B']['mean_recall']>=byroot['A']['mean_recall'],
        'S_zero_recall_root_count_no_increase':byroot['B']['zero_recall_roots']<=byroot['A']['zero_recall_roots'],
        'order_control_not_refuting_claim':b['2']['missed']<c['2']['missed'] and sum(b[str(cl)]['missed'] for cl in (1,2))<sum(c[str(cl)]['missed'] for cl in (1,2)),
        'fit_nonconflict_S_improves':sum(fit_summary[f'fold{fold}_B']['2']['nonconflict_error_rows'] for fold in range(3))<sum(fit_summary[f'fold{fold}_A']['2']['nonconflict_error_rows'] for fold in range(3)),
    })
    # Exact, already-observed 682 header gap; changing to the canonical input
    # can change the old N1 branch while leaving the new body bytes unchanged.
    from v125_model import probabilities as replay_probabilities
    header=sparse.load_npz(PARENT/'B_header_ASA.npz')
    gap=trace.row_position.isin(pd.read_parquet(PARENT/'header_span_ledger.parquet',columns=['row_position']).row_position).to_numpy()
    header_counts=[]
    for fold in range(3):
        _,_,model=load_model(fold,'B',10201,OUT/f'fold{fold}_B')
        p=replay_probabilities(model,header,bodies['B'],lengths,'cuda')
        q=(trace.fold.eq(fold).to_numpy()&gap);old=pred['B'][q];new=p[trace.local.to_numpy()[q]].argmax(1)
        header_counts.append({'fold':fold,'rows':int(q.sum()),'prediction_flips':int((old!=new).sum()),
            'old_errors':int((old!=trace.truth.to_numpy()[q]).sum()),
            'canonical_header_errors':int((new!=trace.truth.to_numpy()[q]).sum())})
    quality['gates']['registered_header_variant_invariance']=sum(z['prediction_flips'] for z in header_counts)==0
    quality['primary_quality_passed']=all(quality['gates'].values())
    slices=comparison_slices(trace,ladder,pred)
    priority=priority_fit_diagnosis(trace,ladder,all_prob)
    source=trace.assign(**{arm+'_wrong':pred[arm]!=trace.truth.to_numpy() for arm in 'ABC'}).groupby(['root','truth']).agg(
        rows=('local','size'),A_errors=('A_wrong','sum'),B_errors=('B_wrong','sum'),C_errors=('C_wrong','sum')).reset_index()
    source.to_csv(OUT/'source_group_changes.csv',index=False)
    result={'status':'nine_arms_source_closed_evaluated','latest_actual_training':'V125',
        'classifier_fits_new':9,'optimizer_steps':sum(z['steps'] for z in fit_receipts),
        'calibration_fits':0,'confirmation_fits':0,'full_rows':n,'ASA_rows':len(trace),
        'complete_official_class_support':{str(cl):int((truth==cl).sum()) for cl in (0,1,2)},
        'fit_receipts':fit_receipts,'model_replays':replays,'checkpoint_fit_class_25':fit_summary,
        'metrics':measured,'quality':quality,'S_group_results':byroot,'registered_header_probe':header_counts,
        'slices':slices,'priority_fit':priority,'frozen_non_ASA_errors':107,
        'primary_quality_passed':bool(quality['primary_quality_passed']),
        'confirmation_allowed':bool(quality['primary_quality_passed']),
        'quality_acceptance':False,'model_promoted':False,
        'limitations':['Development folds have been repeatedly inspected; no new independent blind test.',
            'C shuffling is a destructive mechanistic control, not a valid label-preserving synthetic example.',
            'N1 base can still depend on the old syslog header; residual branch alone does not prove full-model invariance.',
            'No proof that available sanitized fields suffice to determine every organizer M/S label.']}
    save(OUT/'primary_evaluation.json',result)
    print(json.dumps({'stage':result['status'],'ASA_errors':{a:{cl:measured['ASA_expert'][a][cl]['missed'] for cl in ('1','2')} for a in 'ABC'},
                      'gates':quality['gates'],'confirmation_allowed':result['confirmation_allowed']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
