"""Independent actual-model replay and full official-row V131 evaluation."""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_review import class_counts,evaluate_primary,check_bindings
from v131_common import (ROOT,OUT,PLAN,REVIEW,TRACE,OFFICIAL,read,save,load_data,
    fit_context,require_run_seal,require_checkpoint,sha,learning_gates)
from v131_model import make_model,infer
from v107_matched_training import ROWS,FOLDS,FID,DEST as TEACHERS
from v125_evaluate import teacher_predictions,root_stats


def full_folds():
    return pd.read_parquet(FOLDS,columns=['row_position','root','proposed_fold'])


def load_reference(trace):
    official=pd.read_parquet(OFFICIAL,columns=['event_id','label_binary'])
    rows=pd.read_parquet(ROWS,columns=['row_position','event_id','label_index','route'])
    folds=full_folds()
    n=len(official)
    if n!=2056871 or len(rows)!=n or len(folds)!=n:raise ValueError('Incomplete full reference')
    if not np.array_equal(rows.row_position,np.arange(n)) or not np.array_equal(folds.row_position,rows.row_position):
        raise ValueError('Wrong full-row identity')
    if not np.array_equal(official.event_id,rows.event_id):raise ValueError('Event identity mismatch')
    y=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(np.int8)
    if not np.array_equal(y,rows.label_index):raise ValueError('Independent label mismatch')
    asa=np.flatnonzero(rows.route.eq('asa'))
    if not np.array_equal(asa,trace.row_position) or not np.array_equal(y[asa],trace.truth):
        raise ValueError('ASA routing count changed')
    if not np.array_equal(folds.proposed_fold.to_numpy()[asa],trace.fold) or not np.array_equal(folds.root.to_numpy()[asa],trace.root):
        raise ValueError('ASA source closure changed')
    return pd.DataFrame({'row_position':rows.row_position,'truth':y,'route':rows.route,
        'root':folds.root,'fold':folds.proposed_fold})


def verify_fit(folder,x,kind):
    receipt=read(folder/'fit.json');require_checkpoint(receipt,kind)
    ep=receipt['endpoint'];cp=read(folder/'checkpoints.json')
    expected=[0,1,10,100,500,1000,2000] if kind=='probe' else [0,1,2,5,10,15,20,25,35,50,75,100]
    if [v['epoch'] for v in cp]!=expected:raise ValueError('Missing registered checkpoints')
    for c in cp:
        for name,key in [(f"epoch{c['epoch']}_model.pt",'model_sha256'),(f"epoch{c['epoch']}_train_prob.npy",'train_prob_sha256')]:
            if sha(folder/name)!=c[key]:raise ValueError('Checkpoint changed '+str(folder/name))
        if kind!='probe' and sha(folder/f"epoch{c['epoch']}_sealed_all_prob.npy")!=c['sealed_probability_sha256']:
            raise ValueError('Sealed predictions changed')
    for filename,key in [('steps.jsonl','steps_sha256'),('progress.json','progress_sha256'),('checkpoints.json','checkpoints_sha256')]:
        if sha(folder/filename)!=receipt[key]:raise ValueError('Trajectory identity changed')
    state=torch.load(folder/f'epoch{ep}_model.pt',map_location='cpu',weights_only=True)
    if (state['kind'],state['epoch'],state['optimizer_steps'],state['seal_sha256'])!=(
        kind,ep,receipt['optimizer_steps'],sha(OUT/'run_seal.json')):raise ValueError('Endpoint state identity changed')
    if sha(folder/f'epoch{ep}_model.pt')!=receipt['model_sha256']:raise ValueError('Endpoint hash changed')
    model=make_model(receipt['hidden'],'cuda');model.load_state_dict(state['model'])
    ids=np.load(folder/'train_ids.npy');train_prob,*_=infer(model,x,ids)
    stored=np.load(folder/f'epoch{ep}_train_prob.npy')
    gap=float(np.max(np.abs(train_prob-stored)))
    if gap>2e-6 or not np.array_equal(train_prob.argmax(1),stored.argmax(1)):raise ValueError('Fit-probability replay failed')
    countsM=countsS=0;lines=0
    for line in (folder/'steps.jsonl').read_text(encoding='utf-8').splitlines():
        v=json.loads(line);lines+=1
        if v['step']!=lines or v['finite'] is not True:raise ValueError('Bad optimizer journal')
        countsM+=v['M_original_rows'];countsS+=v['S_original_rows']
    if lines!=receipt['optimizer_steps']:raise ValueError('Journal steps do not match fit')
    if (countsM,countsS)!=(receipt['fit_class_mass'][1]*ep,receipt['fit_class_mass'][2]*ep):
        raise ValueError('Original-class exposure mass mismatch')
    summary={'folder':folder.name,'kind':kind,'optimizer_steps':lines,'train_replay_max_abs':gap,
        'original_M_exposures':countsM,'original_S_exposures':countsS,'fit_sha256':sha(folder/'fit.json')}
    all_prob=None
    if kind!='probe':
        all_prob,*_=infer(model,x,np.arange(x.shape[0]))
        stored=np.load(folder/f'epoch{ep}_sealed_all_prob.npy')
        gap=float(np.max(np.abs(all_prob-stored)))
        if gap>2e-6 or not np.array_equal(all_prob.argmax(1),stored.argmax(1)):raise ValueError('All-input model replay failed')
        summary['all_replay_max_abs']=gap
    del model;gc.collect();torch.cuda.empty_cache()
    return receipt,summary,train_prob,all_prob


def main():
    trainer=ROOT/'training/v131_train.py';require_run_seal(trainer)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4)
    if (OUT/'delivery.json').exists():raise FileExistsError('Preserve prior delivery')
    x,d=load_data();ref=load_reference(d);folds=full_folds()
    verified=[];fits=[];primary={};probe_replays={}
    for arm in ['R128','C256']:
        receipt,replay,q,_=verify_fit(OUT/('probe_'+arm),x,'probe')
        probe=pd.read_parquet(REVIEW/'tiny64_train_only_inputs.parquet')
        truth=probe.truth.to_numpy();pred=q.argmax(1)
        stat={'M_errors':int(((truth==1)&(pred!=1)).sum()),
            'S_errors':int(((truth==2)&(pred!=2)).sum()),
            'worst_truth_probability':float(q[np.arange(64),truth].min())}
        if stat!=receipt['endpoint_training']:raise ValueError('Probe decision metadata mismatch')
        probe_replays[arm]=stat;verified.append(replay);fits.append(receipt)
    allowed=any(v['M_errors']==v['S_errors']==0 and v['worst_truth_probability']>=.9 for v in probe_replays.values())
    decision=read(OUT/'probe_decision.json')
    if allowed!=decision['full_trial_allowed']:raise ValueError('Probe gate wrong')
    selection={'selected_arm':None,'arms':{}}
    if allowed:
        selection=read(OUT/'learning_selection.json')
        # Choice must already exist before any newly held output is inspected.
        if selection['new_held_results_read_before_selection'] is not False:raise ValueError('Held-based selection')
        for arm in ['R','O','C','CO']:
            receipt,replay,trainq,allq=verify_fit(OUT/f'fold1_{arm}',x,'primary')
            verified.append(replay);fits.append(receipt);primary[arm]={1:allq}
            fit,c,pure,_,ids=fit_context(d,1)
            # Recompute every TRAIN gate from original rows, not claimed truth.
            panel=pd.read_parquet(REVIEW/'fold1_train_only_panel.parquet')
            y=fit.truth.to_numpy();loc=fit.local.to_numpy();pred=allq[loc].argmax(1)
            old=np.load(ROOT/'artifacts/v124_header_trial_20260929/fold1_B/epoch25_prob.npy')[loc].argmax(1)
            hard=fit.row_position.isin(panel.loc[panel.truth==2,'row_position']).to_numpy()
            matched=fit.row_position.isin(panel.loc[panel.truth==1,'row_position']).to_numpy()
            stats={'M_errors':int(((y==1)&(pred!=1)).sum()),'S_errors':int(((y==2)&(pred!=2)).sum()),
                'pure_S_errors':int(((y==2)&(pred!=2)&pure[loc].astype(bool)).sum()),
                'hard_S_errors':int((hard&(pred!=2)).sum()),'matched_M_errors':int((matched&(pred!=1)).sum()),
                'old_correct_S_regressions':int(((y==2)&(old==2)&(pred!=2)).sum()),
                'hard_error_roots':int(fit.loc[hard&(pred!=2),'root'].nunique())}
            expected=selection['arms'][arm]
            if stats!=expected['stats'] or learning_gates(stats)!=expected['gates']:raise ValueError('TRAIN gate mismatch')
        selected=next((a for a in ['R','C','O','CO'] if all(selection['arms'][a]['gates'].values())),None)
        if selected!=selection['selected_arm']:raise ValueError('Unregistered arm selection')
        if selected:
            for f in [0,2]:
                for arm in (['R'] if selected=='R' else ['R',selected]):
                    receipt,replay,_,allq=verify_fit(OUT/f'fold{f}_{arm}',x,'confirmation')
                    verified.append(replay);fits.append(receipt)
                    primary.setdefault(arm,{})[f]=allq
    selected=selection['selected_arm']; uniform=selected is not None
    teacher=teacher_predictions(len(ref),folds);asa=ref.route.eq('asa').to_numpy()
    if int((teacher[~asa]!=ref.truth.to_numpy()[~asa]).sum())!=107:raise ValueError('Frozen non-ASA drift')
    base=np.empty(len(d),np.int8)
    for f in range(3):
        mask=d.fold.eq(f).to_numpy();ids=d.loc[mask,'local'].to_numpy()
        q=np.load(ROOT/f'artifacts/v125_order_trial_20260929/fold{f}_A/epoch25_prob.npy')
        base[mask]=q[ids].argmax(1)
    base_full=teacher.copy();base_full[d.row_position]=base
    output=d[['row_position','local','root','fold','truth']].copy();output['pred_A0']=base
    full_ledger=pd.DataFrame({'row_position':ref.row_position,'pred_A0':base_full})
    metrics={};evaluations={};source=[]
    ladder=pd.read_parquet(ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet')
    headers=set(pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet').row_position)
    candidates=primary if allowed else {}
    profile={'expected_full_rows':2056871,'asa_error_limits':{'M':318,'S':2074,'total':2170},
        'minimum_improved_folds':2,'protected_S_roots':[21702,20849,29],'required_full_classes':[0,1,2]}
    for arm,predictions in candidates.items():
        pred=base.copy()
        for f,q in predictions.items():
            mask=d.fold.eq(f).to_numpy();pred[mask]=q[d.loc[mask,'local'].to_numpy()].argmax(1)
        output['pred_'+arm]=pred
        complete=teacher.copy();complete[d.row_position]=pred;full_ledger['pred_'+arm]=complete
        quality=evaluate_primary(ref,pd.DataFrame({'row_position':ref.row_position,
            'pred_A':base_full,'pred_B':complete}),profile)
        byroot=root_stats(d,pred);s=d.truth.eq(2).to_numpy();y=d.truth.to_numpy()
        outside=s&~d.root.isin([21702,20849,29]).to_numpy()
        unknown=ladder.diagnostic_bucket.eq('unknown_parameter_pooled_support').to_numpy()&s
        header=d.row_position.isin(headers).to_numpy();root2868=d.root.eq(2868).to_numpy()&(y==1)
        fullmetric=class_counts(ref.assign(pred=complete),'pred')
        basemetric=class_counts(ref.assign(pred=base_full),'pred')
        gates=quality['gates']
        gates.update({'uniform_three_fold_method_completed':len(predictions)==3,
            'TRAIN_qualified_before_held':uniform and arm==selected,
            'full_precision_recall_F1_protected':all(fullmetric[str(c)][m]>=basemetric[str(c)][m]
                for c in [0,1,2] for m in ['precision','recall','f1']),
            'S_source_mean_recall':byroot['mean_recall']>=.1151252713,
            'S_zero_recall_roots':byroot['zero_recall_roots']<=198,
            'S_outside_top3':int((pred[outside]!=2).sum())<1373,
            'unknown186_S':int((pred[unknown]!=2).sum())<=184,
            'header682':int((pred[header]!=y[header]).sum())==0,
            'root2868_M':int((pred[root2868]!=1).sum())<=48})
        metrics[arm]={'full_task':fullmetric,'ASA':class_counts(d.assign(pred=pred),'pred'),
            'S_sources':byroot,'newly_trained_folds':sorted(predictions),
            'scope':'uniform three-fold development model' if len(predictions)==3 else 'hybrid diagnosis: new fold1 only, old A0 on other folds'}
        evaluations[arm]={'gates':gates,'quality_passed':all(gates.values()),
            'paired_changes':quality['paired_changes'],'folds':quality['folds']}
        groups=d.assign(error=pred!=y).groupby(['root','truth']).error.agg(['size','sum']).reset_index()
        groups['arm']=arm;source.append(groups)
    output.to_parquet(OUT/'ASA_prediction_ledger.parquet',index=False)
    full_ledger.to_parquet(OUT/'full_prediction_ledger.parquet',index=False)
    if source:pd.concat(source).to_csv(OUT/'source_errors.csv',index=False)
    quality=bool(selected and evaluations[selected]['quality_passed'])
    fitcount=len(fits);steps=sum(f['optimizer_steps'] for f in fits)
    verification={'status':'actual_endpoint_replay_and_original_row_accounting_verified',
        'all_checks_passed':True,'model_replays':verified,'fit_count':fitcount,'optimizer_steps':steps,
        'official_original_rows':len(ref),'ASA_rows':len(d),'non_ASA_frozen_errors':107,
        'scope':'Endpoint model replay, checkpoint identities, optimizer journal and independently read official labels; not independent blind validation.'}
    save(OUT/'verification.json',verification)
    result={'status':'v131_completed_quality_passed_pending_independent_confirmation' if quality else
        ('v131_completed_learning_failed_quality_failed' if allowed and not selected else
         ('v131_completed_learning_passed_quality_failed' if selected else 'v131_probes_failed_full_training_blocked')),
        'plan_version':'V130-learning-qualification','execution_version':'V131',
        'classifier_fits':fitcount,'optimizer_steps':steps,'probe_fits':2,
        'primary_fits':4 if allowed else 0,'additional_source_fits':fitcount-(6 if allowed else 2),
        'calibration_fits':0,'probe_replays':probe_replays,'learning':selection,
        'selected_arm':selected,'uniform_candidate_completed':uniform,
        'metrics':metrics,'quality':evaluations,'quality_acceptance':quality,'model_promoted':False,
        'original_full_rows':len(ref),'ASA_rows':len(d),'non_ASA_frozen_errors':107,
        'verification_sha256':sha(OUT/'verification.json'),'plan_sha256':sha(PLAN),
        'seal_sha256':sha(OUT/'run_seal.json'),
        'limitations':['Previously observed development folds; no new independent blind data.',
            'Training memorization is not a stable cross-source M/S semantic criterion.',
            'If TRAIN gates fail, full-row counts are explicitly hybrid diagnoses, not a uniform candidate.',
            'No official unlabeled validation submission or external dataset used.']}
    save(OUT/'delivery.json',result)
    print(json.dumps({'status':result['status'],'fits':fitcount,'updates':steps,'selected':selected,
        'ASA_errors':{a:{c:metrics[a]['ASA'][c]['missed'] for c in ['1','2']} for a in metrics},
        'quality_acceptance':quality},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
