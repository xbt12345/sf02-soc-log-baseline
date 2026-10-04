"""Independent official-row quality and saved-model/window replay. No fitting."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_review import class_counts,evaluate_primary
from v134_runtime import (ROOT,OUT,PLAN,ARMS,read,save,sha,require_run_seal,load_data,
    fit_context,stats,guard_sources,window_pass,endpoint,confirmation_allowed)
from v131_model import Classifier,infer,tensor_hash
from v131_evaluate import load_reference,full_folds,teacher_predictions
from v125_evaluate import root_stats


def model(state):
    m=Classifier(128).to('cuda');m.load_state_dict(state['model']);return m


def verify_fit(folder,x,d,kind='primary'):
    r=read(folder/'fit.json');endpoint(r)
    f,c,pure,_,ids=fit_context(d,r['fold']);stored=np.load(folder/'train_ids.npy')
    if not np.array_equal(ids,stored):raise ValueError('TRAIN identity changed')
    for file,key in [('progress.json','progress_sha256'),('steps.jsonl','steps_sha256'),('checkpoints.json','checkpoints_sha256')]:
        if sha(folder/file)!=r[key]:raise ValueError('Fit journal changed '+file)
    cp=read(folder/'checkpoints.json');p=read(PLAN)
    if [z['epoch'] for z in cp]!=p['record']['fixed_checkpoints']:raise ValueError('Missing fixed checkpoints')
    for z in cp:
        if sha(folder/f"epoch{z['epoch']}_model.pt")!=z['model_sha256'] or sha(folder/f"epoch{z['epoch']}_train_prob.npy")!=z['train_prob_sha256']:
            raise ValueError('Checkpoint hash changed')
    if sha(folder/'sealed_all_prob.npy')!=cp[-1]['all_prob_sha256']:raise ValueError('Sealed full probability changed')
    prior=pd.read_parquet(ROOT/'artifacts/v132_training_mastery_review_20260930/training_mastery_ledger.parquet')
    prior=prior[prior.arm.eq('R')&prior.outer_fit_role.eq(r['fold'])]
    if not np.array_equal(prior.row_position,f.row_position):raise ValueError('Reference role mismatch')
    old=prior.pred.to_numpy();expected_source=f.groupby(['root','truth']).size().rename('support')
    history=read(folder/'progress.json');population=0
    if [h['epoch'] for h in history]!=list(range(1,101)):raise ValueError('Missing per-epoch histories')
    for h in history:
        e=h['epoch'];path=folder/'epochs'/f'epoch{e:03d}_rows.parquet';sfile=folder/'epochs'/f'epoch{e:03d}_sources.parquet'
        if sha(path)!=h['rows_sha256'] or sha(sfile)!=h['sources_sha256'] or sha(folder/'epochs'/f'epoch{e:03d}_member_pred.npy')!=h['member_sha256']:
            raise ValueError('Epoch accounting changed')
        rows=pd.read_parquet(path)
        if not np.array_equal(rows.row_position,f.row_position) or not np.array_equal(rows.truth,f.truth) or rows.row_position.duplicated().any():
            raise ValueError('Epoch original truth/row mapping mismatch')
        q=rows[['p0','p1','p2']].to_numpy()
        if not np.array_equal(q.argmax(1),rows.pred) or not np.array_equal(rows.pure_TRAIN_input,pure[f.local].astype(bool)):
            raise ValueError('Epoch decisions/pure definition mismatch')
        if h['class_mass_seen']!=c.sum(0).tolist():raise ValueError('Epoch exposure mismatch')
        src=pd.read_parquet(sfile);guard_sources(src,expected_source,e)
        own=rows.assign(wrong=rows.pred!=rows.truth).groupby(['root','truth']).wrong.sum().sort_index()
        if not own.equals(src.set_index(['root','truth']).errors.sort_index()):raise ValueError('Source error accounting mismatch')
        population+=len(rows)
    counts=np.zeros(3,np.int64);step=0;permutation=hashlib.sha256();rng=np.random.default_rng(r['seed']+r['fold'])
    for _ in range(100):permutation.update(ids[rng.permutation(len(ids))].astype('<i8').tobytes())
    if permutation.hexdigest()!=r['permutation_sha256']:raise ValueError('Batch order drift')
    for line in (folder/'steps.jsonl').read_text(encoding='utf-8').splitlines():
        s=json.loads(line);step+=1
        if s['step']!=step or not s['finite']:raise ValueError('Broken optimizer journal')
        counts[1]+=s['M_original_rows'];counts[2]+=s['S_original_rows']
    if step!=r['expected_steps'] or not np.array_equal(counts,100*c.sum(0)):raise ValueError('Full original exposure mismatch')
    replays=[];endq=None;newhistory=[]
    for e in [96,97,98,99,100]:
        state=torch.load(folder/f'epoch{e}_model.pt',map_location='cpu',weights_only=True)
        if (state['epoch'],state['arm'],state['fold'],state['seed'],state['seal_sha256'])!=(e,r['arm'],r['fold'],r['seed'],sha(OUT/'run_seal.json')):
            raise ValueError('Window model identity changed')
        m=model(state);q,*_=infer(m,x,ids);saved=np.load(folder/f'epoch{e}_train_prob.npy')
        gap=float(np.abs(q-saved).max())
        if gap>2e-6 or not np.array_equal(q.argmax(1),saved.argmax(1)):raise ValueError('Window TRAIN model replay failed')
        full=np.zeros((len(c),3),np.float32);full[ids]=q;s=stats(f,full,pure,old)
        if s!=history[e-1]['stats']:raise ValueError('Window classification stats mismatch')
        replays.append({'epoch':e,'max_abs':gap,'stats':s})
        if e==100:
            endq,*_=infer(m,x,np.arange(len(c)));saved=np.load(folder/'sealed_all_prob.npy')
            gap=float(np.abs(endq-saved).max())
            if gap>2e-6 or not np.array_equal(endq.argmax(1),saved.argmax(1)):raise ValueError('All-input replay failed')
            replays[-1]['all_input_max_abs']=gap
        del m;gc.collect();torch.cuda.empty_cache()
    stable=all(z['stats']['mastered'] for z in replays)
    if stable!=r['fixed_window_mastered'] or stable!=window_pass(history):raise ValueError('Window qualification mismatch')
    state=torch.load(folder/'epoch0_model.pt',map_location='cpu',weights_only=True)
    if tensor_hash(state['model'])!=r['initial_state_sha256']:raise ValueError('Initial identity changed')
    return r,{'folder':folder.name,'mastered':stable,'window_replays':replays,'audited_epoch_original_rows':population,
              'optimizer_steps':step,'M_exposures':int(counts[1]),'S_exposures':int(counts[2]),'fit_sha256':sha(folder/'fit.json')},endq


def quality_for(ref,d,pred,base_full,ladder,headers):
    q=evaluate_primary(ref,pd.DataFrame({'row_position':ref.row_position,'pred_A':base_full,'pred_B':pred}),
        {'expected_full_rows':2056871,'asa_error_limits':{'M':318,'S':2074,'total':2170},
         'minimum_improved_folds':2,'protected_S_roots':[21702,20849,29],'required_full_classes':[0,1,2]})
    asa=pred[d.row_position];y=d.truth.to_numpy();sources=root_stats(d,asa)
    full=class_counts(ref.assign(pred=pred),'pred');base=class_counts(ref.assign(pred=base_full),'pred')
    out=(y==2)&~d.root.isin([21702,20849,29]).to_numpy()
    unk=ladder.diagnostic_bucket.eq('unknown_parameter_pooled_support').to_numpy()&(y==2)
    head=d.row_position.isin(headers).to_numpy();r2868=d.root.eq(2868).to_numpy()&(y==1)
    q['gates'].update(full_precision_recall_F1_protected=all(full[str(cl)][metric]>=base[str(cl)][metric] for cl in (0,1,2) for metric in ('precision','recall','f1')),
        S_source_mean_recall=sources['mean_recall']>=.1151252713,S_zero_recall_roots=sources['zero_recall_roots']<=198,
        S_outside_top3=int((asa[out]!=2).sum())<1373,unknown186_S=int((asa[unk]!=2).sum())<=184,
        header682=int((asa[head]!=y[head]).sum())==0,root2868_M=int((asa[r2868]!=1).sum())<=48,
        non_ASA_frozen_errors=int((pred[ref.route.ne('asa')]!=ref.truth[ref.route.ne('asa')]).sum())==107)
    q['quality_passed']=all(q['gates'].values());q['S_sources']=sources
    return q


def primary():
    require_run_seal(ROOT/'training/v134_train.py');torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4)
    if (OUT/'delivery.json').exists():raise FileExistsError('Preserve delivered results')
    x,d=load_data();ref=load_reference(d);predictions={a:{} for a in ARMS};verified=[];fits=[];training={a:[] for a in ARMS}
    for f in range(3):
        initials=[];orders=[]
        for arm in ARMS:
            r,v,q=verify_fit(OUT/f'fold{f}_{arm}',x,d);fits.append(r);verified.append(v)
            training[arm].append({'fold':f,'mastered':v['mastered'],'endpoint':v['window_replays'][-1]['stats'],
                                   'window_errors':[{'epoch':z['epoch'],'M':z['stats']['M_errors'],'S':z['stats']['S_errors']} for z in v['window_replays']]})
            predictions[arm][f]=q;initials.append(r['initial_state_sha256']);orders.append(r['permutation_sha256'])
        if len(set(initials))!=1 or len(set(orders))!=1:raise ValueError('Factorial initialization/order not matched')
    qualify={'status':'all_role_TRAIN_qualification_frozen_before_new_HELD_read','task_candidate':'R_decay','arms':training,
        'all_role_mastered':{a:all(v['mastered'] for v in values) for a,values in training.items()},
        'R_decay_all_roles_mastered':all(v['mastered'] for v in training['R_decay']),
        'new_held_read_before_qualification':False,'plan_sha256':sha(PLAN),'model_promoted':False}
    save(OUT/'learning_qualification.json',qualify)
    # No new HELD prediction is joined with its truth until TRAIN qualification above is written.
    teacher=teacher_predictions(len(ref),full_folds());base=np.empty(len(d),np.int8)
    for f in range(3):
        mask=d.fold.eq(f).to_numpy();q=np.load(ROOT/f'artifacts/v125_order_trial_20260929/fold{f}_A/epoch25_prob.npy')
        base[mask]=q[d.loc[mask,'local']].argmax(1)
    base_full=teacher.copy();base_full[d.row_position]=base
    ladder=pd.read_parquet(ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet')
    if not np.array_equal(ladder.row_position,d.row_position):raise ValueError('Slice reference misalignment')
    headers=set(pd.read_parquet(ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet').row_position)
    asa_ledger=d[['row_position','local','root','fold','truth']].copy();asa_ledger['pred_A0']=base
    ledger=pd.DataFrame({'row_position':ref.row_position,'pred_A0':base_full});quality={};source=[]
    for a in ARMS:
        pred=teacher.copy();v=np.empty(len(d),np.int8)
        for f,q in predictions[a].items():
            mask=d.fold.eq(f).to_numpy();v[mask]=q[d.loc[mask,'local']].argmax(1)
        pred[d.row_position]=v;ledger['pred_'+a]=pred;asa_ledger['pred_'+a]=v
        quality[a]=quality_for(ref,d,pred,base_full,ladder,headers)
        src=d.assign(wrong=v!=d.truth).groupby(['root','truth']).wrong.agg(['size','sum']).reset_index();src['arm']=a;source.append(src)
    matched=evaluate_primary(ref,pd.DataFrame({'row_position':ref.row_position,'pred_A':ledger.pred_R_const,'pred_B':ledger.pred_R_decay}),
        {'expected_full_rows':2056871,'asa_error_limits':{'M':318,'S':2074,'total':2170},'minimum_improved_folds':2,
         'protected_S_roots':[21702,20849,29],'required_full_classes':[0,1,2]})
    a,b=matched['ASA']['A'],matched['ASA']['B']
    mg={'M_errors_do_not_increase':b['1']['missed']<=a['1']['missed'],'S_errors_do_not_increase':b['2']['missed']<=a['2']['missed'],
        'at_least_one_class_strictly_improves':any(b[str(c)]['missed']<a[str(c)]['missed'] for c in (1,2)),
        'at_least_two_folds_improve':sum(z['B_errors']<z['A_errors'] for z in matched['folds'])>=2}
    passed=qualify['R_decay_all_roles_mastered'] and quality['R_decay']['quality_passed'] and all(mg.values())
    save(OUT/'quality.json',{'candidate':'R_decay','candidate_quality_passed':bool(passed),'arms':quality,
                          'matched_control':matched,'matched_control_gates':mg,'no_other_arm_promotion':True})
    asa_ledger.to_parquet(OUT/'ASA_prediction_ledger.parquet',index=False);ledger.to_parquet(OUT/'full_prediction_ledger.parquet',index=False)
    pd.concat(source).to_csv(OUT/'source_errors.csv',index=False)
    save(OUT/'verification.json',{'status':'actual_window_models_and_full_original_population_verified',
        'model_replays':verified,'fits':len(fits),'optimizer_steps':sum(f['optimizer_steps'] for f in fits),
        'official_original_rows':len(ref),'ASA_original_rows':len(d),'validation_scope':'Actual96-100 models, all100 original-row/source epochs, class mass and independent full official truth; development data, not blind acceptance.'})
    delivery={'status':'v134_primary_completed_quality_passed_pending_confirmation' if passed else 'v134_primary_completed_quality_failed',
        'execution_version':'V134','plan_version':read(PLAN)['version'],'classifier_fits':len(fits),'optimizer_steps':sum(f['optimizer_steps'] for f in fits),
        'candidate':'R_decay','quality_acceptance':bool(passed),'model_promoted':False,'learning':qualify,
        'metrics':{a:{'ASA':q['ASA']['B'],'full_task':q['full_task']['B'],'S_sources':q['S_sources']} for a,q in quality.items()},
        'quality':quality,'matched_control_gates':mg,'verification_sha256':sha(OUT/'verification.json'),
        'plan_sha256':sha(PLAN),'seal_sha256':sha(OUT/'run_seal.json'),
        'limitations':['Previously inspected source folds, no independent blind environment or official submission.',
                       'Only ASA refitted; other formats use the verified frozen component.',
                       'TRAIN mastery is not security truth or stable cross-source M/S semantics.']}
    save(OUT/'delivery.json',delivery)
    print(json.dumps({'status':delivery['status'],'fits':len(fits),'updates':delivery['optimizer_steps'],
        'learning':qualify['all_role_mastered'],'ASA_errors':{a:{c:q['ASA']['B'][c]['missed'] for c in ('1','2')} for a,q in quality.items()},'quality':bool(passed)},ensure_ascii=False),flush=True)


if __name__=='__main__':primary()
