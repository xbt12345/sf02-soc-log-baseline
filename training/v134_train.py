"""Execute the sealed V133 2x2 factorial. All model parameters updated only here."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
import tabm

from v134_runtime import (ROOT,PLAN,CONTRACT,OUT,ARMS,INPUT,TRACE,OFFICIAL,REVIEW,
    read,save,sha,check_plan,seal_run,require_run_seal,load_data,fit_context,
    learning_rate,stats,guard_sources,window_pass,endpoint,confirmation_allowed)
from v131_model import Classifier,batch,objective,tensor_hash,module_norms,infer
from v131_train import gradient_diagnosis
from v131_evaluate import load_reference,full_folds,teacher_predictions,ROWS,FOLDS,FID,TEACHERS

DEVICE='cuda'


def emit(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)


def configure():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4)


def model_for(seed):
    torch.manual_seed(seed)
    return Classifier(128).to(DEVICE)


def sources():
    import v134_evaluate
    paths={PLAN,CONTRACT,INPUT,TRACE,OFFICIAL,ROWS,FOLDS,FID,Path(tabm.__file__),
           REVIEW/'training_role_error_ledger.parquet',ROOT/'training/test_v134_training.py',
           ROOT/'artifacts/v132_training_mastery_review_20260930/training_mastery_ledger.parquet',
           ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet',
           ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet'}
    for rel in read(PLAN)['evidence_sha256']:paths.add(ROOT/rel)
    for f in range(3):
        paths.update([ROOT/f'artifacts/v125_order_trial_20260929/fold{f}_A/epoch25_prob.npy',
                      TEACHERS/f'fold{f}_N1_teacher/scores_all_input_ids.npy',TEACHERS/f'fold{f}_N1_teacher/fit.json'])
    for mod in list(sys.modules.values()):
        fn=getattr(mod,'__file__',None)
        if fn:
            q=Path(fn).resolve()
            if q.suffix=='.py' and q.is_file() and q.is_relative_to(ROOT) and not q.relative_to(ROOT).parts[0].startswith('.venv'):
                paths.add(q)
    return paths


def detailed_infer(model,x,ids,c):
    model.eval();probs=[];members=[];member_sum=np.zeros(3);zeros=np.zeros(2);total=np.zeros(2)
    with torch.no_grad():
        for start in range(0,len(ids),256):
            take=ids[start:start+256];z,h1,h2=batch(model,x,take,True)
            probs.append(torch.softmax(z,-1).mean(1).cpu().numpy())
            members.append(z.argmax(-1).cpu().numpy().astype(np.int8))
            lp=torch.log_softmax(z,-1).mean(1).cpu().numpy()
            member_sum-=np.sum(c[take]*lp,axis=0)
            for j,h in enumerate([h1,h2]):zeros[j]+=int((h==0).sum());total[j]+=h.numel()
    return np.concatenate(probs),np.concatenate(members),member_sum,(zeros/total).tolist()


def register():
    if OUT.exists():raise FileExistsError('Preserve prior run')
    p=check_plan();configure();x,d=load_data();x.sort_indices()
    keys=[hashlib.sha256(x.indices[x.indptr[i]:x.indptr[i+1]].astype('<i8').tobytes()+
                        x.data[x.indptr[i]:x.indptr[i+1]].astype('<f4').tobytes()).hexdigest() for i in range(x.shape[0])]
    if not np.array_equal(d.canonical_key,d.local.map(dict(enumerate(keys)))):raise ValueError('Numerical input map mismatch')
    schedules=[];oldledger=pd.read_parquet(ROOT/'artifacts/v132_training_mastery_review_20260930/training_mastery_ledger.parquet')
    for f in range(3):
        fit,c,pure,totals,used=fit_context(d,f);role=p['roles'][f]
        if len(fit)!=role['original_fit_rows'] or c.sum(0).tolist()!=[0,role['M'],role['S']]:raise ValueError('TRAIN mass changed')
        old=oldledger[oldledger.arm.eq('R')&oldledger.outer_fit_role.eq(f)]
        if not np.array_equal(old.row_position,fit.row_position):raise ValueError('Old role identity changed')
        dummy=np.zeros((len(c),3));dummy[:,1]=1
        floor=stats(fit,dummy,pure,old.pred.to_numpy())['empirical_minimum_errors']
        if floor!=role['empirical_collision_floor']:raise ValueError('Numerical conflict population changed')
        rng=np.random.default_rng(10201+f);h=hashlib.sha256()
        for _ in range(100):h.update(used[rng.permutation(len(used))].astype('<i8').tobytes())
        schedules.append({'fold':f,'fit_rows':len(fit),'class_mass':c.sum(0).tolist(),
            'pure_class_mass':totals.tolist(),'logical_batches':math.ceil(len(used)/256),
            'expected_steps':100*math.ceil(len(used)/256),'floor':floor,'permutation_sha256':h.hexdigest()})
    ref=load_reference(d);teacher=teacher_predictions(len(ref),full_folds());outside=ref.route.ne('asa').to_numpy()
    if int((teacher[outside]!=ref.truth.to_numpy()[outside]).sum())!=107:raise ValueError('Non-ASA frozen population changed')
    model=model_for(10201);old=torch.load(ROOT/'artifacts/v131_learning_trial_20260930/fold1_R/epoch0_model.pt',map_location='cpu',weights_only=True)
    if tensor_hash(model.state_dict())!=tensor_hash(old['model']):raise ValueError('Shared initialization changed')
    fit,c,_,_,used=fit_context(d,1);a,m,_,_=detailed_infer(model,x,used[:8],c);b,*_=infer(model,x,used[:8])
    if not np.array_equal(a.argmax(1),b.argmax(1)) or float(np.abs(a-b).max())>1e-6:raise ValueError('Inference algebra changed')
    size=sum(v.numel()*v.element_size() for v in model.state_dict().values())
    estimated=size*3*len(p['record']['fixed_checkpoints'])*15+3*2**30
    if shutil.disk_usage(ROOT).free<estimated:raise RuntimeError('Insufficient registered snapshot storage')
    packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','pyarrow','tabm']}
    packages.update(torch=torch.__version__,device=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
                    tf32=torch.backends.cuda.matmul.allow_tf32,threads=torch.get_num_threads(),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    OUT.mkdir();save(OUT/'preflight.json',{'status':'passed_before_any_update','roles':schedules,
        'initial_state_sha256':tensor_hash(model.state_dict()),'packages':packages,
        'estimated_snapshot_bytes_upper':estimated,'free_bytes':shutil.disk_usage(ROOT).free,
        'official_rows':len(ref),'input_shape':list(x.shape),'no_training_updates':True})
    del model;gc.collect();torch.cuda.empty_cache()
    seal_run(__file__,sources()|{OUT/'preflight.json'})
    save(OUT/'registration.json',{'status':'registered_before_update','plan_sha256':sha(PLAN),
        'seal_sha256':sha(OUT/'run_seal.json'),'fits':0,'updates':0,'candidate':'R_decay','model_promoted':False})
    emit(stage='registered',roles=schedules,device=packages['device'])


def run_fit(fold,arm,seed=10201,kind='primary'):
    p=require_run_seal(__file__);configure()
    if kind=='confirmation' and (arm!='R_decay' or seed!=13701 or not confirmation_allowed(read(OUT/'learning_qualification.json'),read(OUT/'quality.json'))):
        raise ValueError('Confirmation not authorized by actual quality')
    folder=OUT/(f'fold{fold}_{arm}' if kind=='primary' else f'seed{seed}_fold{fold}_{arm}')
    if folder.exists():raise FileExistsError('Never resume/overwrite a partial fit')
    x,d=load_data();fit,c,pure,totals,used=fit_context(d,fold);nb=math.ceil(len(used)/256)
    prior=pd.read_parquet(ROOT/'artifacts/v132_training_mastery_review_20260930/training_mastery_ledger.parquet')
    prior=prior[prior.arm.eq('R')&prior.outer_fit_role.eq(fold)]
    if not np.array_equal(prior.row_position,fit.row_position):raise ValueError('Regression reference changed')
    old=prior.pred.to_numpy();expected_source=fit.groupby(['root','truth']).size().rename('support')
    model=model_for(seed);initial=tensor_hash(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    rng=np.random.default_rng(seed+fold);permutation=hashlib.sha256();folder.mkdir();np.save(folder/'train_ids.npy',used)
    (folder/'epochs').mkdir()
    save(folder/'started.json',{'kind':kind,'fold':fold,'arm':arm,'seed':seed,'initial_state_sha256':initial,
        'fit_rows':len(fit),'class_mass':c.sum(0).tolist(),'pure_class_mass':totals.tolist(),
        'expected_steps':100*nb,'seal_sha256':sha(OUT/'run_seal.json'),'held_labels_used_for_gradient':0})
    history=[];checkpoints=[];grads=[];steps=0;start=time.monotonic()
    log=(folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1)
    def diagnose(epoch):
        q,member,ce,zeros=detailed_infer(model,x,used,c)
        full=np.zeros((len(c),3),np.float32);full[used]=q
        s=stats(fit,full,pure,old);loc=fit.local.to_numpy();y=fit.truth.to_numpy();pr=pure[loc].astype(bool)
        other=full[loc].copy();other[np.arange(len(fit)),y]=-np.inf
        ledger=fit[['row_position','local','root','truth']].copy();ledger['epoch']=epoch
        ledger['pure_TRAIN_input']=pr;ledger['pred']=full[loc].argmax(1)
        for cl in range(3):ledger[f'p{cl}']=full[loc,cl]
        ledger['probability_margin']=full[loc,y]-other.max(1)
        all_member=np.zeros((len(c),16),np.int8);all_member[used]=member
        ledger['correct_member_count']=(all_member[loc]==y[:,None]).sum(1).astype(np.int8)
        ledger['old_V131_R_pred']=old
        path=folder/'epochs'/f'epoch{epoch:03d}_rows.parquet'
        if path.exists():raise FileExistsError('Overwritten original-row epoch')
        ledger.to_parquet(path,index=False)
        np.save(folder/'epochs'/f'epoch{epoch:03d}_member_pred.npy',member)
        wrong=ledger.pred.to_numpy()!=y
        tmp=ledger.assign(wrong=wrong,pure_wrong=wrong&pr,repair=(old!=y)&~wrong,regression=(old==y)&wrong)
        src=tmp.groupby(['root','truth']).agg(support=('wrong','size'),errors=('wrong','sum'),
            pure_errors=('pure_wrong','sum'),repairs=('repair','sum'),regressions=('regression','sum')).reset_index()
        src['epoch']=epoch;guard_sources(src,expected_source,epoch)
        source_path=folder/'epochs'/f'epoch{epoch:03d}_sources.parquet'
        if source_path.exists():raise FileExistsError('Overwritten source epoch')
        src.to_parquet(source_path,index=False)
        per={str(cl):{'support':int(c[:,cl].sum()),'errors':s['M_errors' if cl==1 else 'S_errors'],
            'member_CE':float(ce[cl]/c[:,cl].sum()),
            'ensemble_CE':float(-(c[used,cl]*np.log(np.maximum(q[:,cl],1e-12))).sum()/c[:,cl].sum()),
            'no_correct_member_rows':int((c[used,cl]*(member!=cl).all(1)).sum())} for cl in (1,2)}
        return dict(prob=q,stats=s,per_class=per,activation_zero_fraction=zeros,
                    rows_sha256=sha(path),sources_sha256=sha(source_path),member_sha256=sha(folder/'epochs'/f'epoch{epoch:03d}_member_pred.npy'))
    def snapshot(epoch,diag):
        torch.save({'kind':kind,'fold':fold,'arm':arm,'seed':seed,'epoch':epoch,'optimizer_steps':steps,
            'model':model.state_dict(),'optimizer':opt.state_dict(),'lr':opt.param_groups[0]['lr'],
            'initial_state_sha256':initial,'seal_sha256':sha(OUT/'run_seal.json')},folder/f'epoch{epoch}_model.pt')
        np.save(folder/f'epoch{epoch}_train_prob.npy',diag['prob'])
        cp={'epoch':epoch,'optimizer_steps':steps,'model_sha256':sha(folder/f'epoch{epoch}_model.pt'),
            'train_prob_sha256':sha(folder/f'epoch{epoch}_train_prob.npy'),'stats':diag['stats']}
        if epoch in p['record']['global_M_S_gradient_diagnosis_epochs']:
            grads.append({'epoch':epoch,**gradient_diagnosis(model,x,c,pure,totals,used)});save(folder/'gradient_diagnosis.json',grads)
        if epoch==100:
            allq,*_=infer(model,x,np.arange(len(c)));np.save(folder/'sealed_all_prob.npy',allq)
            cp['all_prob_sha256']=sha(folder/'sealed_all_prob.npy')
        checkpoints.append(cp);save(folder/'checkpoints.json',checkpoints)
    diag=diagnose(0);snapshot(0,diag)
    for epoch in range(1,101):
        require_run_seal(__file__);lr=learning_rate(arm,epoch)
        for group in opt.param_groups:group['lr']=lr
        model.train();seen=np.zeros_like(c);rowrisk=auxrisk=0.
        order=used[rng.permutation(len(used))];permutation.update(order.astype('<i8').tobytes())
        for start_id in range(0,len(order),256):
            ids=order[start_id:start_id+256];seen[ids]+=c[ids];opt.zero_grad(set_to_none=True)
            z=batch(model,x,ids);mass=torch.as_tensor(c[ids],device=DEVICE,dtype=torch.float32)
            single=torch.as_tensor(pure[ids],device=DEVICE)
            loss,row,aux=objective(z,mass,single,float(c.sum()),totals,nb,arm.startswith('O'))
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in model.parameters()):raise FloatingPointError('Nonfinite gradient')
            gradient=module_norms(model,gradient=True);before={k:v.detach().clone() for k,v in model.named_parameters()}
            opt.step();update=module_norms(model,before=before);del before
            steps+=1;rowrisk+=float(row.detach());auxrisk+=float(aux.detach())
            log.write(json.dumps({'step':steps,'epoch':epoch,'lr':lr,'M_original_rows':int(c[ids,1].sum()),
                'S_original_rows':int(c[ids,2].sum()),'row_risk_contribution':float(row.detach()),
                'pure_class_risk_contribution':float(aux.detach()),'batch_objective':float(loss.detach()),
                'gradient_norm':gradient,'update_norm':update,'finite':True})+'\n')
        if not np.array_equal(seen,c) or steps!=epoch*nb:raise ValueError('Original-row coverage failed')
        diag=diagnose(epoch);record={k:v for k,v in diag.items() if k!='prob'}
        record.update(epoch=epoch,optimizer_steps=steps,lr=lr,class_mass_seen=seen.sum(0).tolist(),
                      row_risk_online=rowrisk,pure_class_risk_online=auxrisk,seconds=time.monotonic()-start)
        history.append(record);save(folder/'progress.json',history)
        if epoch in p['record']['fixed_checkpoints']:snapshot(epoch,diag)
        if epoch%10==0 or epoch in [1,2,5,96,97,98,99]:
            emit(stage='fit_progress',kind=kind,fold=fold,arm=arm,epoch=epoch,steps=steps,lr=lr,stats=diag['stats'],seconds=round(time.monotonic()-start,1))
    log.close();require_run_seal(__file__)
    result={'status':'fit_executed','kind':kind,'fold':fold,'arm':arm,'seed':seed,'completed_epochs':100,
        'prediction_epoch':100,'optimizer_steps':steps,'expected_steps':100*nb,'initial_state_sha256':initial,
        'permutation_sha256':permutation.hexdigest(),'class_mass':c.sum(0).tolist(),
        'fixed_window_mastered':window_pass(history),'endpoint_training':diag['stats'],
        'seconds':time.monotonic()-start,'fit_rows':len(fit),'seal_sha256':sha(OUT/'run_seal.json'),
        'progress_sha256':sha(folder/'progress.json'),'steps_sha256':sha(folder/'steps.jsonl'),
        'checkpoints_sha256':sha(folder/'checkpoints.json'),'model_promoted':False}
    endpoint(result);save(folder/'fit.json',result)
    emit(stage='fit_complete',kind=kind,fold=fold,arm=arm,steps=steps,mastered=result['fixed_window_mastered'],stats=diag['stats'],seconds=round(result['seconds'],1))
    del model,opt;gc.collect();torch.cuda.empty_cache()


def primary():
    for fold in range(3):
        for arm in ARMS:run_fit(fold,arm)


def confirm():
    if not confirmation_allowed(read(OUT/'learning_qualification.json'),read(OUT/'quality.json')):
        save(OUT/'confirmation_decision.json',{'status':'not_run_quality_or_learning_failed','fits':0,'updates':0,'model_promoted':False})
        emit(stage='confirmation_blocked');return
    for f in range(3):run_fit(f,'R_decay',13701,'confirmation')
    save(OUT/'confirmation_decision.json',{'status':'registered_three_fits_completed','fits':3,'updates':17800,'model_promoted':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','primary','confirm']);stage=parser.parse_args().stage
    try:globals()[stage]()
    except Exception as e:
        if OUT.exists():save(OUT/f'{stage}_failure.json',{'status':'execution_failed','stage':stage,'error':str(e),'traceback':traceback.format_exc(),'model_promoted':False})
        raise


if __name__=='__main__':main()
