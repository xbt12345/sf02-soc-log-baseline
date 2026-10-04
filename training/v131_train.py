"""Execute the fixed V130 learning qualification plan. No held-based selection."""
import argparse
import gc
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import tabm

from v131_common import (ROOT,PLAN,CONTRACT,OUT,REVIEW,INPUT,TRACE,OFFICIAL,MANIFEST,
    read,save,check_plan,seal_run,require_run_seal,require_checkpoint,load_data,fit_context,learning_gates,sha)
from v131_model import (Classifier,ActiveFirst,make_model,batch,objective,infer,tensor_hash,module_norms)

DEVICE='cuda'
ARMS={'R':(128,False),'O':(128,True),'C':(256,False),'CO':(256,True)}


def emit(**kw):
    print(json.dumps(kw,ensure_ascii=False),flush=True)


def sources():
    import v131_evaluate as ev
    paths={Path(__file__),PLAN,CONTRACT,INPUT,TRACE,OFFICIAL,MANIFEST,
        REVIEW/'audit.json',REVIEW/'verification.json',REVIEW/'training_role_error_ledger.parquet',
        REVIEW/'tiny64_train_only_inputs.parquet',REVIEW/'fold1_train_only_panel.parquet',
        ROOT/'training/test_v131_training.py',ROOT/'training/v131_evaluate.py',
        ROOT/'artifacts/v124_header_trial_20260929/header_span_ledger.parquet',
        ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet',
        ROOT/'artifacts/v128_nested_score_trial_20260929_r3/delivery.json',
        ev.ROWS,ev.FOLDS,ev.FID,Path(tabm.__file__)}
    for f in range(3):
        paths.update([ROOT/f'artifacts/v125_order_trial_20260929/fold{f}_A/epoch25_prob.npy',
            ROOT/f'artifacts/v124_header_trial_20260929/fold{f}_B/epoch25_prob.npy',
            ev.TEACHERS/f'fold{f}_N1_teacher/scores_all_input_ids.npy',
            ev.TEACHERS/f'fold{f}_N1_teacher/fit.json'])
    for mod in list(sys.modules.values()):
        fn=getattr(mod,'__file__',None)
        if fn:
            p=Path(fn).resolve()
            if p.suffix=='.py' and p.is_file() and p.is_relative_to(ROOT) and not p.relative_to(ROOT).parts[0].startswith('.venv'):
                paths.add(p)
    return paths


def configure():
    if not torch.cuda.is_available():raise RuntimeError('Registered local CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4)


def register():
    if OUT.exists():raise FileExistsError('Preserve existing training run: '+str(OUT))
    p=check_plan();configure();x,d=load_data()
    ledger=pd.read_parquet(REVIEW/'training_role_error_ledger.parquet')
    import hashlib
    x.sort_indices()
    keys=[hashlib.sha256(x.indices[x.indptr[i]:x.indptr[i+1]].astype('<i8').tobytes()+
        x.data[x.indptr[i]:x.indptr[i+1]].astype('<f4').tobytes()).hexdigest() for i in range(x.shape[0])]
    if not np.array_equal(d.canonical_key,d.local.map(dict(enumerate(keys)))):
        raise ValueError('Actual numerical input identity mismatch')
    schedules=[]
    for f in range(3):
        fit,c,pure,totals,used=fit_context(d,f)
        if not np.array_equal(fit.row_position,ledger.loc[ledger.outer_fit_role==f,'row_position']):
            raise ValueError('Training reference roles changed')
        schedules.append({'fold':f,'fit_rows':len(fit),'class_mass':c.sum(0).tolist(),
            'pure_class_mass':totals.tolist(),'logical_batches':math.ceil(len(used)/256),
            'optimizer_steps_per_fit':100*math.ceil(len(used)/256)})
    probe=pd.read_parquet(REVIEW/'tiny64_train_only_inputs.parquet')
    r=make_model(128,DEVICE).eval();w=make_model(256,DEVICE).eval()
    ids=probe.local.to_numpy(np.int64)
    with torch.no_grad():
        zr=batch(r,x,ids);zw=batch(w,x,ids)
    initial=float((zr-zw).abs().max())
    if initial>1e-6:raise ValueError('Wide/small initial outputs mismatch '+str(initial))
    # Compare the implemented computation against the historical exact full
    # sparse first layer on the real selected inputs, both forward and gradient.
    from v104_phase_b import SparseTabM,csr_tensor
    torch.manual_seed(10201);legacy=SparseTabM().to(DEVICE)
    if tensor_hash(r.state_dict())!=tensor_hash(legacy.state_dict()):raise ValueError('R initialization drift')
    block=x[ids[:8]]
    from v75_views import BYTE_FEATURES
    xt=csr_tensor(block,DEVICE);fact=torch.as_tensor(block[:,BYTE_FEATURES:].toarray(),device=DEVICE)
    loss=[];grad=[];out=[]
    for m in [r,legacy]:
        m.zero_grad(set_to_none=True)
        z=m(xt,fact);val=-torch.log_softmax(z,-1)[:,:,2].mean();val.backward()
        out.append(z.detach().cpu().numpy());loss.append(float(val.detach()))
        grad.append({k:v.grad.detach().cpu().numpy().copy() for k,v in m.named_parameters()})
    forward=float(np.max(np.abs(out[0]-out[1])))
    gradient=max(float(np.max(np.abs(grad[0][k]-grad[1][k]))) for k in grad[0])
    if max(forward,abs(loss[0]-loss[1]),gradient)>1e-6:raise ValueError('Sparse algebra mismatch')
    del r,w,legacy,grad;gc.collect();torch.cuda.empty_cache()
    import v131_evaluate as ev
    full=ev.load_reference(d)
    teacher=ev.teacher_predictions(len(full),ev.full_folds())
    outside=full.route.ne('asa').to_numpy()
    if int((teacher[outside]!=full.truth.to_numpy()[outside]).sum())!=107:
        raise ValueError('Frozen non-ASA population changed')
    OUT.mkdir()
    package={'torch':torch.__version__,'tabm':importlib.metadata.version('tabm'),
        'numpy':np.__version__,'scipy':importlib.metadata.version('scipy'),'pandas':pd.__version__,
        'device':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
        'allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    save(OUT/'preflight.json',{'status':'passed_before_any_update','schedules':schedules,
        'full_rows':len(full),'ASA_rows':len(d),'probe_rows':len(probe),
        'wide_initial_max_abs':initial,'active_algebra_forward_max_abs':forward,
        'active_algebra_gradient_max_abs':gradient,'packages':package,
        'class_collision_counterexample':read(REVIEW/'loss_collision_counterexamples.json')})
    seal_run(Path(__file__),sources()|{OUT/'preflight.json'})
    save(OUT/'registration.json',{'status':'registered_before_any_optimizer_step','plan_sha256':sha(PLAN),
        'seal_sha256':sha(OUT/'run_seal.json'),'completed_fits':0,'optimizer_steps':0,
        'package':package,'quality_acceptance':False,'model_promoted':False})
    emit(stage='registered',schedules=schedules,initial_gap=initial,sparse_forward_gap=forward,sparse_gradient_gap=gradient)


def train_stats(frame,p,pure,old,hard_rows,matched_rows):
    pred=p[frame.local.to_numpy(np.int64)].argmax(1)
    y=frame.truth.to_numpy();pure_row=pure[frame.local.to_numpy(np.int64)].astype(bool)
    oldpred=old[frame.local.to_numpy(np.int64)].argmax(1)
    hard=frame.row_position.isin(hard_rows).to_numpy();matched=frame.row_position.isin(matched_rows).to_numpy()
    return {'M_errors':int(((y==1)&(pred!=1)).sum()),'S_errors':int(((y==2)&(pred!=2)).sum()),
        'pure_S_errors':int(((y==2)&(pred!=2)&pure_row).sum()),
        'hard_S_errors':int(((pred!=2)&hard).sum()),'matched_M_errors':int(((pred!=1)&matched).sum()),
        'old_correct_S_regressions':int(((y==2)&(oldpred==2)&(pred!=2)).sum()),
        'hard_error_roots':int(frame.loc[hard&(pred!=2),'root'].nunique())}


def gradient_diagnosis(model,x,c,pure,totals,used):
    model.eval();result={}
    for weighting in ['original','pure_class']:
        previous=None;norms={}
        for cl in [1,2]:
            model.zero_grad(set_to_none=True)
            denom=c.sum() if weighting=='original' else totals[cl]
            for i in range(0,len(used),256):
                ids=used[i:i+256];mass=c[ids,cl].astype(np.float32)
                if weighting=='pure_class':mass*=pure[ids]
                if not mass.any():continue
                ce=-torch.log_softmax(batch(model,x,ids),-1).mean(1)[:,cl]
                (ce*torch.as_tensor(mass,device=DEVICE)/denom).sum().backward()
            norms[str(cl)]=module_norms(model,gradient=True)
            grads={k:(v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v)) for k,v in model.named_parameters()}
            if previous is not None:
                cos={}
                for name,_ in model.named_children():
                    names=[k for k in grads if k.startswith(name+'.')]
                    dot=sum((previous[k]*grads[k]).sum() for k in names)
                    a=sum(previous[k].square().sum() for k in names).sqrt()
                    b=sum(grads[k].square().sum() for k in names).sqrt()
                    cos[name]=float(dot/(a*b)) if float(a*b)>0 else None
                result[weighting]={'norms':norms,'M_S_cosine':cos}
            previous=grads
        del previous,grads
    model.zero_grad(set_to_none=True)
    return result


def run_fit(kind,fold,arm):
    p=require_run_seal(Path(__file__));configure()
    reg=read(OUT/'registration.json')
    if reg['seal_sha256']!=sha(OUT/'run_seal.json'):raise ValueError('Run identity changed')
    folder=OUT/(('probe_'+arm) if kind=='probe' else f'fold{fold}_{arm}')
    if folder.exists():raise FileExistsError('Do not overwrite/resume a partial fit '+str(folder))
    if kind=='primary':
        decision=read(OUT/'probe_decision.json')
        if not decision['full_trial_allowed']:raise ValueError('Both probes failed; full fitting blocked')
        for before in p['primary_trial']['fit_order'][:p['primary_trial']['fit_order'].index(arm)]:
            require_checkpoint(read(OUT/f'fold1_{before}/fit.json'),'primary')
    if kind=='confirmation':
        selection=read(OUT/'learning_selection.json')
        selected=selection['selected_arm']
        if not selected or arm not in [selected,'R'] or fold not in [0,2]:
            raise ValueError('Unauthorized confirmation')
    x,d=load_data();fit,c,pure,totals,used=fit_context(d,fold)
    if kind=='probe':
        probe=pd.read_parquet(REVIEW/'tiny64_train_only_inputs.parquet')
        c=np.zeros_like(c)
        c[probe.local,probe.truth]=probe.original_row_mass
        used=probe.local.to_numpy(np.int64);pure=np.ones(len(c),np.float32)
        totals=c.sum(0).astype(np.float64)
        width=128 if arm=='R128' else 256
        focused=False;epochs=2000;nb=1;checkpoints=set(p['tiny_probe']['checkpoint_updates']);wd=0.0
    else:
        width,focused=ARMS[arm];epochs=100;nb=math.ceil(len(used)/256)
        checkpoints=set(p['primary_trial']['checkpoints_epochs']);wd=.0003
    expected=epochs*nb
    model=make_model(width,DEVICE);initial=tensor_hash(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=wd)
    rng=np.random.default_rng(10201+fold)
    old=np.load(ROOT/f'artifacts/v124_header_trial_20260929/fold{fold}_B/epoch25_prob.npy')
    panel=pd.read_parquet(REVIEW/'fold1_train_only_panel.parquet')
    hard=set(panel.loc[panel.truth==2,'row_position']);matched=set(panel.loc[panel.truth==1,'row_position'])
    folder.mkdir();np.save(folder/'train_ids.npy',used)
    save(folder/'started.json',{'kind':kind,'fold':fold,'arm':arm,'width':width,
        'focused':focused,'initial_state_sha256':initial,'expected_steps':expected,
        'class_mass':c.sum(0).tolist(),'pure_class_mass':totals.tolist(),
        'seal_sha256':sha(OUT/'run_seal.json'),'held_labels_used_for_gradient':0})
    log=open(folder/'steps.jsonl','w',encoding='utf-8',buffering=1)
    history=[];saved=[];gradient_history=[];start=time.monotonic();steps=0
    def snapshot(epoch,diagnostic):
        state={'kind':kind,'fold':fold,'arm':arm,'hidden':width,'epoch':epoch,'optimizer_steps':steps,
            'model':model.state_dict(),'initial_state_sha256':initial,'seal_sha256':sha(OUT/'run_seal.json')}
        torch.save(state,folder/f'epoch{epoch}_model.pt')
        train_prob=diagnostic['prob']
        np.save(folder/f'epoch{epoch}_train_prob.npy',train_prob)
        saved.append({'epoch':epoch,'optimizer_steps':steps,'stats':diagnostic['stats'],
            'model_sha256':sha(folder/f'epoch{epoch}_model.pt'),
            'train_prob_sha256':sha(folder/f'epoch{epoch}_train_prob.npy')})
        if kind!='probe':
            if epoch in [0,25,50,100]:
                gradient_history.append({'epoch':epoch,**gradient_diagnosis(model,x,c,pure,totals,used)})
                save(folder/'gradient_diagnosis.json',gradient_history)
            # Predictions are produced but no held truth or quality is read.
            all_prob,*_=infer(model,x,np.arange(len(c)))
            np.save(folder/f'epoch{epoch}_sealed_all_prob.npy',all_prob)
            saved[-1]['sealed_probability_sha256']=sha(folder/f'epoch{epoch}_sealed_all_prob.npy')
        save(folder/'checkpoints.json',saved)
    def diagnose():
        prob,no_member,margins,member_sum,zeros=infer(model,x,used,c)
        full=np.zeros((len(c),3),np.float32);full[used]=prob
        mass=c[used].astype(np.float64)
        per={str(cl):{'support':int(c[:,cl].sum()),
            'errors':int((mass[:,cl]*(prob.argmax(1)!=cl)).sum()),
            'no_correct_member_rows':int((mass[:,cl]*no_member[:,cl]).sum()),
            'member_CE':float(member_sum[cl]/c[:,cl].sum()),
            'ensemble_CE':float(-(mass[:,cl]*np.log(np.maximum(prob[:,cl],1e-12))).sum()/c[:,cl].sum())}
            for cl in [1,2]}
        if kind=='probe':
            truth=probe.truth.to_numpy();stats={'M_errors':per['1']['errors'],'S_errors':per['2']['errors'],
                'worst_truth_probability':float(prob[np.arange(len(probe)),truth].min())}
        else:stats=train_stats(fit,full,pure,old,hard,matched)
        return {'prob':prob,'per_class':per,'stats':stats,'activation_zero_fraction':zeros}
    diag=diagnose();snapshot(0,diag)
    for epoch in range(1,epochs+1):
        if kind!='probe' or epoch in checkpoints:require_run_seal(Path(__file__))
        model.train();seen=np.zeros_like(c);numrow=0.;numaux=0.
        order=used if kind=='probe' else used[rng.permutation(len(used))]
        batches=[order] if kind=='probe' else [order[i:i+256] for i in range(0,len(order),256)]
        for ids in batches:
            seen[ids]+=c[ids];opt.zero_grad(set_to_none=True)
            z=batch(model,x,ids)
            mass=torch.as_tensor(c[ids],device=DEVICE,dtype=torch.float32)
            single=torch.as_tensor(pure[ids],device=DEVICE)
            loss,row,aux=objective(z,mass,single,float(c.sum()),totals,nb,focused)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite objective')
            loss.backward()
            if any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in model.parameters()):
                raise FloatingPointError('Nonfinite parameter gradient')
            gradient=module_norms(model,gradient=True)
            before={k:v.detach().clone() for k,v in model.named_parameters()}
            opt.step();update=module_norms(model,before=before);del before
            steps+=1;numrow+=float(row.detach());numaux+=float(aux.detach())
            log.write(json.dumps({'step':steps,'epoch':epoch,'lr':.002,
                'M_original_rows':int(c[ids,1].sum()),'S_original_rows':int(c[ids,2].sum()),
                'batch_scaled_objective':float(loss.detach()),'row_risk_contribution':float(row.detach()),
                'pure_class_risk_contribution':float(aux.detach()),'gradient_norm':gradient,
                'update_norm':update,'finite':True})+'\n')
        if not np.array_equal(seen,c) or steps!=epoch*nb:raise ValueError('Original-row coverage/step conservation failed')
        # Full fit-role classification every epoch; never a held metric.
        if kind!='probe' or epoch in checkpoints:
            diag=diagnose();entry={'epoch':epoch,'optimizer_steps':steps,
                'class_mass_seen':seen.sum(0).tolist(),'row_risk_online':numrow,
                'pure_class_risk_online':numaux,'per_class':diag['per_class'],'stats':diag['stats'],
                'activation_zero_fraction':diag['activation_zero_fraction'],'seconds':time.monotonic()-start}
            history.append(entry);save(folder/'progress.json',history)
            if epoch in checkpoints:snapshot(epoch,diag)
            if kind=='probe' or epoch%10==0 or epoch in [1,2,5]:
                emit(stage='fit_progress',kind=kind,fold=fold,arm=arm,epoch=epoch,steps=steps,
                    stats=diag['stats'],seconds=round(time.monotonic()-start,1))
            if kind!='probe':
                pred=diag['prob'].argmax(1);full=np.full(len(c),-1,np.int8);full[used]=pred
                src=fit.assign(wrong=full[fit.local.to_numpy()]!=fit.truth.to_numpy()).groupby(['root','truth']).wrong.agg(['size','sum'])
                src.to_csv(folder/'current_training_source_errors.csv')
    log.close()
    result={'status':'fit_executed','kind':kind,'fold':fold,'arm':arm,'hidden':width,
        'endpoint':epochs,'optimizer_steps':steps,'expected_steps':expected,
        'fit_class_mass':c.sum(0).tolist(),'initial_state_sha256':initial,'seconds':time.monotonic()-start,
        'endpoint_training':diag['stats'],'endpoint_per_class':diag['per_class'],
        'seal_sha256':sha(OUT/'run_seal.json'),'model_promoted':False,
        'model_sha256':sha(folder/f'epoch{epochs}_model.pt'),
        'train_prob_sha256':sha(folder/f'epoch{epochs}_train_prob.npy'),
        'steps_sha256':sha(folder/'steps.jsonl'),'progress_sha256':sha(folder/'progress.json'),
        'checkpoints_sha256':sha(folder/'checkpoints.json')}
    require_checkpoint(result,kind);save(folder/'fit.json',result)
    emit(stage='fit_complete',kind=kind,fold=fold,arm=arm,steps=steps,stats=diag['stats'],seconds=round(result['seconds'],1))
    del model,opt;gc.collect();torch.cuda.empty_cache()


def probes():
    require_run_seal(Path(__file__))
    for arm in ['R128','C256']:run_fit('probe',1,arm)
    r={}
    for arm in ['R128','C256']:
        receipt=read(OUT/f'probe_{arm}/fit.json');require_checkpoint(receipt,'probe')
        s=receipt['endpoint_training'];r[arm]=s['M_errors']==s['S_errors']==0 and s['worst_truth_probability']>=.9
    save(OUT/'probe_decision.json',{'status':'probes_completed','per_model_passed':r,
        'full_trial_allowed':any(r.values()),'classifier_fits':2,'optimizer_steps':4000,
        'quality_acceptance':False,'model_promoted':False})
    emit(stage='probe_decision',passes=r,full_trial_allowed=any(r.values()))


def primary():
    if not read(OUT/'probe_decision.json')['full_trial_allowed']:
        emit(stage='primary_blocked',reason='Both actual-input probes failed');return
    for arm in ['R','O','C','CO']:run_fit('primary',1,arm)


def select():
    require_run_seal(Path(__file__));configure();x,d=load_data();fit,c,pure,_,used=fit_context(d,1)
    panel=pd.read_parquet(REVIEW/'fold1_train_only_panel.parquet')
    old=np.load(ROOT/'artifacts/v124_header_trial_20260929/fold1_B/epoch25_prob.npy')
    hard=set(panel.loc[panel.truth==2,'row_position']);matched=set(panel.loc[panel.truth==1,'row_position'])
    result={};ledger=fit[['row_position','local','root','truth']].copy()
    for arm in ['R','O','C','CO']:
        folder=OUT/f'fold1_{arm}';receipt=read(folder/'fit.json');require_checkpoint(receipt,'primary')
        if sha(folder/'epoch100_model.pt')!=receipt['model_sha256']:raise ValueError('Endpoint changed')
        state=torch.load(folder/'epoch100_model.pt',map_location='cpu',weights_only=True)
        model=make_model(ARMS[arm][0],DEVICE);model.load_state_dict(state['model'])
        replay,*_=infer(model,x,used)
        saved=np.load(folder/'epoch100_train_prob.npy')
        gap=float(np.max(np.abs(replay-saved)))
        if gap>2e-6 or not np.array_equal(replay.argmax(1),saved.argmax(1)):raise ValueError('Training model replay failed')
        full=np.zeros((len(c),3),np.float32);full[used]=replay
        stats=train_stats(fit,full,pure,old,hard,matched);gates=learning_gates(stats)
        result[arm]={'stats':stats,'gates':gates,'learning_qualified':all(gates.values()),
            'train_replay_max_abs':gap,'fit_sha256':sha(folder/'fit.json')}
        ledger['pred_'+arm]=full[ledger.local.to_numpy()].argmax(1)
        del model;gc.collect();torch.cuda.empty_cache()
    selected=next((arm for arm in ['R','C','O','CO'] if result[arm]['learning_qualified']),None)
    ledger.to_parquet(OUT/'primary_training_prediction_ledger.parquet',index=False)
    save(OUT/'learning_selection.json',{'status':'training_only_selection_recorded','selected_arm':selected,
        'registered_preference':['R','C','O','CO'],'arms':result,'new_held_results_read_before_selection':False,
        'plan_sha256':sha(PLAN),'seal_sha256':sha(OUT/'run_seal.json'),
        'confirmation_allowed':selected is not None,'quality_acceptance':False,'model_promoted':False})
    emit(stage='learning_selection',selected=selected,arms=result)


def confirm():
    selection=read(OUT/'learning_selection.json');arm=selection['selected_arm']
    if arm is None:emit(stage='confirmation_blocked',reason='No arm met TRAIN qualification');return
    for fold in [0,2]:
        for name in (['R'] if arm=='R' else ['R',arm]):run_fit('confirmation',fold,name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','probes','primary','select','confirm'])
    stage=parser.parse_args().stage
    try:globals()[stage]()
    except Exception as e:
        if OUT.exists():save(OUT/f'{stage}_failure.json',{'status':'execution_failed','stage':stage,
            'error':str(e),'traceback':traceback.format_exc(),'model_promoted':False})
        raise


if __name__=='__main__':main()
