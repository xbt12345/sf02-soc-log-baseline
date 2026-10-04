"""Conditional fixed-epoch rotation of the preselected v87 arm and A0 control.

Primary source stays frozen. The objective, model, solver and pair eligibility
are shared with the registered primary implementation; only the fold-local data,
teacher and output directory change. No new epoch/arm selection occurs here.
"""
import copy
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,read,save,sha
import v87r2_execute as primary
from v85_protection import (baseline_objective,Residual,csr_tensor,raw_counts,cm_from_counts,
                           changes,margin,supervised_loss,protection_penalty,inference,ALPHA,sha_tensor_state)
from v81_training_contract import compare

DEST=primary.DEST


def register():
    reg=primary.check();cont=read(DEST/'continuation.json');chosen=cont['primary_selected']
    assert chosen is not None and cont['rotation_authorized']
    target=DEST/'rotation';target.mkdir(exist_ok=True)
    path=target/'registration.json'
    if path.exists():return read(path)
    spec={'source_sha256':sha(__file__),'primary_registration_sha256':sha(DEST/'registration.json'),
          'primary_selection_sha256':sha(DEST/'continuation.json'),'winner':chosen,'folds':[3,4],
          'arms':sorted(set(['A0',chosen['arm']])),'fixed_epoch':chosen['epoch'],
          'teacher_solver':'Same cold-start fold-local L-BFGS OVR objective, alpha1e-6, maxiter1000/gtol1e-6/ftol1e-12/maxcor10; no held-out training.',
          'shared_model_objective_solver_pair_policy':reg['source_sha256'],'new_selection':False,
          'continuation':'Both rotation folds at fixed primary epoch must pass fit/inner zero regression and true repair. C/H diagnostics do not select parameters. Otherwise stop full final fitting.',
          'C_H_used_for_hyperparameters':False,'support_withdrawal_refits_after_rotation_pass_only':True}
    save(path,spec);primary.emit(stage='rotation_registered',**spec);return spec


def setup(h,spec):
    r,y,fid,x,cc,full,vc,fit,inner,selected=primary.data(h)
    folder=DEST/'rotation'/f'fold{h}';folder.mkdir(exist_ok=True)
    if (folder/'teacher_fit.json').exists():return r,y,fid,x,cc,full,vc,fit,inner,selected,folder
    used=np.flatnonzero(cc.sum(1));xx=x[used];counts=cc[used]
    v=np.zeros((x.shape[1]+1)*3);v[-3:]=np.log(np.maximum(counts.sum(0)/counts.sum(),1e-9))
    start=time.monotonic()
    result=minimize(baseline_objective,v,args=(xx,counts),jac=True,method='L-BFGS-B',
                    options={'maxiter':1000,'gtol':1e-6,'ftol':1e-12,'maxcor':10,'maxls':30})
    assert result.success and abs(result.jac).max()<=1e-5
    w=result.x.reshape(x.shape[1]+1,3);teacher={'coef':w[:-1].copy(),'intercept':w[-1].copy()}
    joblib.dump(teacher,folder/'teacher.joblib');z0=np.asarray(x@teacher['coef'])+teacher['intercept'];old=z0.argmax(1).astype(np.int8)
    np.save(folder/'teacher_scores.npy',z0);np.save(folder/'teacher_prediction.npy',old)
    protected=full[np.arange(len(old)),old];ids=np.flatnonzero(protected)
    oldm=z0[ids,old[ids]]-np.max(np.where(np.eye(3,dtype=bool)[old[ids]],-np.inf,z0[ids]),axis=1)
    assert (oldm>0).all()
    np.save(folder/'selected_rows.npy',selected.astype(np.int32));np.save(folder/'protection_fids.npy',ids.astype(np.int32))
    np.save(folder/'protection_original_mass.npy',protected[ids].astype(np.int32));np.save(folder/'protection_epsilon.npy',np.minimum(.001,oldm/2))
    # The shared helper names the local inner fold 1. Remap only fold names for
    # that diagnostic output, retaining all original row positions/components.
    local=r.copy();local['fold']=local.fold.replace({1:h,h:1})
    primary.prepare_pairs(local,y,fid,full,selected,folder)
    role=np.where(r.fold.eq(0),'H',np.where(r.fold.eq(2),'C',np.where(inner,'inner','train')))
    for name in ['component','body_group','source_symbol']:
        take=r[name]>=0
        assert pd.DataFrame({'id':r.loc[take,name],'role':role[take]}).groupby('id').role.nunique().max()==1
    pd.DataFrame({'row_position':r.row_position,'canonical_fold':r.fold,'component':r.component,'role':role,
                  'selected':np.isin(r.row_position,selected)}).to_parquet(folder/'manifest.parquet',index=False)
    save(folder/'teacher_fit.json',{'actual_classifier_fits':1,'actual_calibration_fits':0,'converged':True,
         'iterations':int(result.nit),'gradient_inf':float(abs(result.jac).max()),'objective':float(result.fun),
         'seconds':time.monotonic()-start,'model_sha256':sha(folder/'teacher.joblib'),'source_sha256':sha(__file__),
         'canonical_inner_fold':h,'train_population_rows':int(fit.sum()),'selected_original_rows':len(selected),
         'selected_per_class':cc.sum(0).tolist(),'protected_original_rows':int(protected.sum()),
         'protected_unique_inputs':len(ids),'all_role_M_S_in_loss':True,
         'input_and_preparation_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}})
    primary.emit(stage='rotation_teacher_complete',fold=h,selected_rows=len(selected),seconds=round(time.monotonic()-start,1))
    return r,y,fid,x,cc,full,vc,fit,inner,selected,folder


def train(h,arm,spec):
    assert sha(__file__)==spec['source_sha256'];reg=primary.check()
    r,y,fid,x,cc,full,vc,fit,inner,selected,folder=setup(h,spec)
    if (folder/(arm+'_fit.json')).exists():return
    epoch_limit=spec['fixed_epoch'];used=np.flatnonzero(cc.sum(1));vused=np.flatnonzero(vc.sum(1))
    z0=np.load(folder/'teacher_scores.npy',mmap_mode='r');old=np.load(folder/'teacher_prediction.npy')
    xt=csr_tensor(x[used]);ct=torch.as_tensor(cc[used],dtype=torch.float32,device='cuda');zt=torch.as_tensor(np.asarray(z0[used]),dtype=torch.float32,device='cuda')
    labels=torch.as_tensor(old[used].astype(np.int64),device='cuda');pt=ct.gather(1,labels[:,None]).squeeze(1)
    eps=torch.minimum(torch.full_like(pt,.001),margin(zt,labels)/2);protection=primary.Protection(x,z0,old,folder)
    pairs=pd.read_parquet(folder/'pairs.parquet');pi=[torch.as_tensor(np.searchsorted(used,pairs[n]),dtype=torch.long,device='cuda') for n in ['anchor_fid','positive_fid','negative_fid']]
    pl=torch.as_tensor(pairs.anchor_class.to_numpy(),device='cuda')
    torch.manual_seed(primary.SEED);torch.cuda.manual_seed_all(primary.SEED);model=Residual().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=reg['lr'])
    initialhash=sha_tensor_state({k:v.detach().cpu() for k,v in model.state_dict().items()})
    assert initialhash==read(primary.OLD/'fold1/C_fit.json')['initial_state_hash']
    trace=[];progress=[];saved=[];start=time.monotonic();stagnation=0;fatal=None
    def snapshot(epoch,early=False):
        with torch.no_grad():delta=model(xt).cpu().numpy()
        tp=(np.asarray(z0[used])+delta).argmax(1).astype(np.int8);vp=(np.asarray(z0[vused])+inference(model,x,vused)).argmax(1).astype(np.int8)
        fm=changes(cc[used],old[used],tp);vm=changes(vc[vused],old[vused],vp);pr=protection.margins_report(model)
        guard=compare(cm_from_counts(vc,old),np.asarray(vm['cm']),True)
        eligible=(epoch==epoch_limit and pr['violating_input_groups']==0 and pr['protected_negative_flips']==0
                  and fm['negative_flips']==0 and fm['positive_flips']>0 and vm['negative_flips']==0 and vm['positive_flips']>0 and guard['eligible'])
        name=f'{arm}_epoch{epoch:03}';bundle={'model':{k:t.detach().cpu().clone() for k,t in model.state_dict().items()},
          'optimizer':optimizer.state_dict(),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
          'epoch':epoch,'arm':arm,'canonical_inner_fold':h,'source_sha256':sha(__file__)}
        torch.save(bundle,folder/(name+'.pt'));np.savez_compressed(folder/(name+'_predictions.npz'),fit_feature_ids=used,fit_prediction=tp,inner_feature_ids=vused,inner_prediction=vp)
        item={'name':name,'epoch':epoch,'early_stop':early,'selected_fit':fm,'inner':vm,'protection':pr,'eligible':bool(eligible),'inner_metric_guard':guard}
        trace.append(item);saved.append(name);save(folder/(arm+'_selection_trace.json'),trace)
        primary.emit(stage='rotation_snapshot',fold=h,arm=arm,epoch=epoch,fit_errors=fm['errors'],inner_errors=vm['errors'],inner_repairs=vm['positive_flips'],inner_regressions=vm['negative_flips'],eligible=eligible)
    snapshot(0)
    for epoch in range(1,epoch_limit+1):
        optimizer.zero_grad(set_to_none=True);hidd=primary.hidden(model,xt);z=zt+model.last(hidd)
        main=supervised_loss(z,ct);l2=ALPHA/2*sum(p.square().sum() for p in model.parameters())
        contrast=primary.contrastive_loss(hidd,pi,pl) if reg['arms'][arm]['contrastive'] else z.new_zeros(())
        multiplier=10. if epoch<=20 else 100. if epoch<=40 else 1000.
        hinge=protection_penalty(z,labels,eps,pt,multiplier);loss=main+l2+primary.PAIR_WEIGHT*contrast+hinge;loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        base=[p.detach().clone() for p in model.parameters()];before=copy.deepcopy(optimizer.state_dict());rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
        optimizer.step();proposal=[p.detach().clone() for p in model.parameters()];records=[];stop=None;fraction=1.
        if reg['arms'][arm]['solver']=='active_QP':accepted,fraction,records,stop=primary.projected_update(model,base,proposal,protection)
        else:
            direction=[b-a for a,b in zip(base,proposal)];accepted=False
            for k in range(primary.MAX_TRUST+1):
                fraction=2.**-k;primary.assign(model,base,direction,fraction);pr,_,_=protection.evaluate(model,for_search=True)
                if pr['violating_input_groups']==0:accepted=True;break
        postcheck,_,_=protection.evaluate(model)
        if accepted and (postcheck['violating_input_groups'] or postcheck['protected_negative_flips']):
            accepted=False
            records.append({'numerical_postcheck_rejected':True,'postcheck':postcheck})
        if not accepted:
            primary.assign(model,base,[torch.zeros_like(t) for t in base]);optimizer.load_state_dict(before)
            torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda']);fraction=0.
        displacement=float(torch.sqrt(sum((p-a).square().sum() for p,a in zip(model.parameters(),base))))
        stagnation=stagnation+1 if not accepted or displacement<1e-9 else 0
        pr,_,_=protection.evaluate(model);assert pr['protected_negative_flips']==pr['violating_input_groups']==0
        with torch.no_grad():pred=(np.asarray(z0[used])+model(xt).cpu().numpy()).argmax(1)
        fm=changes(cc[used],old[used],pred)
        item={'epoch':epoch,'main_before':float(main),'contrastive_before':float(contrast),'hinge_before':float(hinge),'objective_before':float(loss),
          'accepted':accepted,'proposal_fraction':fraction,'parameter_displacement':displacement,'consecutive_no_progress':stagnation,
          'selected_fit_errors':fm['errors'],'selected_fit_positive':fm['positive_flips'],'selected_fit_negative':fm['negative_flips'],
          'protection':pr,'QP_records':records,'seconds':time.monotonic()-start,'stop':stop}
        progress.append(item);save(folder/(arm+'_progress.json'),progress)
        if epoch in primary.CHECKPOINTS or epoch==epoch_limit:snapshot(epoch)
        if stop or stagnation>=3 or time.monotonic()-start>primary.ARM_SECONDS:
            fatal=stop or {'reason':'three_consecutive_no_progress' if stagnation>=3 else 'registered_runtime_limit'}
            if epoch not in primary.CHECKPOINTS and epoch!=epoch_limit:snapshot(epoch,True)
            break
        del base,before,proposal,hidd,z,loss,main,l2,contrast,hinge
    last=trace[-1]
    info={'arm':arm,'canonical_inner_fold':h,'actual_classifier_fits':1,'actual_calibration_fits':0,
          'attempts':len(progress),'accepted_updates':sum(p['accepted'] for p in progress),'saved_states':saved,
          'fixed_primary_epoch':epoch_limit,'eligible_at_fixed_epoch':last['eligible'],'stopped_reason':fatal,
          'initial_state_hash':initialhash,'source_sha256':sha(__file__),'teacher_sha256':sha(folder/'teacher.joblib'),
          'seconds':time.monotonic()-start,'new_checkpoint_selection':False}
    save(folder/(arm+'_fit.json'),info);primary.emit(stage='rotation_arm_complete',**info)
    del model,optimizer,protection,xt;gc.collect();torch.cuda.empty_cache()


def diagnose(h,arms):
    r,y,fid,x,cc,full,vc,fit,inner,selected=primary.data(h);folder=DEST/'rotation'/f'fold{h}'
    z0=np.load(folder/'teacher_scores.npy',mmap_mode='r');old=np.load(folder/'teacher_prediction.npy');result=[]
    roles={'fit_full':fit,'inner':inner,'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    for arm in arms:
        info=read(folder/(arm+'_fit.json'));name=info['saved_states'][-1]
        bundle=torch.load(folder/(name+'.pt'),map_location='cuda',weights_only=True);model=Residual().cuda();model.load_state_dict(bundle['model'])
        p=np.empty(x.shape[0],np.int8)
        for start in range(0,len(p),8192):
            ids=np.arange(start,min(start+8192,len(p)));p[ids]=(np.asarray(z0[ids])+inference(model,x,ids)).argmax(1)
        np.save(folder/(name+'_all_prediction.npy'),p);tables=[];classwise=[]
        for role,mask in roles.items():
            counts=raw_counts(fid,y,mask,len(p));m=changes(counts,old,p);m['role']=role;tables.append(m)
            for route in sorted(r.route.unique()):
                for cls in range(3):
                    take=mask&r.route.eq(route).to_numpy()&(y==cls);base=old[fid];new=p[fid]
                    classwise.append({'role':role,'route':route,'class':cls,'support':int(take.sum()),
                         'old_correct':int((take&(base==y)).sum()),'new_correct':int((take&(new==y)).sum()),
                         'repairs':int((take&(base!=y)&(new==y)).sum()),'regressions':int((take&(base==y)&(new!=y)).sum())})
        pd.DataFrame(classwise).to_csv(folder/(name+'_classwise.csv'),index=False)
        result.append({'name':name,'roles':tables});primary.emit(stage='rotation_locked_diagnostic',fold=h,name=name,errors={p['role']:p['errors'] for p in tables})
        del bundle,model;torch.cuda.empty_cache()
    save(folder/'diagnosis.json',result)


def main():
    spec=register()
    for h in spec['folds']:
        for arm in spec['arms']:train(h,arm,spec)
        diagnose(h,spec['arms'])
    checks=[]
    for h in spec['folds']:
        report=read(DEST/'rotation'/f'fold{h}'/(spec['winner']['arm']+'_fit.json'))
        checks.append({'fold':h,'fixed_epoch':spec['fixed_epoch'],'eligible':report['eligible_at_fixed_epoch'],'new_selection':False})
    out={'winner':spec['winner'],'checks':checks,'rotation_passed':all(c['eligible'] for c in checks),
         'actual_classifier_fits':len(spec['folds'])*(1+len(spec['arms'])),'actual_calibration_fits':0,
         'final_full_fit_authorized':False,'source_sha256':sha(__file__)}
    save(DEST/'rotation/continuation.json',out);primary.emit(stage='rotation_complete',**out)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with threadpool_limits(limits=4):main()
