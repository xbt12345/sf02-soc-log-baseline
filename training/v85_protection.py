"""Fold-local frozen teacher and three registered protection residual arms."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, read, save, sha, load_sparse, metrics
from v79_execute import rows
from v82_capacity import LAST, DEST as V82
from v81_training_contract import compare

DEST=ROOT/'artifacts/v85_protection_20260927'
SEED=8501
EPOCHS=60
CHECKPOINTS=[0,10,20,40,60]
WIDTH=66287
ALPHA=1e-6


def emit(**a):print(json.dumps(a,ensure_ascii=False),flush=True)


def raw_counts(fid,y,mask,n):
    return np.bincount(fid[mask]*3+y[mask],minlength=n*3).reshape(n,3)


def cm_from_counts(counts,pred):
    return np.stack([np.bincount(pred,weights=counts[:,i],minlength=3).astype(np.int64) for i in range(3)])


def changes(counts,old,new):
    ix=np.arange(len(old));oldright=counts[ix,old];newright=counts[ix,new];changed=old!=new
    nf=oldright*changed;pf=newright*changed
    wrong_to_wrong=(counts.sum(1)-oldright-newright)*changed
    n=int(counts.sum());cm=cm_from_counts(counts,new)
    out=metrics(cm)
    out.update(positive_flips=int(pf.sum()),negative_flips=int(nf.sum()),
       negative_flips_by_class=np.bincount(old,weights=nf,minlength=3).astype(int).tolist(),
       positive_flips_by_class=np.bincount(new,weights=pf,minlength=3).astype(int).tolist(),
       wrong_to_different_wrong=int(wrong_to_wrong.sum()),changed_decisions=int((counts.sum(1)*changed).sum()),
       negative_flip_rate=float(nf.sum()/n) if n else None)
    return out


def baseline_objective(flat,x,cc):
    w=flat.reshape(WIDTH+1,3);z=np.asarray(x@w[:-1])+w[-1];total=cc.sum(1);den=total.sum()
    loss=(total@np.logaddexp(0,z).sum(1)-(cc*z).sum())/den+ALPHA/2*np.square(w[:-1]).sum()
    residual=(expit(z)*total[:,None]-cc)/den
    g=np.vstack([np.asarray(x.T@residual)+ALPHA*w[:-1],residual.sum(0)])
    return float(loss),g.ravel()


class Residual(nn.Module):
    def __init__(self,width=WIDTH,h1=256,h2=64):
        super().__init__();self.first=nn.Linear(width,h1,bias=False);self.second=nn.Linear(h1,h2)
        self.last=nn.Linear(h2,3);nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)
    def forward(self,x):
        h=torch.sparse.mm(x,self.first.weight.T) if x.layout!=torch.strided else self.first(x)
        return self.last(nn.functional.gelu(self.second(nn.functional.gelu(h,approximate='tanh')),approximate='tanh'))


def csr_tensor(x,device='cuda'):
    return torch.sparse_csr_tensor(torch.tensor(x.indptr,dtype=torch.int64,device=device),
        torch.tensor(x.indices,dtype=torch.int64,device=device),torch.tensor(x.data,dtype=torch.float32,device=device),
        size=x.shape,device=device)


def margin(z,label):
    true=z.gather(1,label[:,None]).squeeze(1)
    other=z.masked_fill(nn.functional.one_hot(label,3).bool(),-torch.inf).max(1).values
    return true-other


def supervised_loss(z,cc):
    return ((nn.functional.softplus(z).sum(1)*cc.sum(1))-(z*cc).sum(1)).sum()/cc.sum()


def selective_kd(z,teacher,pmass):
    kl=(torch.softmax(teacher,1)*(torch.log_softmax(teacher,1)-torch.log_softmax(z,1))).sum(1)
    return (kl*pmass).sum()/pmass.sum()


def protection_penalty(z,oldlabel,epsilon,pmass,multiplier):
    violation=(epsilon-margin(z,oldlabel)).clamp_min(0);total=z.new_zeros(())
    for cls in range(3):
        mass=pmass*(oldlabel==cls)
        if mass.sum()>0:total+=(mass*violation.square()).sum()/mass.sum()
    active=pmass>0
    return multiplier*(total+violation[active].max().square())


def feasibility(z,oldlabel,epsilon,pmass,tolerance=1e-7):
    m=margin(z,oldlabel);active=pmass>0;v=(epsilon-m).clamp_min(0)
    return {'violating_input_groups':int(((v>tolerance)&active).sum().item()),
        'violating_original_rows':int(pmass[(v>tolerance)&active].sum().item()),
        'max_violation':float(v[active].max().item()),'min_protected_margin':float(m[active].min().item()),
        'protected_negative_flips':int(pmass[(z.argmax(1)!=oldlabel)&active].sum().item())}


def register():
    if DEST.exists():raise FileExistsError(DEST)
    DEST.mkdir();r=rows();fid=np.load(LAST/'row_feature_id.npy');selected=np.load(V82/'selected_rows.npy')
    mask=np.zeros(len(r),bool);mask[selected]=True;outer_fit=~r.fold.isin([0,2]).to_numpy()
    assert not (mask&~outer_fit).any()
    role=np.where(~outer_fit,'locked',np.where(r.fold.eq(1),'inner','train'))
    for field in ['component','body_group','source_symbol']:
        take=r[field].ge(0);assert pd.DataFrame({'id':r.loc[take,field],'role':role[take]}).groupby('id').role.nunique().max()==1
    spec={'version':'v85-protection-1','source_sha256':sha(__file__),'plan_sha256':sha(ROOT/'docs/V84_PRESERVE_CORRECT_AND_REPAIR_PLAN.md'),
      'primary_inner_fold':1,'secondary_inner_folds':[3,4],'outer_locked_folds':[0,2],
      'source_folds':'Primary train -1/3/4; inner1; C2/H0 locked. Rotation retrains fold-local teachers.',
      'input':'Unchanged fixed 66287-column R0; no fitted vocabulary, IDF or scale. Canonical input indexing is label-free; never an inference lookup.',
      'exposure':'Filter fixed v82 selected rows by current training role; every role M/S included with original frequency. Unsampled fit normals used for regression only.',
      'teacher':{'loss':'original row mean summed OVR BCE + alpha/2 coefficient norm','alpha':ALPHA,'solver':'L-BFGS-B',
          'maxiter':1000,'gtol':1e-6,'ftol':1e-12,'maxcor':10,'initialized_from_old_label_weights':False},
      'residual':{'dimensions':[WIDTH,256,64,3],'activation':'GELU tanh','last_layer_initialization':'zero',
          'seed':SEED,'optimizer':'full-batch Adam','lr':.003,'weight_decay':0.,'epochs':EPOCHS,'checkpoints':CHECKPOINTS,
          'main_loss':'exact original-row-frequency OVR BCE on all selected rows; no resampling/reweighting'},
      'arms':{'A':'freeze teacher, primary OVR + alpha/2 residual parameter square',
          'B':'same + 1.0 selective KL at T=1 on correctly predicted actual fit rows; no KL on old wrong rows',
          'C':'same + per-class mean squared margin violation and max violation; coefficients10/100/1000 at epochs1/21/41, plus full-P feasible backtracking'},
      'constraints':{'epsilon':'min(1e-3, old positive margin/2)','tolerance':1e-7,'backtrack_max':16,
          'backtrack':'interpolate all residual parameters between previous feasible state and Adam proposal; halve fraction until full P feasible, otherwise reject. Not a converged constrained-optimum claim.',
          'optimizer_state':'Adam moments advance with gradient even if parameter proposal is interpolated; restore prior moments on fully rejected proposal.',
          'P':'actual selected fit teacher-correct rows only, not validation labels','ties':'no strict-margin protection claim if teacher has tie; registration/fold preparation rejects such case'},
      'selection':'Eligible saved stages require P margin feasibility + selected fit positive flips>0/negative flips=0 + inner full positive flips>0/negative flips=0 + each-class exact compare protection. Pick inner errors, then earlier epoch; never H.',
      'continuation':'Only primary eligible candidate proceeds with A on folds3/4 at the fixed primary winning epoch. If none, stop three-arm trial; fixed final snapshots may be diagnosed on C/H after selection frozen, never replacement. Further contrastive training conditional on protection, rotation and fit-full/C/H regression success; final fitting remains separately gated.',
      'unknown_data':'All folds previously inspected development, not a new blind test; no guarantee on unknown input.',
      'new_external_data':False,'pseudo_labels':False,'target_answers_read':False,
      'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'data/official/train.parquet',OUT/'rows.parquet',LAST/'X.data',LAST/'X.indices',LAST/'X.indptr',LAST/'row_feature_id.npy',V82/'selected_rows.npy',ROOT/'training/v75_views.py']},
      'fixed_schema_proof':{'byte_space':65792,'facts_fixed_schema':477,'independent_metadata':18,'no_fold_fitted_transform':True},
      'actual_runtime':{'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
      'original_official_class_counts':np.bincount(r.label_index,minlength=3).tolist()}
    pd.DataFrame({'row_position':r.row_position,'fold':r.fold,'component':r.component,'selected':mask,'primary_role':role}).to_parquet(DEST/'manifest.parquet',index=False)
    spec['manifest_sha256']=sha(DEST/'manifest.parquet');save(DEST/'registration.json',spec)
    emit(stage='registered',version=spec['version'],runtime=spec['actual_runtime'])


def check():
    c=read(DEST/'registration.json');assert c['source_sha256']==sha(__file__),'Source changed after registration'
    return c


def fold_data(h):
    check();r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    selected=np.zeros(len(r),bool);selected[np.load(V82/'selected_rows.npy')]=True
    pool=~r.fold.isin([0,2]).to_numpy();fit=pool&~r.fold.eq(h).to_numpy();inner=r.fold.eq(h).to_numpy()
    active=selected&fit
    assert np.array_equal(active&(y>0),fit&(y>0))
    counts=raw_counts(fid,y,active,x.shape[0]);used=np.flatnonzero(counts.sum(1));valcounts=raw_counts(fid,y,inner,x.shape[0])
    folder=DEST/f'fold{h}';folder.mkdir(exist_ok=True)
    path=folder/'exposure.json'
    if not path.exists():
        np.save(folder/'selected_rows.npy',np.flatnonzero(active).astype(np.int32))
        save(path,{'inner_fold':h,'train_population_rows':int(fit.sum()),'inner_rows':int(inner.sum()),
            'selected_original_rows':int(active.sum()),'selected_per_class':counts.sum(0).tolist(),'unique_selected_inputs':len(used),
            'all_role_M_S_selected':True,'selected_rows_sha256':sha(folder/'selected_rows.npy')})
    return r,y,fid,x,counts,used,valcounts,fit,inner,folder


def teacher(h):
    r,y,fid,x,counts,used,valcounts,fit,inner,folder=fold_data(h)
    if (folder/'teacher_fit.json').exists():return
    cc=counts[used];xx=x[used];v=np.zeros((WIDTH+1)*3);v[-3:]=np.log(np.maximum(cc.sum(0)/cc.sum(),1e-9))
    rng=np.random.default_rng(SEED);direction=rng.normal(size=len(v));direction/=np.linalg.norm(direction)
    _,g=baseline_objective(v,xx[:31],cc[:31]);eps=1e-5
    fd=(baseline_objective(v+eps*direction,xx[:31],cc[:31])[0]-baseline_objective(v-eps*direction,xx[:31],cc[:31])[0])/(2*eps)
    error=float(abs(fd-g@direction));assert error<1e-6
    start=time.monotonic();iterations=[0]
    def callback(v):
        iterations[0]+=1
        if iterations[0]%50==0:emit(stage='teacher',fold=h,iteration=iterations[0],seconds=round(time.monotonic()-start,1))
    res=minimize(baseline_objective,v,args=(xx,cc),jac=True,method='L-BFGS-B',callback=callback,
       options={'maxiter':1000,'maxcor':10,'gtol':1e-6,'ftol':1e-12,'maxls':30})
    w=res.x.reshape(WIDTH+1,3);model={'coef':w[:-1].copy(),'intercept':w[-1].copy()}
    joblib.dump(model,folder/'teacher.joblib');z=np.asarray(x@model['coef'])+model['intercept']
    np.save(folder/'teacher_scores.npy',z);p=z.argmax(1).astype(np.int8);np.save(folder/'teacher_prediction.npy',p)
    report={'actual_classifier_fits':1,'converged':bool(res.success and abs(res.jac).max()<=1e-5),
       'iteration':int(res.nit),'objective':float(res.fun),'gradient_inf':float(abs(res.jac).max()),'message':str(res.message),
       'seconds':time.monotonic()-start,'model_sha256':sha(folder/'teacher.joblib'),'directional_gradient_error':error,
       'train':metrics(cm_from_counts(counts,p)),'inner':metrics(cm_from_counts(valcounts,p)),
       'source_sha256':sha(__file__)}
    save(folder/'teacher_fit.json',report);assert report['converged'],report
    emit(stage='teacher_complete',fold=h,fit_errors=report['train']['errors'],inner_errors=report['inner']['errors'],seconds=report['seconds'])


def inference(model,x,ids,device='cuda',batch=4096):
    out=np.empty((len(ids),3),np.float32);model.eval()
    with torch.no_grad():
        for start in range(0,len(ids),batch):out[start:start+batch]=model(csr_tensor(x[ids[start:start+batch]],device)).cpu().numpy()
    return out


def train_arm(h,arm):
    reg=check();r,y,fid,x,counts,used,valcounts,fit,inner,folder=fold_data(h)
    if (folder/(arm+'_fit.json')).exists():return
    assert read(folder/'teacher_fit.json')['converged']
    z0=np.load(folder/'teacher_scores.npy',mmap_mode='r');p0=np.load(folder/'teacher_prediction.npy')
    cc=counts[used];old=p0[used];pmass=cc[np.arange(len(used)),old].astype(np.float32)
    zt=torch.tensor(z0[used],dtype=torch.float32,device='cuda');ct=torch.tensor(cc,dtype=torch.float32,device='cuda')
    pt=torch.tensor(pmass,device='cuda');labels=torch.tensor(old.astype(np.int64),device='cuda')
    assert np.array_equal(zt.argmax(1).cpu().numpy(),old)
    oldmargin=margin(zt,labels);assert bool((oldmargin[pt>0]>0).all())
    et=torch.minimum(torch.full_like(oldmargin,1e-3),oldmargin/2)
    xt=csr_tensor(x[used]);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=Residual().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    initial={name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
    initialhash=sha_tensor_state(initial)
    vused=np.flatnonzero(valcounts.sum(1));pbase_v=p0[vused];basecm=cm_from_counts(valcounts[vused],pbase_v)
    trace=[];logs=[];start=time.monotonic();saved=[]
    def snapshot(epoch):
        model.eval()
        with torch.no_grad():
            residual=model(xt);ztnew=zt+residual;train_delta=residual.cpu().numpy()
        # Use original teacher double logits for saved decision arithmetic.
        train_z=np.asarray(z0[used])+train_delta;tp=train_z.argmax(1).astype(np.int8)
        vp=(np.asarray(z0[vused])+inference(model,x,vused)).argmax(1).astype(np.int8)
        feas=feasibility(ztnew,labels,et,pt)
        fitmetric=changes(cc,old,tp);vm=changes(valcounts[vused],pbase_v,vp)
        gate=compare(basecm,np.asarray(vm['cm']),True)
        eligible=feas['violating_input_groups']==0 and fitmetric['negative_flips']==0 and fitmetric['positive_flips']>0 and vm['negative_flips']==0 and vm['positive_flips']>0 and gate['eligible']
        name=f'{arm}_epoch{epoch:03}'
        state={name:t.detach().cpu().clone() for name,t in model.state_dict().items()};torch.save(state,folder/(name+'.pt'))
        np.savez_compressed(folder/(name+'_predictions.npz'),fit_feature_ids=used,fit_prediction=tp,inner_feature_ids=vused,inner_prediction=vp)
        item={'name':name,'epoch':epoch,'selected_fit':fitmetric,'inner':vm,'protection':feas,'eligible':bool(eligible),'inner_metric_guard':gate}
        trace.append(item);save(folder/(arm+'_selection_trace.json'),trace);saved.append(name)
        emit(stage='snapshot',fold=h,arm=arm,epoch=epoch,fit_errors=fitmetric['errors'],fit_negative=fitmetric['negative_flips'],
             inner_errors=vm['errors'],inner_negative=vm['negative_flips'],inner_positive=vm['positive_flips'],margin_violations=feas['violating_input_groups'],eligible=eligible)
    snapshot(0)
    for epoch in range(1,EPOCHS+1):
        model.train();optimizer.zero_grad(set_to_none=True);z=zt+model(xt)
        main=supervised_loss(z,ct);l2=ALPHA/2*sum(t.square().sum() for t in model.parameters());extra=z.new_zeros(())
        if arm=='B':extra=selective_kd(z,zt,pt)
        multiplier=10. if epoch<=20 else 100. if epoch<=40 else 1000.
        if arm=='C':extra=protection_penalty(z,labels,et,pt,multiplier)
        objective=main+l2+extra;objective.backward()
        if not all(t.grad is None or torch.isfinite(t.grad).all() for t in model.parameters()):raise FloatingPointError('Nonfinite gradient')
        previous=[t.detach().clone() for t in model.parameters()] if arm=='C' else None
        optimizer_before=copy.deepcopy(optimizer.state_dict()) if arm=='C' else None
        optimizer.step();fraction=1.;retries=0;rejected=False
        if arm=='C':
            with torch.no_grad():
                proposed=[t.detach().clone() for t in model.parameters()]
                for retries in range(17):
                    if retries:
                        fraction=2.**(-retries)
                        for t,a,b in zip(model.parameters(),previous,proposed):t.copy_(a+fraction*(b-a))
                    feas=feasibility(zt+model(xt),labels,et,pt)
                    if feas['violating_input_groups']==0:break
                else:
                    rejected=True;fraction=0.
                    for t,a in zip(model.parameters(),previous):t.copy_(a)
                    optimizer.load_state_dict(optimizer_before)
                assert feasibility(zt+model(xt),labels,et,pt)['violating_input_groups']==0
            del previous,proposed,optimizer_before
        with torch.no_grad():
            now=zt+model(xt);feas=feasibility(now,labels,et,pt);p=now.argmax(1)
            fiterr=int((ct.sum(1)-ct.gather(1,p[:,None]).squeeze(1)).sum().item())
        item={'epoch':epoch,'objective_before_step':float(objective.item()),'main_before_step':float(main.item()),'aux_before_step':float(extra.item()),
             'fit_errors_after_step':fiterr,'protection_after_step':feas,'step_fraction':fraction,'backtracks':retries,'fully_rejected':rejected,'seconds':time.monotonic()-start}
        logs.append(item)
        if epoch%10==0:
            save(folder/(arm+'_progress.json'),logs);emit(stage='epoch',fold=h,arm=arm,epoch=epoch,errors=fiterr,
               protected_negative=feas['protected_negative_flips'],max_violation=feas['max_violation'],fraction=fraction,seconds=round(time.monotonic()-start,1))
        if epoch in CHECKPOINTS:snapshot(epoch)
    valid=[v for v in trace if v['eligible']];chosen=min(valid,key=lambda v:(v['inner']['errors'],v['epoch'])) if valid else None
    report={'arm':arm,'inner_fold':h,'actual_classifier_fits':1,'epochs_executed':EPOCHS,'snapshots':saved,
       'selected':None if chosen is None else chosen['name'],'selection_frozen_before_C_H':True,
       'initial_state_hash':initialhash,'seconds':time.monotonic()-start,'teacher_model_sha256':sha(folder/'teacher.joblib'),
       'protection_violating_stages':sum(v['protection']['violating_input_groups']>0 for v in trace),
       'backtracked_steps':sum(v['backtracks']>0 for v in logs),'fully_rejected_steps':sum(v['fully_rejected'] for v in logs),
       'source_sha256':sha(__file__),'constrained_optimum_claimed':False,'external_labels_in_protection':False}
    save(folder/(arm+'_progress.json'),logs);save(folder/(arm+'_fit.json'),report)
    emit(stage='arm_complete',**report)
    del xt,model,optimizer;gc.collect();torch.cuda.empty_cache()


def sha_tensor_state(state):
    import hashlib
    digest=hashlib.sha256()
    for key,t in sorted(state.items()):digest.update(key.encode());digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def finish_primary():
    check();folder=DEST/'fold1';candidates=[]
    for arm in ['A','B','C']:
        info=read(folder/(arm+'_fit.json'))
        if info['selected']:
            s=next(i for i in read(folder/(arm+'_selection_trace.json')) if i['name']==info['selected'])
            candidates.append({'arm':arm,'name':s['name'],'epoch':s['epoch'],'inner_errors':s['inner']['errors']})
    chosen=min(candidates,key=lambda s:(s['inner_errors'],s['epoch'],s['arm'])) if candidates else None
    out={'primary_selected':chosen,'secondary_authorized':chosen is not None,'reason':'eligible_primary' if chosen else 'no_stage_passed_P_and_inner_zero_negative_flip_with_gain',
       'contrastive_training_authorized':False,'final_full_fit_authorized':False,'H_used_for_selection':False}
    save(DEST/'continuation.json',out);emit(stage='continuation',**out)


def finish_rotation():
    check();out=read(DEST/'continuation.json');chosen=out['primary_selected']
    assert chosen is not None and out['secondary_authorized']
    checks=[]
    for h in [3,4]:
        folder=DEST/f'fold{h}'
        for arm in sorted(set(['A',chosen['arm']])):
            read(folder/(arm+'_fit.json'))
        # No new checkpoint selection: replay the primary-registered winning epoch.
        stage=next(s for s in read(folder/(chosen['arm']+'_selection_trace.json')) if s['epoch']==chosen['epoch'])
        checks.append({'fold':h,'fixed_primary_epoch':chosen['epoch'],'eligible':stage['eligible'],'stage':stage})
    out['rotation_checks']=checks;out['rotation_passed']=all(c['eligible'] for c in checks)
    out['contrastive_training_authorized']=False
    out['pending_gate']='fit_full/C/H paired regression after fixed selection'
    save(DEST/'continuation.json',out);emit(stage='rotation_complete',passed=out['rotation_passed'])


def complete_gates():
    check();out=read(DEST/'continuation.json');chosen=out['primary_selected']
    if chosen is None or not out.get('rotation_passed',False):
        out['contrastive_training_authorized']=False
        save(DEST/'continuation.json',out);return
    gates=[]
    for h in [1,3,4]:
        folder=DEST/f'fold{h}';name=f"{chosen['arm']}_epoch{chosen['epoch']:03}"
        entry=next(d for d in read(folder/'diagnosis.json') if d['name']==name)
        r,y,fid,x,cc,used,vc,fit,inner,_=fold_data(h);old=np.load(folder/'teacher_prediction.npy')
        for item in entry['roles']:
            if item['role'] not in ['fit_full','C','H']:continue
            mask=fit if item['role']=='fit_full' else r.fold.eq(2 if item['role']=='C' else 0).to_numpy()
            cr=raw_counts(fid,y,mask,x.shape[0]);check_metrics=compare(cm_from_counts(cr,old),np.asarray(item['cm']),item['role']!='fit_full')
            passed=item['negative_flips']==0 and check_metrics['eligible']
            gates.append({'fold':h,'role':item['role'],'passed':passed,'positive':item['positive_flips'],'negative':item['negative_flips'],'metric_guard':check_metrics})
    out['locked_regression_checks']=gates;out['locked_regression_passed']=all(g['passed'] for g in gates)
    out['contrastive_training_authorized']=out['locked_regression_passed'];out.pop('pending_gate',None)
    save(DEST/'continuation.json',out);emit(stage='locked_gate_complete',passed=out['locked_regression_passed'])


def diagnose_final(h,arms):
    r,y,fid,x,cc,used,vc,fit,inner,folder=fold_data(h);old=np.load(folder/'teacher_prediction.npy');z0=np.load(folder/'teacher_scores.npy',mmap_mode='r')
    roles={'fit_full':fit,'inner':inner,'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()};results=[]
    # Freeze primary selections before any locked role diagnostics.
    assert (DEST/'continuation.json').exists()
    for arm in arms:
        info=read(folder/(arm+'_fit.json'));names=list(dict.fromkeys([info['snapshots'][-1]]+([] if info['selected'] is None else [info['selected']])))
        for name in names:
            model=Residual().cuda();model.load_state_dict(torch.load(folder/(name+'.pt'),map_location='cuda',weights_only=True))
            p=np.empty(x.shape[0],np.int8)
            for start in range(0,x.shape[0],8192):
                ids=np.arange(start,min(start+8192,x.shape[0]));p[ids]=(z0[ids]+inference(model,x,ids)).argmax(1)
            np.save(folder/(name+'_all_prediction.npy'),p);rp=p[fid];base=old[fid];table=[]
            for role,mask in roles.items():
                cr=raw_counts(fid,y,mask,x.shape[0]);item=changes(cr,old,p);item.update(role=role)
                table.append(item)
            classrows=[]
            for role,mask in roles.items():
                for route in sorted(r.route.unique()):
                    take=mask&r.route.eq(route).to_numpy()
                    for cls in range(3):
                        ix=take&(y==cls);classrows.append({'role':role,'route':route,'class':cls,'support':int(ix.sum()),
                           'old_correct':int(((base==y)&ix).sum()),'new_correct':int(((rp==y)&ix).sum()),
                           'positive_flips':int(((base!=y)&(rp==y)&ix).sum()),'negative_flips':int(((base==y)&(rp!=y)&ix).sum())})
            pd.DataFrame(classrows).to_csv(folder/(name+'_classwise.csv'),index=False)
            results.append({'name':name,'selected_on_inner':name==info['selected'],'roles':table})
            emit(stage='final_diagnosis',fold=h,name=name,errors={t['role']:t['errors'] for t in table},negative={t['role']:t['negative_flips'] for t in table})
            del model;torch.cuda.empty_cache()
    save(folder/'diagnosis.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','teacher','train','finish','rotation','complete','diagnose']);p.add_argument('--fold',type=int,default=1);p.add_argument('--arm',choices=['A','B','C']);p.add_argument('--arms',nargs='+',choices=['A','B','C'],default=['A','B','C']);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with threadpool_limits(limits=4):
        if a.stage=='register':register()
        elif a.stage=='teacher':teacher(a.fold)
        elif a.stage=='train':train_arm(a.fold,a.arm)
        elif a.stage=='finish':finish_primary()
        elif a.stage=='rotation':finish_rotation()
        elif a.stage=='complete':complete_gates()
        else:diagnose_final(a.fold,a.arms)
