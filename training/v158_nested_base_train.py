"""Fresh full supervised pipelines excluding both inner query and outer roots."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,math,time,traceback
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from v158_nested_runtime import ROOT,OUT,PLAN,QUAL,read,sha,save,review,seal,require,require_epoch_sources
from v158_fusion_contract import STAGES,role,counts,validate_stage_roots
from v131_common import load_data,INPUT,TRACE,OFFICIAL
from v135_model import Classifier,batch,tensor_hash,module_norms
from v138_train import configure,quick_stats
from v138_readout import Readout,full_gradient,bounded_lbfgs_step,GradientBudgetExhausted
from v141_second_model import SecondRepresentation
from v146_optimizer import assign,restore,acceptable

def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)

class Counted:
    def __init__(self,model,limit,folder):
        self.count=0;self.completed=0;self.limit=limit;self.feature_calls=0
        self.trace=(folder/'classifier_calls.jsonl').open('w',encoding='utf-8',buffering=1)
        self.hook=model.register_forward_pre_hook(self.before)
        self.after_hook=model.register_forward_hook(self.after)
        self.original_feature=getattr(model,'features',None)
        if self.original_feature:
            def features(packed):self.feature_calls+=1;return self.original_feature(packed)
            model.features=features
        self.model=model
    def before(self,*args):
        if self.count>=self.limit:raise RuntimeError('Prospective classifier forward budget exhausted')
        self.count+=1
        self.trace.write(json.dumps(dict(event='attempt',attempt=self.count,grad_enabled=torch.is_grad_enabled()))+'\n')
    def after(self,*args):
        self.completed+=1
        self.trace.write(json.dumps(dict(event='completed',completed=self.completed,features=self.feature_calls))+'\n')
    def close(self):
        self.hook.remove();self.after_hook.remove();self.trace.close()
        if self.original_feature:self.model.features=self.original_feature

def load_role(f,j):
    x,d=load_data();d=d.reset_index(drop=True);fit,query,held=role(d,f,j)
    expected=read(QUAL/f'outer{f}_inner{j}_all_supervised_stage_roots.json')
    validate_stage_roots(set(fit.root),set(query.root),set(held.root),expected['supervised_stage_roots'])
    c=counts(fit);used=np.flatnonzero(c.sum(1));mix=fit.groupby('canonical_key').truth.nunique()
    pure=np.zeros(22546,np.float64)
    for loc,key in fit[['local','canonical_key']].drop_duplicates().itertuples(index=False):pure[loc]=mix[key]==1
    label_counts=fit.groupby(['canonical_key','truth']).size().unstack(fill_value=0)
    floor=int((label_counts.sum(1)-label_counts.max(1)).sum())
    return x,d,fit,c,used,pure,floor

def folder_for(f,j,stage):return OUT/f'outer{f}_inner{j}'/stage

def expected_next():
    for f in range(3):
        for j in range(3):
            for stage in STAGES:
                if not (folder_for(f,j,stage)/'fit.json').exists():return (f,j,stage)
    return None

def register():
    p=review();configure();assert not OUT.exists();OUT.mkdir()
    import v158_legacy_nested_train
    # Resolve lazy optimizer dependencies before any official model function.
    dummy=torch.nn.Linear(1,1,dtype=torch.float64).cuda()
    opt=torch.optim.AdamW(dummy.parameters(),lr=.002);opt2=torch.optim.LBFGS(dummy.parameters(),max_iter=1)
    value=dummy(torch.ones(1,1,dtype=torch.float64,device='cuda')).sum();value.backward()
    del opt,opt2,dummy,value
    extra={INPUT,TRACE,OFFICIAL,QUAL/'qualification.json',QUAL/'nested_source_roles.parquet',QUAL/'pre_execution_bindings.json'}
    for f in range(3):
        for j in range(3):extra.add(QUAL/f'outer{f}_inner{j}_all_supervised_stage_roots.json')
    extra|={ROOT/k for k in read(QUAL/'pre_execution_bindings.json')['source_sha256']}
    extra|={ROOT/k for k in read(ROOT/'artifacts/v158_legacy_expert_bank_qualification_20261001/pre_saved_array_bindings.json')['source_sha256']}
    seal(__file__,extra)
    save(OUT/'registration.json',dict(status='base_phase_registered_before_any_official_model_call_or_update',new_fits=0,updates=0,
        official_classifier_forward_calls=0,setup_dummy_forwards=1,setup_dummy_gradients=1,setup_dummy_optimizer_updates=0,
        current_base_phase_fits_max=45,legacy_phase_fits_max=9,total_base_fits_max=54,reserved_fusion_fits_max=6,total_new_fits_max=60,full_network_updates_max=35600,dense_full_gradients_max=7200,legacy_full_gradients_max=9000,
        physical_sealed_files=len(read(OUT/'run_seal.json')['source_sha256']),seal_sha256=sha(OUT/'run_seal.json'),
        fusion_activation='Separate implemented fusion entry, current arrays source role verification and forward/gradient budget seal required; no automatic fusion fit.'))
    emit(stage='nested_base_phase_registered',base_phase_fits=45,new_fits=0)

def log_rows(folder,frame,q,stage,update):
    r=frame[['row_position','local','root','fold','truth','canonical_key']].copy();r['pred']=q[frame.local].argmax(1)
    for cl in range(3):r[f'p{cl}']=q[frame.local,cl]
    r['update']=update;r['stage']=stage
    path=folder/f'update{update}_FIT_rows.parquet';assert not path.exists();r.to_parquet(path,index=False)
    return path

def source_stats(folder,frame,q,update):
    r=frame[['root','truth']].copy();r['wrong']=q[frame.local].argmax(1)!=frame.truth.to_numpy()
    r['CE']=-np.log(np.maximum(q[frame.local,frame.truth],np.finfo(np.float64).tiny))
    r['Brier']=((q[frame.local]-np.eye(3)[frame.truth])**2).sum(1)
    src=r.groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),CE_sum=('CE','sum'),Brier_sum=('Brier','sum')).reset_index()
    src.to_parquet(folder/f'update{update}_FIT_sources.parquet',index=False)
    return src.groupby('truth')[['support','errors','CE_sum','Brier_sum']].sum().to_dict('index')

def full_network(f,j,folder,p,x,d,frame,c,used,pure,floor):
    torch.manual_seed(10201);model=Classifier(128).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    nb=math.ceil(len(used)/256);budget=p['role_budgets'][f*3+j]['full_network_forward_cap']
    counter=Counted(model,budget,folder);rng=np.random.default_rng(10201+f);history=[];updates=0;start=time.monotonic();oldpred=None
    log=(folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1)
    def diagnose(epoch):
        nonlocal oldpred
        q=np.zeros((22546,3),np.float64)
        with torch.no_grad():
            for offset in range(0,len(used),256):
                ii=used[offset:offset+256];q[ii]=torch.softmax(batch(model,x,ii),-1).mean(1).cpu().numpy()
        if oldpred is None:oldpred=q[frame.local].argmax(1).copy()
        state=tensor_hash(model.state_dict());s=quick_stats(frame,q,pure,oldpred,floor)
        assert np.isfinite(q[used]).all()
        metrics=source_stats(folder,frame,q,epoch)
        if epoch in p['full_network_row_checkpoints']:
            log_rows(folder,frame,q,'full_network',epoch)
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch),folder/f'epoch{epoch}_model.pt')
        return dict(epoch=epoch,updates=updates,stats=s,class_risk=metrics,parameter_sha256=state,classifier_forward_calls=counter.count)
    history.append(diagnose(0))
    for epoch in range(1,101):
        # Full source/data/runtime check at each fit entry and terminal; project
        # Python sources and plan checked each epoch without rereading all DLLs.
        require_epoch_sources()
        lr=.002 if epoch<=80 else .00002+.5*(.002-.00002)*(1+math.cos(math.pi*(epoch-80)/20))
        for group in opt.param_groups:group['lr']=lr
        seen=np.zeros_like(c);order=used[rng.permutation(len(used))]
        for offset in range(0,len(order),256):
            ii=order[offset:offset+256];seen[ii]+=c[ii];opt.zero_grad(set_to_none=True)
            z=batch(model,x,ii);mass=torch.as_tensor(c[ii],device='cuda',dtype=torch.float32)
            loss=-(torch.log_softmax(z,-1).mean(1)*mass).sum()/float(c.sum())*nb
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite network CE')
            log.write(json.dumps(dict(event='batch_gradient_attempt',epoch=epoch,planned_update=updates+1,original_class_mass=c[ii].sum(0).tolist()))+'\n')
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError('Nonfinite network gradient')
            gradient=module_norms(model,gradient=True);before={k:v.detach().clone() for k,v in model.named_parameters()}
            opt.step();delta=module_norms(model,before=before);del before;updates+=1
            log.write(json.dumps(dict(event='batch_gradient_and_update_completed',epoch=epoch,update=updates,lr=lr,loss=float(loss),original_class_mass=c[ii].sum(0).tolist(),gradient_norm=gradient,update_norm=delta))+'\n')
        assert np.array_equal(seen,c);history.append(diagnose(epoch));save(folder/'progress.json',history)
        if epoch in [1,20,40,60,80,96,97,98,99,100]:emit(stage='nested_network_epoch',outer=f,inner=j,epoch=epoch,updates=updates,stats=history[-1]['stats'])
    log.close();h1=[];h2=[];logits=[];probabilities=[]
    with torch.no_grad():
        for offset in range(0,22546,256):
            ii=np.arange(offset,min(offset+256,22546));z,a,b=batch(model,x,ii,True)
            logits.append(z.cpu().numpy());h1.append(a.cpu().numpy());h2.append(b.cpu().numpy())
            probabilities.append(torch.softmax(z,-1).mean(1).cpu().numpy())
    for name,values in [('h1',h1),('h2',h2),('member_logits',logits)]:np.save(folder/(name+'.npy'),np.concatenate(values))
    np.save(folder/'mean_probability.npy',np.concatenate(probabilities))
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=10201),folder/'endpoint.pt')
    assert updates==100*nb and counter.count==budget
    receipt=dict(full_gradients=0,batch_gradients=updates,accepted_updates=updates,classifier_forward_calls=counter.count,internal_features_calls=0,termination='fixed_100_epochs',last5_FIT_stats=[z['stats'] for z in history[-5:]],seconds=time.monotonic()-start)
    counter.close();del model,opt;gc.collect();torch.cuda.empty_cache();return receipt

def dense(f,j,stage,folder,p,frame,c,used,pure,floor):
    parent=folder.parent;backbone=torch.load(parent/STAGES[0]/'endpoint.pt',map_location='cpu',weights_only=True)['model']
    h2=torch.as_tensor(np.load(parent/STAGES[0]/'h2.npy'),device='cuda',dtype=torch.float64)
    x,_=load_data();facts=torch.as_tensor(x[:,65792:].toarray(),device='cuda',dtype=torch.float64);del x
    index=STAGES.index(stage)
    if index<=2:
        model=Readout(backbone).cuda();h=h2
        if index==2:model.load_state_dict(torch.load(parent/STAGES[1]/'endpoint.pt',map_location='cpu',weights_only=True)['state'])
    else:
        head=torch.load(parent/STAGES[2]/'endpoint.pt',map_location='cpu',weights_only=True)['state']
        model=SecondRepresentation(backbone,head).cuda()
        if index==4:model.load_state_dict(torch.load(parent/STAGES[3]/'endpoint.pt',map_location='cpu',weights_only=True)['state'])
        h1=torch.as_tensor(np.load(parent/STAGES[0]/'h1.npy'),device='cuda',dtype=torch.float64);h=torch.cat([h1,h2],-1)
    budget=p['role_budgets'][f*3+j]['LBFGS_stage_forward_cap' if index<4 else 'Armijo_stage_forward_cap']
    counter=Counted(model,budget,folder);mass=torch.as_tensor(c,device='cuda',dtype=torch.float64);ids=used
    def qfit():
        q=np.zeros((22546,3),np.float64)
        with torch.no_grad():
            for offset in range(0,len(ids),2048):
                ii=ids[offset:offset+2048];q[ii]=torch.softmax(model(h[ii],facts[ii]),-1).mean(1).cpu().numpy()
        assert np.isfinite(q[ids]).all()
        return q
    initial=qfit();previous_q=np.load(parent/STAGES[index-1]/'mean_probability.npy')
    zero_gap=float(np.abs(initial[ids]-previous_q[ids]).max());zero_tolerance=2e-6 if index==1 else 2e-12
    assert zero_gap<=zero_tolerance and np.array_equal(initial[ids].argmax(1),previous_q[ids].argmax(1))
    old=initial[frame.local].argmax(1);protected=old==frame.truth.to_numpy();before={k:v.detach().clone() for k,v in model.state_dict().items()}
    log_rows(folder,frame,initial,stage,0);source_stats(folder,frame,initial,0)
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},update=0),folder/'accepted0.pt')
    gradients=accepted=proposals=0;history=[];last=deque(maxlen=5);start=time.monotonic();step=.01
    log=(folder/'gradients.jsonl').open('w',encoding='utf-8',buffering=1);trials=(folder/'proposals.jsonl').open('w',encoding='utf-8',buffering=1)
    opt=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=1,max_eval=200,tolerance_grad=1e-7,tolerance_change=1e-9,history_size=20,line_search_fn='strong_wolfe') if index<4 else None
    def gradient():
        nonlocal gradients
        if gradients>=200:raise GradientBudgetExhausted('Nested full-gradient cap')
        gradients+=1;log.write(json.dumps(dict(event='full_gradient_attempt',full_gradient=gradients))+'\n')
        loss,seen=full_gradient(model,h,facts,mass,ids)
        log.write(json.dumps(dict(event='full_gradient_completed',full_gradient=gradients,member_CE=loss,class_mass=seen,state_sha256=tensor_hash(model.state_dict()),gradient_norm=float(sum(v.grad.square().sum() for v in model.parameters()).sqrt())))+'\n')
        return loss
    def closure():return torch.tensor(gradient(),device='cuda',dtype=torch.float64)
    def value():
        risk=0.
        with torch.no_grad():
            for offset in range(0,len(ids),2048):
                ii=ids[offset:offset+2048];risk+=float(-(torch.log_softmax(model(h[ii],facts[ii]),-1).mean(1)*mass[ii]).sum()/mass.sum())
        return risk
    termination='gradient_budget'
    while gradients<200 and accepted<200:
        if index<4:
            status=bounded_lbfgs_step(opt,model,closure)
            if status!='accepted_changed_state':termination=status;break
            q=qfit()
        else:
            if proposals>=600:termination='proposal_budget';break
            loss=gradient();parameters=tuple(model.parameters());norm=float(sum(v.grad.square().sum() for v in parameters).sqrt())
            if norm<=1e-15:termination='zero_gradient';break
            base=tuple(v.detach().clone() for v in parameters);direction=tuple(v.grad.detach().clone()/norm for v in parameters);ok=False
            for backtrack in range(20):
                if proposals>=600 or step<1e-8:break
                assign(parameters,base,direction,step);proposals+=1;trial=value();q=qfit()
                guard=bool(np.all(q[frame.local].argmax(1)[protected]==frame.truth.to_numpy()[protected])) and quick_stats(frame,q,pure,old,floor)['mastered']
                ok=acceptable(loss,trial,step,norm,guard,.0001)
                trials.write(json.dumps(dict(proposal=proposals,base_CE=loss,trial_CE=trial,step=step,classification_guard=guard,accepted=ok))+'\n')
                if ok:step=min(.01,step*2);break
                restore(parameters,base);step*=.5
            if not ok:restore(parameters,base);termination='no_feasible_step';break
        accepted+=1;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};identity=tensor_hash(state)
        assert not last or last[-1]['state_sha256']!=identity
        s=quick_stats(frame,q,pure,old,floor);last.append(dict(update=accepted,state=state,state_sha256=identity,stats=s,probability=q.copy()))
        metrics=source_stats(folder,frame,q,accepted)
        history.append(dict(update=accepted,gradients=gradients,stats=s,class_risk=metrics,state_sha256=identity,forward_calls=counter.count));save(folder/'progress.json',history)
        if accepted in p['dense_row_checkpoints']:
            log_rows(folder,frame,q,stage,accepted)
            torch.save(dict(state=state,update=accepted),folder/f'accepted{accepted}.pt')
        if accepted in [1,50,100,150,200]:emit(stage='nested_dense_update',outer=f,inner=j,phase=stage,accepted=accepted,gradients=gradients,stats=s)
    if accepted==200:termination='accepted_update_budget'
    log.close();trials.close();parts=[];probabilities=[]
    with torch.no_grad():
        for offset in range(0,22546,2048):
            ii=np.arange(offset,min(offset+2048,22546));z=model(h[ii],facts[ii]);parts.append(z.cpu().numpy());probabilities.append(torch.softmax(z,-1).mean(1).cpu().numpy())
    logits=np.concatenate(parts);np.save(folder/'member_logits.npy',logits)
    np.save(folder/'mean_probability.npy',np.concatenate(probabilities))
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()}),folder/'endpoint.pt')
    for z in last:
        if not (folder/f"accepted{z['update']}.pt").exists():torch.save(dict(state=z['state'],update=z['update']),folder/f"accepted{z['update']}.pt")
        if not (folder/f"update{z['update']}_FIT_rows.parquet").exists():log_rows(folder,frame,z['probability'],stage,z['update'])
    if index>=3:
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith(('reference_','head_')))
    q=qfit();s=quick_stats(frame,q,pure,old,floor)
    receipt=dict(full_gradients=gradients,batch_gradients=0,accepted_updates=accepted,proposal_evaluations=proposals,classifier_forward_calls=counter.count,internal_features_calls=counter.feature_calls,
        termination=termination,last5_FIT_stats=[z['stats'] for z in last],endpoint_FIT_stats=s,
        actual_FIT_zero_step_probability_max_gap=zero_gap,zero_step_tolerance=zero_tolerance,zero_step_argmax_preserved=True,
        baseline_correct_FIT_rows=int(protected.sum()),baseline_correct_FIT_regressions=int((q[frame.local].argmax(1)[protected]!=frame.truth.to_numpy()[protected]).sum()),
        protection_policy='Post-fit report for historical LBFGS; actual fit-only acceptance guard for final V146 A stage.',seconds=time.monotonic()-start)
    counter.close();del model,h,facts,mass;gc.collect();torch.cuda.empty_cache();return receipt

def fit(f,j,stage):
    p=require(__file__);configure();assert expected_next()==(f,j,stage)
    folder=folder_for(f,j,stage);assert not folder.exists();folder.mkdir(parents=True)
    x,d,frame,c,used,pure,floor=load_role(f,j)
    assert p['role_budgets'][f*3+j]['canonical_floor']==floor
    frame[['row_position','local','root','fold','truth','canonical_key']].to_parquet(folder/'FIT_reference.parquet',index=False)
    save(folder/'started.json',dict(outer_fold=f,excluded_inner=j,stage=stage,fit_rows=len(frame),class_mass=c.sum(0).tolist(),fit_roots=sorted(set(frame.root)),query_or_outer_labels_used=0,seal_sha256=sha(OUT/'run_seal.json')))
    result=full_network(f,j,folder,p,x,d,frame,c,used,pure,floor) if stage==STAGES[0] else dense(f,j,stage,folder,p,frame,c,used,pure,floor)
    require(__file__)
    result.update(status='base_stage_fit_executed',outer_fold=f,excluded_inner=j,stage=stage,selected_by_score=False,fit_rows=len(frame),class_mass=c.sum(0).tolist(),canonical_label_collision_floor=floor,
        source_model_sha256=sha(folder/'endpoint.pt'),logits_sha256=sha(folder/'member_logits.npy'),seal_sha256=sha(OUT/'run_seal.json'),new_task_candidate=False,quality_acceptance=False)
    save(folder/'fit.json',result);emit(stage='nested_base_stage_complete',**result)

def all_base():
    for f in range(3):
        for j in range(3):
            for stage in STAGES:fit(f,j,stage)
    receipts=[read(folder_for(f,j,s)/'fit.json') for f in range(3) for j in range(3) for s in STAGES]
    save(OUT/'base_phase_completion.json',dict(status='all_45_new_inner_supervised_stages_completed',base_stage_fits=45,fusion_fits=0,
        batch_gradients=sum(z['batch_gradients'] for z in receipts),full_gradients=sum(z['full_gradients'] for z in receipts),accepted_updates=sum(z['accepted_updates'] for z in receipts),
        classifier_forward_calls=sum(z['classifier_forward_calls'] for z in receipts),internal_features_calls=sum(z['internal_features_calls'] for z in receipts),
        quality_acceptance=False,scope='OOF supervision generation only; no task model promoted.',source_sha256={folder_for(f,j,s).joinpath('fit.json').relative_to(ROOT).as_posix():sha(folder_for(f,j,s)/'fit.json') for f in range(3) for j in range(3) for s in STAGES}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','fit','all-base']);ap.add_argument('--outer',type=int);ap.add_argument('--inner',type=int);ap.add_argument('--stage',choices=STAGES);args=ap.parse_args()
    try:
        if args.mode=='register':register()
        elif args.mode=='all-base':all_base()
        else:fit(args.outer,args.inner,args.stage)
    except Exception as e:
        if OUT.exists():save(OUT/'execution_failure.json',dict(status='preserve_partial_pipeline_no_automatic_restart',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),entry_sha256=sha(Path(__file__))))
        raise
