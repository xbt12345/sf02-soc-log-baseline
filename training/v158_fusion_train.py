"""One fixed paired source-excluded mean-probability learner, six fits maximum."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,time,traceback
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from v158_fusion_runtime import ROOT,OUT,BANK,PLAN,CONTRACT,read,sha,save,review,seal,require,endpoint
from v158_fusion_prototype_v2 import SharedProbabilityScorer
from v138_train import configure,quick_stats
from v135_model import tensor_hash
from v146_optimizer import assign,restore,acceptable
from v142_retention_check import check as retention

def emit(**v):print(json.dumps(v,ensure_ascii=False),flush=True)

class Counter:
    def __init__(self,model,limit,path):
        self.attempted=self.completed=0;self.limit=limit;self.log=path.open('w',encoding='utf-8',buffering=1)
        self.pre=model.register_forward_pre_hook(self.before);self.post=model.register_forward_hook(self.after)
    def before(self,*args):
        if self.attempted>=self.limit:raise RuntimeError('Sealed fusion forward limit')
        self.attempted+=1;self.log.write(json.dumps(dict(event='attempt',attempt=self.attempted,grad_enabled=torch.is_grad_enabled()))+'\n')
    def after(self,*args):
        self.completed+=1;self.log.write(json.dumps(dict(event='completed',completed=self.completed))+'\n')
    def close(self):self.pre.remove();self.post.remove();self.log.close()

def arrays(f):
    frame=pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet');ids=np.sort(frame.local.unique())
    c=np.bincount(frame.local.to_numpy(np.int64)*3+frame.truth.to_numpy(np.int64),minlength=22546*3).reshape(22546,3)
    assert not c[:,0].any() and c.sum()==len(frame)
    pure=np.zeros(22546,np.float64);mix=frame.groupby('canonical_key').truth.nunique()
    for loc,key in frame[['local','canonical_key']].drop_duplicates().itertuples(index=False):pure[loc]=mix[key]==1
    oof=torch.as_tensor(np.load(BANK/f'fold{f}/OOF_probabilities.npy'),device='cuda',dtype=torch.float64)
    deploy=torch.as_tensor(np.load(BANK/f'fold{f}/deployment_probabilities.npy'),device='cuda',dtype=torch.float64)
    cond=torch.as_tensor(np.load(BANK/'conditions.npy'),device='cuda',dtype=torch.float64)
    prior=torch.as_tensor(np.load(BANK/f'fold{f}/prior.npy'),device='cuda',dtype=torch.float64)
    mass=torch.as_tensor(c,device='cuda',dtype=torch.float64)
    current=deploy[:,:16].mean(1).cpu().numpy();old=current[frame.local].argmax(1);protected=old==frame.truth.to_numpy()
    return frame,ids,c,pure,oof,deploy,cond,prior,mass,old,protected

def model_for(arm):
    model=SharedProbabilityScorer(0 if arm=='A' else 527,16).cuda()
    model.load_state_dict(torch.load(OUT/f'initial_{arm}.pt',map_location='cpu',weights_only=True)['state']);return model

def probabilities(model,bank,condition,prior,ids):
    q=np.full((22546,3),np.nan,np.float64)
    with torch.no_grad():
        for start in range(0,len(ids),2048):
            ii=ids[start:start+2048];value,_=model(bank[ii],condition[ii] if model.condition_dim else None,prior)
            q[ii]=value.cpu().numpy()
    assert np.isfinite(q[ids]).all();return q

def risk(model,bank,condition,prior,mass,ids,gradient=False):
    if gradient:model.zero_grad(set_to_none=True)
    q=np.full((22546,3),np.nan,np.float64);total=mass.sum();seen=torch.zeros(3,dtype=torch.float64,device='cuda');value=0.
    with torch.enable_grad() if gradient else torch.no_grad():
        for start in range(0,len(ids),2048):
            ii=ids[start:start+2048];prob,_=model(bank[ii],condition[ii] if model.condition_dim else None,prior)
            loss=-(prob.clamp_min(torch.finfo(prob.dtype).tiny).log()*mass[ii]).sum()/total
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fusion original-row mean-probability CE')
            if gradient:loss.backward()
            value+=float(loss.detach());seen+=mass[ii].sum(0);q[ii]=prob.detach().cpu().numpy()
    assert torch.equal(seen,mass.sum(0))
    if gradient and any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError('Nonfinite fusion gradient')
    return value,q,seen.cpu().tolist()

def rows(frame,q,f):
    rr=frame[['row_position','local','root','fold','truth','canonical_key']].copy();rr['training_role']=f;rr['pred']=q[frame.local].argmax(1)
    for cl in range(3):rr[f'p{cl}']=q[frame.local,cl]
    return rr

def record_sources(folder,frame,q,update):
    rr=frame[['root','truth']].copy();rr['wrong']=q[frame.local].argmax(1)!=frame.truth.to_numpy()
    rr['CE']=-np.log(np.maximum(q[frame.local,frame.truth],np.finfo(float).tiny));rr['Brier']=((q[frame.local]-np.eye(3)[frame.truth])**2).sum(1)
    summary=rr.groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),CE_sum=('CE','sum'),Brier_sum=('Brier','sum')).reset_index()
    summary.to_parquet(folder/f'accepted{update}_deployment_FIT_sources.parquet',index=False)
    return summary.groupby('truth')[['support','errors','CE_sum','Brier_sum']].sum().to_dict('index')

def register():
    p,c=review();configure();assert not OUT.exists();OUT.mkdir()
    # Initialize exactly matched common columns and outputs; no classifier call.
    torch.manual_seed(15801);a=SharedProbabilityScorer(0,16)
    torch.manual_seed(15801);b=SharedProbabilityScorer(527,16)
    with torch.no_grad():b.hidden.weight[:,:11].copy_(a.hidden.weight);b.hidden.bias.copy_(a.hidden.bias);b.output.weight.copy_(a.output.weight)
    for arm,model in [('A',a),('B',b)]:
        count=sum(v.numel() for v in model.parameters());assert count==p[f'fusion_{arm}_parameters']
        torch.save(dict(state=model.state_dict(),parameters=count,seed=15801),OUT/f'initial_{arm}.pt')
    # Resolve lazy torch dependencies using a separately reported dummy.
    dummy=torch.nn.Linear(1,1,dtype=torch.float64).cuda();dummy(torch.ones(1,1,dtype=torch.float64,device='cuda')).sum().backward()
    import v158_fusion_evaluate
    seal(__file__,{OUT/'initial_A.pt',OUT/'initial_B.pt'})
    save(OUT/'registration.json',dict(status='six_fusion_fits_sealed_before_classifier_preflight',new_fits=0,new_gradients=0,new_updates=0,
        setup_dummy_forwards=1,setup_dummy_gradients=1,setup_dummy_updates=0,preflight_gradients=0,
        classifier_fit_limit=6,gradient_limit=1200,proposal_limit=3600,update_limit=1200,seal_sha256=sha(OUT/'run_seal.json'),quality_acceptance=False))
    emit(event='fusion_registered',fits_max=6,updates=0)

def preflight():
    _,c=require(__file__);configure();assert not (OUT/'preflight.json').exists();report=[];all_rows=[];current_rows=[];total=0
    for f in range(3):
        frame,ids,count,pure,oof,deploy,cond,prior,mass,old,protected=arrays(f);previous=None
        current_rows.append(rows(frame,deploy[:,:16].mean(1).cpu().numpy(),f))
        for arm in ['A','B']:
            folder=OUT/f'preflight{f}_{arm}';folder.mkdir();model=model_for(arm);before=tensor_hash(model.state_dict())
            counter=Counter(model,c['role_call_budgets'][f]['preflight_classifier_forward_cap_per_arm'],folder/'classifier_calls.jsonl')
            oo=probabilities(model,oof,cond,prior,ids);repeat=probabilities(model,oof,cond,prior,ids)
            de=probabilities(model,deploy,cond,prior,np.arange(22546));dr=probabilities(model,deploy,cond,prior,np.arange(22546))
            assert np.array_equal(oo[ids],repeat[ids]) and np.array_equal(de,dr) and tensor_hash(model.state_dict())==before
            expected=(deploy*prior[None,:,None]).sum(1).cpu().numpy();gap=float(np.abs(de-expected).max());assert gap<=1e-12
            assert np.all(de[frame.local].argmax(1)[protected]==frame.truth.to_numpy()[protected])
            if previous is not None:assert np.array_equal(de,previous[0]) and np.array_equal(oo[ids],previous[1][ids])
            previous=(de,oo);np.save(folder/'OOF_probability.npy',oo);np.save(folder/'deployment_probability.npy',de)
            if arm=='A':all_rows.append(rows(frame,de,f))
            stats=quick_stats(frame,de,pure,old,[22,6,28][f]);assert stats['mastered'] and stats['new_errors_vs_start']==0
            report.append(dict(fold=f,arm=arm,forward_calls=counter.attempted,original_FIT_rows=len(frame),protected_rows=int(protected.sum()),
                baseline_preserving_max_gap=gap,stats=stats,parameters_unchanged=True))
            total+=counter.attempted;counter.close();del model;gc.collect();torch.cuda.empty_cache()
    joint=retention(pd.concat(all_rows,ignore_index=True));current_joint=retention(pd.concat(current_rows,ignore_index=True));assert joint['passed'] and current_joint['passed']
    save(OUT/'preflight.json',dict(status='actual_zero_step_pair_and_joint_TRAIN_replayed_before_first_fusion_fit',classifier_forward_calls=total,
        new_fits=0,new_gradients=0,new_updates=0,folds=report,joint_TRAIN_retention=joint,current16_joint_TRAIN_retention=current_joint,
        accepted_guard_entails_joint_by_protecting_every_current_correct_row=True,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in OUT.glob('preflight*/*.npy')}))
    emit(event='fusion_zero_step_passed',classifier_forward_calls=total,fits=0,gradients=0,updates=0)

def fit(f,arm):
    p,c=require(__file__);configure();activation=read(OUT/'preflight.json');check=activation['joint_TRAIN_retention'];assert check['passed']
    from experiment_review import check_bindings
    check_bindings(activation['source_sha256']);assert f in [0,1,2] and arm in ['A','B']
    folder=OUT/f'fold{f}_{arm}';assert not folder.exists() and len(list(OUT.glob('fold*_*/started.json')))<6;folder.mkdir()
    frame,ids,count,pure,oof,deploy,cond,prior,mass,old,protected=arrays(f);model=model_for(arm)
    counter=Counter(model,c['role_call_budgets'][f]['fit_classifier_forward_cap_per_arm'],folder/'classifier_calls.jsonl')
    initial=tensor_hash(model.state_dict());initial_loss,qo,_=risk(model,oof,cond,prior,mass,ids);qd=probabilities(model,deploy,cond,prior,ids)
    assert np.array_equal(qd[ids],np.load(OUT/f'preflight{f}_{arm}/deployment_probability.npy')[ids])
    rows(frame,qd,f).to_parquet(folder/'initial_deployment_FIT_rows.parquet',index=False)
    save(folder/'started.json',dict(fold=f,arm=arm,initial_parameter_sha256=initial,original_class_mass=count.sum(0).tolist(),
        fit_original_rows=len(frame),OOF_gradient_source='all_54_fit_source_verified_inner_query_outputs',outer_labels_used=0,
        seal_sha256=sha(OUT/'run_seal.json'),initial_OOF_mean_probability_CE=initial_loss))
    gradients=accepted=proposals=0;step=.01;history=[];last=deque(maxlen=5);start=time.monotonic();termination='gradient_budget'
    gl=(folder/'gradients.jsonl').open('w',encoding='utf-8',buffering=1);tl=(folder/'proposals.jsonl').open('w',encoding='utf-8',buffering=1)
    while gradients<200 and accepted<200:
        if proposals>=600:termination='proposal_budget';break
        gradients+=1;gl.write(json.dumps(dict(event='attempt',full_gradient=gradients))+'\n')
        loss,qbase,seen=risk(model,oof,cond,prior,mass,ids,True);parameters=tuple(model.parameters());norm=float(sum(v.grad.square().sum() for v in parameters).sqrt())
        gl.write(json.dumps(dict(event='completed',full_gradient=gradients,original_class_mass=seen,OOF_mean_probability_CE=loss,gradient_norm=norm,parameter_sha256=tensor_hash(model.state_dict())))+'\n')
        if norm<=1e-15:termination='zero_gradient';break
        base=tuple(v.detach().clone() for v in parameters);direction=tuple(v.grad.detach().clone()/norm for v in parameters);ok=False
        for backtrack in range(20):
            if proposals>=600 or step<1e-8:break
            assign(parameters,base,direction,step);proposals+=1;trial,qt,_=risk(model,oof,cond,prior,mass,ids)
            qd=probabilities(model,deploy,cond,prior,ids);stats=quick_stats(frame,qd,pure,old,[22,6,28][f])
            guard=bool(np.all(qd[frame.local].argmax(1)[protected]==frame.truth.to_numpy()[protected])) and stats['mastered']
            ok=acceptable(loss,trial,step,norm,guard,.0001);identity=tensor_hash(model.state_dict())
            tl.write(json.dumps(dict(proposal=proposals,base_CE=loss,trial_CE=trial,step=step,gradient_norm=norm,Armijo_bound=loss-.0001*step*norm,
                classification_guard=guard,accepted=ok,parameter_sha256=identity))+'\n')
            if ok:step=min(.01,step*2);break
            restore(parameters,base);step*=.5
        if not ok:restore(parameters,base);termination='no_feasible_step';break
        accepted+=1;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};assert not last or last[-1]['identity']!=identity
        source=record_sources(folder,frame,qd,accepted);last.append(dict(update=accepted,state=state,identity=identity,stats=stats,q_deployment=qd.copy(),q_OOF=qt.copy()))
        history.append(dict(update=accepted,gradients=gradients,proposals=proposals,OOF_CE=trial,stats=stats,class_risk=source,parameter_sha256=identity));save(folder/'progress.json',history)
        if accepted in [1,20,50,100,150,200]:
            torch.save(dict(state=state,update=accepted),folder/f'accepted{accepted}.pt');rows(frame,qd,f).to_parquet(folder/f'accepted{accepted}_deployment_FIT_rows.parquet',index=False)
            rows(frame,qt,f).to_parquet(folder/f'accepted{accepted}_OOF_rows.parquet',index=False)
            emit(event='fusion_accepted',fold=f,arm=arm,accepted_updates=accepted,full_gradients=gradients,OOF_CE=trial,stats=stats)
    if accepted==200:termination='accepted_update_budget'
    gl.close();tl.close();qd=probabilities(model,deploy,cond,prior,np.arange(22546));final_loss,qo,_=risk(model,oof,cond,prior,mass,ids)
    assert np.all(qd[frame.local].argmax(1)[protected]==frame.truth.to_numpy()[protected]) and quick_stats(frame,qd,pure,old,[22,6,28][f])['mastered']
    np.save(folder/'endpoint_deployment_probability.npy',qd);np.save(folder/'endpoint_OOF_probability.npy',qo)
    rows(frame,qd,f).to_parquet(folder/'endpoint_original_FIT_rows.parquet',index=False);rows(frame,qo,f).to_parquet(folder/'endpoint_original_OOF_rows.parquet',index=False)
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},fold=f,arm=arm,seal_sha256=sha(OUT/'run_seal.json')),folder/'endpoint.pt')
    for item in last:
        update=item['update']
        if not (folder/f'accepted{update}.pt').exists():torch.save(dict(state=item['state'],update=update),folder/f'accepted{update}.pt')
        if not (folder/f'accepted{update}_deployment_FIT_rows.parquet').exists():rows(frame,item['q_deployment'],f).to_parquet(folder/f'accepted{update}_deployment_FIT_rows.parquet',index=False)
        if not (folder/f'accepted{update}_OOF_rows.parquet').exists():rows(frame,item['q_OOF'],f).to_parquet(folder/f'accepted{update}_OOF_rows.parquet',index=False)
    result=dict(status='fusion_fit_executed',fold=f,arm=arm,full_gradients=gradients,proposal_evaluations=proposals,accepted_updates=accepted,
        classifier_forward_calls=counter.attempted,termination=termination,selected_by_score=False,initial_OOF_CE=initial_loss,final_OOF_CE=final_loss,
        endpoint_FIT_stats=quick_stats(frame,qd,pure,old,[22,6,28][f]),last5=[dict(update=z['update'],parameter_sha256=z['identity'],stats=z['stats']) for z in last],
        protected_current_correct_original_FIT_rows=int(protected.sum()),initial_parameter_sha256=initial,endpoint_parameter_sha256=tensor_hash(model.state_dict()),
        model_sha256=sha(folder/'endpoint.pt'),deployment_probability_sha256=sha(folder/'endpoint_deployment_probability.npy'),OOF_probability_sha256=sha(folder/'endpoint_OOF_probability.npy'),
        seconds=time.monotonic()-start,seal_sha256=sha(OUT/'run_seal.json'),quality_acceptance=False)
    endpoint(result);counter.close();require(__file__);save(folder/'fit.json',result);emit(event='fusion_fit_complete',fold=f,arm=arm,accepted_updates=accepted,full_gradients=gradients,termination=termination)
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','preflight','fit','all-fusion']);ap.add_argument('--fold',type=int);ap.add_argument('--arm',choices=['A','B']);a=ap.parse_args()
    try:
        if a.mode=='register':register()
        elif a.mode=='preflight':preflight()
        elif a.mode=='fit':fit(a.fold,a.arm)
        else:
            for f in range(3):
                for arm in ['A','B']:fit(f,arm)
    except Exception as e:
        if OUT.exists():save(OUT/'execution_failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),source_sha256=sha(Path(__file__)),no_automatic_restart=True))
        raise
