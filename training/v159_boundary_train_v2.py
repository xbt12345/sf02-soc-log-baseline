"""Counted register/preflight/fit entry; six fixed fits, no automatic restart."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,time,traceback
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from v159_boundary_runtime_v2 import ROOT,OUT,BANK,PREP,INPUT,PLAN,review,seal,require,endpoint,read,save,sha
from v159_current_input_boundary_v3 import CurrentInputBoundary
from v159_class_direction import class_direction,finite_armijo
from v138_train import configure
from v135_model import tensor_hash
from v142_retention_check import check as retention

def emit(**v):print(json.dumps(v,ensure_ascii=False),flush=True)

class Counter:
    def __init__(self,model,cap,path,gradient_cap=0):
        self.cap=cap;self.gradient_cap=gradient_cap;self.head_attempts=self.head_completed=self.feature_attempts=self.feature_completed=self.gradient_attempts=self.gradient_completed=0
        self.log=Path(path).open('w',encoding='utf-8',buffering=1)
        self.hooks=[model.register_forward_pre_hook(lambda *a:self.before('head')),model.register_forward_hook(lambda *a:self.after('head')),
            model.opinions.register_forward_pre_hook(lambda *a:self.before('feature')),model.opinions.register_forward_hook(lambda *a:self.after('feature'))]
    def write(self,**v):self.log.write(json.dumps(v)+'\n')
    def before(self,kind):
        name=kind+'_attempts'
        if getattr(self,name)>=self.cap:raise RuntimeError('Sealed '+kind+' budget exceeded')
        setattr(self,name,getattr(self,name)+1);self.write(event='attempt',kind=kind,ordinal=getattr(self,name),grad_enabled=torch.is_grad_enabled())
    def after(self,kind):
        name=kind+'_completed';setattr(self,name,getattr(self,name)+1);self.write(event='completed',kind=kind,ordinal=getattr(self,name))
    def gradient_before(self,cls,mass):
        if self.gradient_attempts>=self.gradient_cap:raise RuntimeError('Sealed complete class gradient budget exceeded')
        self.gradient_attempts+=1;self.write(event='attempt',kind='full_class_gradient',ordinal=self.gradient_attempts,class_id=cls,original_class_mass=mass)
    def gradient_after(self,cls,mass):
        self.gradient_completed+=1;self.write(event='completed',kind='full_class_gradient',ordinal=self.gradient_completed,class_id=cls,original_class_mass=mass)
    def counts(self):return {k:getattr(self,k) for k in ['head_attempts','head_completed','feature_attempts','feature_completed','gradient_attempts','gradient_completed']}
    def close(self):
        for h in self.hooks:h.remove()
        self.log.close()

def context(f):
    x=load_npz(INPUT).tocsr();x.sort_indices()
    frame=pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet');ids=np.sort(frame.local.unique())
    counts=np.bincount(frame.local.to_numpy(np.int64)*3+frame.truth.to_numpy(np.int64),minlength=22546*3).reshape(22546,3)
    oof=np.load(BANK/f'fold{f}/OOF_probabilities.npy',mmap_mode='r')[:,:16];deploy=np.load(BANK/f'fold{f}/deployment_probabilities.npy',mmap_mode='r')[:,:16]
    oo=pd.read_parquet(PREP/f'fold{f}/OOF_visible_input_rows.parquet');de=pd.read_parquet(PREP/f'fold{f}/deployment_visible_input_rows.parquet')
    assert np.array_equal(frame.row_position,oo.row_position) and np.array_equal(frame.truth,oo.truth) and np.array_equal(frame.row_position,de.row_position)
    assert counts.sum()==len(frame) and not counts[:,0].any()
    return dict(fold=f,x=x,frame=frame,ids=ids,counts=counts,mass=counts.sum(0),OOF=oof,deployment=deploy,OOF_rows=oo,deployment_rows=de,info=read(PREP/'qualification.json')['roles'][f]['scopes'])

def inputs(ctx,scope,ids):
    x=ctx['x'][ids];xx=torch.sparse_csr_tensor(torch.tensor(x.indptr,dtype=torch.int64,device='cuda'),torch.tensor(x.indices,dtype=torch.int64,device='cuda'),torch.tensor(x.data,dtype=torch.float64,device='cuda'),size=x.shape,device='cuda')
    p=torch.tensor(np.array(ctx[scope][ids]),dtype=torch.float64,device='cuda');return xx,p

def model_for():
    model=CurrentInputBoundary().cuda();model.load_state_dict(torch.load(OUT/'initial.pt',map_location='cpu',weights_only=True)['state']);return model

def probabilities(model,ctx,scope,ids,return_log=False):
    q=np.full((22546,3),np.nan,np.float64);lp=np.full_like(q,np.nan)
    with torch.no_grad():
        for start in range(0,len(ids),2048):
            ii=ids[start:start+2048];value,logq,_=model(*inputs(ctx,scope,ii));q[ii]=value.cpu().numpy();lp[ii]=logq.cpu().numpy()
    assert np.isfinite(q[ids]).all();return (q,lp) if return_log else q

def risk(model,ctx,scope,ids,counter=None,gradient_class=None):
    """Accumulate numerator/global original mass, including zero-class chunks."""
    mass=ctx['mass'];assert mass[1]>0 and mass[2]>0
    if gradient_class is not None:
        assert gradient_class in [1,2] and counter is not None
        counter.gradient_before(gradient_class,mass.tolist());model.zero_grad(set_to_none=True)
    q=np.full((22546,3),np.nan,np.float64);lp=np.full_like(q,np.nan);numerators=np.zeros(3);seen=np.zeros(3,np.int64)
    with torch.enable_grad() if gradient_class is not None else torch.no_grad():
        for start in range(0,len(ids),2048):
            ii=ids[start:start+2048];value,logq,_=model(*inputs(ctx,scope,ii));c=torch.tensor(ctx['counts'][ii],dtype=torch.float64,device=logq.device)
            if gradient_class is not None:
                loss=-(c[:,gradient_class]*logq[:,gradient_class]).sum()/float(mass[gradient_class])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite stable class loss')
                loss.backward()
            numerators+=(-(c*logq).sum(0)).detach().cpu().numpy();seen+=ctx['counts'][ii].sum(0);q[ii]=value.detach().cpu().numpy();lp[ii]=logq.detach().cpu().numpy()
    assert np.array_equal(seen,mass)
    risks=numerators[1:]/mass[1:];assert np.isfinite(risks).all()
    gradient=None
    if gradient_class is not None:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        gradient=np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in model.parameters()]);counter.gradient_after(gradient_class,mass.tolist())
    return risks,q,lp,gradient

def stats(ctx,q,scope):
    rr=ctx[scope+'_rows'];pred=q[rr.local].argmax(1);wrong=pred!=rr.truth.to_numpy();base=rr.initial_pred.to_numpy();info=ctx['info'][scope]
    result=dict(original_rows=len(rr),M_errors=int((wrong&rr.truth.eq(1)).sum()),S_errors=int((wrong&rr.truth.eq(2)).sum()),
        N_predictions=int((pred==0).sum()),pure_errors=int((wrong&rr.pure_current_input).sum()),total_errors=int(wrong.sum()),
        new_errors_vs_initial=int((wrong&rr.initial_correct).sum()),protected_regressions=int((wrong&rr.protected_correct).sum()),repairs_vs_initial=int((~wrong&~rr.initial_correct).sum()),
        minimum_original_errors=info['current_input_minimum_errors'],complete_visible_input_minimum_errors=info['unconstrained_minimum_original_errors'],guard_constrained_minimum_errors=info['initial_correct_guard_constrained_minimum_errors'],all_initial_correct_guard_constrained_minimum_errors=info['all_initial_correct_guard_constrained_minimum_errors'])
    result['mastered']=result['pure_errors']==result['protected_regressions']==0 and result['total_errors']<=result['minimum_original_errors']
    return result

def rows(ctx,q,lp,scope):
    r=ctx[scope+'_rows'].copy();r['pred']=q[r.local].argmax(1)
    for c in range(3):r[f'p{c}']=q[r.local,c];r[f'logp{c}']=lp[r.local,c] if lp is not None else np.nan
    r['stable_CE']=-r[['logp0','logp1','logp2']].to_numpy()[np.arange(len(r)),r.truth]
    r['probability_clip_CE']=-np.log(np.maximum(q[r.local,r.truth],1e-300));return r

def record(folder,ctx,scope,q,lp,update):
    r=rows(ctx,q,lp,scope);r.to_parquet(folder/f'accepted{update}_{scope}_rows.parquet',index=False)
    source=r.assign(wrong=r.pred.ne(r.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()
    source.to_parquet(folder/f'accepted{update}_{scope}_sources.parquet',index=False)
    return stats(ctx,q,scope)

def assign(model,base,direction,step):
    with torch.no_grad():
        offset=0
        for p,v in zip(model.parameters(),base):
            p.copy_(v+step*torch.tensor(direction[offset:offset+p.numel()].reshape(tuple(p.shape)),dtype=p.dtype,device=p.device));offset+=p.numel()
    assert offset==len(direction)

def restore(model,base):
    with torch.no_grad():
        for p,v in zip(model.parameters(),base):p.copy_(v)

def register():
    plan=review();configure();assert not OUT.exists() and read(PREP/'qualification.json')['input_qualification_passed'];OUT.mkdir()
    model=CurrentInputBoundary();assert sum(p.numel() for p in model.parameters())==1060832
    torch.save(dict(state=model.state_dict(),seed=15901,parameters=1060832),OUT/'initial.pt')
    # Warm only a separately counted dummy, never a candidate/data forward.
    dummy=torch.nn.Linear(1,1,dtype=torch.float64).cuda();dummy(torch.ones(1,1,dtype=torch.float64,device='cuda')).sum().backward()
    import v159_boundary_evaluate_v2
    seal(__file__,{OUT/'initial.pt'})
    save(OUT/'registration.json',dict(status='V159_six_fixed_fits_sealed_before_official_calls',official_fits=0,official_classifier_calls=0,official_features_calls=0,official_gradients=0,official_updates=0,setup_dummy_forwards=1,setup_dummy_gradients=1,setup_dummy_updates=0,seal_sha256=sha(OUT/'run_seal.json')))
    emit(event='V159_registered_without_official_forward',fits=0)

def preflight():
    plan=require(__file__);configure();assert not (OUT/'preflight.json').exists();reports=[];joint=[]
    for f in range(3):
        ctx=context(f);budget=plan['role_call_budgets'][f];previous=None
        for arm in ['A','B']:
            folder=OUT/f'preflight{f}_{arm}';folder.mkdir();model=model_for();before=tensor_hash(model.state_dict())
            counter=Counter(model,budget['preflight_classifier_cap_'+arm],folder/'calls.jsonl',4 if arm=='A' else 0)
            oo=probabilities(model,ctx,'OOF',ctx['ids']);repeat=probabilities(model,ctx,'OOF',ctx['ids']);de=probabilities(model,ctx,'deployment',np.arange(22546));dr=probabilities(model,ctx,'deployment',np.arange(22546))
            assert np.array_equal(oo[ctx['ids']],repeat[ctx['ids']]) and np.array_equal(de,dr)
            if previous is not None:assert np.array_equal(oo[ctx['ids']],previous[0][ctx['ids']]) and np.array_equal(de,previous[1])
            previous=(oo,de);gaps={}
            for scope,q,ids in [('OOF',oo,ctx['ids']),('deployment',de,np.arange(22546))]:
                raw=ctx[scope][ids].mean(1);gap=float(np.abs(raw-q[ids]).max());assert gap<=3e-12 and np.array_equal(q[ids].argmax(1),raw.argmax(1));gaps[scope]=gap
            assert stats(ctx,de,'deployment')['mastered'] and stats(ctx,oo,'OOF')['protected_regressions']==0
            grad_report=None
            if arm=='A':
                gs=[]
                for repetition in range(2):
                    gg=[]
                    for cls in [1,2]:
                        riskv,q,lp,g=risk(model,ctx,'OOF',ctx['ids'],counter,cls);assert np.array_equal(q[ctx['ids']],oo[ctx['ids']]);gg.append(g)
                    gs.append(gg)
                assert all(np.array_equal(gs[0][j],gs[1][j]) for j in [0,1])
                grad_report=dict(class_norms=[float(np.linalg.norm(g)) for g in gs[0]],repeated_complete_class_gradients_identical=True,hidden_initial_gradients_zero=all(np.count_nonzero(g[:-48])==0 for g in gs[0]),output_gradient_nonzero=all(np.linalg.norm(g[-48:])>0 for g in gs[0]))
                assert grad_report['hidden_initial_gradients_zero'] and grad_report['output_gradient_nonzero']
                joint.append(rows(ctx,de,None,'deployment'))
            assert tensor_hash(model.state_dict())==before
            np.save(folder/'OOF_probability.npy',oo);np.save(folder/'deployment_probability.npy',de)
            reports.append(dict(fold=f,arm=arm,origin_gaps=gaps,counts=counter.counts(),OOF_stats=stats(ctx,oo,'OOF'),deployment_stats=stats(ctx,de,'deployment'),gradients=grad_report,parameters_unchanged=True))
            counter.close();del model;gc.collect();torch.cuda.empty_cache()
    protected=retention(pd.concat(joint,ignore_index=True));assert protected['passed']
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in OUT.glob('preflight*/*')}
    save(OUT/'preflight.json',dict(status='actual_all_role_origin_and_repeated_full_class_gradient_preflight_passed',reports=reports,joint_TRAIN_retention=protected,
        actual_head_forward_calls=sum(z['counts']['head_attempts'] for z in reports),actual_opinion_feature_calls=sum(z['counts']['feature_attempts'] for z in reports),actual_full_class_gradients=sum(z['counts']['gradient_attempts'] for z in reports),official_fits=0,official_updates=0,source_sha256=sources,quality_acceptance=False))
    emit(event='V159_real_preflight_complete',head_calls=sum(z['counts']['head_attempts'] for z in reports),full_class_gradients=12)

def fit(f,arm):
    plan=require(__file__);configure();activation=read(OUT/'preflight.json');assert activation['joint_TRAIN_retention']['passed']
    from experiment_review import check_bindings
    check_bindings(activation['source_sha256']);assert f in [0,1,2] and arm in ['A','B']
    folder=OUT/f'fold{f}_{arm}';assert not folder.exists() and len(list(OUT.glob('fold*_*/started.json')))<6;folder.mkdir()
    ctx=context(f);model=model_for();budget=plan['role_call_budgets'][f];counter=Counter(model,budget['fit_classifier_cap_per_arm'],folder/'calls.jsonl',400)
    initial=tensor_hash(model.state_dict());save(folder/'started.json',dict(fold=f,arm=arm,source_seal_sha256=sha(OUT/'run_seal.json'),original_class_mass=ctx['mass'].tolist(),initial_parameter_sha256=initial,score_selected=False))
    initial_risk,qo,lpo,_=risk(model,ctx,'OOF',ctx['ids']);de,ld=probabilities(model,ctx,'deployment',ctx['ids'],True)
    assert np.array_equal(qo[ctx['ids']],np.load(OUT/f'preflight{f}_{arm}/OOF_probability.npy')[ctx['ids']]) and np.array_equal(de[ctx['ids']],np.load(OUT/f'preflight{f}_{arm}/deployment_probability.npy')[ctx['ids']])
    record(folder,ctx,'OOF',qo,lpo,0);record(folder,ctx,'deployment',de,ld,0)
    accepted=proposals=iterations=0;step=1.;last=deque(maxlen=5);history=[];termination='gradient_iteration_budget';start=time.monotonic()
    proposals_log=(folder/'proposals.jsonl').open('w',encoding='utf-8',buffering=1)
    gradients_log=(folder/'directions.jsonl').open('w',encoding='utf-8',buffering=1)
    while iterations<200 and accepted<200:
        if proposals>=600:termination='proposal_budget';break
        iterations+=1;gs=[];base_risks=None
        for cls in [1,2]:
            value,q,lp,g=risk(model,ctx,'OOF',ctx['ids'],counter,cls);gs.append(g)
            if base_risks is not None:assert np.array_equal(value,base_risks)
            base_risks=value
        direction=class_direction(*gs,float(ctx['mass'][1]),float(ctx['mass'][2]),arm)
        gradients_log.write(json.dumps(dict(iteration=iterations,full_class_gradients=counter.gradient_attempts,base_risks=base_risks.tolist(),class_norms=[float(np.linalg.norm(g)) for g in gs],direction_status=direction['status'],M_weight=direction['M_weight'],raw_infinity_norm=direction['raw_infinity_norm'],class_slopes=direction['class_slopes'],parameter_sha256=tensor_hash(model.state_dict())))+'\n')
        if direction['status']!='direction_qualified_for_finite_guarded_proposal':termination=direction['status'];break
        base=tuple(p.detach().clone() for p in model.parameters());ok=False
        for backtrack in range(40):
            if proposals>=600 or step<2**-40:break
            assign(model,base,direction['direction'],step);proposals+=1
            trial,qt,lpt,_=risk(model,ctx,'OOF',ctx['ids']);qd,lpd=probabilities(model,ctx,'deployment',ctx['ids'],True)
            os=stats(ctx,qt,'OOF');ds=stats(ctx,qd,'deployment');guard=ds['mastered'] and ds['new_errors_vs_initial']==0 and (arm=='A' or os['protected_regressions']==0)
            ok=finite_armijo(base_risks,trial,direction['class_slopes'],*ctx['mass'][1:],arm,step,guard)
            identity=tensor_hash(model.state_dict());proposals_log.write(json.dumps(dict(proposal=proposals,iteration=iterations,backtrack=backtrack,step=step,base_risks=base_risks.tolist(),trial_risks=trial.tolist(),class_slopes=direction['class_slopes'],Armijo_bounds=(base_risks+1e-4*step*np.asarray(direction['class_slopes'])).tolist(),classification_guard=guard,OOF_stats=os,deployment_stats=ds,accepted=ok,parameter_sha256=identity))+'\n')
            if ok:step=min(1.,step*2);break
            restore(model,base);step*=.5
        if not ok:
            restore(model,base);termination='proposal_budget' if proposals>=600 else 'no_feasible_step';break
        assert any(not torch.equal(p,v) for p,v in zip(model.parameters(),base))
        accepted+=1;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        os=record(folder,ctx,'OOF',qt,lpt,accepted);ds=record(folder,ctx,'deployment',qd,lpd,accepted)
        history.append(dict(update=accepted,iteration=iterations,proposals=proposals,full_class_gradients=counter.gradient_attempts,stable_class_risks=trial.tolist(),OOF_stats=os,deployment_stats=ds,parameter_sha256=identity));save(folder/'progress.json',history)
        last.append(dict(update=accepted,state=state,identity=identity,OOF_stats=os,deployment_stats=ds))
        if accepted in [1,20,50,100,150,200]:torch.save(dict(state=state,update=accepted),folder/f'accepted{accepted}.pt')
        emit(event='V159_accepted',fold=f,arm=arm,update=accepted,iteration=iterations,proposals=proposals,OOF=os,stable_class_risks=trial.tolist())
    if accepted==200:termination='accepted_update_budget'
    gradients_log.close();proposals_log.close()
    final_risks,qo,lpo,_=risk(model,ctx,'OOF',ctx['ids']);qd,lpd=probabilities(model,ctx,'deployment',np.arange(22546),True);assert stats(ctx,qd,'deployment')['mastered']
    if arm=='B':assert stats(ctx,qo,'OOF')['protected_regressions']==0
    for scope,q,lp in [('OOF',qo,lpo),('deployment',qd,lpd)]:
        np.save(folder/f'endpoint_{scope}_probability.npy',q);rows(ctx,q,lp,scope).to_parquet(folder/f'endpoint_{scope}_rows.parquet',index=False)
    for item in last:
        target=folder/f"accepted{item['update']}.pt"
        if not target.exists():torch.save(dict(state=item['state'],update=item['update']),target)
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},fold=f,arm=arm,seal_sha256=sha(OUT/'run_seal.json')),folder/'endpoint.pt')
    r=dict(status='V159_boundary_fit_executed',fold=f,arm=arm,gradient_iterations=iterations,full_class_gradients=counter.gradient_attempts,proposal_evaluations=proposals,accepted_updates=accepted,termination=termination,selected_by_score=False,
        counts=counter.counts(),initial_parameter_sha256=initial,endpoint_parameter_sha256=tensor_hash(model.state_dict()),initial_stable_class_risks=initial_risk.tolist(),final_stable_class_risks=final_risks.tolist(),OOF_stats=stats(ctx,qo,'OOF'),deployment_stats=stats(ctx,qd,'deployment'),
        last5=[dict(update=i['update'],parameter_sha256=i['identity'],OOF_stats=i['OOF_stats'],deployment_stats=i['deployment_stats']) for i in last],elapsed_seconds=time.monotonic()-start,
        model_sha256=sha(folder/'endpoint.pt'),OOF_probability_sha256=sha(folder/'endpoint_OOF_probability.npy'),deployment_probability_sha256=sha(folder/'endpoint_deployment_probability.npy'),quality_acceptance=False)
    endpoint(r);counter.close();require(__file__);save(folder/'fit.json',r);emit(event='V159_fit_terminal',fold=f,arm=arm,termination=termination,updates=accepted)
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','preflight','fit','all-fits']);ap.add_argument('--fold',type=int);ap.add_argument('--arm',choices=['A','B']);a=ap.parse_args()
    try:
        if a.mode=='register':register()
        elif a.mode=='preflight':preflight()
        elif a.mode=='fit':fit(a.fold,a.arm)
        else:
            for f in range(3):
                for arm in ['A','B']:fit(f,arm)
    except Exception as e:
        if OUT.exists():save(OUT/f'execution_failure_{a.mode}_{a.fold}_{a.arm}.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),no_automatic_restart=True,source_sha256=sha(Path(__file__))))
        raise
