"""Equal-role CE controls: full R0 residual, named increment, and strict P."""
import argparse,copy,time,traceback
import numpy as np,pandas as pd,torch
from torch import nn
from scipy import sparse
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from v92_common import *
from v81_training_contract import compare

class Branch(nn.Module):
    def __init__(self,width,linear=False):
        super().__init__();self.linear=linear
        if linear:self.out=nn.Linear(width,3)
        else:self.first=nn.Linear(width,64,bias=False);self.second=nn.Linear(64,16);self.out=nn.Linear(16,3)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x):
        if self.linear:return torch.sparse.mm(x,self.out.weight.T)+self.out.bias
        h=torch.sparse.mm(x,self.first.weight.T)
        return self.out(nn.functional.gelu(self.second(nn.functional.gelu(h,approximate='tanh')),approximate='tanh'))

def csr(x,device):
    x=x.tocsr();return torch.sparse_csr_tensor(torch.as_tensor(x.indptr.astype(np.int64),device=device),
        torch.as_tensor(x.indices.astype(np.int64),device=device),torch.as_tensor(x.data,device=device),size=x.shape,device=device)

def margin(z,label):
    return z.gather(1,label[:,None]).ravel()-z.masked_fill(nn.functional.one_hot(label,3).bool(),-torch.inf).max(1).values

def register():
    assert not (DEST/'registration.json').exists();prep=read(DEST/'preparation.json');assert prep['status']=='complete'
    assert read(DEST/'execution_scope_audit.json')['ASA_target_identity_failures']==0
    assert read(DEST/'representation_contract.json')['known_missing_rows_in_target']==0
    assert prep['S_input_groups_traced']==26 and prep['M_regression_rows_traced']==668
    spec={'version':'v92-v91-plan-execution-1','seed':SEED,'arms':['L0','U0','P0'],
        'classifier_fits':3,'calibration_fits':0,'teacher_fits':0,'domain_classifier_fits':0,
        'teacher':'Reuse same fold-local v85 teacher, parameter rows folds -1/3/4, inner1, C2/H0. No mismatched role teacher.',
        'scope':'ASA-gated residual on full R0. Exact raw-row 922324-population softmax CE: all M/S and all benign. Non-ASA losses constant with unchanged predictions, explicitly supervised.',
        'input_increment':'U1/P1 skipped: no target known-field omission verified. Prepared expression-only dictionary is not fed to any fit. Full R0 retained.',
        'equal_capacity':'U/P hidden64/16 and identical input width/seed/zero output. L0 is separately declared linear capacity reference.',
        'protection':'P actual full-fit old-correct rows. epsilon=min(.001,oldmargin/2), float64 logits, tolerance1e-10. Full P checked every proposal, <=16 halvings; reject and restore optimizer on failure.',
        'loss':'Exact full original-row mean 3-class cross entropy; no E slack constraint, class/repeat weights, pseudo-label, selective distillation or calibration.',
        'optimizer':'full-batch Adam','lr':.003,'max_steps':400,'seconds_per_fit':480,'fully_rejected_stop':10,
        'convergence':'ASA-normalized gradient infinity<=1e-4 and last20 ASA-CE range<1e-6. Fixed max step/wall stops do not prove optimum; P effective steps required.',
        'selection':'Final training snapshot only; no validation-selected epoch. Inner full zero negative flips, all-class guard, ASA repair>0, full-fit zero negative flips; free arms remain diagnostic only. Pick P inner errors then lexicographic id. Freeze selection before C/H diagnosis.',
        'continuation':'Only numerically completed, protected eligible arm proceeds to rotation/final gates; no eligible arm stops expansion and final fitting.',
        'roles':'All component-disjoint existing development; no new blind-test claim, no forced rare A/B/V split. No new constraint-label feedback.',
        'source_bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'training/v92_common.py',ROOT/'training/v92_prepare.py',ROOT/'training/v92_train.py',ROOT/'training/v81_training_contract.py',ROOT/'training/v85_protection.py',ROOT/'training/v89_common.py',ROOT/'docs/V91_FIRST_PRINCIPLES_RESEARCH_AND_PLAN.md']},
        'input_bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in [OUT/'rows.parquet',ROOT/'data/official/train.parquet',OLD/'teacher_scores.npy',DEST/'ASA_R0.npz',DEST/'ASA_increment.npz',DEST/'ASA_input_ids.npy',DEST/'ASA_fit_counts.npy',DEST/'fit_counts.npy',DEST/'increment_encoder.joblib',DEST/'preparation.json']}}
    save(DEST/'registration.json',spec);event('registration','completed',arms=spec['arms'],classifier_fits=3);emit(stage='registered',arms=spec['arms'])

def fit_branch(name):
    reg=check();assert name in reg['arms'];assert not (DEST/(name+'_fit.json')).exists();event('classifier_fit','started',name=name)
    start=time.monotonic();torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(SEED)
    r,y,fid,z0,old,sel,fit=data();ids=np.load(DEST/'ASA_input_ids.npy');cc=np.load(DEST/'ASA_fit_counts.npy');used=np.flatnonzero(cc.sum(1));total=int(fit.sum());asaN=int(cc.sum())
    xx=sparse.load_npz(DEST/'ASA_R0.npz');xt=csr(xx[used],device)
    model=Branch(xx.shape[1],name=='L0').to(device);opt=torch.optim.Adam(model.parameters(),lr=reg['lr'])
    teacher=torch.as_tensor(np.asarray(z0[ids[used]]),dtype=torch.float64,device=device);truth=torch.as_tensor(cc[used],dtype=torch.float64,device=device)
    oldlabel=torch.as_tensor(old[ids[used]],dtype=torch.long,device=device);pmass=truth.gather(1,oldlabel[:,None]).ravel();P=pmass>0
    eps=torch.minimum(torch.full_like(pmass,.001),margin(teacher,oldlabel)/2);assert torch.all(eps[P]>0)
    allcc=np.load(DEST/'fit_counts.npy');outside=np.ones(len(z0),bool);outside[ids]=False
    non=np.flatnonzero(outside&(allcc.sum(1)>0));constant=float((allcc[non].sum(1)*logsumexp(z0[non],axis=1)-(allcc[non]*z0[non]).sum(1)).sum()/total)
    protected=name.startswith('P');logs=[];accepted=0;rejected=0;consecutive=0;losses=[];converged=False;stop='max_steps';gradient=None
    def feasibility():
        with torch.no_grad():
            zz=teacher+model(xt).double();v=(eps[P]-margin(zz,oldlabel)[P]).clamp_min(0)
            return float(v.max().item()),int(pmass[(zz.argmax(1)!=oldlabel)&P].sum().item())
    assert feasibility()[1]==0
    for step in range(1,reg['max_steps']+1):
        model.train();opt.zero_grad();zz=teacher+model(xt).double();roi=((-torch.log_softmax(zz,1)*truth).sum())/asaN
        loss=roi*(asaN/total)+constant;loss.backward();gradient=max(float(p.grad.abs().max().item()) for p in model.parameters() if p.grad is not None)*(total/asaN)
        previous=copy.deepcopy(model.state_dict());moments=copy.deepcopy(opt.state_dict()) if protected else None;opt.step();fraction=1.;wasaccepted=True
        if protected:
            proposal=copy.deepcopy(model.state_dict());wasaccepted=False
            for bt in range(17):
                model.load_state_dict({k:previous[k]+fraction*(proposal[k]-previous[k]) for k in previous})
                v,nf=feasibility()
                if v<=1e-10 and nf==0:wasaccepted=True;break
                fraction/=2
            if not wasaccepted:model.load_state_dict(previous);opt.load_state_dict(moments)
        if wasaccepted:accepted+=1;consecutive=0
        else:rejected+=1;consecutive+=1
        with torch.no_grad():
            finalz=teacher+model(xt).double();ce=float((-torch.log_softmax(finalz,1)*truth).sum().item()/asaN)
            pred=finalz.argmax(1);errors=int((truth.sum(1)-truth.gather(1,pred[:,None]).ravel()).sum().item())
        losses.append(ce)
        if step==1 or step%10==0:
            v,nf=feasibility();rec={'step':step,'ASA_ce':ce,'full_population_ce':constant+ce*asaN/total,'ASA_fit_errors':errors,'ASA_gradient_inf':gradient,'accepted_steps':accepted,'rejected_steps':rejected,'last_step_fraction':fraction if wasaccepted else 0.,'protected_margin_violation':v,'P_negative_flips':nf,'seconds':time.monotonic()-start}
            logs.append(rec);save(DEST/(name+'_progress.json'),logs);emit(stage='training',name=name,**rec)
        if consecutive>=reg['fully_rejected_stop']:stop='no_feasible_update_path';break
        if step>=30 and gradient<=1e-4 and max(losses[-20:])-min(losses[-20:])<1e-6:
            converged=True;stop='training_convergence_criterion';break
        if time.monotonic()-start>=reg['seconds_per_fit']:stop='registered_wall_budget';break
    model.eval();torch.save({'state_dict':model.cpu().state_dict(),'width':xx.shape[1],'linear':name=='L0','increment':name in ['U1','P1'],'seed':SEED},DEST/(name+'_model.pt'));model.to(device)
    scores=np.empty((len(ids),3),np.float64)
    with torch.no_grad():
        for beg in range(0,len(ids),4096):scores[beg:beg+4096]=np.asarray(z0[ids[beg:beg+4096]])+model(csr(xx[beg:beg+4096],device)).cpu().double().numpy()
    np.save(DEST/(name+'_ASA_scores.npy'),scores);new=old.copy();new[ids]=scores.argmax(1).astype(np.int8);np.save(DEST/(name+'_prediction.npy'),new)
    roles,cells=evaluate(r,y,fid,old,new,role_names=['fit','inner']);primary={k:roles[k] for k in ['fit','inner']}
    cells=[c for c in cells if c['role'] in ['fit','inner']];pd.DataFrame(cells).to_csv(DEST/(name+'_primary_classwise.csv'),index=False)
    asa=[c for c in cells if c['role']=='inner' and c['route']=='asa'];v,nf=feasibility();guard=compare(cm_from_counts(raw_counts(fid,y,r.fold.eq(1).to_numpy(),len(old)),old),np.asarray(primary['inner']['cm']),True)
    numeric_valid=converged and accepted>0 and (not protected or (v<=1e-10 and nf==0))
    eligible=protected and numeric_valid and primary['fit']['negative_flips']==0 and primary['inner']['negative_flips']==0 and guard['eligible'] and sum(c['repairs'] for c in asa)>0
    report={'status':'completed_converged' if converged else 'stopped_not_convergence_proven','name':name,'new_classifier_fits':1,'new_calibration_fits':0,'converged':converged,'stop_reason':stop,'steps':step,'accepted_steps':accepted,'rejected_steps':rejected,'ASA_gradient_inf':gradient,'last_ASA_ce':ce,'full_population_ce':constant+ce*asaN/total,'ASA_training_original_rows':asaN,'full_population_original_rows':total,'all_fit_M_S_used':True,'all_fit_benign_used':True,'protected':protected,'P_original_rows':int(allcc[np.arange(len(old)),old].sum()),'P_ASA_original_rows':int(pmass.sum().item()),'final_P_margin_violation':v,'final_P_negative_flips':nf,'numeric_comparison_valid':numeric_valid,'primary_eligible':bool(eligible),'inner_guard':guard,'roles':primary,'seconds':time.monotonic()-start,'device':device,'source_sha256':sha(__file__),'model_sha256':sha(DEST/(name+'_model.pt'))}
    save(DEST/(name+'_fit.json'),report);event('classifier_fit',report['status'],name=name,converged=converged,accepted_steps=accepted,stop_reason=stop,eligible=bool(eligible));emit(stage='fit_complete',name=name,converged=converged,stop=stop,inner=primary['inner'],eligible=bool(eligible))
    del model,xt;torch.cuda.empty_cache()

def summarize():
    reg=check();reports={n:read(DEST/(n+'_fit.json')) for n in reg['arms']};assert not (DEST/'selection.json').exists()
    eligible=[n for n,a in reports.items() if a['primary_eligible']];selected=min(eligible,key=lambda n:(reports[n]['roles']['inner']['errors'],n)) if eligible else None
    save(DEST/'selection.json',{'selected':selected,'selected_before_C_H_diagnosis':True,'quality_promoted':False,'reason':'No numerically valid protected arm passed strict inner guards' if selected is None else 'Proceed to rotation gates, not promoted'})
    event('selection','frozen',selected=selected)
    r,y,fid,z,old,sel,fit=data();cells=[];full={}
    for name in reg['arms']:
        roles,c=evaluate(r,y,fid,old,np.load(DEST/(name+'_prediction.npy')));full[name]=roles;cells.extend([dict(arm=name,**a) for a in c])
    pd.DataFrame(cells).to_csv(DEST/'all_roles_classwise.csv',index=False);save(DEST/'full_role_diagnosis.json',full)
    result={'status':'registered_experiments_executed','actual_classifier_fits':len(reports),'actual_calibration_fits':0,'actual_teacher_fits':0,'selected':selected,'quality_acceptance':False,'model_promoted':False,'continued_to_rotation':False,'new_full_training':False,'new_full_task_replay':False,'reports':reports,
        'scope':'Existing inspected official-data roles. C/H diagnosis after primary selection locked; not independent blind validation. Conditional expansion requires eligible protected candidate.'}
    save(DEST/'results.json',result);event('summary','completed',new_classifier_fits=len(reports),selected=selected);emit(stage='summary',selected=selected,new_fits=len(reports))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['register','fit','summarize']);ap.add_argument('--arm');a=ap.parse_args()
    with threadpool_limits(limits=4):
        try:register() if a.stage=='register' else fit_branch(a.arm) if a.stage=='fit' else summarize()
        except Exception as exc:
            if DEST.exists():event('runtime_problem','failed',exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            raise
