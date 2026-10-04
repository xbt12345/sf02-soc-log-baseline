"""Training-only numerical completion, prospectively logged as two extra fits."""
import argparse,time,numpy as np,pandas as pd,torch
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from v92_common import *
from v92_train import Branch,csr

def register():
    check();assert not (DEST/'numerical_completion_registration.json').exists()
    assert all(not read(DEST/(a+'_fit.json'))['converged'] for a in ['L0','U0'])
    reg={'source_sha256':sha(__file__),'arms':['L0R','U0R'],'additional_classifier_fits':2,
        'trigger':'Training-only400 steps failed gradient/CE stabilization; not triggered by C/H performance.',
        'input_and_loss':'Same R0/ASA gate/teacher/raw-row CE and original training roles; no validation labels or new expression.',
        'L0R':'Fresh zero linear residual, double SciPy L-BFGS, max1000, gtol1e-7, ftol1e-12, no added regularizer.',
        'U0R':'Continue saved U0 weights in double with full-batch strong-Wolfe L-BFGS, history20,max500; count as extra fit. P0 not resumed or used to compare protection cost.',
        'budget_seconds':480,'optimizer_convergence_grad_inf':1e-5,
        'promotion':'Both diagnostic only. Existing failed protected selection unchanged; no rotation/final fitting of these candidates.',
        'adaptive_not_blind':True,'role_labels_used_for_optimizer':'fit only'}
    save(DEST/'numerical_completion_registration.json',reg);event('numerical_completion_registration','completed',receipt=reg)

def main(name):
    check();reg=read(DEST/'numerical_completion_registration.json');assert reg['source_sha256']==sha(__file__);assert name in reg['arms'] and not (DEST/(name+'_fit.json')).exists()
    event('classifier_fit','started',name=name,reason='numerical_completion');start=time.monotonic()
    r,y,fid,z0,old,sel,fit=data();ids=np.load(DEST/'ASA_input_ids.npy');counts=np.load(DEST/'ASA_fit_counts.npy');used=np.flatnonzero(counts.sum(1));cc=counts[used].astype(float);xx=sparse.load_npz(DEST/'ASA_R0.npz');x=xx[used].astype(float);z=np.asarray(z0[ids[used]]);n=cc.sum();logs=[]
    if name=='L0R':
        width=x.shape[1]
        def objective(theta):
            w=theta.reshape(width+1,3);scores=z+x@w[:-1]+w[-1];p=np.exp(scores-logsumexp(scores,axis=1)[:,None]);res=(p*cc.sum(1)[:,None]-cc)/n
            return float((cc.sum(1)@logsumexp(scores,axis=1)-(cc*scores).sum())/n),np.vstack([x.T@res,res.sum(0)]).ravel()
        steps=[0];saved=[np.zeros((width+1)*3)]
        def cb(v):
            steps[0]+=1;saved[0]=v.copy()
            if steps[0]%50==0:
                f,g=objective(v);rec={'step':steps[0],'ASA_ce':f,'gradient_inf':float(np.abs(g).max()),'seconds':time.monotonic()-start};logs.append(rec);save(DEST/(name+'_progress.json'),logs);emit(stage='numerical_completion',name=name,**rec)
            if time.monotonic()-start>reg['budget_seconds']:raise TimeoutError('registered_fit_budget')
        try:
            res=minimize(objective,saved[0],jac=True,method='L-BFGS-B',callback=cb,options={'maxiter':1000,'maxcor':10,'maxls':30,'gtol':1e-7,'ftol':1e-12})
            theta=res.x;message=str(res.message);success=bool(res.success)
        except TimeoutError:theta=saved[0];message='registered_wall_budget';success=False
        f,g=objective(theta);gradient=float(np.abs(g).max());converged=success and gradient<=reg['optimizer_convergence_grad_inf'];w=theta.reshape(width+1,3)
        np.savez_compressed(DEST/(name+'_model.npz'),coef=w[:-1],intercept=w[-1]);scores=np.asarray(z0[ids])+xx@w[:-1]+w[-1];step=steps[0];loss=f
    else:
        torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';saved=torch.load(DEST/'U0_model.pt',map_location='cpu',weights_only=True)
        model=Branch(saved['width']);model.load_state_dict(saved['state_dict']);model=model.to(device).double();xt=csr(x,device);tz=torch.as_tensor(z,dtype=torch.float64,device=device);tc=torch.as_tensor(cc,dtype=torch.float64,device=device)
        opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=500,history_size=20,tolerance_grad=1e-7,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=[0]
        def closure():
            opt.zero_grad();zz=tz+model(xt);loss=(-torch.log_softmax(zz,1)*tc).sum()/n;loss.backward();calls[0]+=1
            if calls[0]%25==0:
                gradient=max(float(p.grad.abs().max().item()) for p in model.parameters());rec={'closure':calls[0],'ASA_ce':float(loss.item()),'gradient_inf':gradient,'seconds':time.monotonic()-start};logs.append(rec);save(DEST/(name+'_progress.json'),logs);emit(stage='numerical_completion',name=name,**rec)
            if time.monotonic()-start>reg['budget_seconds']:raise TimeoutError('registered_fit_budget')
            return loss
        try:opt.step(closure);message='LBFGS returned normally';success=True
        except TimeoutError:message='registered_wall_budget';success=False
        opt.zero_grad();zz=tz+model(xt);loss=(-torch.log_softmax(zz,1)*tc).sum()/n;loss.backward();gradient=max(float(p.grad.abs().max().item()) for p in model.parameters());loss=float(loss.item());converged=success and gradient<=reg['optimizer_convergence_grad_inf'];step=calls[0]
        scores=np.empty((len(ids),3))
        with torch.no_grad():
            for beg in range(0,len(ids),4096):scores[beg:beg+4096]=np.asarray(z0[ids[beg:beg+4096]])+model(csr(xx[beg:beg+4096].astype(float),device)).cpu().numpy()
        torch.save({'state_dict':model.cpu().state_dict(),'width':saved['width'],'linear':False,'dtype':'float64','continued_from':'U0'},DEST/(name+'_model.pt'));del model,xt;torch.cuda.empty_cache()
    np.save(DEST/(name+'_ASA_scores.npy'),scores);new=old.copy();new[ids]=scores.argmax(1).astype(np.int8);np.save(DEST/(name+'_prediction.npy'),new)
    roles,cells=evaluate(r,y,fid,old,new);pd.DataFrame(cells).to_csv(DEST/(name+'_classwise.csv'),index=False)
    report={'status':'completed_converged' if converged else 'stopped_not_convergence_proven','name':name,'new_classifier_fits':1,'converged':converged,'gradient_inf':gradient,'steps_or_closures':step,'message':message,'ASA_ce':loss,'roles':roles,'primary_eligible':False,'diagnostic_only':True,'selection_changed':False,'source_sha256':sha(__file__),'seconds':time.monotonic()-start}
    save(DEST/(name+'_fit.json'),report);event('classifier_fit',report['status'],name=name,converged=converged,diagnostic_only=True);emit(stage='numerical_complete',name=name,converged=converged,fit_errors=roles['fit']['errors'],inner_errors=roles['inner']['errors'],gradient=gradient)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['register','fit']);ap.add_argument('--arm');a=ap.parse_args()
    with threadpool_limits(limits=4):register() if a.stage=='register' else main(a.arm)
