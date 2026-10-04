"""Four fresh training-side component-withheld diagnostics, no reused teacher."""
import argparse,time,numpy as np,pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from v92_common import *

def design():
    check();assert not (DEST/'rare_rotation_registration.json').exists();r,y,fid,z,old,sel,fit=data()
    pool=np.load(V89/'support_full_rows.npy');facts=[__import__('json').loads(s) for s in pd.read_parquet(OUT/'projections.parquet').facts]
    fields=['action','outcome','transport_protocol','src_role','dst_role','dst_port_fixed'];keys=[tuple(str(f[k]) for k in fields) if all(k in f for k in fields) else None for f in facts]
    rare=[a for a in read(ROOT/'artifacts/v91_first_principles_20260928/audit.json')['fine_S_independent_support'] if a['S_components']==2];conditions=[]
    for number,rec in enumerate(rare):
        key=tuple(rec['key']);target=np.array([i for i in pool if y[i]==2 and keys[int(r.projection_id.iloc[i])]==key]);comps=sorted(r.component.iloc[target].unique());assert len(comps)==2
        for direction,component in enumerate(comps):
            name=f'rare{number}_leave{direction}';train=pool[~r.component.iloc[pool].eq(component).to_numpy()&r.route.iloc[pool].eq('asa').to_numpy()]
            test=pool[r.component.iloc[pool].eq(component).to_numpy()&r.route.iloc[pool].eq('asa').to_numpy()];target_test=target[r.component.iloc[target].eq(component).to_numpy()]
            assert len(test)>0 and not set(r.component.iloc[train])&set(r.component.iloc[test]);assert len(set(r.component.iloc[target[~np.isin(target,target_test)]]))==1
            np.savez_compressed(DEST/(name+'_roles.npz'),train=train,test=test,target=target_test)
            conditions.append({'name':name,'behavior_key':list(key),'heldout_component':int(component),'remaining_same_class_behavior_components':1,'train_class_counts':np.bincount(y[train],minlength=3).tolist(),'test_class_counts':np.bincount(y[test],minlength=3).tolist(),'target_S_rows':len(target_test)})
    reg={'source_sha256':sha(__file__),'conditions':conditions,'additional_classifier_fits':len(conditions),'teacher_fits':0,'initialization':'Fresh zero linear coefficients, role-local prior intercept; no old-model parameters or logits.',
        'loss':'Role-local original ASA-row-frequency softmax CE + 1e-6/2 coefficient square; fixed R0, no vocabulary fitting.',
        'solver':'L-BFGS-B max1000,gtol1e-6,ftol1e-12,maxcor10,maxls30; success and grad_inf<=1e-5 for convergence.',
        'purpose':'Two directional diagnostics per two-component rare slice; no label feedback or promotion selection. A second validation/constraint role cannot be invented.',
        'scope':'Predefined training population excludes v89 six support holdout components. Does not estimate general-world probability; some test directions lack competing-class support.'}
    save(DEST/'rare_rotation_registration.json',reg);event('rare_rotation_registration','completed',receipt=reg);emit(stage='rare_rotation_registered',conditions=conditions)

def main(name):
    check();reg=read(DEST/'rare_rotation_registration.json');assert reg['source_sha256']==sha(__file__);spec=next(a for a in reg['conditions'] if a['name']==name);assert not (DEST/(name+'_fit.json')).exists();event('classifier_fit','started',name=name,reason='rare_component_rotation');start=time.monotonic()
    r,y,fid,z,old,sel,fit=data();ids=np.load(DEST/'ASA_input_ids.npy');x=sparse.load_npz(DEST/'ASA_R0.npz').astype(float);roles=np.load(DEST/(name+'_roles.npz'));train=roles['train'];test=roles['test'];target=roles['target']
    code=np.searchsorted(ids,fid);cc=np.bincount(code[train]*3+y[train],minlength=len(ids)*3).reshape(-1,3);used=np.flatnonzero(cc.sum(1));xx=x[used];counts=cc[used];n=counts.sum();width=x.shape[1];initial=np.zeros((width+1,3));initial[-1]=np.log(np.maximum(counts.sum(0)/n,1e-12));steps=[0]
    def objective(theta):
        w=theta.reshape(width+1,3);z=np.asarray(xx@w[:-1])+w[-1];res=(np.exp(z-logsumexp(z,axis=1)[:,None])*counts.sum(1)[:,None]-counts)/n
        return float((counts.sum(1)@logsumexp(z,axis=1)-(counts*z).sum())/n+1e-6/2*np.square(w[:-1]).sum()),np.vstack([xx.T@res+1e-6*w[:-1],res.sum(0)]).ravel()
    def cb(theta):
        steps[0]+=1
        if steps[0]%100==0:
            f,g=objective(theta);emit(stage='rare_rotation_fit',name=name,iteration=steps[0],loss=f,gradient=float(np.abs(g).max()),seconds=time.monotonic()-start)
    res=minimize(objective,initial.ravel(),jac=True,method='L-BFGS-B',callback=cb,options={'maxiter':1000,'gtol':1e-6,'ftol':1e-12,'maxcor':10,'maxls':30});f,g=objective(res.x);w=res.x.reshape(width+1,3);conv=bool(res.success and np.abs(g).max()<=1e-5)
    np.savez_compressed(DEST/(name+'_model.npz'),coef=w[:-1],intercept=w[-1]);score=x[code[test]]@w[:-1]+w[-1];pred=score.argmax(1).astype(np.int8);np.savez_compressed(DEST/(name+'_predictions.npz'),rows=test,scores=score,prediction=pred)
    mask=np.isin(test,target);cm=lambda t,p:np.bincount(t*3+p,minlength=9).reshape(3,3).tolist()
    result={'name':name,'status':'completed_converged' if conv else 'stopped_not_convergence_proven','new_classifier_fits':1,'converged':conv,'objective':f,'gradient_inf':float(np.abs(g).max()),'iterations':steps[0],'message':str(res.message),'component_disjoint':True,'all_heldout_cm':cm(y[test],pred),'target_cm':cm(y[test[mask]],pred[mask]),'target_S_correct':int((pred[mask]==2).sum()),'target_S_total':int(mask.sum()),'target_row_ids':target.tolist(),'diagnostic_only':True,'source_sha256':sha(__file__),'seconds':time.monotonic()-start}
    save(DEST/(name+'_fit.json'),result);event('classifier_fit',result['status'],name=name,converged=conv,diagnostic_only=True);emit(stage='rare_rotation_complete',**result)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['register','fit']);ap.add_argument('--condition');a=ap.parse_args()
    with threadpool_limits(limits=4):design() if a.stage=='register' else main(a.condition)
