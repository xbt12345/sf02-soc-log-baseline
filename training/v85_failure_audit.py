"""No-fit problem ledger and local feasible-direction diagnosis of trained arms."""
import json
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,read,save,sha,metrics
from v85_protection import DEST,fold_data,Residual,csr_tensor,supervised_loss,margin,feasibility,ALPHA


def main():
    if (DEST/'execution_receipt.json').exists():raise FileExistsError('Published evidence frozen')
    assert read(DEST/'verification.json')['status']=='passed'
    r,y,fid,x,cc,used,vc,fit,inner,folder=fold_data(1)
    old=np.load(folder/'teacher_prediction.npy');zs=np.load(folder/'teacher_scores.npy',mmap_mode='r');cohorts=[];summaries=[]
    cases=[]
    for d in read(folder/'diagnosis.json'):
        p=np.load(folder/(d['name']+'_all_prediction.npy'));oldrow=old[fid];newrow=p[fid]
        for role,mask in [('selected_fit',np.isin(r.row_position,np.load(folder/'selected_rows.npy'))),('inner',inner),('C',r.fold.eq(2)),('H',r.fold.eq(0))]:
            for kind,q in [('positive',(oldrow!=y)&(newrow==y)),('negative',(oldrow==y)&(newrow!=y)),('remaining_old_error',(oldrow!=y)&(newrow!=y))]:
                ix=np.flatnonzero(np.asarray(mask)&q)
                if not len(ix):continue
                part=r.iloc[ix][['row_position','fold','component','route','label_index']].copy()
                part['name']=d['name'];part['role']=role;part['kind']=kind;part['fid']=fid[ix];part['old']=oldrow[ix];part['new']=newrow[ix]
                cases.append(part)
    table=pd.concat(cases,ignore_index=True);table.to_parquet(DEST/'problem_casebook.parquet',index=False)
    grouped=table.groupby(['name','role','kind','route','label_index']).agg(rows=('row_position','size'),input_groups=('fid','nunique'),components=('component','nunique')).reset_index()
    grouped.to_csv(DEST/'problem_slices.csv',index=False)
    # Perturb saved C state once along its normalized data-loss gradient. No optimizer,
    # no adaptation, no persisted perturbed classifier; purely local train-only diagnosis.
    model=Residual().cuda();state=torch.load(folder/'C_epoch060.pt',weights_only=True,map_location='cuda');model.load_state_dict(state)
    xt=csr_tensor(x[used]);zt=torch.tensor(zs[used],dtype=torch.float32,device='cuda');ct=torch.tensor(cc[used],dtype=torch.float32,device='cuda')
    labels=torch.tensor(old[used].astype(np.int64),device='cuda');pmass=ct.gather(1,labels[:,None]).squeeze(1)
    eps=torch.minimum(torch.full_like(pmass,.001),margin(zt,labels)/2)
    loss=supervised_loss(zt+model(xt),ct);params=list(model.parameters());grads=torch.autograd.grad(loss,params)
    norm=torch.sqrt(sum(g.square().sum() for g in grads));original=[p.detach().clone() for p in params]
    directions=[-g.detach()/norm for g in grads]
    base_m=margin(zt+model(xt),labels).detach();slack=(base_m-eps).cpu().numpy();pactive=pmass.cpu().numpy()>0
    activeids=used[pactive&(slack<=1e-5)];active=[]
    for f in activeids:
        take=np.flatnonzero((fid==f)&fit)
        active.append({'fid':int(f),'protected_original_selected_rows':int(cc[f,old[f]]),'protected_class':int(old[f]),
          'slack':float(slack[np.searchsorted(used,f)]),'routes':r.iloc[take].route.unique().tolist(),'components':r.iloc[take].component.unique().tolist()})
    closest=[]
    for j in np.flatnonzero(pactive)[np.argsort(slack[pactive])[:10]]:
        f=used[j];take=np.flatnonzero((fid==f)&fit)
        closest.append({'fid':int(f),'class':int(old[f]),'original_selected_rows':int(cc[f,old[f]]),'slack':float(slack[j]),
                       'routes':r.iloc[take].route.unique().tolist(),'components':r.iloc[take].component.unique().tolist()})
    pcc=torch.zeros_like(ct);pcc.scatter_(1,labels[:,None],pmass[:,None]);ecc=ct-pcc
    scores=zt+model(xt);p_loss=supervised_loss(scores,pcc);e_loss=supervised_loss(scores,ecc)
    gp=torch.autograd.grad(p_loss,params,retain_graph=True);ge=torch.autograd.grad(e_loss,params)
    pn=torch.sqrt(sum(g.square().sum() for g in gp));en=torch.sqrt(sum(g.square().sum() for g in ge))
    cosine=sum((a*b).sum() for a,b in zip(gp,ge))/(pn*en)
    diagnostic_teacher=torch.tensor(zs[used],dtype=torch.float64,device='cuda');diagnostic_counts=ct.double()
    def stable_loss():return supervised_loss(diagnostic_teacher+model(xt).double(),diagnostic_counts)
    with torch.no_grad():stable_base=float(stable_loss())
    rays=[];projected=[]
    with torch.no_grad():
        for step in [1e-4,1e-3,1e-2]:
            for p,a,direction in zip(params,original,directions):p.copy_(a+step*direction)
            z=zt+model(xt);f=feasibility(z,labels,eps,pmass)
            violation=((eps-margin(z,labels))>1e-7)&(pmass>0)
            blocked=[]
            for j in torch.nonzero(violation,as_tuple=True)[0].cpu().numpy():
                f_id=used[j];take=np.flatnonzero((fid==f_id)&fit)
                blocked.append({'fid':int(f_id),'protected_class':int(old[f_id]),'original_rows':int(pmass[j].item()),
                   'routes':r.iloc[take].route.unique().tolist(),'components':r.iloc[take].component.unique().tolist()})
            data_loss=float(stable_loss())
            rays.append({'normalized_data_loss_gradient_step':step,'data_loss':data_loss,
                         'loss_change':data_loss-stable_base,'protection':f,'blocked_protected_inputs':blocked})
        for p,a in zip(params,original):p.copy_(a)
    # A single analytic tangent projection tests the search-direction hypothesis.
    # Not a new training run, full QP solver, or saved/selected model candidate.
    blocker=rays[0]['blocked_protected_inputs'][0]['fid'] if rays[0]['blocked_protected_inputs'] else None
    projection=None
    if blocker is not None:
        j=int(np.searchsorted(used,blocker));one_z=zt[j:j+1]+model(csr_tensor(x[used[j:j+1]]))
        boundary=margin(one_z,labels[j:j+1])[0]
        gh=torch.autograd.grad(boundary,params);hdot=sum((h*d).sum() for h,d in zip(gh,directions));hn2=sum(h.square().sum() for h in gh)
        tangent=[d-(hdot/hn2)*h for d,h in zip(directions,gh)]
        dot=sum((h*d).sum() for h,d in zip(gh,tangent))
        projection={'blocker_fid':int(blocker),'original_margin_directional_derivative':float(hdot),
           'projected_margin_directional_derivative':float(dot),'projection_scope':'One train-only constraint, local diagnostic; every P still checked.'}
        with torch.no_grad():
            for step in [1e-3,1e-2]:
                for p,a,direction in zip(params,original,tangent):p.copy_(a+step*direction)
                z=zt+model(xt);f=feasibility(z,labels,eps,pmass)
                projected.append({'step':step,'data_loss_change':float(stable_loss())-stable_base,'protection':f})
            for p,a in zip(params,original):p.copy_(a)
    # Pinned-state input-conflict lower bound is a finite empirical obstruction only.
    oldcc=cc[used];pm=oldcc[np.arange(len(used)),old[used]]
    floor=int((oldcc.sum(1)-oldcc.max(1)).sum());forced=int(np.where(pm>0,oldcc.sum(1)-pm,oldcc.sum(1)-oldcc.max(1)).sum())
    roles=[]
    for role,mask in [('selected_fit',np.isin(r.row_position,np.load(folder/'selected_rows.npy'))),('fit_full',fit),('inner',inner),('C',r.fold.eq(2)),('H',r.fold.eq(0))]:
        mask=np.asarray(mask);base=metrics(np.bincount(y[mask]*3+old[fid[mask]],minlength=9).reshape(3,3))
        roles.append({'role':role,'baseline':base})
    cprogress=read(folder/'C_progress.json')
    out={'source_sha256':sha(__file__),'new_classifier_fits_in_audit':0,'verification_sha256':sha(DEST/'verification.json'),
         'local_data_gradient_norm':float(norm),'C_final_data_loss':stable_base,'C_optimizer_float32_data_loss':float(loss.detach()),
         'diagnostic_loss_precision':'Float64 original teacher plus direct residual, float64 OVR accumulation; gradient direction remains registered float32 training loss.',
         'train_only_gradient_ray_checks':rays,
         'tight_P_input_groups_slack_le_1e_5':active,'tight_P_input_group_count':len(active),'closest_P_constraints':closest,
         'local_E_vs_P_loss_gradient_cosine':float(cosine),'P_data_loss':float(p_loss),'E_data_loss':float(e_loss),
         'single_constraint_tangent_projection':projection,'projected_direction_checks':projected,
         'fit_empirical_unrestricted_lookup_floor':floor,'fit_empirical_protected_lookup_floor':forced,
         'C_rejected_steps':sum(p['fully_rejected'] for p in cprogress),'C_first_rejected_epoch':next((p['epoch'] for p in cprogress if p['fully_rejected']),None),
         'C_epoch40_to60_weights_unchanged':all(torch.equal(torch.load(folder/'C_epoch040.pt',weights_only=True,map_location='cpu')[k],
                                                  torch.load(folder/'C_epoch060.pt',weights_only=True,map_location='cpu')[k]) for k in state),
         'baseline_roles':roles,'all_role_M_S_in_loss':True,'original_native_flow_M_preserved':int(((r.route=='native_flow')&(y==1)).sum()),
         'fold_training_native_flow_M':int(((r.route=='native_flow')&(y==1)&fit).sum()),
         'issues':[
          {'id':'P01','problem':'Freezing teacher does not protect final decisions','status':'confirmed_and_controlled_only_in_C_train','evidence':'A/B negative flips; C selected-fit zero negative flips and all saved margins independently verified.'},
          {'id':'P02','problem':'Selective KL is only average soft protection','status':'unresolved_as_safety_mechanism','evidence':'B saved stages still have negative flips; no promotion.'},
          {'id':'P03','problem':'Feasible backtracking stalls without changing search direction','status':'unresolved_optimizer_limit','evidence':'C38 backtracked,28 rejected; final local gradient perturbations recorded; not proof of no feasible nonlinear solution.'},
          {'id':'P04','problem':'Training protection does not transfer to held-out correct decisions','status':'unresolved','evidence':'C inner1 positive/negative=1/4; original validation M regressions remain.'},
          {'id':'P05','problem':'Full normal population not in primary sampled-normal loss','status':'tracked_not_claimed_complete','evidence':'B one extra normal regression in full fit; all unselected fit normals replayed, expansion stopped by gate.'},
          {'id':'P06','problem':'Complete-input mixed labels make part of error irreducible under fixed encoding','status':'recorded_not_deleted','evidence':{'unrestricted_empirical_floor':floor,'protected_empirical_floor':forced,'scope':'Finite current encoded training input only; not raw-data Bayes floor.'}},
          {'id':'P07','problem':'Crosscomponent M/S evidence, VPC M absence and unseen-format supervision','status':'still_unresolved','evidence':'No new external samples, pseudo labels or target-answer training; contrastive/zero-M/full-fit phases stopped by primary gate.'}
         ],
         'implementation_repairs':[
            {'phase':'before_training_registration','issue':'Snapshot subtracting two float32 teacher logits lost tiny residuals','repair':'Persist decisions using direct residual plus original float64 teacher; mathematical cancellation test passed.'},
            {'phase':'post_training_no_fit_audit','issue':'torch.flatnonzero is not an available API in installed torch','repair':'Use torch.nonzero(as_tuple=True); failed log retained; classifiers and registered source unchanged.'}
            ,{'phase':'independent_no_fit_audit_verification','issue':'Float32 BCE reduction order dominated tiny local loss differences','repair':'Use float64 original teacher plus direct residual and float64 loss accumulation in both local diagnostic formulas. Training protocol and model weights unchanged.'}
         ],
         'next_step':'Do not extend failed Adam/backtracking budget. First diagnose and solve active-constraint tangent direction on train-only data, then require fixed component-fold gains with zero negative flips before adding crosscomponent supervision. No label-aware inference gate.',
         'scope':'No-fit arithmetic, case counts and finite local train-only gradient perturbations of saved C. No new optimized candidate, no unknown-data guarantee.'}
    save(DEST/'problem_ledger.json',out)
    print(json.dumps({k:v for k,v in out.items() if k in ['fit_empirical_unrestricted_lookup_floor','fit_empirical_protected_lookup_floor','tight_P_input_group_count','C_rejected_steps','C_first_rejected_epoch','C_epoch40_to60_weights_unchanged','original_native_flow_M_preserved','fold_training_native_flow_M','train_only_gradient_ray_checks']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=4):main()
