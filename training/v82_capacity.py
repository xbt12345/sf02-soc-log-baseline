"""Official-only matched full-R0 capacity trial. Never opens target answers.

Original sparse columns are retained verbatim; a fit-only Nystrom RBF branch
adds interactions, not new evidence. Original row/class counts survive grouping.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, read, save, sha, load_sparse, metrics
from v79_execute import rows
from v81_training_contract import accept_candidate, compare

DEST=ROOT/'artifacts/v82_capacity_20260927'
LAST=ROOT/'artifacts/v79_execution_20260927'
OLD=ROOT/'artifacts/v78_boundary_20260922'
SEED=8201
ALPHA=1e-6
CHECKPOINTS=[25,100,300,1000]


def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)


def counts_for(fid,y,selected,n):
    return np.bincount(fid[selected]*3+y[selected],minlength=n*3).reshape(n,3)


def support_bucket(ids,y,selected,width):
    counts=counts_for(ids,y,selected,width);matched=counts[ids]
    total=matched.sum(1);truth=matched[np.arange(len(y)),y]
    return np.where(total==0,0,np.where(truth==0,1,np.where((matched>0).sum(1)==1,2,3))).astype(np.int8)


def objective(flat,x,k,counts,denom):
    width=x.shape[1];d=0 if k is None else k.shape[1]
    w=flat[:width*3].reshape(width,3)
    u=flat[width*3:(width+d)*3].reshape(d,3);bias=flat[-3:]
    z=np.asarray(x@w)+bias
    if d:z+=k@u
    total=counts.sum(1)
    loss=(np.sum(total[:,None]*np.logaddexp(0,z))-np.sum(counts*z))/denom
    loss+=ALPHA/2*(np.square(w).sum()+np.square(u).sum())
    residual=(expit(z)*total[:,None]-counts)/denom
    grad=np.concatenate([(np.asarray(x.T@residual)+ALPHA*w).ravel(),
                         ((k.T@residual+ALPHA*u).ravel() if d else np.empty(0)),residual.sum(0)])
    return float(loss),grad


def decode(flat,width,d):
    return {'coef':flat[:width*3].reshape(width,3).copy(),
            'kernel_coef':flat[width*3:(width+d)*3].reshape(d,3).copy(),
            'intercept':flat[-3:].copy()}


def predictions(x,k,model):
    out=np.empty(x.shape[0],np.int8)
    for beg in range(0,len(out),4096):
        z=x[beg:beg+4096]@model['coef']+model['intercept']
        if model['kernel_coef'].shape[0]:z+=k[beg:beg+4096]@model['kernel_coef']
        out[beg:beg+4096]=z.argmax(1)
    return out


def cm(y,p,mask):return np.bincount(y[mask]*3+p[mask],minlength=9).reshape(3,3)


def register():
    if DEST.exists():raise FileExistsError(DEST)
    DEST.mkdir()
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');n=int(fid.max())+1
    selected=pd.read_parquet(OLD/'manifest.parquet',columns=['optimizer_diagnostic']).optimizer_diagnostic.to_numpy()
    fit=~r.fold.isin([0,2]).to_numpy();cal=r.fold.eq(2).to_numpy()
    assert selected.sum()==104544 and not (selected&~fit).any()
    for field in ['component','body_group','source_symbol']:
        take=r[field].ge(0)
        roles=np.where(cal,'C',np.where(r.fold.eq(0),'H','fit'))
        assert pd.DataFrame({'group':r.loc[take,field],'role':roles[take]}).groupby('group').role.nunique().max()==1
    projections=pd.read_parquet(OUT/'projections.parquet',columns=['facts'])
    canonical=[json.dumps(json.loads(s),sort_keys=True,separators=(',',':')) for s in projections.facts]
    ids,unique=pd.factorize(np.asarray(canonical,dtype=object),sort=False)
    buckets=support_bucket(ids[r.projection_id.to_numpy()],y,selected,len(unique))
    complete=support_bucket(fid,y,selected,n)
    asa=r.route.eq('asa').to_numpy();component=r.component.to_numpy()
    big=int(read(ROOT/'artifacts/v81_diagnosis_20260927/support_diagnosis.json')['roles']['C']['asa']['top_error_components'][0]['component'])
    masks={'non_ASA':cal&~asa,'ASA_largest_component':cal&asa&(component==big),'ASA_other_components':cal&asa&(component!=big)}
    for code,name in enumerate(['unseen_combination','opposite_label_support','same_label_only','mixed_label_support']):
        masks['ASA_'+name]=cal&asa&(buckets==code)
    facts=[json.loads(s) for s in projections.facts]
    action=np.asarray([f.get('action') for f in facts],object)[r.projection_id.to_numpy()]
    vpc=r.route.eq('vpc_v2').to_numpy()
    masks['VPC_with_action']=cal&vpc&(action!='unknown')&(pd.notna(action))
    masks['VPC_without_action']=cal&vpc&~masks['VPC_with_action']
    # Missing conditions are explicitly NA and cannot establish transfer success.
    supported={name:mask for name,mask in masks.items() if mask.any()}
    np.savez_compressed(DEST/'validation_masks.npz',**supported)
    np.save(DEST/'selected_rows.npy',np.flatnonzero(selected).astype(np.int32))
    pd.DataFrame({'row_position':r.row_position,'fold':r.fold,'route':r.route,'component':r.component,
                  'selected':selected,'fact_support_bucket':buckets,'complete_support_bucket':complete}).to_parquet(DEST/'manifest.parquet',index=False)
    # The known consistent-label fit error cohort is a diagnostic, never resampled.
    oldpred=np.load(LAST/'P1_B_ovr_feature_prediction.npy')[fid]
    cohort=fit&asa&(y==2)&(oldpred!=y)&(complete==2)
    assert cohort.sum()==457
    np.save(DEST/'fit_457_cohort.npy',np.flatnonzero(cohort).astype(np.int32))
    contract={'version':'v82-full-R0-capacity-1','source_sha256':sha(__file__),'seed':SEED,
        'input':'Unchanged 66287-column full R0; original sparse columns always retained. Nonlinear adds fit-only 256-dimensional Nystrom RBF features.',
        'kernel':'256 uniform unique actually exposed fit landmarks; gamma=1/median positive squared pair distance of 512 fit-only unique inputs. No target/H data in basis.',
        'limitation':'Extra basis changes regularization geometry as well as effective capacity; success does not isolate architecture alone. Byte bag still loses order>2.',
        'roles':{'fit':int(fit.sum()),'C':int(cal.sum()),'H':int(r.fold.eq(0).sum())},
        'exposure':np.bincount(y[selected],minlength=3).tolist(),'selected_rows':int(selected.sum()),
        'sample_sha256':sha(DEST/'selected_rows.npy'),'manifest_sha256':sha(DEST/'manifest.parquet'),
        'loss':'Identical summed OVR binary logistic mean + alpha/2 coefficient square; unregularized intercept; original row frequency/class counts, alpha1e-6.',
        'budget':{'solver':'L-BFGS','maxiter':1000,'gtol':1e-6,'ftol':1e-12,'maxcor':10,'checkpoints':CHECKPOINTS},
        'selection':'Same C protocol independently per arm: checkpoints eligible vs historical SGD under v81 exact-integer main-improve/subgroups-nonregress contract. Minimum C errors, then earlier iteration. No H alternate selection.',
        'required_subgroups':list(supported),'NA_subgroups':[name for name in masks if name not in supported],
        'stress':'Both arms retrain from scratch after removing ASA-M fit rows while retaining all ASA-S. Basis rebuilt on restricted actual fit; H carrier M/S + C support reports; zero B is NA.',
        'continuation':'Primary nonlinear candidate must pass historical SGD C, matched linear C and fixed H protection; zero-M ASA/CEF transfer must not regress; then shadow expanded-fit H stability before any final full-data fit.',
        'four_arms':'Conditional on useful evidence-preserving supervised views; trivial whitespace variation does not itself justify an additional training arm.',
        'target_read':False,'external_data':False,'pseudo_labels':False,'full_data_fit':False,
        'protected_original_class_counts':np.bincount(y,minlength=3).tolist(),
        'input_sha256':{'train':sha(ROOT/'data/official/train.parquet'),'rows':sha(OUT/'rows.parquet'),
                        'X_data':sha(LAST/'X.data'),'X_indices':sha(LAST/'X.indices'),'X_indptr':sha(LAST/'X.indptr'),
                        'feature_ids':sha(LAST/'row_feature_id.npy'),'historical_contract':sha(OLD/'contract.json')},
        'subgroup_support':{name:np.bincount(y[m],minlength=3).tolist() for name,m in supported.items()}}
    save(DEST/'registration.json',contract)
    emit(stage='registered',exposure=contract['exposure'],required_subgroups=contract['subgroup_support'])


def kernel_features(x,used,protocol):
    path=DEST/(protocol+'_kernel.npy')
    if path.exists():return np.load(path,mmap_mode='r')
    rng=np.random.default_rng(SEED);sample=rng.choice(used,min(512,len(used)),replace=False)
    dist=euclidean_distances(x[sample],squared=True)
    positive=dist[np.triu_indices(len(sample),1)];positive=positive[positive>1e-10]
    gamma=float(1/np.median(positive))
    basis=Nystroem(kernel='rbf',gamma=gamma,n_components=256,random_state=SEED,n_jobs=1).fit(x[used])
    landmarks=used[basis.component_indices_]
    assert set(landmarks).issubset(set(used))
    joblib.dump(basis,DEST/(protocol+'_basis.joblib'))
    np.save(DEST/(protocol+'_landmark_ids.npy'),landmarks)
    result=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(x.shape[0],256))
    start=time.monotonic()
    for beg in range(0,x.shape[0],2048):
        result[beg:beg+2048]=basis.transform(x[beg:beg+2048]).astype(np.float32)
        if beg%65536==0:emit(stage='kernel',protocol=protocol,rows=beg,seconds=round(time.monotonic()-start,1))
    result.flush();del result
    save(DEST/(protocol+'_basis.json'),{'fit_unique_inputs':len(used),'gamma':gamma,'landmarks':len(landmarks),
         'fit_only_landmarks':True,'source_sha256':sha(DEST/(protocol+'_basis.joblib')),
         'kernel_sha256':sha(path),'seconds':time.monotonic()-start})
    return np.load(path,mmap_mode='r')


def train_one(x,k,counts,name):
    if (DEST/(name+'_fit.json')).exists():return
    used=np.flatnonzero(counts.sum(1));xx=x[used];kk=None if k is None else np.asarray(k[used],np.float64)
    cc=counts[used];den=int(cc.sum());width=x.shape[1];d=0 if kk is None else kk.shape[1]
    w=np.zeros((width+d+1)*3);w[-3:]=np.log(np.maximum(cc.sum(0)/den,1e-9))
    rng=np.random.default_rng(SEED);direction=rng.normal(size=len(w));direction/=np.linalg.norm(direction)
    args=(xx[:31],None if kk is None else kk[:31],cc[:31],den)
    _,grad=objective(w,*args);eps=1e-5
    numeric=(objective(w+eps*direction,*args)[0]-objective(w-eps*direction,*args)[0])/(2*eps)
    gradient_error=float(abs(numeric-grad@direction));assert gradient_error<1e-6
    iterations=[0];start=time.monotonic();states=[]
    def checkpoint(v,it):
        label=f'{name}_iter{it:04}';model=decode(v,width,d)
        joblib.dump(model,DEST/(label+'.joblib'))
        np.save(DEST/(label+'_prediction.npy'),predictions(x,k,model))
        states.append({'name':label,'iteration':it})
    def callback(v):
        iterations[0]+=1;it=iterations[0]
        if it in CHECKPOINTS:checkpoint(v,it)
        if it%25==0:
            value,g=objective(v,xx,kk,cc,den)
            emit(stage='optimization',model=name,iteration=it,objective=value,gradient_inf=float(abs(g).max()),seconds=round(time.monotonic()-start,1))
    result=minimize(objective,w,args=(xx,kk,cc,den),method='L-BFGS-B',jac=True,callback=callback,
                    options={'maxiter':1000,'gtol':1e-6,'ftol':1e-12,'maxcor':10,'maxls':30})
    if int(result.nit) not in [s['iteration'] for s in states]:checkpoint(result.x,int(result.nit))
    info={'name':name,'states':states,'iterations':int(result.nit),'objective':float(result.fun),
          'gradient_inf':float(abs(result.jac).max()),'converged':bool(result.success and abs(result.jac).max()<=1e-5),
          'solver_message':str(result.message),'directional_gradient_error':gradient_error,
          'exposed_rows':den,'exposed_per_class':cc.sum(0).tolist(),'unique_inputs':len(used),
          'seconds':time.monotonic()-start,'source_sha256':sha(__file__)}
    save(DEST/(name+'_fit.json'),info);emit(stage='fit_complete',**{k:v for k,v in info.items() if k!='states'})
    del xx,kk;gc.collect()


def evaluate(r,fid,name,baseline):
    y=r.label_index.to_numpy();cal=r.fold.eq(2).to_numpy();held=r.fold.eq(0).to_numpy();fit=~(cal|held)
    masks=dict(np.load(DEST/'validation_masks.npz'));b=baseline[fid]
    reference_cm=cm(y,b,cal);basegroups={g:cm(y,b,m) for g,m in masks.items()}
    trace=[]
    for state in read(DEST/(name+'_fit.json'))['states']:
        pred=np.load(DEST/(state['name']+'_prediction.npy'))[fid]
        gate=accept_candidate(reference_cm,cm(y,pred,cal),{g:(basegroups[g],cm(y,pred,m)) for g,m in masks.items()})
        trace.append({**state,'C':metrics(cm(y,pred,cal)),'fit':metrics(cm(y,pred,fit)),
                      'groups':{g:metrics(cm(y,pred,m)) for g,m in masks.items()},'gate':gate})
    eligible=[s for s in trace if s['gate']['eligible']]
    selected=min(eligible,key=lambda t:(t['C']['errors'],t['iteration'])) if eligible else None
    final=trace[-1]
    # Freeze selection before inspecting H and never pick a replacement using H.
    save(DEST/(name+'_selection.json'),{'selected':None if selected is None else selected['name'],
        'trace':trace,'eligible_states':len(eligible),'H_used_for_selection':False})
    evaluation=[];cohort=np.load(DEST/'fit_457_cohort.npy')
    for label in dict.fromkeys([final['name']]+([] if selected is None else [selected['name']])):
        p=np.load(DEST/(label+'_prediction.npy'))[fid]
        table=[]
        for role,mask in [('fit',fit),('C',cal),('H',held)]:
            for route in sorted(r.route.unique()):
                m=mask&r.route.eq(route).to_numpy();c=cm(y,p,m)
                for cl in range(3):table.append({'model':label,'role':role,'route':route,'class':cl,
                    'support':int(c[cl].sum()),'correct':int(c[cl,cl]),'pred_B':int(c[cl,0]),'pred_M':int(c[cl,1]),'pred_S':int(c[cl,2])})
        pd.DataFrame(table).to_csv(DEST/(label+'_classwise.csv'),index=False)
        evaluation.append({'name':label,'H':metrics(cm(y,p,held)),'H_historical_protection':compare(cm(y,b,held),cm(y,p,held),False),
                           'same_label_fit_457_correct':int((p[cohort]==y[cohort]).sum()),'same_label_fit_457_rows':len(cohort)})
    save(DEST/(name+'_evaluation.json'),evaluation)
    emit(stage='selection',model=name,selected=None if selected is None else selected['name'],C_final_errors=final['C']['errors'],evaluation=evaluation)


def run(protocol):
    contract=read(DEST/'registration.json');assert contract['source_sha256']==sha(__file__),'Changed after registration'
    r=rows();x=load_sparse(LAST/'X');fid=np.load(LAST/'row_feature_id.npy');y=r.label_index.to_numpy()
    selected=np.zeros(len(r),bool);selected[np.load(DEST/'selected_rows.npy')]=True
    if protocol in ['ASA_zero_M','CEF_zero_M']:
        route='asa' if protocol=='ASA_zero_M' else 'cef_fields'
        removed=selected&r.route.eq(route).to_numpy()&(y==1);selected[removed]=False
        save(DEST/(protocol+'_exposure.json'),{'removed_M':int(removed.sum()),'retained_carrier_classes':np.bincount(y[selected&r.route.eq(route).to_numpy()],minlength=3).tolist(),
            'all_label_parameters_reset':True,'fit_only_basis_rebuilt':True,'carrier':route})
    counts=counts_for(fid,y,selected,x.shape[0]);used=np.flatnonzero(counts.sum(1))
    kernel=kernel_features(x,used,protocol)
    old=joblib.load(OLD/'O_sgd.joblib');baseline=(x@old.coef_.T+old.intercept_).argmax(1).astype(np.int8)
    for arm,kk in [('linear',None),('nonlinear',kernel)]:
        name=protocol+'_'+arm;train_one(x,kk,counts,name);evaluate(r,fid,name,baseline)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','primary','ASA_zero_M','CEF_zero_M']);args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage=='register':register()
        else:run(args.stage)
