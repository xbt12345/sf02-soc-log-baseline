"""Registered, classwise boundary and matched-objective learning experiments."""
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, BATCH, NAMES, read, save, sha, adapter, matrices, features, load_sparse, new_model, metrics
from v75_views import view, byte_matrix
from v75_corrective import stable
from v75_metadata import encode

DEST=ROOT/'artifacts/v78_boundary_20260922'
SEED=7801
DELTAS=np.array([-2.,-1.,-.5,-.25,-.125,0.,.125,.25,.5,1.,2.])


def data():
    r=pd.read_parquet(OUT/'rows.parquet')
    m=matrices();m['new_text']=load_sparse(OUT/'corrective/text')
    return r,m


def masks(r):
    return {'fit':~r.fold.isin([0,2]).to_numpy(),'calibration':r.fold.eq(2).to_numpy(),'evaluation':r.fold.eq(0).to_numpy()}


def summarize(y,p):return metrics(np.bincount(y*3+p,minlength=9).reshape(3,3))


def prepare():
    if DEST.exists():raise FileExistsError(DEST)
    for name,h in read(ROOT/'evidence/2026-09-21/v75_four_arm/delivery.json')['artifact_sha256'].items():
        assert sha(ROOT/name)==h,name
    DEST.mkdir()
    r,m=data();roles=masks(r);role=np.where(roles['fit'],'fit',np.where(roles['calibration'],'calibration','evaluation'))
    manifest=r[['row_position','event_id','component','body_group','source_symbol','route','label_index']].copy();manifest['role']=role
    assert manifest.groupby('component').role.nunique().max()==1
    assert manifest.groupby('body_group').role.nunique().max()==1
    assert manifest[manifest.source_symbol>=0].groupby('source_symbol').role.nunique().max()==1
    rng=np.random.default_rng(SEED);selected=[]
    for (_,yi),z in r[roles['fit']].groupby(['route','label_index']):
        ix=z.index.to_numpy()
        selected.extend(ix if yi!=0 or len(ix)<=2048 else rng.choice(ix,2048,replace=False))
    selected=np.array(sorted(selected),np.int64);manifest['optimizer_diagnostic']=False;manifest.loc[selected,'optimizer_diagnostic']=True
    manifest.to_parquet(DEST/'manifest.parquet',index=False)
    support=manifest.groupby(['route','label_index','role']).size().rename('rows').reset_index()
    support.to_csv(DEST/'support.csv',index=False)
    wanted=set()
    for _,z in r.groupby(['route','label_index']):
        if len(z)<=50:wanted.update(z.index)
        else:
            for mask in roles.values():wanted.update(z.index[mask[z.index]][:4])
    # Prior errors are inspected diagnostic material, not a new holdout definition.
    prior=pd.read_parquet(OUT/'corrective/predictions.parquet',columns=['E_pred'])
    for yi in [1,2]:wanted.update(r.index[r.route.eq('asa')&r.label_index.eq(yi)&prior.E_pred.ne(yi)][:60])
    wanted=np.array(sorted(wanted));raw={};offset=0
    for b in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized','src_port'],use_threads=False):
        for pos in wanted[np.searchsorted(wanted,offset):np.searchsorted(wanted,offset+len(b))]:
            raw[int(pos)]=(b.column(0)[int(pos)-offset].as_py(),b.column(1)[int(pos)-offset].as_py())
        offset+=len(b)
    v=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');messages=[raw[int(i)][0] or '' for i in wanted]
    parsed=[v.prepare_record({'message_sanitized':s}) for s in messages];facts=[p['facts'] for p in parsed]
    fx=normalize(enc.transform(facts).astype(np.float32),norm='l2',copy=False)
    extra,_,_=encode([raw[int(i)][1] for i in wanted],[f.get('src_port_fixed',65536) for f in facts])
    xx=sparse.hstack([byte_matrix([stable(view(s)[0]) for s in messages]),fx,extra],format='csr',dtype=np.float32)
    cached=features(r,wanted,m,'new');delta=xx-cached
    maxdiff=float(abs(delta.data).max()) if delta.nnz else 0.;assert maxdiff<1e-7
    # Preserve all rare malignant evidence locally with original roles and hashes.
    evidence=[]
    for pos in wanted:
        if r.route.iat[pos]=='unsupported' and r.label_index.iat[pos]==1:
            s=raw[int(pos)][0] or '';cooked=stable(view(s)[0])
            evidence.append({'row_position':int(pos),'component':int(r.component.iat[pos]),'role':role[pos],
                'raw':s,'model_text':cooked,'facts':parsed[list(wanted).index(pos)]['facts'],
                'raw_sha256':__import__('hashlib').sha256(s.encode()).hexdigest()})
    save(DEST/'unsupported_evidence.json',evidence)
    contract={'version':'v78-boundary-1','source_sha256':sha(__file__),'new_fits_before_registration':0,
        'roles':'fit folds -1,1,3,4; calibration fold2; evaluation fold0; all inherited inspected-development, not fresh blind.',
        'role_counts':{k:int(v.sum()) for k,v in roles.items()},'optimizer_subset_rows':len(selected),
        'optimizer_subset':'All nonbenign fit rows; normal up to2048 per route sampled with fixed seed. No weights. Diagnostic subset only, never full-data training claim.',
        'K_baseline':'R0 corrected fixed-byte/facts/metadata, original SGD six epochs on ALL fit rows; calibration only changes one scalar M/S bias.',
        'K_delta_grid':DELTAS.tolist(),'K_selection':'Calibration: strict fewer errors, every class recall/precision/F1 nondecreasing and normal FP nonincreasing. Choose smallest errors then abs(delta); same guards each source-component hash half. Otherwise delta0 no candidate.',
        'O':'Exactly same subset/OVR summed binary logistic mean loss + alpha/2*coef^2, unregularized intercept, original weight1. SGD6 vs L-BFGS max1000; no heldout early stopping.',
        'alpha':1e-6,'epochs':6,'seed':SEED,'eta0':.05,'lbfgs_gtol':1e-6,'lbfgs_ftol':1e-12,
        'convergence_acceptance':'scipy success AND gradient infinity norm <=1e-5; never claim approximation is exact optimum.',
        'promotion':'No private-answer tuning or unconditional full fit; any quality candidate must pass classwise evaluation and shadow-refit gates.',
        'raw_replay':{'rows':len(wanted),'all_rare_cells_le50':True,'all_routes':int(r.loc[wanted,'route'].nunique()),'max_feature_delta':maxdiff},
        'input_sha256':{'train':sha(ROOT/'data/official/train.parquet'),'rows':sha(OUT/'rows.parquet')},
        'manifest_sha256':sha(DEST/'manifest.parquet')}
    save(DEST/'contract.json',contract);print(json.dumps(contract,ensure_ascii=False,indent=2))


def check():
    c=read(DEST/'contract.json');assert c['source_sha256']==sha(__file__);assert c['manifest_sha256']==sha(DEST/'manifest.parquet');return c


def fit_sgd(r,m,indices,path):
    c=check();model=new_model(c);rng=np.random.default_rng(SEED);mask=np.zeros(len(r),bool);mask[indices]=True
    start=time.monotonic()
    for ep in range(6):
        count=0
        for beg in rng.permutation(np.arange(0,len(r),BATCH)):
            ix=np.arange(beg,min(beg+BATCH,len(r)));rng.shuffle(ix);ix=ix[mask[ix]]
            if not len(ix):continue
            model.partial_fit(features(r,ix,m,'new'),r.label_index.to_numpy()[ix],classes=np.arange(3));count+=len(ix)
        assert count==len(indices)
        print({'stage':path.stem,'epoch':ep+1,'rows':count,'seconds':round(time.monotonic()-start,1)},flush=True)
    joblib.dump(model,path)
    return model


def logits_all(r,m,model):
    z=np.empty((len(r),3),np.float64)
    for start in range(0,len(r),BATCH):
        ix=np.arange(start,min(start+BATCH,len(r)));x=features(r,ix,m,'new')
        z[ix]=model.decision_function(x) if hasattr(model,'decision_function') else x@model['coef']+model['intercept']
    return z


def gates(a,b):
    good={'fewer_errors':b['errors']<a['errors'],'normal_FP_not_higher':b['normal_false_alerts']<=a['normal_false_alerts']}
    for k in ['recall','precision','f1']:
        good[k+'_not_lower']=all(y is not None and y>=x-1e-12 for x,y in zip(a[k],b[k]) if x is not None)
    return good


def class_table(r,z,name):
    y=r.label_index.to_numpy();p=z.argmax(1);rows=[]
    for role,mask in masks(r).items():
        for route in ['ALL']+sorted(r.route.unique().tolist()):
            take=mask if route=='ALL' else mask&r.route.eq(route).to_numpy();cm=np.bincount(y[take]*3+p[take],minlength=9).reshape(3,3)
            for k in range(3):
                n=int(cm[k].sum());pp=int(cm[:,k].sum());tp=int(cm[k,k])
                rows.append({'model':name,'role':role,'route':route,'label':NAMES[k],'support':n,'correct':tp,'errors':n-tp,
                    'recall':tp/n if n else None,'precision':tp/pp if pp else None,'f1':2*tp/(n+pp) if n+pp else None,
                    **{'pred_'+NAMES[j]:int(cm[k,j]) for j in range(3)}})
    pd.DataFrame(rows).to_csv(DEST/(name+'_classwise.csv'),index=False)


def baseline():
    check();r,m=data();roles=masks(r)
    assert not (DEST/'K_base.joblib').exists()
    with threadpool_limits(limits=4):
        model=fit_sgd(r,m,np.flatnonzero(roles['fit']),DEST/'K_base.joblib');z=logits_all(r,m,model)
    np.save(DEST/'K_base_logits.npy',z);class_table(r,z,'K_base')
    save(DEST/'baseline_complete.json',{'classifier_fits':1,'calibration_fits':0,'fit_rows':int(roles['fit'].sum()),'model_sha256':sha(DEST/'K_base.joblib'),'logits_sha256':sha(DEST/'K_base_logits.npy')})


def calibrate():
    check();assert not (DEST/'calibration.json').exists();r=pd.read_parquet(OUT/'rows.parquet');roles=masks(r);z=np.load(DEST/'K_base_logits.npy');y=r.label_index.to_numpy()
    mask=roles['calibration'];part=(r.component.to_numpy().astype(np.int64)*2654435761)%2
    masks_c=[mask,mask&(part==0),mask&(part==1)];base=[summarize(y[x],z[x].argmax(1)) for x in masks_c]
    candidates=[]
    for delta in DELTAS:
        q=z[mask]+[0,delta/2,-delta/2];out=summarize(y[mask],q.argmax(1));g=gates(base[0],out)
        stability=[]
        for i,s in enumerate(masks_c[1:]):
            v=summarize(y[s],(z[s]+[0,delta/2,-delta/2]).argmax(1));gg=gates(base[i+1],v)
            # Separate subgroups may tie; retain all classwise guards and total no-regression.
            gg['fewer_errors']=v['errors']<=base[i+1]['errors'];stability.append(all(gg.values()))
        candidates.append({'delta':float(delta),'metrics':out,'gates':g,'half_guards':stability,'eligible':all(g.values()) and all(stability)})
    eligible=[a for a in candidates if a['eligible']];chosen=min(eligible,key=lambda a:(a['metrics']['errors'],abs(a['delta']))) if eligible else None
    delta=chosen['delta'] if chosen else 0.;q=z+[0,delta/2,-delta/2]
    np.save(DEST/'K_logits.npy',q);class_table(r,q,'K')
    report={'classifier_fits':0,'calibration_fits':1,'selected_delta':delta,'candidate_exists':chosen is not None,
        'calibration_baseline':base[0],'candidates':candidates,'evaluation_baseline':summarize(y[roles['evaluation']],z[roles['evaluation']].argmax(1)),
        'evaluation_candidate':summarize(y[roles['evaluation']],q[roles['evaluation']].argmax(1)),
        'scope':'Only C labels used to choose delta. H is previously inspected development; no private answers. No eligible bias => retain delta0, not a successful repair.'}
    report['evaluation_gates']=gates(report['evaluation_baseline'],report['evaluation_candidate'])
    save(DEST/'calibration.json',report);print(json.dumps({k:v for k,v in report.items() if k!='candidates'},indent=2))


def objective(flat,x,y,alpha):
    w=flat.reshape(x.shape[1]+1,3);z=x@w[:-1]+w[-1]
    loss=float((np.logaddexp(0,z).sum()-z[np.arange(len(y)),y].sum())/len(y)+alpha/2*np.square(w[:-1]).sum())
    residual=expit(z);residual[np.arange(len(y)),y]-=1
    grad=np.empty_like(w);grad[:-1]=np.asarray(x.T@residual)/len(y)+alpha*w[:-1];grad[-1]=residual.mean(0)
    return loss,grad.ravel()


def optimizer():
    c=check();r,m=data();manifest=pd.read_parquet(DEST/'manifest.parquet');ix=np.flatnonzero(manifest.optimizer_diagnostic)
    assert not (DEST/'O_sgd.joblib').exists();start=time.monotonic()
    with threadpool_limits(limits=4):
        model=fit_sgd(r,m,ix,DEST/'O_sgd.joblib')
        x=sparse.vstack([features(r,ix[a:a+BATCH],m,'new') for a in range(0,len(ix),BATCH)],format='csr').astype(np.float64)
        y=r.label_index.to_numpy()[ix]
        w=np.vstack([model.coef_.T,model.intercept_]).astype(np.float64).ravel()
        # Real finite-difference checks of both coefficient and intercept derivatives.
        small=x[:min(100,len(x.indptr)-1)];ys=y[:small.shape[0]];_,g=objective(w,small,ys,c['alpha'])
        rng=np.random.default_rng(SEED);direction=rng.normal(size=len(w));direction/=np.linalg.norm(direction);eps=1e-5
        numeric=(objective(w+eps*direction,small,ys,c['alpha'])[0]-objective(w-eps*direction,small,ys,c['alpha'])[0])/(2*eps)
        analytic=float(g@direction);assert abs(numeric-analytic)<1e-6
        before,gb=objective(w,x,y,c['alpha']);trace=[];counter=[0]
        def callback(ww):
            counter[0]+=1
            if counter[0]%25==0:
                f,g=objective(ww,x,y,c['alpha']);d={'iteration':counter[0],'objective':f,'gradient_max':float(abs(g).max()),'seconds':round(time.monotonic()-start,1)}
                trace.append(d);save(DEST/'optimizer_progress.json',trace);print(d,flush=True)
        result=minimize(objective,w,args=(x,y,c['alpha']),jac=True,method='L-BFGS-B',callback=callback,
                        options={'maxiter':1000,'maxls':30,'gtol':c['lbfgs_gtol'],'ftol':c['lbfgs_ftol'],'maxcor':10})
        converged=bool(result.success and np.isfinite(result.fun) and abs(result.jac).max()<=1e-5)
        ww=result.x.reshape(x.shape[1]+1,3);conv={'coef':ww[:-1],'intercept':ww[-1]};joblib.dump(conv,DEST/'O_lbfgs.joblib')
        del x;gc.collect()
        zs=logits_all(r,m,model);zl=logits_all(r,m,conv)
    np.save(DEST/'O_sgd_logits.npy',zs);np.save(DEST/'O_lbfgs_logits.npy',zl)
    class_table(r,zs,'O_sgd');class_table(r,zl,'O_lbfgs')
    roles=masks(r);all_y=r.label_index.to_numpy();comp={}
    for role,take in [('optimizer_fit_subset',manifest.optimizer_diagnostic.to_numpy()),*roles.items()]:
        a=summarize(all_y[take],zs[take].argmax(1));b=summarize(all_y[take],zl[take].argmax(1));comp[role]={'SGD':a,'LBFGS':b,'gates':gates(a,b)}
    report={'classifier_fits':2,'calibration_fits':0,'subset_rows':len(ix),'converged':converged,'scipy_success':bool(result.success),
        'iterations':int(result.nit),'message':str(result.message),'objective_before':before,'objective_after':float(result.fun),
        'gradient_before':float(abs(gb).max()),'gradient_after':float(abs(result.jac).max()),
        'directional_gradient_error':abs(numeric-analytic),'objective':'mean(sum_OVR_binary_logloss)+alpha/2*sum(coef^2); intercept unregularized; weight1',
        'comparisons':comp,'seconds':time.monotonic()-start,'source_sha256':sha(__file__)}
    save(DEST/'optimizer.json',report);print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},indent=2));print(json.dumps(comp['evaluation'],indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','baseline','calibrate','optimizer']);a=ap.parse_args()
    {'prepare':prepare,'baseline':baseline,'calibrate':calibrate,'optimizer':optimizer}[a.stage]()
