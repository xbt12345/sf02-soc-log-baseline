"""Controlled expansion of the registered optimizer diagnostic, H always excluded."""
import time
import gc
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from v78_boundary import DEST, SEED, data, masks, objective, logits_all, class_table, gates, summarize
from run_v75 import ROOT, OUT, BATCH, read, save, sha, features


def main():
    parent=read(DEST/'optimizer.json')
    assert parent['converged'] and all(parent['comparisons']['calibration']['gates'].values()) and all(parent['comparisons']['evaluation']['gates'].values())
    assert not (DEST/'shadow_contract.json').exists()
    r,m=data();roles=masks(r);manifest=pd.read_parquet(DEST/'manifest.parquet');old=np.flatnonzero(manifest.optimizer_diagnostic)
    rng=np.random.default_rng(SEED+1);added=[]
    for (_,label),z in r[roles['calibration']].groupby(['route','label_index']):
        ix=z.index.to_numpy();added.extend(ix if label!=0 or len(ix)<=2048 else rng.choice(ix,2048,replace=False))
    ix=np.sort(np.concatenate([old,np.array(added)]));assert not roles['evaluation'][ix].any() and len(np.unique(ix))==len(ix)
    pd.DataFrame({'row_position':ix,'was_original_fit':np.isin(ix,old)}).to_parquet(DEST/'shadow_manifest.parquet',index=False)
    save(DEST/'shadow_contract.json',{'new_fits_before_registration':0,'original_subset_rows':len(old),'added_calibration_subset_rows':len(added),'expanded_rows':len(ix),
        'purpose':'Same objective/alpha/R0/unit weights; add C subset to T subset, converge; evaluate same excluded H. No new threshold.',
        'subset_limitation':'ALL nonbenign T/C and up to2048 benign per route per role. Not full official training; no removed original diagnostic rows.',
        'acceptance':'Strict lower H errors; each class recall/precision/F1 and normal FP no regression; source has been inspected, not blind.',
        'source_sha256':sha(__file__),'parent_optimizer_sha256':sha(DEST/'optimizer.json'),'manifest_sha256':sha(DEST/'shadow_manifest.parquet')})
    start=time.monotonic();model=joblib.load(DEST/'O_lbfgs.joblib')
    with threadpool_limits(limits=4):
        x=sparse.vstack([features(r,ix[a:a+BATCH],m,'new') for a in range(0,len(ix),BATCH)],format='csr').astype(np.float64)
        y=r.label_index.to_numpy()[ix];w=np.vstack([model['coef'],model['intercept']]).ravel()
        initial=objective(w,x,y,1e-6)[0];it=[0];trace=[]
        def callback(ww):
            it[0]+=1
            if it[0]%50==0:
                f,g=objective(ww,x,y,1e-6);v={'iteration':it[0],'objective':f,'gradient_max':float(abs(g).max()),'seconds':round(time.monotonic()-start,1)}
                trace.append(v);save(DEST/'shadow_progress.json',trace);print(v,flush=True)
        result=minimize(objective,w,args=(x,y,1e-6),jac=True,method='L-BFGS-B',callback=callback,
            options={'maxiter':1000,'maxls':30,'gtol':1e-6,'ftol':1e-12,'maxcor':10})
        w=result.x.reshape(x.shape[1]+1,3);model={'coef':w[:-1],'intercept':w[-1]};joblib.dump(model,DEST/'shadow.joblib')
        del x;gc.collect();z=logits_all(r,m,model)
    np.save(DEST/'shadow_logits.npy',z);class_table(r,z,'shadow')
    h=roles['evaluation'];y=r.label_index.to_numpy();before=np.load(DEST/'O_lbfgs_logits.npy');a=summarize(y[h],before[h].argmax(1));b=summarize(y[h],z[h].argmax(1))
    report={'classifier_fits':1,'calibration_fits':0,'expanded_rows':len(ix),'converged':bool(result.success and abs(result.jac).max()<=1e-5),
        'iterations':int(result.nit),'objective_before':initial,'objective_after':float(result.fun),'gradient_after':float(abs(result.jac).max()),
        'evaluation_before':a,'evaluation_after':b,'gates':gates(a,b),'fixed':int(((before[h].argmax(1)!=y[h])&(z[h].argmax(1)==y[h])).sum()),
        'broken':int(((before[h].argmax(1)==y[h])&(z[h].argmax(1)!=y[h])).sum()),'seconds':time.monotonic()-start,'source_sha256':sha(__file__)}
    save(DEST/'shadow.json',report);print(report,flush=True)


if __name__=='__main__':main()
