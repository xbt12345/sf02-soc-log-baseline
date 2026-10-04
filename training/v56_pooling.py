"""Convex multiclass residual shrinkage around a frozen neural score model.

This is a deterministic regularized adaptation, NOT a Bayesian posterior or LMMNN.
Identical observations are aggregated by summing their original class counts.
"""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
import run_v54 as v
import run_v55 as n

LEVELS=['coarse','fine','exact']

def keys(x):return {'coarse':n.behavior(x,False),'fine':n.behavior(x,True),'exact':v.frame_keys(x[v.FIELDS])}

def fit_design(x,bodies):
 kk=keys(x);maps={};support=[];count=[];feature_keys=[];offset=0
 for level in LEVELS:
  groups=pd.DataFrame({'key':kk[level],'body':np.asarray(bodies)}).groupby('key',sort=True).agg(rows=('body','size'),bodies=('body','nunique'))
  maps[level]={k:i+offset for i,k in enumerate(groups.index)};offset+=len(groups);count.extend(groups.rows);support.extend(groups.bodies);feature_keys.extend([[level,k] for k in groups.index])
 return {'maps':maps,'rows':np.asarray(count,dtype=float),'bodies':np.asarray(support,dtype=float),'feature_keys':feature_keys,'fit_rows':len(x),'levels':LEVELS}

def design(b,x):
 kk=keys(x);ri=[];ci=[]
 for level in LEVELS:
  mapping=b['maps'][level]
  for i,k in enumerate(kk[level]):
   if k in mapping:ri.append(i);ci.append(mapping[k])
 return sparse.csr_matrix((np.ones(len(ri)),(ri,ci)),shape=(len(x),len(b['feature_keys'])))

def penalty(b,kind,strength):
 # Frequency normalization keeps repeated rows in CE without interpreting
 # their repetition as extra support for an unconstrained group parameter.
 p=float(strength)*b['rows']/b['fit_rows']
 if kind=='support':p=p/np.sqrt(b['bodies'])
 elif kind!='uniform':raise ValueError(kind)
 return p

def compress(x,y):
 codes,_=pd.factorize(v.frame_keys(x[v.FIELDS]),sort=False);first=np.unique(codes,return_index=True)[1];counts=np.zeros((len(first),3));np.add.at(counts,(codes,np.asarray(y,dtype=int)),1)
 return x.iloc[first].reset_index(drop=True),counts,codes

def objective(theta,A,offset,counts,precision):
 w=theta.reshape(-1,3);z=offset+A@w;total=counts.sum();mass=counts.sum(1);logp=z-logsumexp(z,axis=1,keepdims=True)
 loss=-np.sum(counts*logp)/total+.5*np.sum(precision[:,None]*w*w)
 dz=(softmax(z,axis=1)*mass[:,None]-counts)/total;gradient=A.T@dz+precision[:,None]*w
 return float(loss),np.asarray(gradient).ravel()

def fit(base,x,y,bodies,kind,strength,options):
 b=fit_design(x,bodies);xx,c,ids=compress(x,y);A=design(b,xx);_,p=n.predict(base,xx);offset=np.log(np.maximum(p.astype(np.float64),1e-30));prec=penalty(b,kind,strength)
 start=np.zeros(A.shape[1]*3);initial=objective(start,A,offset,c,prec)[0];scale=np.sqrt(prec);scaled=A.multiply(1/scale).tocsr()
 res=minimize(objective,start,args=(scaled,offset,c,np.ones(len(prec))),jac=True,method='L-BFGS-B',options=options)
 w=res.x.reshape(-1,3)/scale[:,None];w-=w.mean(1,keepdims=True);value,grad=objective(w.ravel(),A,offset,c,prec)
 report={'success':bool(res.success),'message':str(res.message),'iterations':int(res.nit),'function_evaluations':int(res.nfev),'objective_start':initial,'objective_end':value,'max_abs_gradient':float(abs(grad).max()),'max_abs_scaled_gradient':float(abs(grad.reshape(-1,3)/scale[:,None]).max()),'original_rows':len(x),'compressed_observations':len(xx),'class_count_sums':c.sum(0).astype(int).tolist(),'features':A.shape[1]}
 return {'design':b,'weights':w,'kind':kind,'strength':strength,'optimizer':report,'base':base},report

def predict(bundle,x):
 xx,_,ids=compress(x,np.zeros(len(x),dtype=int));_,p=n.predict(bundle['base'],xx);A=design(bundle['design'],xx);z=np.log(np.maximum(p.astype(np.float64),1e-30))+A@bundle['weights'];out=softmax(z,axis=1)[ids]
 return out.argmax(1),out
