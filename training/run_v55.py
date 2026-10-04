"""Fit-side factorial: regularization, group risk and partial observations.

Public sklearn weighted Adam solves each epoch's frozen reweighted objective.
The adversary uses full-fit group means once per epoch, not the paper's batch update.
"""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import run_v54 as v

ROOT=Path(__file__).resolve().parents[1]
SPLITS=['body','behavior'];EPOCHS=[10,30,60]
VIEWS={'full':[],'no_src_role':['src_role'],'no_dst_role':['dst_role']}
SPECS={'E0':('erm',.01,False),'E1':('erm',1.,False),'D0':('dro',.01,False),'D1':('dro',1.,False),
       'B1':('balanced',1.,False),'E1V':('erm',1.,True),'D1V':('dro',1.,True),'B1V':('balanced',1.,True)}

def behavior(x,fine=False):
 z=x[['transport_protocol','src_role','dst_role']].copy()
 if fine:z[['icmp_type','icmp_code']]=x[['icmp_type','icmp_code']]
 z['src_port_observed']=x.src_port_fixed.ne(v.MISSING).astype(str);z['dst_port_observed']=x.dst_port_fixed.ne(v.MISSING).astype(str)
 return z.astype(str).agg('|'.join,axis=1).to_numpy()

def domain_bucket(key):return int(hashlib.sha256(('v55-behavior|'+key).encode()).hexdigest()[:8],16)%3

def expanded(x,y,partial):
 names=list(VIEWS) if partial else ['full'];prior=[.5,.25,.25] if partial else [1.]
 frames=[];labels=[];viewids=[];groups=[];base=behavior(x)
 for j,name in enumerate(names):
  a=x.copy();a[VIEWS[name]]=v.MISSING;frames.append(a);labels.extend(y);viewids.extend([j]*len(a))
  groups.extend([json.dumps([name,k,int(t)],separators=(',',':')) for k,t in zip(base,y)])
 xx=pd.concat(frames,ignore_index=True);yy=np.asarray(labels,dtype=int);vi=np.asarray(viewids)
 keys=sorted(set(groups));lookup={k:i for i,k in enumerate(keys)};gi=np.asarray([lookup[k] for k in groups]);counts=np.bincount(gi,minlength=len(keys))
 gv=np.asarray([names.index(json.loads(k)[0]) for k in keys]);prior=np.asarray(prior)
 return xx,yy,vi,gi,keys,counts,gv,prior

def group_means(loss,gi,counts):return np.bincount(gi,weights=loss,minlength=len(counts))/counts

def distribution(mode,means,previous,gv,prior,eta=.1):
 out=np.empty(len(means),dtype=float)
 for j in range(len(prior)):
  ix=gv==j
  if mode=='dro':
   a=np.log(previous[ix]/prior[j])+eta*means[ix];a-=a.max();q=np.exp(a);q/=q.sum()
  else:q=np.ones(ix.sum())/ix.sum()
  out[ix]=prior[j]*q
 assert np.isclose(out.sum(),1)
 return out

def risk_weights(mode,gi,counts,gv,prior,q):
 n=len(gi)
 if mode=='erm':
  masses=np.array([counts[gv==j].sum() for j in range(len(prior))]);w=n*prior[gv[gi]]/masses[gv[gi]]
 else:w=n*q[gi]/counts[gi]
 assert np.isclose(w.sum(),n)
 return w

def predict(bundle,x):
 return v.predict({'family':'neural','input':bundle['input'],'models':[bundle['model']]},x)

def prepare(root,out):
 assert not out.exists();out.mkdir(parents=True)
 source=root/'artifacts/v54_method_change_20260914';r=pd.read_parquet(source/'rows.parquet');x=pd.read_parquet(source/'observations.parquet')
 mask=r.inner_role.ne(2);oldsplit=pd.read_parquet(source/'pressure_split.parquet').loc[mask].reset_index(drop=True)
 r=r.loc[mask].reset_index(drop=True);x=x.loc[mask].reset_index(drop=True)
 assert len(r)==99398 and r.groupby('body_group').projection_id.nunique().max()==1
 r.to_parquet(out/'rows.parquet',index=False);x.to_parquet(out/'observations.parquet',index=False)
 normal=r.route.eq('asa_acl').to_numpy();body=oldsplit.role.eq('inner_validation').to_numpy();keys=behavior(x,True)
 beh=np.array([domain_bucket(k)==0 for k in keys]);beh[normal]=body[normal]
 counts={};support=[]
 for name,va in [('body',body),('behavior',beh)]:
  assert set(r.loc[va,'body_group']).isdisjoint(r.loc[~va,'body_group'])
  if name=='behavior':assert set(keys[va&~normal]).isdisjoint(keys[~va&~normal])
  manifest=r[['row_position','label_index','body_group']].copy();manifest['role']=np.where(va,'validation','fit');manifest['behavior_key']=keys
  manifest.to_parquet(out/(name+'_split.parquet'),index=False)
  for role,m in [('fit',~va),('validation',va)]:
   assert set(r.loc[m,'label_index'])=={0,1,2}
   z=r.loc[m,['row_position','label_index','body_group']].copy();z['behavior_group']=behavior(x.loc[m]);a=z.groupby(['behavior_group','label_index']).agg(rows=('row_position','size'),bodies=('body_group','nunique')).reset_index();a['split']=name;a['role']=role;support.append(a)
  counts[name]={'fit_rows':int((~va).sum()),'validation_rows':int(va.sum()),'fit_labels':r.loc[~va,'label_index'].value_counts().to_dict(),'validation_labels':r.loc[va,'label_index'].value_counts().to_dict()}
 pd.concat(support).to_parquet(out/'group_support.parquet',index=False)
 for src in [Path(__file__),root/'training/run_v54.py']:shutil.copyfile(src,out/src.name)
 cfg={'version':'v55-fit-side-risk-1','source_protocol':'v54 pressure fit rows only; prior pressure evaluation rows excluded physically',
  'specs':{k:{'risk':a,'alpha':b,'partial_views':c} for k,(a,b,c) in SPECS.items()},'epochs':EPOCHS,'seed':20260914,'splits':counts,
  'network':{'hidden_layer_sizes':[64,32],'activation':'relu','solver':'adam','batch_size':1024,'learning_rate_init':.001,'shuffle':True,'early_stopping':False},
  'views':VIEWS,'partial_view_mass':[.5,.25,.25],'adversary_step':.1,
  'risk_definition':'Behavior groups use protocol, directions and port visibility. Loss components are behavior by fit label by view; labels supervise losses only, never inputs. Every original row retained. ERM follows row frequencies. BALANCED is static uniform over observed components within each view. DRO starts balanced, then updates exponential group weights from full-fit means once per epoch. sklearn weighted mini-batches normalize by their own weight sum; this is an epoch-reweighted adaptation, not bitwise reproduction or an unbiased-gradient claim for the paper algorithm.',
  'validation_definition':'BODY reuses the previous pressure-fit inner body split. BEHAVIOR holds ASA keys protocol/directions/ICMP type+code/port visibility with SHA256 v55-behavior bucket0 of3 entirely out. ACL normal shares the same body split in both, so behavior disjointness applies to ASA only. No split chosen from model results. A single behavior bucket is a limited stress control, not all-domain cross-validation.',
  'selection':'For each candidate choose one epoch jointly across two validation protocols and three input views: minimize maximum ASA-to-normal rate, then maximum M/S error rate, then maximum normal error rate, then worst observed behavior-class mean NLL, then average error rate. Missing classes are reported, not assigned zero loss. Break ties by earlier checkpoint.',
  'continuation':'Versus E0 under the same joint epoch selector: full-view M and S error counts no worse in EACH validation protocol, at least one strict improvement; all three views in both protocols have no more ASA-to-normal and no more normal-control errors; worst behavior-class mean NLL does not increase. If none pass, stop before outer fitting. Passing authorizes further outer nested evaluation, never automatic promotion.',
  'source_bindings':{p.relative_to(root).as_posix():v.sha(p) for p in [source/'configuration.json',source/'rows.parquet',source/'observations.parquet',source/'pressure_split.parquet',root/'evidence/2026-09-14/v54_delivery/delivery.json']}}
 cfg['local_bindings']={p.name:v.sha(p) for p in out.iterdir() if p.is_file()};v.save(out/'configuration.json',cfg);v.save(out/'preregistered.json',{'sha256':v.sha(out/'configuration.json'),'fits':0});print(json.dumps(counts),flush=True)

def load(root,out):
 c=v.read(out/'configuration.json');assert v.sha(out/'configuration.json')==v.read(out/'preregistered.json')['sha256']
 for p,h in c['local_bindings'].items():assert v.sha(out/p)==h,p
 for p,h in c['source_bindings'].items():assert v.sha(root/p)==h,p
 assert v.sha(__file__)==c['local_bindings']['run_v55.py']
 return c,pd.read_parquet(out/'rows.parquet'),pd.read_parquet(out/'observations.parquet')

def fit(root,out,split,candidate):
 c,r,x=load(root,out);folder=out/(split+'_'+candidate);assert not folder.exists();folder.mkdir();started=time.monotonic()
 role=pd.read_parquet(out/(split+'_split.parquet')).role;tr=role.eq('fit').to_numpy();va=~tr;s=c['specs'][candidate];y=r.label_index.to_numpy()
 xx,yy,vi,gi,keys,counts,gv,prior=expanded(x.loc[tr],y[tr],s['partial_views']);inp=v.fit_encoder(xx);mat=v.encode(inp,xx).astype(np.float32)
 cold=np.flatnonzero(np.asarray(mat.sum(0)).ravel()==0);m=MLPClassifier(**c['network'],alpha=s['alpha'],random_state=c['seed'])
 q=distribution('balanced',np.zeros(len(keys)),None,gv,prior);log=[];allpred=[]
 v.save(folder/'risk_components.json',{'keys':keys,'counts':counts.tolist(),'view_ids':gv.tolist(),'view_prior':prior.tolist(),'fit_original_rows':int(tr.sum()),'expanded_rows':len(xx)})
 for epoch in range(1,max(c['epochs'])+1):
  before=q.copy();weights=risk_weights(s['risk'],gi,counts,gv,prior,q)
  m.partial_fit(mat,yy,classes=np.array([0,1,2]),sample_weight=weights);m.coefs_[0][cold]=0
  bundle={'input':inp,'model':m,'candidate':candidate,'epoch':epoch,'split':split,'configuration_sha256':v.sha(out/'configuration.json')}
  pp=m.predict_proba(mat);loss=-np.log(np.maximum(pp[np.arange(len(yy)),yy],1e-30));means=group_means(loss,gi,counts)
  if s['risk']=='dro':q=distribution('dro',means,q,gv,prior,c['adversary_step'])
  log.append({'epoch':epoch,'group_mean_nll':means.tolist(),'used_q':before.tolist(),'next_q':q.tolist(),'weighted_fit_nll':float(np.average(loss,weights=weights)),'sklearn_last_epoch_loss':float(m.loss_)})
  if epoch in c['epochs']:
   joblib.dump(bundle,folder/('epoch_'+str(epoch)+'.joblib'))
   reports={}
   for name,cols in VIEWS.items():
    a=x.loc[va].copy();a[cols]=v.MISSING;pred,prob=predict(bundle,a);d=r.loc[va].copy();d['scenario']=name;d['epoch']=epoch;d['pred']=pred
    for k in range(3):d['p_'+str(k)]=prob[:,k]
    d['behavior_group']=behavior(x.loc[va]);d['eligible_stress']=True if not cols else x.loc[va,cols].ne(v.MISSING).any(axis=1).to_numpy();allpred.append(d)
    aa=d[(d.route=='asa')&d.eligible_stress];reports[name]=v.metrics(aa.label_index.to_numpy(),aa.pred.to_numpy())
   print(json.dumps({'split':split,'candidate':candidate,'epoch':epoch,'fit_nll':log[-1]['weighted_fit_nll'],'validation':reports,'seconds':time.monotonic()-started}),flush=True)
 pd.concat(allpred,ignore_index=True).to_parquet(folder/'validation.parquet',index=False);v.save(folder/'learning.json',log)
 v.save(folder/'complete.json',{'seconds':time.monotonic()-started,'bindings':{p.name:v.sha(p) for p in folder.iterdir() if p.is_file()},'outer_evaluation_executed':False})

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','fit']);a.add_argument('--root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--split',choices=SPLITS);a.add_argument('--candidate',choices=list(SPECS));t=a.parse_args()
 if t.mode=='prepare':prepare(t.root.resolve(),t.out.resolve())
 else:fit(t.root.resolve(),t.out.resolve(),t.split,t.candidate)
