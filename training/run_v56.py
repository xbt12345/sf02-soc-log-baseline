"""Bounded support-shrinkage study; every fit/selection is separated from evaluation."""
import argparse,json,shutil,sys,time,importlib.metadata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
import run_v54 as v
import run_v55 as n
import v56_pooling as m
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/v55_risk_validation_20260914';PREV=ROOT/'artifacts/v55_calibration_control_r2_20260914';RUN=ROOT/'artifacts/v56_support_pooling_20260914'
PROTOCOLS=['body','behavior_0','behavior_1','behavior_2'];STRENGTHS=[.1,1.,10.]

def prepare():
 assert not RUN.exists();RUN.mkdir(parents=True);r=pd.read_parquet(BASE/'rows.parquet');x=pd.read_parquet(BASE/'observations.parquet');old=pd.read_parquet(BASE/'body_split.parquet');normal=r.route.eq('asa_acl').to_numpy();b=np.array([n.domain_bucket(k) for k in n.behavior(x,True)])
 info={}
 for protocol in PROTOCOLS:
  ev=old.role.eq('validation').to_numpy(copy=True) if protocol=='body' else b==int(protocol[-1]);ev[normal]=old.role.eq('validation').to_numpy()[normal];idx=np.flatnonzero(~ev);rr=r.iloc[idx]
  a,c=next(StratifiedGroupKFold(3,shuffle=True,random_state=20260915).split(rr,rr.label_index,rr.body_group));z=r[['row_position','label_index','body_group']].copy();z['role']='evaluation';z.loc[idx[a],'role']='fit';z.loc[idx[c],'role']='calibration'
  for left,right in [('fit','calibration'),('fit','evaluation'),('calibration','evaluation')]:assert set(z.loc[z.role==left,'body_group']).isdisjoint(z.loc[z.role==right,'body_group'])
  for role in ['fit','calibration','evaluation']:assert set(z.loc[z.role==role,'label_index'])=={0,1,2}
  if protocol in ['body','behavior_0']:pd.testing.assert_frame_equal(z,pd.read_parquet(PREV/('body_split.parquet' if protocol=='body' else 'behavior_split.parquet')))
  z.to_parquet(RUN/(protocol+'_split.parquet'),index=False);info[protocol]={role:z.loc[z.role==role,'label_index'].value_counts().to_dict() for role in ['fit','calibration','evaluation']}
 for name in ['run_v56.py','v56_pooling.py','run_v54.py','run_v55.py','test_v56.py']:shutil.copyfile(ROOT/'training'/name,RUN/name)
 c={'version':'v56-support-shrinkage-1','protocols':PROTOCOLS,'views':v.VIEWS,'strengths':STRENGTHS,'kinds':['uniform','support'],'splits':info,'base_network':v.read(BASE/'configuration.json')['network'],'base_alpha':1.,'base_epochs':10,'base_seed':20260914,
 'optimizer':{'maxiter':400,'gtol':1e-7,'ftol':1e-12,'maxls':40},
 'model':'Frozen E1 logits plus additive learned corrections for coarse behavior, fine behavior, exact 13-field observation. Unknown keys contribute zero correction at that level. No label/identity/time/product/body hash is a prediction feature. Full observations and original class labels retained.',
 'objective':'Original row-mean multiclass cross entropy plus 0.5 sum_j precision_j ||correction_j||^2. Uniform precision=lambda*n_j/N. Support precision=lambda*n_j/(N*sqrt(B_j)), B_j is number of distinct fit body groups. Coarse/fine/exact levels all fitted jointly. B_j is a conservative support proxy, not established incident independence. This is fixed ridge shrinkage inspired by partial pooling, not a Bayesian posterior, learned prior or reproduction of LMMNN. No record loss downweighting.',
 'solver':'Exact aggregation of identical observations into three original label counts. Optimize rescaled parameters sqrt(precision)*weights with L-BFGS-B; no change to mathematical objective. Same options and zero initialization for each candidate, no warm start. success and scaled gradient <=0.000005 needed for eligibility.',
 'selection':'Per protocol and kind, choose strength only on calibration full/src-role-missing/dst-role-missing: minimize maximum ASA->normal rate, then maximum M/S class error rate, then maximum normal error rate, then average ASA NLL, then total ASA errors. Prefer larger strength on exact ties. Argmax only, no threshold fitting; both calibration and evaluation use same model, no refit. Port-deletion views are recorded and enter evaluation protection checks but not parameter selection.',
 'gate':'For each candidate family, full-input M and S errors must be no greater than the same-split frozen E1 in BODY and each of three behavior buckets, with strict total improvement in BODY and pooled nonoverlapping behavior ASA. Every evaluation view must have no more ASA->normal and normal-control errors. Worst M/S error rate across the three core views and four protocols must not increase. If none pass, stop before original pressure/full-data expansion. Passing only permits further development, never real-transfer certification.',
 'scope':'Only v55 fit-side 99398 records, prior pressure evaluation physically absent. Three behavior buckets partition ASA evaluation, BODY overlaps them. Same four normal evaluation rows reused across all protocols; no extra independent normal samples. Previously viewed sources remain adaptive development, not new blind testing. Models for BODY and bucket0 reuse exact existing E1 subfit models after split/source binding; buckets1,2 train new bases.',
 'sources':[{'url':'https://arxiv.org/abs/2206.03314','use':'random effects plus general predictor motivation; predominantly regression, not direct SOC evidence'},{'url':'https://github.com/gsimchoni/lmmnn','use':'author implementation inspected at README level; not installed or reproduced'},{'url':'https://mc-stan.org/rstanarm/articles/pooling.html','use':'partial pooling and prediction/model checking principle; fixed support penalty is our testable adaptation'}],
 'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','joblib','pyarrow']},
 'source_bindings':{p.relative_to(ROOT).as_posix():v.sha(p) for p in [BASE/'configuration.json',BASE/'rows.parquet',BASE/'observations.parquet',BASE/'body_split.parquet',PREV/'configuration.json',PREV/'body_split.parquet',PREV/'behavior_split.parquet',PREV/'body/model.joblib',PREV/'behavior/model.joblib',ROOT/'evidence/2026-09-14/v55_delivery/delivery.json']}}
 c['local_bindings']={p.name:v.sha(p) for p in RUN.iterdir() if p.is_file()};v.save(RUN/'configuration.json',c);v.save(RUN/'preregistered.json',{'sha256':v.sha(RUN/'configuration.json'),'new_fits':0});print(json.dumps(info),flush=True)

def load():
 c=v.read(RUN/'configuration.json');assert v.sha(RUN/'configuration.json')==v.read(RUN/'preregistered.json')['sha256']
 for base,key in [(ROOT,'source_bindings'),(RUN,'local_bindings')]:
  for p,h in c[key].items():assert v.sha(base/p)==h,p
 for name in ['run_v56.py','v56_pooling.py']:assert v.sha(ROOT/'training'/name)==c['local_bindings'][name]
 return c,pd.read_parquet(BASE/'rows.parquet'),pd.read_parquet(BASE/'observations.parquet')

def predictions(bundle,r,x,mask,baseline=False):
 out=[]
 for view,cols in v.VIEWS.items():
  a=x.loc[mask].copy();a[cols]=v.MISSING;yp,p=n.predict(bundle,a) if baseline else m.predict(bundle,a);z=r.loc[mask].copy();z['scenario']=view;z['pred']=yp
  for j in range(3):z['p_'+str(j)]=p[:,j]
  z['eligible_stress']=True if not cols else x.loc[mask,cols].ne(v.MISSING).any(axis=1).to_numpy();out.append(z)
 return pd.concat(out,ignore_index=True)

def rank(d):
 bad=[];rates=[];normal=[];loss=[];err=0
 for view in n.VIEWS:
  a=d[(d.scenario==view)&(d.route=='asa')&d.eligible_stress];b=d[(d.scenario==view)&(d.route=='asa_acl')];mm=v.metrics(a.label_index.to_numpy(),a.pred.to_numpy());bad.append(mm['ASA_to_normal']/len(a));rates.extend([mm['class_errors_B_M_S'][k]/int((a.label_index==k).sum()) for k in [1,2]]);normal.append(float((b.pred!=b.label_index).mean()));pp=a[['p_0','p_1','p_2']].to_numpy();loss.append(float(-np.log(np.maximum(pp[np.arange(len(a)),a.label_index],1e-30)).mean()));err+=mm['errors']
 return [max(bad),max(rates),max(normal),float(np.mean(loss)),err]

def run(protocol):
 c,r,x=load();folder=RUN/protocol;assert not folder.exists();folder.mkdir();z=pd.read_parquet(RUN/(protocol+'_split.parquet'));tr=z.role.eq('fit').to_numpy();cal=z.role.eq('calibration').to_numpy();ev=z.role.eq('evaluation').to_numpy();y=r.label_index.to_numpy();start=time.monotonic()
 if protocol in ['body','behavior_0']:
  p=PREV/('body' if protocol=='body' else 'behavior')/'model.joblib';base=joblib.load(p);base_origin={'reused':p.relative_to(ROOT).as_posix(),'sha256':v.sha(p),'new_base_fit':False}
 else:
  enc=v.fit_encoder(x.loc[tr]);a=v.encode(enc,x.loc[tr]).astype(np.float32);cold=np.flatnonzero(np.asarray(a.sum(0)).ravel()==0);net=MLPClassifier(**c['base_network'],alpha=1.,random_state=c['base_seed'])
  for _ in range(c['base_epochs']):net.partial_fit(a,y[tr],classes=[0,1,2]);net.coefs_[0][cold]=0
  base={'input':enc,'model':net};base_origin={'new_base_fit':True,'epochs':10,'fit_rows':int(tr.sum())}
 for j,f in enumerate(v.FIELDS):assert base['input']['seen'][j]==set(x.loc[tr,f])
 joblib.dump(base,folder/'base.joblib');v.save(folder/'base_origin.json',base_origin);predictions(base,r,x,cal,True).to_parquet(folder/'base_calibration.parquet',index=False);choices={};allfits={}
 for kind in c['kinds']:
  candidates={}
  for strength in c['strengths']:
   name=kind+'_'+str(strength);bundle,fit=m.fit(base,x.loc[tr],y[tr],r.loc[tr,'body_group'],kind,strength,c['optimizer']);joblib.dump(bundle,folder/(name+'.joblib'));d=predictions(bundle,r,x,cal);d.to_parquet(folder/(name+'_calibration.parquet'),index=False);score=rank(d);ok=fit['success'] and fit['max_abs_scaled_gradient']<=5e-6
   candidates[name]={'rank':score,'optimizer':fit,'eligible':bool(ok),'strength':strength};allfits[name]=fit;print(json.dumps({'protocol':protocol,'candidate':name,'calibration_rank':score,'optimizer':fit,'seconds':time.monotonic()-start}),flush=True)
  eligible=[k for k,a in candidates.items() if a['eligible']];chosen=min(eligible,key=lambda k:candidates[k]['rank']+[-candidates[k]['strength']]) if eligible else None;choices[kind]={'selected':chosen,'candidates':candidates}
 v.save(folder/'selection.json',choices);v.save(folder/'before_evaluation.json',{'bindings':{p.name:v.sha(p) for p in folder.iterdir() if p.is_file()},'evaluation_generated':False})
 predictions(base,r,x,ev,True).to_parquet(folder/'base_evaluation.parquet',index=False)
 for kind,z in choices.items():
  if z['selected'] is not None:predictions(joblib.load(folder/(z['selected']+'.joblib')),r,x,ev).to_parquet(folder/(kind+'_evaluation.parquet'),index=False)
 v.save(folder/'complete.json',{'bindings':{p.name:v.sha(p) for p in folder.iterdir() if p.is_file()},'new_base_fits':int(base_origin['new_base_fit']),'residual_fits':len(allfits),'seconds':time.monotonic()-start});print(json.dumps({'completed':protocol,'selections':{k:z['selected'] for k,z in choices.items()}}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--protocol',choices=PROTOCOLS);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.protocol)
