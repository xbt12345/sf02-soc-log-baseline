"""Independent input/matrix/layer replay and epoch risk update reconstruction."""
import json,sys,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/v55_risk_validation_20260914';OUT=ROOT/'evidence/2026-09-14/v55_verification'
sys.path.insert(0,str(RUN));import run_v55 as n;import run_v54 as v

def encode(bundle,x):
 enc=bundle['input'];offset=np.r_[0,np.cumsum([len(c) for c in enc['encoder'].categories_])];ri=[];ci=[]
 for j,k in enumerate(v.FIELDS):
  lookup={s:i+offset[j] for i,s in enumerate(enc['encoder'].categories_[j])}
  for i,value in enumerate(x[k]):
   value=value if value in enc['seen'][j] or value==v.MISSING else v.UNKNOWN;ri.append(i);ci.append(lookup[value])
 a=sparse.csr_matrix((np.ones(len(ri),dtype=np.float32),(ri,ci)),shape=(len(x),offset[-1]));assert (a!=v.encode(enc,x)).nnz==0
 return a

def manual(bundle,x):
 keys=v.frame_keys(x);ids,_=pd.factorize(keys,sort=False);first=np.unique(ids,return_index=True)[1];a=encode(bundle,x.iloc[first]);m=bundle['model'];z=a
 for i in range(len(m.coefs_)):
  z=z@m.coefs_[i]+m.intercepts_[i]
  if i<len(m.coefs_)-1:z=np.maximum(z,0)
 return softmax(z,axis=1)[ids]

def main():
 assert not OUT.exists() or not any(OUT.iterdir()),'Refuse to overwrite verification evidence';OUT.mkdir(parents=True,exist_ok=True);cfg,r,x=n.load(ROOT,RUN)
 allrows=pd.read_parquet(ROOT/'artifacts/v54_method_change_20260914/rows.parquet');allowed=allrows.inner_role.ne(2);orig=allrows.loc[allowed].reset_index(drop=True);pd.testing.assert_frame_equal(r,orig)
 original_inputs=pd.read_parquet(ROOT/'artifacts/v54_method_change_20260914/observations.parquet');pd.testing.assert_frame_equal(x,original_inputs.loc[allowed].reset_index(drop=True))
 y=r.label_index.to_numpy();done=[];rows=0;delta=0.;riskdelta=0.;control=[]
 for split in n.SPLITS:
  manifest=pd.read_parquet(RUN/(split+'_split.parquet'));tr=manifest.role.eq('fit').to_numpy();va=~tr
  np.testing.assert_array_equal(manifest.row_position,r.row_position);assert set(r.loc[tr,'body_group']).isdisjoint(r.loc[va,'body_group'])
  old=pd.read_parquet(ROOT/'artifacts/v54_method_change_20260914/pressure_split.parquet');old=old[old.role!='outer_evaluation'].reset_index(drop=True);body=old.role.eq('inner_validation').to_numpy()
  normal=r.route.eq('asa_acl').to_numpy()
  if split=='body':np.testing.assert_array_equal(va,body)
  else:
   z=x[['transport_protocol','src_role','dst_role','icmp_type','icmp_code']].astype(str).agg('|'.join,axis=1)+'|'+x.src_port_fixed.ne(v.MISSING).astype(str)+'|'+x.dst_port_fixed.ne(v.MISSING).astype(str)
   import hashlib
   expected=z.map(lambda k:int(hashlib.sha256(('v55-behavior|'+k).encode()).hexdigest()[:8],16)%3==0).to_numpy(copy=True);expected[normal]=body[normal];np.testing.assert_array_equal(va,expected)
   assert set(z[va&~normal]).isdisjoint(z[tr&~normal])
  for candidate,s in cfg['specs'].items():
   folder=RUN/(split+'_'+candidate)
   if not (folder/'complete.json').exists():
    print(json.dumps({'waiting_for_completed_training':folder.name}),flush=True);deadline=time.monotonic()+1800
    while not (folder/'complete.json').exists():
     if time.monotonic()>deadline:raise TimeoutError(folder.name)
     time.sleep(2)
   for p,h in v.read(folder/'complete.json')['bindings'].items():assert v.sha(folder/p)==h
   c=v.read(folder/'risk_components.json');log=v.read(folder/'learning.json');d=pd.read_parquet(folder/'validation.parquet');keys=c['keys'];gv=np.array(c['view_ids']);prior=np.array(c['view_prior']);counts=np.array(c['counts'])
   if split=='body' and candidate=='E0':
    for e in cfg['epochs']:
     oldpred=pd.read_parquet(ROOT/'artifacts/v54_method_change_20260914/pressure_neural'/('inner_'+str(e)+'_decisions.parquet'));now=d[(d.epoch==e)&(d.scenario=='full')]
     np.testing.assert_array_equal(now.row_position,oldpred.row_position);np.testing.assert_array_equal(now.pred,oldpred.pred)
     gap=float(abs(now[['p_0','p_1','p_2']].to_numpy()-oldpred[['p_benign','p_malicious','p_suspicious']].to_numpy()).max());assert gap<3e-6
     control.append({'epoch':e,'all_decisions_match_v54':True,'max_probability_difference':gap})
   xx,yy,vi,gi,k,cc,vv,pr=n.expanded(x.loc[tr],y[tr],s['partial_views']);assert k==keys;np.testing.assert_array_equal(cc,counts)
   # Reconstruct component counts independently from original row labels and facts.
   base=x.loc[tr,['transport_protocol','src_role','dst_role']].astype(str).agg('|'.join,axis=1)+'|'+x.loc[tr,'src_port_fixed'].ne(v.MISSING).astype(str)+'|'+x.loc[tr,'dst_port_fixed'].ne(v.MISSING).astype(str)
   direct=[]
   for view in (list(n.VIEWS) if s['partial_views'] else ['full']):direct.extend([json.dumps([view,b,int(t)],separators=(',',':')) for b,t in zip(base,y[tr])])
   independent=pd.Series(direct).value_counts();np.testing.assert_array_equal([independent[k] for k in keys],counts)
   previous=np.array([prior[j]/(gv==j).sum() for j in gv])
   for epoch,item in enumerate(log,1):
    np.testing.assert_allclose(item['used_q'],previous,rtol=0,atol=1e-12);means=np.array(item['group_mean_nll']);q=previous.copy()
    if s['risk']=='dro':
     for j in range(len(prior)):
      ix=gv==j;zz=previous[ix]*np.exp(cfg['adversary_step']*means[ix]);q[ix]=prior[j]*zz/zz.sum()
    np.testing.assert_allclose(item['next_q'],q,rtol=0,atol=1e-12)
    w=n.risk_weights(s['risk'],gi,counts,gv,prior,previous)
    obj=sum(prior[j]*np.average(means[gv==j],weights=counts[gv==j]) for j in range(len(prior))) if s['risk']=='erm' else previous@means
    np.testing.assert_allclose(obj,item['weighted_fit_nll'],rtol=0,atol=1e-7);previous=q
    if epoch not in cfg['epochs']:continue
    bundle=joblib.load(folder/('epoch_'+str(epoch)+'.joblib'));m=bundle['model'];assert m.alpha==s['alpha'] and m.random_state==cfg['seed']
    for j,field in enumerate(v.FIELDS):assert bundle['input']['seen'][j]==set(xx[field])
    for view,cols in n.VIEWS.items():
     a=x.loc[va].copy();a[cols]=v.MISSING;p=manual(bundle,a);saved=d[(d.epoch==epoch)&(d.scenario==view)]
     np.testing.assert_array_equal(saved.row_position,r.loc[va,'row_position']);np.testing.assert_array_equal(saved.label_index,y[va]);ref=saved[['p_0','p_1','p_2']].to_numpy()
     delta=max(delta,float(abs(p-ref).max()));np.testing.assert_allclose(p,ref,atol=3e-6,rtol=0);np.testing.assert_array_equal(p.argmax(1),saved.pred);rows+=len(p)
     eligible=np.ones(len(a),dtype=bool) if not cols else x.loc[va,cols].ne(v.MISSING).any(axis=1).to_numpy();np.testing.assert_array_equal(saved.eligible_stress,eligible)
    p=manual(bundle,xx);loss=-np.log(np.maximum(p[np.arange(len(yy)),yy],1e-30));indmeans=np.bincount(gi,weights=loss,minlength=len(counts))/counts
    riskdelta=max(riskdelta,float(abs(indmeans-means).max()));np.testing.assert_allclose(indmeans,means,atol=1e-4,rtol=0);rows+=len(p)
    aa=encode(bundle,xx.iloc[np.unique(pd.factorize(v.frame_keys(xx))[0],return_index=True)[1]]);cold=np.flatnonzero(np.asarray(aa.sum(0)).ravel()==0);assert np.all(m.coefs_[0][cold]==0)
   done.append({'split':split,'candidate':candidate,'epochs_checked':len(log),'saved_models_checked':3});print(json.dumps(done[-1]),flush=True)
 v.save(OUT/'verification.json',{'all_checks_passed':True,'runs':done,'score_rows_replayed':rows,'max_probability_difference':delta,'max_fit_group_loss_difference':riskdelta,'same_BODY_ERM_control_reproduced':control,
  'scope':'Physical exclusion of prior pressure evaluation, split/input/layer replay, recorded full-fit risks and all adversary update arithmetic; not a claim that every Adam mini-batch was independently rerun or that validation is blind.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}})
if __name__=='__main__':main()
