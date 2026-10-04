"""Post-hoc structural repair: transferable single-fact effects before group effects."""
import argparse,json,shutil,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import softmax
import run_v56 as t
import run_v54 as v
import run_v55 as n
import v56_pooling as old
ROOT=t.ROOT;BASE=t.RUN;RUN=ROOT/'artifacts/v56_shared_control_20260914'
SHARED=['transport_protocol','src_role','dst_role','src_port_range','dst_port_range','icmp_type','icmp_code','icmp_message','icmp_unreachable']

def keys(x):return {**{'field:'+k:x[k].to_numpy() for k in SHARED},**old.keys(x)}

def fit_design(x,bodies):
 maps={};counts=[];supports=[];features=[];offset=0
 for level,k in keys(x).items():
  z=pd.DataFrame({'key':k,'body':np.asarray(bodies)}).groupby('key',sort=True).agg(rows=('body','size'),bodies=('body','nunique'));maps[level]={k:i+offset for i,k in enumerate(z.index)};offset+=len(z);counts.extend(z.rows);supports.extend(z.bodies);features.extend([[level,k] for k in z.index])
 return {'maps':maps,'rows':np.array(counts,dtype=float),'bodies':np.array(supports,dtype=float),'feature_keys':features,'fit_rows':len(x)}

def design(b,x):
 ri=[];ci=[]
 for level,kk in keys(x).items():
  lookup=b['maps'][level]
  for i,k in enumerate(kk):
   if k in lookup:ri.append(i);ci.append(lookup[k])
 return sparse.csr_matrix((np.ones(len(ri)),(ri,ci)),shape=(len(x),len(b['feature_keys'])))

def prepare():
 assert not RUN.exists();RUN.mkdir(parents=True);c,r,x=t.load();sources=[BASE/'configuration.json',ROOT/'evidence/2026-09-14/v56_diagnosis/receipt.json']
 for protocol in c['protocols']:
  sources.extend([BASE/(protocol+'_split.parquet'),BASE/protocol/'base.joblib',BASE/protocol/'base_calibration.parquet',BASE/protocol/'base_evaluation.parquet',BASE/protocol/'selection.json'])
  roles=pd.read_parquet(BASE/(protocol+'_split.parquet')).role;tr=roles.eq('fit').to_numpy();ev=roles.eq('evaluation').to_numpy();b=fit_design(x.loc[tr],r.loc[tr,'body_group']);kk=keys(x.loc[ev]);covered=np.zeros(ev.sum(),dtype=bool)
  for field in SHARED:
   level='field:'+field
   if len(b['maps'][level])>1:covered|=np.array([k in b['maps'][level] for k in kk[level]])
  asa=r.loc[ev,'route'].eq('asa').to_numpy();v.save(RUN/(protocol+'_coverage.json'),{'ASA_rows':int(asa.sum()),'ASA_with_observed_nonconstant_shared_fact':int((covered&asa).sum()),'scope':'Unlabeled fit-vocabulary coverage only, not classification benefit.'})
 sources.extend([ROOT/'training/v56_pooling.py',ROOT/'training/run_v56.py',ROOT/'training/run_v54.py',ROOT/'training/run_v55.py']);shutil.copyfile(__file__,RUN/Path(__file__).name)
 cfg={'version':'v56-posthoc-shared-1','protocols':c['protocols'],'strengths':c['strengths'],'optimizer':c['optimizer'],'shared_fields':SHARED,'selection':c['selection'],'gate':c['gate'],'objective':c['objective'],'structural_delta':'Add single-field effects for 9 shared semantic fields before old coarse/fine/exact effects. No raw fixed port number in new marginal branch; all original 13 facts remain in base and exact branch. Same original class-count CE, frequency/sqrt(body support) precision, frozen E1, lambda grid and calibration-only selector. Unknown levels contribute zero.',
  'scope':'Supplement introduced after v56 group-only failure and diagnostic. Same adaptive development, no new blind test, no old pressure. Fit vocabulary coverage was checked before these fits. Not a Bayesian posterior or LMMNN reproduction.',
  'source_bindings':{p.relative_to(ROOT).as_posix():v.sha(p) for p in sources}}
 cfg['local_bindings']={p.name:v.sha(p) for p in RUN.iterdir() if p.is_file()};v.save(RUN/'configuration.json',cfg);v.save(RUN/'preregistered.json',{'sha256':v.sha(RUN/'configuration.json'),'fits':0});print(json.dumps({'prepared':True,'new_fits':0}),flush=True)

def load():
 c=v.read(RUN/'configuration.json');assert v.sha(RUN/'configuration.json')==v.read(RUN/'preregistered.json')['sha256']
 for root,key in [(ROOT,'source_bindings'),(RUN,'local_bindings')]:
  for p,h in c[key].items():assert v.sha(root/p)==h
 assert v.sha(__file__)==c['local_bindings'][Path(__file__).name]
 _,r,x=t.load();return c,r,x

def cache_predictions(protocol,role,b,r,x,mask):
 d=pd.read_parquet(BASE/protocol/('base_'+role+'.parquet'));parts=[]
 for view,cols in v.VIEWS.items():
  a=x.loc[mask].copy();a[cols]=v.MISSING;z=d[d.scenario==view].reset_index(drop=True);pd.testing.assert_frame_equal(z[r.columns],r.loc[mask].reset_index(drop=True));parts.append((z,design(b,a),np.log(np.maximum(z[['p_0','p_1','p_2']].to_numpy().astype(np.float64),1e-30))))
 return parts

def predict(cache,w):
 parts=[]
 for z,a,offset in cache:
  p=softmax(offset+a@w,axis=1);d=z.copy();d['pred']=p.argmax(1)
  for j in range(3):d['p_'+str(j)]=p[:,j]
  parts.append(d)
 return pd.concat(parts,ignore_index=True)

def run(protocol):
 c,r,x=load();folder=RUN/protocol;assert not folder.exists();folder.mkdir();roles=pd.read_parquet(BASE/(protocol+'_split.parquet')).role;tr=roles.eq('fit').to_numpy();cal=roles.eq('calibration').to_numpy();ev=roles.eq('evaluation').to_numpy();base=joblib.load(BASE/protocol/'base.joblib');b=fit_design(x.loc[tr],r.loc[tr,'body_group']);xx,counts,ids=old.compress(x.loc[tr],r.loc[tr,'label_index']);A=design(b,xx);_,p=n.predict(base,xx);offset=np.log(np.maximum(p.astype(np.float64),1e-30));cache=cache_predictions(protocol,'calibration',b,r,x,cal);choices={};start=time.monotonic()
 for strength in c['strengths']:
  precision=strength*b['rows']/b['fit_rows']/np.sqrt(b['bodies']);scale=np.sqrt(precision);scaled=A.multiply(1/scale).tocsr();res=minimize(old.objective,np.zeros(A.shape[1]*3),args=(scaled,offset,counts,np.ones(len(precision))),jac=True,method='L-BFGS-B',options=c['optimizer']);w=res.x.reshape(-1,3)/scale[:,None];w-=w.mean(1,keepdims=True);val,grad=old.objective(w.ravel(),A,offset,counts,precision);gmax=float(abs(grad.reshape(-1,3)/scale[:,None]).max());name='shared_'+str(strength)
  fit={'success':bool(res.success),'iterations':int(res.nit),'message':str(res.message),'objective':val,'scaled_gradient':gmax};bundle={'design':b,'weights':w,'strength':strength,'optimizer':fit,'base_source':(BASE/protocol/'base.joblib').relative_to(ROOT).as_posix(),'base_sha256':v.sha(BASE/protocol/'base.joblib')};joblib.dump(bundle,folder/(name+'.joblib'));d=predict(cache,w);d.to_parquet(folder/(name+'_calibration.parquet'),index=False);choices[name]={'rank':t.rank(d),'strength':strength,'optimizer':fit,'eligible':bool(res.success and gmax<=5e-6)};print(json.dumps({'protocol':protocol,'candidate':name,**choices[name],'seconds':time.monotonic()-start}),flush=True)
 eligible=[k for k,a in choices.items() if a['eligible']];chosen=min(eligible,key=lambda k:choices[k]['rank']+[-choices[k]['strength']]) if eligible else None;v.save(folder/'selection.json',{'candidates':choices,'selected':chosen});v.save(folder/'before_evaluation.json',{'bindings':{p.name:v.sha(p) for p in folder.iterdir() if p.is_file()},'evaluation_generated':False})
 if chosen:
  bundle=joblib.load(folder/(chosen+'.joblib'));cache=cache_predictions(protocol,'evaluation',b,r,x,ev);predict(cache,bundle['weights']).to_parquet(folder/'evaluation.parquet',index=False)
 v.save(folder/'complete.json',{'bindings':{p.name:v.sha(p) for p in folder.iterdir() if p.is_file()},'new_residual_fits':len(choices),'new_base_fits':0,'seconds':time.monotonic()-start});print(json.dumps({'complete':protocol,'selected':chosen}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--protocol',choices=t.PROTOCOLS);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.protocol)
