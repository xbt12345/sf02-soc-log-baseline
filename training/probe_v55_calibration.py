"""Supplemental E1 threshold-transfer control with separate fit/calibration roles.

Chosen after the main inner study, hence descriptive development, not a new blind test.
It never reads or evaluates the original outer pressure records.
"""
import argparse,json,shutil,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/v55_risk_validation_20260914';OUT=ROOT/'artifacts/v55_calibration_control_r2_20260914'
sys.path.insert(0,str(BASE));import run_v55 as n;import run_v54 as v

def choose_threshold(d):
 initial=[];support=[];events=[]
 for j,view in enumerate(n.VIEWS):
  a=d[(d.scenario==view)&(d.route=='asa')&d.eligible_stress];y=a.label_index.to_numpy();p=a[['p_0','p_1','p_2']].to_numpy();keep=a.pred.to_numpy()!=0;s=p[:,1]/np.maximum(p[:,1]+p[:,2],1e-30)
  initial.extend([int(((y==1)&~keep).sum()),int((y==2).sum())]);support.extend([int((y==1).sum()),int((y==2).sum())]);events.extend([(float(z),j,int(t)) for z,t in zip(s[keep],y[keep])])
 assert min(support)>0;events.sort();scores=np.array([z[0] for z in events]);inc=np.zeros((len(events),6),dtype=int)
 for k,(_,j,t) in enumerate(events):inc[k,2*j+(t==2)]=1 if t==1 else -1
 cum=np.vstack([np.zeros(6,dtype=int),inc.cumsum(0)])+initial
 thresholds=np.unique(np.r_[0.,.5,np.nextafter(np.unique(scores),np.inf)]);errors=cum[np.searchsorted(scores,thresholds,side='left')];rates=errors/np.array(support)
 index=min(range(len(thresholds)),key=lambda i:(rates[i].max(),rates[i].mean(),abs(thresholds[i]-.5)))
 return {'threshold':float(thresholds[index]),'calibration_errors_full_src_dst_M_S':errors[index].tolist(),'calibration_class_support_full_src_dst_M_S':support,'calibration_worst_class_error_rate':float(rates[index].max()),'candidates':len(thresholds)}

def output(bundle,r,x,mask):
 out=[]
 for view,cols in n.VIEWS.items():
  a=x.loc[mask].copy();a[cols]=v.MISSING;pred,p=n.predict(bundle,a);d=r.loc[mask].copy();d['scenario']=view;d['pred']=pred
  d['eligible_stress']=True if not cols else x.loc[mask,cols].ne(v.MISSING).any(axis=1).to_numpy()
  for j in range(3):d['p_'+str(j)]=p[:,j]
  out.append(d)
 return pd.concat(out,ignore_index=True)

def prepare():
 assert not OUT.exists();OUT.mkdir(parents=True);c,r,x=n.load(ROOT,BASE);manifests={}
 for split in n.SPLITS:
  original=pd.read_parquet(BASE/(split+'_split.parquet'));outerfit=original.role.eq('fit').to_numpy();idx=np.flatnonzero(outerfit);rr=r.loc[outerfit].reset_index(drop=True)
  a,b=next(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20260915).split(rr,rr.label_index,rr.body_group))
  m=r[['row_position','label_index','body_group']].copy();m['role']='evaluation';m.loc[idx[a],'role']='fit';m.loc[idx[b],'role']='calibration'
  for role in ['fit','calibration','evaluation']:assert set(m.loc[m.role==role,'label_index'])=={0,1,2}
  m.to_parquet(OUT/(split+'_split.parquet'),index=False);manifests[split]=m.role.value_counts().to_dict()
 shutil.copyfile(__file__,OUT/Path(__file__).name)
 cfg={'version':'v55-posthoc-separated-calibration-2','network':c['network'],'alpha':1.,'epochs':10,'seed':20260914,'splits':manifests,
  'selection':'Each original inner FIT is split again by body groups, fixed SGKF3 first fold seed20260915. Fit E1 only on subfit. Select one M/S threshold on calibration full/src-missing/dst-missing views by smallest worst class error rate, then mean rate, then proximity to 0.5. Preserve original argmax-normal decisions. Save model and threshold before scoring original inner evaluation. Never refit after calibration.',
  'scope':'Supplement chosen after E1 exploratory frontier on inspected inner development. Same-model argmax versus calibrated hard decisions isolates threshold effect with identical reduced fit data. Scores remain uncalibrated; this is decision selection, not probability calibration. Does not override main-study stop gate or perform outer fitting.',
  'source_bindings':{p.relative_to(ROOT).as_posix():v.sha(p) for p in [BASE/'configuration.json',BASE/'rows.parquet',BASE/'observations.parquet',BASE/'body_split.parquet',BASE/'behavior_split.parquet',ROOT/'evidence/2026-09-14/v55_review/receipt.json',ROOT/'evidence/2026-09-14/v55_diagnosis/receipt.json']}}
 cfg['local_bindings']={p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()};v.save(OUT/'configuration.json',cfg);v.save(OUT/'preregistered.json',{'sha256':v.sha(OUT/'configuration.json')});print(json.dumps(manifests),flush=True)

def fit(split):
 c=v.read(OUT/'configuration.json');assert v.sha(OUT/'configuration.json')==v.read(OUT/'preregistered.json')['sha256'];assert v.sha(__file__)==c['local_bindings'][Path(__file__).name]
 for p,h in c['source_bindings'].items():assert v.sha(ROOT/p)==h
 for p,h in c['local_bindings'].items():assert v.sha(OUT/p)==h
 _,r,x=n.load(ROOT,BASE);m=pd.read_parquet(OUT/(split+'_split.parquet'));tr=m.role.eq('fit').to_numpy();cal=m.role.eq('calibration').to_numpy();ev=m.role.eq('evaluation').to_numpy()
 folder=OUT/split;assert not folder.exists();folder.mkdir();inp=v.fit_encoder(x.loc[tr]);mat=v.encode(inp,x.loc[tr]).astype(np.float32);cold=np.flatnonzero(np.asarray(mat.sum(0)).ravel()==0)
 model=MLPClassifier(**c['network'],alpha=c['alpha'],random_state=c['seed'])
 for epoch in range(c['epochs']):model.partial_fit(mat,r.loc[tr,'label_index'],classes=[0,1,2]);model.coefs_[0][cold]=0
 bundle={'model':model,'input':inp,'split':split,'configuration_sha256':v.sha(OUT/'configuration.json')};joblib.dump(bundle,folder/'model.joblib')
 d=output(bundle,r,x,cal);d.to_parquet(folder/'calibration.parquet',index=False);selection=choose_threshold(d);v.save(folder/'selection.json',selection)
 v.save(folder/'before_evaluation.json',{'bindings':{p.name:v.sha(p) for p in folder.iterdir() if p.is_file()},'evaluation_generated':False})
 e=output(bundle,r,x,ev);p=e[['p_0','p_1','p_2']].to_numpy();s=(p[:,1]/np.maximum(p[:,1]+p[:,2],1e-30)).astype(np.float64)
 e['threshold_pred']=np.where(e.pred.to_numpy()==0,0,np.where(s>=selection['threshold'],1,2));e.to_parquet(folder/'evaluation.parquet',index=False)
 summary={}
 for view in n.VIEWS:
  a=e[(e.scenario==view)&e.eligible_stress&(e.route=='asa')];b=e[(e.scenario==view)&(e.route=='asa_acl')]
  summary[view]={'argmax':v.metrics(a.label_index.to_numpy(),a.pred.to_numpy()),'threshold':v.metrics(a.label_index.to_numpy(),a.threshold_pred.to_numpy()),'normal_argmax':v.metrics(b.label_index.to_numpy(),b.pred.to_numpy()),'normal_threshold':v.metrics(b.label_index.to_numpy(),b.threshold_pred.to_numpy())}
 v.save(folder/'summary.json',summary);v.save(folder/'complete.json',{'bindings':{p.name:v.sha(p) for p in folder.iterdir() if p.is_file()}});print(json.dumps({'split':split,'threshold':selection['threshold'],'summary':summary}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit']);p.add_argument('--split',choices=n.SPLITS);a=p.parse_args();prepare() if a.mode=='prepare' else fit(a.split)
