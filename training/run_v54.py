"""ASA method change: nested kernel selection and neural learning-curve control.

Exact observed categories are retained. Outer evaluation is never read by fit/selection.
"""
import argparse,copy,hashlib,importlib.metadata,json,shutil,sys,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

FIELDS=['action','outcome','transport_protocol','src_role','dst_role','src_port_fixed','dst_port_fixed','src_port_range','dst_port_range','icmp_type','icmp_code','icmp_message','icmp_unreachable']
VIEWS={'full':[],'no_src':['src_port_fixed','src_port_range'],'no_dst':['dst_port_fixed','dst_port_range'],'no_ports':['src_port_fixed','src_port_range','dst_port_fixed','dst_port_range'],'no_src_role':['src_role'],'no_dst_role':['dst_role']}
MISSING='__MISSING__';UNKNOWN='__UNKNOWN__';PROTOCOLS=['pressure','fold_0','fold_1','fold_2']
KERNELS={'LINEAR':{'kernel':'linear'},'SMOOTH':{'kernel':'rbf','gamma':.125},'LOCAL':{'kernel':'rbf','gamma':.5}}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def select(r,protocol):return (r.inner_role!=2).to_numpy() if protocol=='pressure' else (r.fold!=int(protocol[-1])).to_numpy()
def frame_keys(x):return np.asarray([json.dumps(list(row),separators=(',',':')) for row in x.itertuples(index=False,name=None)],dtype=object)
def fit_encoder(x):
 assert not x.isin([UNKNOWN]).to_numpy().any()
 vocab=[sorted(set(x[col])|{MISSING,UNKNOWN}) for col in FIELDS]
 enc=OneHotEncoder(categories=vocab,handle_unknown='error',dtype=np.float64,sparse_output=True)
 enc.fit(x)
 seen=[set(x[c]) for c in FIELDS]
 return {'encoder':enc,'seen':seen,'fields':FIELDS}
def encode(b,x):
 z=x.copy()
 for j,col in enumerate(FIELDS):
  z[col]=[v if v in b['seen'][j] or v==MISSING else UNKNOWN for v in z[col]]
 a=b['encoder'].transform(z)
 assert np.all(np.asarray(a.sum(1)).ravel()==len(FIELDS))
 return a
def compress(x,y):
 # Sum the original convex hinge-loss terms exactly, including mixed labels.
 a=x.copy();a['target']=y;g=a.groupby(FIELDS+['target'],sort=True,dropna=False).size().reset_index(name='weight')
 assert int(g.weight.sum())==len(y)
 for k in [0,1,2]:assert int(g.loc[g.target==k,'weight'].sum())==int((y==k).sum())
 return g[FIELDS],g.target.to_numpy(),g.weight.to_numpy(dtype=float)
def metrics(y,pred):
 cm=np.zeros((3,3),dtype=np.int64);np.add.at(cm,(y,pred),1);support=cm.sum(1)
 recall=np.divide(cm.diagonal(),support,out=np.zeros(3),where=support>0)
 den=cm.sum(0)+support;f=np.divide(2*cm.diagonal(),den,out=np.zeros(3),where=den>0)
 return {'rows':len(y),'errors':int((y!=pred).sum()),'confusion_B_M_S':cm.tolist(),'recall_B_M_S':recall.tolist(),'class_errors_B_M_S':(support-cm.diagonal()).tolist(),'macro_f1_M_S':float(f[1:].mean()),'ASA_to_normal':int(cm[1:,0].sum())}
def rank(y,pred,unseen):
 full=metrics(y,pred);u=metrics(y[unseen],pred[unseen])
 assert set(y[unseen])>={1,2}
 # Protect severe false normal predictions, then improve the weakest class across
 # full and unseen-combination validation. Normal counterexamples break later ties.
 worst=min(full['recall_B_M_S'][1:]+u['recall_B_M_S'][1:])
 return [full['ASA_to_normal'],-float(worst),-full['macro_f1_M_S'],full['class_errors_B_M_S'][0],full['errors']]
def predict(bundle,x):
 ids,_=pd.factorize(frame_keys(x),sort=False);first=np.unique(ids,return_index=True)[1]
 z=encode(bundle['input'],x.iloc[first])
 if bundle['family']=='kernel':
  m=bundle['model'];return m.predict(z)[ids],m.decision_function(z)[ids]
 pp=np.mean([m.predict_proba(z.astype(np.float32)) for m in bundle['models']],axis=0)
 return pp.argmax(1)[ids],pp[ids]

def prepare(root,out):
 assert not out.exists();out.mkdir(parents=True)
 source=root/'artifacts/v53_asa_factorial_r2_20260914'
 r=pd.read_parquet(source/'rows.parquet');x=pd.read_parquet(source/'observations.parquet').astype(object).fillna(MISSING)
 assert len(r)==106969 and x.columns.tolist()==FIELDS and r.groupby('body_group').projection_id.nunique().max()==1
 r.to_parquet(out/'rows.parquet',index=False);x.to_parquet(out/'observations.parquet',index=False)
 manifests={}
 for name in PROTOCOLS:
  outer=select(r,name);ix=np.flatnonzero(outer);rr=r.iloc[ix].reset_index(drop=True)
  a,b=next(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20260914).split(rr,rr.label_index,rr.body_group))
  assert set(rr.iloc[a].label_index)=={0,1,2} and set(rr.iloc[b].label_index)=={0,1,2}
  assert set(rr.iloc[a].body_group).isdisjoint(rr.iloc[b].body_group)
  manifest=r[['row_position','label_index','body_group','union_group']].copy();manifest['role']='outer_evaluation';manifest.loc[ix[a],'role']='inner_fit';manifest.loc[ix[b],'role']='inner_validation'
  manifest.to_parquet(out/(name+'_split.parquet'),index=False)
  manifests[name]={'inner_fit':len(a),'inner_validation':len(b),'outer_fit':int(outer.sum()),'outer_evaluation':int((~outer).sum())}
 shutil.copyfile(__file__,out/'run_v54.py')
 cfg={'version':'v54-kernel-neural-nested-1','fields':FIELDS,'views':VIEWS,'protocols':PROTOCOLS,'manifests':manifests,
  'kernel_candidates':KERNELS,'svc':{'C':1.,'tol':.0001,'cache_size':512,'max_iter':-1,'shrinking':True,'probability':False,'decision_function_shape':'ovr','break_ties':True,'random_state':20260914},
  'neural':{'hidden_layer_sizes':[64,32],'activation':'relu','solver':'adam','alpha':.01,'batch_size':1024,'learning_rate_init':.001,'shuffle':True,'early_stopping':False},
  'neural_epochs':[10,30,60],'neural_inner_seed':20260914,'neural_final_seeds':[20260914,20260915],
  'inner_selection':'Fixed first StratifiedGroupKFold(3) split inside each outer fit. All vocabularies and models refit on inner fit only. Rank by severe ASA-to-normal errors, then worst M/S recall over full and unseen-combination inner validation, then M/S macro-F1, normal errors, total errors. Fixed candidate order breaks exact ties. No outer evaluation is opened until selection and final fits are saved.',
  'kernel_refit':'Select one kernel inside each outer fit, refit on all original outer-fit rows. Also refit LINEAR as an objective-matched capacity control; reuse when selected. Grouping identical input+label only compresses identical hinge-loss terms; sample weights are exact original row counts. No duplicate downweighting or relabeling.',
  'neural_refit':'Select epoch inside outer fit, then fit same architecture from scratch with two fixed seeds using every original outer-fit row. Mean probabilities is the fixed final model. Never choose seed on outer outcomes. Unobserved input columns have first-layer weights set to zero, so unseen categories do not retain random untrained effects.',
  'unknown_categories':'Each field has explicit missing and novel categories; novel is distinct from naturally missing. Fixed one-hot norm protects kernel distance from the all-zero unknown shortcut. No external value enters a fit vocabulary.',
  'score_contract':'Kernel outputs are margins, not probabilities. Neural scores are uncalibrated probabilities. SVC probability=True forbidden because its internal calibration is not our grouped split.',
  'scope':'ASA and existing 16 ACL normal controls, official development only. No raw identity/time/product features; no augmentation of labels or row weights. No deployment or transfer guarantee. Prior evaluated folds remain development.',
  'quality_gates':['Primary ASA errors lower than v51, both class errors no greater, at least 2 folds improve','Pressure ASA total and both class errors no greater than v51','No new ASA-to-normal errors','ACL normal errors no greater than v51','For a new-generalization claim, both primary unseen-combination slices together must improve and pressure must not regress'],
  'source_bindings':{p.relative_to(root).as_posix():sha(p) for p in [source/'rows.parquet',source/'observations.parquet',source/'configuration.json',root/'evidence/2026-09-14/v53_delivery/delivery.json']},
  'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','scikit-learn','joblib','pyarrow']}}
 cfg['local_bindings']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
 save(out/'configuration.json',cfg);save(out/'preregistered.json',{'configuration_sha256':sha(out/'configuration.json'),'new_models_fit':0})
 print(json.dumps({'prepared':str(out),'splits':manifests}),flush=True)
def load(root,out):
 c=read(out/'configuration.json');assert sha(out/'configuration.json')==read(out/'preregistered.json')['configuration_sha256']
 assert sha(__file__)==c['local_bindings']['run_v54.py']
 for p,h in c['source_bindings'].items():assert sha(root/p)==h,p
 for p,h in c['local_bindings'].items():assert sha(out/p)==h,p
 return c,pd.read_parquet(out/'rows.parquet'),pd.read_parquet(out/'observations.parquet')
def fit(root,out,protocol,family):
 c,r,x=load(root,out);folder=out/(protocol+'_'+family);assert not folder.exists();folder.mkdir()
 split=pd.read_parquet(out/(protocol+'_split.parquet'));a=split.role=='inner_fit';b=split.role=='inner_validation';outer=select(r,protocol);y=r.label_index.to_numpy()
 trainkey=set(frame_keys(x.loc[a]));unseen=np.asarray([k not in trainkey for k in frame_keys(x.loc[b])]);assert set(y[b][unseen])>={1,2}
 enc=fit_encoder(x.loc[a]);inner_reports={};rankings={};started=time.monotonic()
 if family=='kernel':
  xx,yy,weight=compress(x.loc[a],y[a]);mat=encode(enc,xx);val=encode(enc,x.loc[b])
  pd.DataFrame({'label':yy,'weight':weight,'input_key':frame_keys(xx)}).to_parquet(folder/'inner_compression.parquet',index=False)
  for name,kw in c['kernel_candidates'].items():
   t=time.monotonic();m=SVC(**c['svc'],**kw);m.fit(mat,yy,sample_weight=weight);assert m.fit_status_==0
   pred,_=predict({'family':'kernel','input':enc,'model':m},x.loc[b]);rankings[name]=rank(y[b],pred,unseen)
   inner_reports[name]={'full':metrics(y[b],pred),'unseen':metrics(y[b][unseen],pred[unseen]),'rank':rankings[name],'seconds':time.monotonic()-t,'n_iter':m.n_iter_.tolist()}
   joblib.dump({'family':'kernel','input':enc,'model':m},folder/('inner_'+name+'.joblib'))
   dd=r.loc[b].copy();dd['pred']=pred;dd['unseen_fit_key']=unseen;dd.to_parquet(folder/('inner_'+name+'_decisions.parquet'),index=False)
   print(json.dumps({'protocol':protocol,'family':family,'inner':name,**inner_reports[name]}),flush=True)
  chosen=min(rankings,key=rankings.get);save(folder/'selection.json',{'selected':chosen,'ranking':rankings,'selected_before_outer_evaluation':True})
  enc=fit_encoder(x.loc[outer]);xx,yy,weight=compress(x.loc[outer],y[outer]);mat=encode(enc,xx)
  pd.DataFrame({'label':yy,'weight':weight,'input_key':frame_keys(xx)}).to_parquet(folder/'outer_compression.parquet',index=False)
  for name in dict.fromkeys(['LINEAR',chosen]):
   m=SVC(**c['svc'],**c['kernel_candidates'][name]);m.fit(mat,yy,sample_weight=weight);assert m.fit_status_==0
   joblib.dump({'family':'kernel','input':enc,'model':m,'choice':name,'protocol':protocol,'configuration_sha256':sha(out/'configuration.json')},folder/('final_'+name+'.joblib'))
   print(json.dumps({'protocol':protocol,'family':family,'outer_fitted':name,'n_support':m.n_support_.tolist()}),flush=True)
 else:
  mat=encode(enc,x.loc[a]).astype(np.float32);val=encode(enc,x.loc[b]).astype(np.float32);cold=np.flatnonzero(np.asarray(mat.sum(0)).ravel()==0)
  m=MLPClassifier(**c['neural'],random_state=c['neural_inner_seed']);learning=[]
  for epoch in range(1,max(c['neural_epochs'])+1):
   m.partial_fit(mat,y[a],classes=np.array([0,1,2]));m.coefs_[0][cold]=0
   if epoch in c['neural_epochs']:
    pp=m.predict_proba(val);pred=pp.argmax(1);name=str(epoch);rankings[name]=rank(y[b],pred,unseen)
    inner_reports[name]={'full':metrics(y[b],pred),'unseen':metrics(y[b][unseen],pred[unseen]),'rank':rankings[name],'training_loss':float(m.loss_),'seconds_since_start':time.monotonic()-started}
    joblib.dump({'family':'neural','input':enc,'models':[m],'epochs':epoch},folder/('inner_'+name+'.joblib'))
    dd=r.loc[b].copy();dd['pred']=pred;dd['unseen_fit_key']=unseen
    for j,k in enumerate(['p_benign','p_malicious','p_suspicious']):dd[k]=pp[:,j]
    dd.to_parquet(folder/('inner_'+name+'_decisions.parquet'),index=False)
    print(json.dumps({'protocol':protocol,'family':family,'inner_epoch':epoch,**inner_reports[name]}),flush=True)
   learning.append({'epoch':epoch,'fit_loss':float(m.loss_)})
  chosen=int(min(rankings,key=rankings.get));save(folder/'selection.json',{'selected_epochs':chosen,'ranking':rankings,'selected_before_outer_evaluation':True})
  enc=fit_encoder(x.loc[outer]);mat=encode(enc,x.loc[outer]).astype(np.float32);cold=np.flatnonzero(np.asarray(mat.sum(0)).ravel()==0);models=[]
  for seed in c['neural_final_seeds']:
   m=MLPClassifier(**c['neural'],random_state=seed)
   for epoch in range(chosen):m.partial_fit(mat,y[outer],classes=np.array([0,1,2]));m.coefs_[0][cold]=0
   models.append(m);print(json.dumps({'protocol':protocol,'family':family,'outer_seed_fitted':seed,'epochs':chosen,'loss':float(m.loss_)}),flush=True)
  joblib.dump({'family':'neural','input':enc,'models':models,'epochs':chosen,'protocol':protocol,'configuration_sha256':sha(out/'configuration.json')},folder/'final_NEURAL.joblib')
  save(folder/'learning_curve.json',learning)
 save(folder/'inner_report.json',inner_reports)
 save(folder/'fit_complete.json',{'seconds':time.monotonic()-started,'bindings':{p.name:sha(p) for p in folder.iterdir() if p.is_file()},'outer_evaluation_opened':False})
def evaluate(root,out,protocol,family):
 c,r,x=load(root,out);folder=out/(protocol+'_'+family);assert not (folder/'evaluation_complete.json').exists()
 for p,h in read(folder/'fit_complete.json')['bindings'].items():assert sha(folder/p)==h,p
 ev=~select(r,protocol);rr=r.loc[ev].reset_index(drop=True);reports={}
 for path in folder.glob('final_*.joblib'):
  name=path.stem[6:];bundle=joblib.load(path);allout=[]
  for scenario,cols in VIEWS.items():
   xx=x.loc[ev].copy();xx[cols]=MISSING;pred,score=predict(bundle,xx)
   d=rr.copy();d['scenario']=scenario;d['eligible_stress']=True if not cols else x.loc[ev,cols].ne(MISSING).any(axis=1).to_numpy();d['pred']=pred
   for j in range(3):d['score_'+str(j)]=score[:,j]
   if family=='neural':
    for k,m in enumerate(bundle['models']):d['seed_'+str(k)+'_pred']=m.predict(encode(bundle['input'],xx).astype(np.float32))
   allout.append(d)
  d=pd.concat(allout,ignore_index=True);d.to_parquet(folder/(name+'_evaluation.parquet'),index=False)
  a=d[(d.scenario=='full')&(d.route=='asa')];normal=d[(d.scenario=='full')&(d.route=='asa_acl')]
  reports[name]={'ASA':metrics(a.label_index.to_numpy(),a.pred.to_numpy()),'normal_ACL':metrics(normal.label_index.to_numpy(),normal.pred.to_numpy()) if len(normal) else {'rows':0},'score_type':'margins_not_probabilities' if family=='kernel' else 'uncalibrated_probabilities'}
  print(json.dumps({'protocol':protocol,'family':family,'model':name,**reports[name]}),flush=True)
 save(folder/'outer_report.json',reports);save(folder/'evaluation_complete.json',{'bindings':{p.name:sha(p) for p in folder.iterdir() if p.is_file()},'model_promoted':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit','evaluate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--protocol',choices=PROTOCOLS);p.add_argument('--family',choices=['kernel','neural']);a=p.parse_args()
 if a.mode=='prepare':prepare(a.root.resolve(),a.out.resolve())
 else:globals()[a.mode](a.root.resolve(),a.out.resolve(),a.protocol,a.family)
