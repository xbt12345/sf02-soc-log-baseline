"""Fixed 2x2 ASA experiment: additive/interactions x natural/coarsened training."""
import argparse,hashlib,importlib.metadata,json,shutil,time,warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import confusion_matrix,f1_score,log_loss

FIELDS=['action','outcome','transport_protocol','src_role','dst_role','src_port_fixed','dst_port_fixed','src_port_range','dst_port_range','icmp_type','icmp_code','icmp_message','icmp_unreachable']
LIMITS={'src_port_fixed':65536,'dst_port_fixed':65536,'icmp_type':256,'icmp_code':256}
VIEWS={'full':[],'no_src':['src_port_fixed','src_port_range'],'no_dst':['dst_port_fixed','dst_port_range'],
 'no_ports':['src_port_fixed','src_port_range','dst_port_fixed','dst_port_range'],'no_src_role':['src_role'],'no_dst_role':['dst_role']}
PAIRS=[(2,3),(2,4),(3,4),(2,5),(2,6),(3,5),(4,6),(5,6),(9,10),(2,3,4)]
NAMES=['N_MAIN','N_PAIR','A_MAIN','A_PAIR']
PROTOCOLS=['pressure','fold_0','fold_1','fold_2']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def observed(f):
 z={}
 for k in FIELDS:
  v=f.get(k)
  if v is None or v=='':z[k]=np.nan
  elif k in LIMITS and (not isinstance(v,int) or isinstance(v,bool) or not 0<=v<LIMITS[k]):z[k]=np.nan
  else:z[k]=str(v)
 return z
def data(root):
 prep=root/'artifacts/v48_information_repair_r2_20260914'
 r=pd.read_parquet(prep/'rows.parquet');r=r[r.route.isin(['asa','asa_acl'])].sort_values('row_position').reset_index(drop=True)
 p=pd.read_parquet(prep/'projections.parquet');active,ids=np.unique(r.projection_id.to_numpy(),return_inverse=True)
 fact=[json.loads(p.iloc[int(i)].facts) for i in active]
 assert all(set(f)<=set(FIELDS) for f in fact)
 x=pd.DataFrame([observed(f) for f in fact],columns=FIELDS,dtype=object).iloc[ids].reset_index(drop=True)
 assert len(r)==106969 and set(r.label_index)=={0,1,2}
 assert r.groupby('body_group').fold.nunique().max()==1
 return r,x
def select(r,protocol):return (r.inner_role!=2).to_numpy() if protocol=='pressure' else (r.fold!=int(protocol[-1])).to_numpy()
def view_data(x,y,augmented):
 if not augmented:return x.reset_index(drop=True),y.copy(),np.ones(len(y))
 parts=[];weights=[]
 for n,cols in VIEWS.items():
  z=x.copy();z[cols]=np.nan;parts.append(z);weights.append(np.full(len(y),.5 if n=='full' else .1))
 w=np.concatenate(weights);np.testing.assert_allclose(w.reshape(len(VIEWS),len(y)).sum(axis=0),1,atol=1e-15)
 return pd.concat(parts,ignore_index=True),np.tile(y,len(VIEWS)),w
def expanded(x,paired):
 """Sparse observed combinations, never dense Cartesian port-pair allocation."""
 if not paired:return x
 z=x.copy()
 for pair in PAIRS:
  cols=[FIELDS[i] for i in pair]
  # All input values are validated finite integers or fixed enum strings, no '|'.
  s=z[cols[0]].astype('string')
  for k in cols[1:]:s=s.str.cat(z[k].astype('string'),sep='|')
  z['joint__'+'__'.join(cols)]=s.astype(object).where(s.notna(),np.nan)
 return z
def metrics(y,p):
 pred=p.argmax(axis=1);c=confusion_matrix(y,pred,labels=[0,1,2])
 return {'rows':len(y),'errors':int((pred!=y).sum()),'confusion_B_M_S':c.tolist(),
   'recall_B_M_S':[float(c[i,i]/c[i].sum()) if c[i].sum() else None for i in range(3)],
   'macro_f1_M_S':float(f1_score(y,pred,labels=[1,2],average='macro',zero_division=0)),
   'log_loss':float(log_loss(y,p,labels=[0,1,2])) if len(y) else None}

def prepare(root,out):
 assert not out.exists(),'Do not overwrite a run.'
 audit=root/'evidence/2026-09-14/v53_raw_audit';a=read(audit/'summary.json')
 assert not a['unmatched_positions'] and not a['observable_field_disagreements']
 out.mkdir(parents=True);shutil.copyfile(__file__,out/Path(__file__).name)
 r,x=data(root);r.to_parquet(out/'rows.parquet',index=False);x.to_parquet(out/'observations.parquet',index=False)
 rt=out/'input_runtime';rt.mkdir()
 for p in (root/'artifacts/v51_fact_residual_20260914/runtime').glob('*.py'):shutil.copyfile(p,rt/p.name)
 bindings={}
 for p in [root/'artifacts/v48_information_repair_r2_20260914'/n for n in ['rows.parquet','projections.parquet']]+[audit/'summary.json',audit/'raw_fields.parquet']:
  bindings[p.relative_to(root).as_posix()]=sha(p)
 for protocol in PROTOCOLS:
  p=root/'artifacts/v51_fact_residual_20260914'/protocol/'evaluation.parquet';bindings[p.relative_to(root).as_posix()]=sha(p)
 carry=root/'artifacts/v53_asa_factorial_20260914'
 for protocol in ['pressure','fold_0']:
  origin=carry/protocol
  assert (origin/'N_MAIN_report.json').exists()
  assert sha(origin/'N_MAIN.joblib')==read(origin/'N_MAIN_report.json')['model_sha256']
  dest=out/'reused_controls'/protocol;dest.mkdir(parents=True)
  for n in ['N_MAIN.joblib','N_MAIN_report.json','N_MAIN_evaluation.parquet']:shutil.copyfile(origin/n,dest/n)
 cfg={'version':'v53-ASA-factorial-2-sparse-interactions','scope':'ASA plus existing 16 ASA ACL normal controls only. Other records remain untouched. Previously inspected official development data; not blind.',
  'rows':len(r),'labels_B_M_S':[int((r.label_index==k).sum()) for k in range(3)],'fields':FIELDS,'views':VIEWS,'augmented_view_weights':[.5,.1,.1,.1,.1,.1],
  'candidates':NAMES,'protocols':PROTOCOLS,'planned_models':16,'reused_N_MAIN_protocols':['pressure','fold_0'],'new_fits_planned':14,
  'ebm':{'feature_types':['nominal']*len(FIELDS),'max_bins':65536,'max_interaction_bins':128,'validation_size':0,'outer_bags':1,'inner_bags':0,
     'max_rounds':1200,'early_stopping_rounds':0,'learning_rate':.05,'min_samples_leaf':1,'min_cat_samples':1,'cat_smooth':10.,'reg_lambda':1.,
     'smoothing_rounds':0,'interaction_smoothing_rounds':0,'gain_scale':1.,'greedy_ratio':0.,'max_leaves':2,'n_jobs':1,'random_state':20260914},
  'interaction_terms':PAIRS,'interaction_implementation':'Observed tuple categorical terms instead of dense Cartesian interactions. All original main fields retained. Unobserved tuples have no learned tuple category. Views recompute tuples after coarsening; no hidden original field is retained.',
  'resource_revision':'Dense categorical port-pair expansion consumed >8 GB in one worker; interrupted the two own v53 processes. Two completed additive controls retained, no completed interaction model. Sequential bag work replaces unnecessary joblib workers. Input data, seed, rounds, labels and main model parameters unchanged apart from n_jobs.',
  'training':'Actual original rows, without deduplication or relabeling. Natural weight 1; augmented six views sum to weight 1 per original row. No data-dependent target removal, threshold tuning, early stopping or external labels.',
  'comparison':'Four predeclared fixed candidates on all four protocols. No hyperparameter selection on evaluation labels. Descriptive development comparison, no automatic promotion.',
  'continuation_gates':['ASA primary total errors lower than v51; both M and S errors no greater; at least two folds improve','No additional malicious or suspicious to normal errors on original inputs','ACL normal controls no worse than v51','Pressure ASA total and each class errors no greater than v51'],
  'stress':'All six views evaluated; role deletion is information loss, not an identity invariance requirement. Compare augmented to its matched natural candidate, report all class errors and losses.',
  'limitations':['Same-observation conflicts cannot be resolved by deterministic feature interactions.','Only 16 normal ACL controls, no normal pressure evaluation.','Coarsening objective is a controlled missing-information experiment, not assumed target missingness mechanism.'],
  'packages':{n:importlib.metadata.version(n) for n in ['interpret-core','numpy','pandas','scikit-learn','scipy','joblib','pyarrow']},'source_bindings':bindings,
  'local_bindings':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}}
 save(out/'configuration.json',cfg);save(out/'preregistered.json',{'configuration_sha256':sha(out/'configuration.json'),'new_models_fitted':0})
 print(json.dumps({'prepared':str(out),'rows':len(r),'planned_models':16}),flush=True)

def fit(root,out,protocol):
 c=read(out/'configuration.json');assert sha(out/'configuration.json')==read(out/'preregistered.json')['configuration_sha256']
 for p,h in c['source_bindings'].items():assert sha(root/p)==h,p
 for p,h in c['local_bindings'].items():assert sha(out/p)==h,p
 assert sha(__file__)==c['local_bindings'][Path(__file__).name]
 folder=out/protocol;assert not folder.exists();folder.mkdir()
 r=pd.read_parquet(out/'rows.parquet');x=pd.read_parquet(out/'observations.parquet').astype(object);x=x.where(pd.notna(x),np.nan)
 tr=select(r,protocol);ev=~tr;y=r.label_index.to_numpy()
 assert set(r.loc[tr,'body_group']).isdisjoint(r.loc[ev,'body_group'])
 if protocol=='pressure':assert set(r.loc[tr,'union_group']).isdisjoint(r.loc[ev,'union_group'])
 assert set(y[tr])=={0,1,2}
 r.loc[tr].to_parquet(folder/'fit_manifest.parquet',index=False)
 b=pd.read_parquet(root/'artifacts/v51_fact_residual_20260914'/protocol/'evaluation.parquet')
 b=b[b.route.isin(['asa','asa_acl'])].sort_values('row_position').reset_index(drop=True)
 np.testing.assert_array_equal(b.row_position,r.loc[ev,'row_position']);np.testing.assert_array_equal(b.label_index,y[ev])
 bp=b[['p_benign','p_malicious','p_suspicious']].to_numpy();asa=b.route.to_numpy()=='asa';reports={};logs=[]
 save(folder/'started.json',{'configuration_sha256':sha(out/'configuration.json'),'fit_rows':int(tr.sum()),'evaluation_rows':int(ev.sum())})
 for name in NAMES:
  xx,yy,w=view_data(x.loc[tr],y[tr],name.startswith('A_'));xx=expanded(xx,name.endswith('PAIR'));kw=dict(c['ebm']);kw['feature_names']=xx.columns.tolist();kw['feature_types']=['nominal']*len(xx.columns);kw['interactions']=0
  start=time.monotonic();reuse=out/'reused_controls'/protocol/'N_MAIN.joblib';reused=name=='N_MAIN' and reuse.exists()
  warn=[]
  if reused:
   model=joblib.load(reuse)
   for k,v in kw.items():
    if k!='n_jobs':assert model.get_params()[k]==v,(k,model.get_params()[k],v)
  else:
   model=ExplainableBoostingClassifier(**kw)
   with warnings.catch_warnings(record=True) as warn:
    warnings.simplefilter('always');model.fit(xx,yy,sample_weight=w)
  assert model.classes_.tolist()==[0,1,2]
  # Main bins must preserve each distinct fit-side observed category.
  per_feature=[]
  for j,k in enumerate(xx.columns):
   known=set(str(v) for v in xx[k].dropna().unique());mapping=model.bins_[j][0]
   assert set(mapping)==known,(name,k,'fit vocabulary differs')
   assert len(set(mapping.values()))==len(known),(name,k,'main bin merged distinct observed values')
   per_feature.append({'field':k,'observed_fit_values':len(known),'distinct_main_bins':len(set(mapping.values()))})
  path=folder/(name+'.joblib');joblib.dump(model,path);loaded=joblib.load(path)
  pp=model.predict_proba(expanded(x.loc[ev],name.endswith('PAIR')));np.testing.assert_array_equal(pp,loaded.predict_proba(expanded(x.loc[ev],name.endswith('PAIR'))))
  if reused:
   oldeval=pd.read_parquet(reuse.parent/'N_MAIN_evaluation.parquet');oldeval=oldeval[oldeval.scenario=='full']
   np.testing.assert_array_equal(pp,oldeval[['p_benign','p_malicious','p_suspicious']].to_numpy())
  elapsed=time.monotonic()-start;predictions=[]
  for scenario,cols in VIEWS.items():
   xe=x.loc[ev].copy();xe[cols]=np.nan;p=loaded.predict_proba(expanded(xe,name.endswith('PAIR')))
   d=r.loc[ev].copy().reset_index(drop=True);d['scenario']=scenario;d['eligible_stress']=True if not cols else x.loc[ev,cols].notna().any(axis=1).to_numpy()
   for j,k in enumerate(['p_benign','p_malicious','p_suspicious']):d[k]=p[:,j]
   predictions.append(d)
  dd=pd.concat(predictions,ignore_index=True);dd.to_parquet(folder/(name+'_evaluation.parquet'),index=False)
  reports[name]={'ASA':metrics(y[ev][asa],pp[asa]),'all_with_ACL':metrics(y[ev],pp),
    'ACL_normal':metrics(y[ev][~asa],pp[~asa]) if (~asa).any() else {'rows':0,'verified':False},
    'stress':{str(s):metrics(g.label_index.to_numpy(),g[['p_benign','p_malicious','p_suspicious']].to_numpy()) for s,g in dd[(dd.route=='asa')&dd.eligible_stress].groupby('scenario')},
    'fit':{'rows':len(yy),'original_rows':int(tr.sum()),'weight_sum':float(w.sum()),'seconds':elapsed,'new_fit':not reused,'reused_from':str(reuse.relative_to(root)) if reused else None,'best_iteration':np.asarray(model.best_iteration_).tolist(),'warnings':[str(z.message) for z in warn]},
    'main_category_coverage':per_feature,'model_sha256':sha(path)}
  save(folder/(name+'_report.json'),reports[name])
  print(json.dumps({'protocol':protocol,'model':name,'seconds':elapsed,'ASA_errors':reports[name]['ASA']['errors'],'confusion':reports[name]['ASA']['confusion_B_M_S'],'ACL':reports[name]['ACL_normal']}),flush=True)
  del model,loaded,xx,yy,w
 save(folder/'report.json',{'baseline_ASA':metrics(y[ev][asa],bp[asa]),'baseline_all_with_ACL':metrics(y[ev],bp),'candidates':reports})
 save(folder/'complete.json',{'bindings':{p.name:sha(p) for p in folder.iterdir() if p.is_file()},'model_promoted':False})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--protocol',choices=PROTOCOLS);a=p.parse_args()
 if a.mode=='prepare':prepare(a.root.resolve(),a.out.resolve())
 else:fit(a.root.resolve(),a.out.resolve(),a.protocol)
