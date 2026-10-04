"""Fixed complete-observation control: empirical joint distribution with backoff.

This isolates complete-combination capacity from generalization to unknown inputs.
It is not a new-label generator or an external-transfer claim.
"""
import hashlib,json,sys,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v53_asa_factorial_r2_20260914'
OUT=ROOT/'artifacts/v53_complete_joint_20260914'
sys.path.insert(0,str(RUN));sys.path.insert(0,str(RUN/'input_runtime'))
import run_v53_asa as v
import v51_residual

def key(f):return json.dumps([None if pd.isna(z) else str(z) for z in v.observed(f).values()],separators=(',',':'))
def combine(prior,counts):
 total=counts.sum(axis=1);p=prior.copy();seen=total>0
 p[seen]=(counts[seen]+prior[seen])/(total[seen,None]+1.)
 return p
def base_prob(bundle,facts):
 b=bundle['base'];x=sparse.hstack([b['text_encoder'].transform(['']*len(facts)),b['fact_encoder'].transform(facts)],format='csr')
 assert not v51_residual.applicable(facts).any()
 return b['model'].predict_proba(x)
def predict(bundle,facts,prior):
 count=bundle['counts'];n=np.array([count.get(key(f),[0,0,0]) for f in facts],dtype=np.int64)
 return combine(prior,n),n.sum(1)
def prepare():
 assert not OUT.exists();OUT.mkdir(parents=True)
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 config={'version':'v53-complete-joint-control-1','scope':'Previously inspected development only. Complete safe facts, original labels and multiplicities. No external identity, timestamps, product or raw placeholder keys.',
  'question':'Does a model using complete fact combinations recover seen-input errors that main and pairwise models miss? What is the price on contradictory seen inputs and unseen combinations?',
  'models':['JOINT_V51','JOINT_A_PAIR'],'protocols':v.PROTOCOLS,'views':v.VIEWS,'fields':v.FIELDS,'prior_mass':1,
  'training':'Count each original fit row once into its entire observed-fact key and original class. Mixed labels remain a distribution. No pure-label filter or label rewriting. Four fit-side count models, no new base or EBM optimization.',
  'prediction':'On a seen key, (fit class counts + backoff probability)/(fit row count + 1). On an unseen key, exact unchanged backoff probability. This is fixed pseudo-count smoothing, not a calibration or independent Bayesian-evidence claim.',
  'backoffs':{'JOINT_V51':'Frozen v51 probabilities, recomputed on actually visible facts for each stress view','JOINT_A_PAIR':'Already frozen augmented EBM with sparse combinations, recomputed on actually visible facts'},
  'decision':'Argmax, no threshold selection, no evaluation-target branch. Full joint lookup cannot establish novel-combination generalization. Apply the existing factorial continuation gates; no automatic promotion.',
  'bindings':{str(p.relative_to(ROOT)):v.sha(p) for p in [RUN/'configuration.json',RUN/'rows.parquet',RUN/'observations.parquet',ROOT/'artifacts/v48_information_repair_r2_20260914/projections.parquet']+list(RUN.glob('*/A_PAIR.joblib'))+list((ROOT/'artifacts/v51_fact_residual_20260914').glob('*/model.joblib'))},
  'script_sha256':v.sha(__file__)}
 v.save(OUT/'configuration.json',config);v.save(OUT/'preregistered.json',{'configuration_sha256':v.sha(OUT/'configuration.json'),'new_count_models':0,'new_gradient_models':0})
 print(json.dumps({'prepared':str(OUT),'fixed_candidates':config['models']}),flush=True)
def fit():
 c=v.read(OUT/'configuration.json');assert v.sha(__file__)==c['script_sha256'];assert v.sha(OUT/'configuration.json')==v.read(OUT/'preregistered.json')['configuration_sha256']
 for path,h in c['bindings'].items():assert v.sha(ROOT/path)==h,path
 r=pd.read_parquet(RUN/'rows.parquet');pp=pd.read_parquet(ROOT/'artifacts/v48_information_repair_r2_20260914/projections.parquet')
 facts=[json.loads(pp.iloc[int(i)].facts) for i in r.projection_id];keys=np.array([key(f) for f in facts],dtype=object)
 for protocol in c['protocols']:
  folder=OUT/protocol;assert not folder.exists();folder.mkdir();start=time.monotonic()
  tr=v.select(r,protocol);ev=~tr;rr=r.loc[ev].reset_index(drop=True);evalfacts=[facts[i] for i in np.flatnonzero(ev)]
  table=pd.crosstab(keys[tr],r.loc[tr,'label_index'].to_numpy()).reindex(columns=[0,1,2],fill_value=0)
  count={k:[int(x) for x in row] for k,row in zip(table.index,table.to_numpy())}
  bundle={'version':c['version'],'counts':count,'fields':v.FIELDS,'protocol':protocol,'configuration_sha256':v.sha(OUT/'configuration.json')}
  joblib.dump(bundle,folder/'joint_counts.joblib');r.loc[tr].to_parquet(folder/'fit_manifest.parquet',index=False)
  b=joblib.load(ROOT/'artifacts/v51_fact_residual_20260914'/protocol/'model.joblib');ebm=joblib.load(RUN/protocol/'A_PAIR.joblib')
  allout={n:[] for n in c['models']};backouts=[];reports={}
  for scenario,cols in v.VIEWS.items():
   fs=[{k:z for k,z in f.items() if k not in cols} for f in evalfacts]
   bp=base_prob(b,fs);x=pd.DataFrame([v.observed(f) for f in fs],columns=v.FIELDS,dtype=object);ap=ebm.predict_proba(v.expanded(x,True))
   # The full-view backoffs must reproduce the already frozen official development outputs.
   if scenario=='full':
    old=pd.read_parquet(ROOT/'artifacts/v51_fact_residual_20260914'/protocol/'evaluation.parquet');old=old[old.route.isin(['asa','asa_acl'])].sort_values('row_position')
    np.testing.assert_array_equal(old.row_position,rr.row_position);np.testing.assert_allclose(bp,old[['p_benign','p_malicious','p_suspicious']],rtol=0,atol=1e-12)
    old=pd.read_parquet(RUN/protocol/'A_PAIR_evaluation.parquet');old=old[old.scenario=='full'];np.testing.assert_array_equal(ap,old[['p_benign','p_malicious','p_suspicious']].to_numpy())
   eligible=np.ones(len(rr),dtype=bool) if not cols else pd.DataFrame([v.observed(f) for f in evalfacts])[cols].notna().any(axis=1).to_numpy()
   bb=rr.copy();bb['scenario']=scenario;bb['eligible_stress']=eligible
   for j,k in enumerate(['p_benign','p_malicious','p_suspicious']):bb[k]=bp[:,j]
   backouts.append(bb)
   for name,prior in [('JOINT_V51',bp),('JOINT_A_PAIR',ap)]:
    q,n=predict(bundle,fs,prior);np.testing.assert_array_equal(q[n==0],prior[n==0])
    d=rr.copy();d['scenario']=scenario;d['eligible_stress']=eligible;d['fit_key_rows']=n
    for j,k in enumerate(['p_benign','p_malicious','p_suspicious']):d[k]=q[:,j]
    allout[name].append(d)
  pd.concat(backouts,ignore_index=True).to_parquet(folder/'V51_backoff_evaluation.parquet',index=False)
  for name,parts in allout.items():
   d=pd.concat(parts,ignore_index=True);d.to_parquet(folder/(name+'_evaluation.parquet'),index=False)
   a=d[(d.route=='asa')&(d.scenario=='full')];q=a[['p_benign','p_malicious','p_suspicious']].to_numpy()
   reports[name]={'ASA':v.metrics(a.label_index.to_numpy(),q),'seen_inputs':int(a.fit_key_rows.gt(0).sum()),'unseen_inputs':int(a.fit_key_rows.eq(0).sum())}
  v.save(folder/'report.json',{'count_keys':len(count),'fit_rows':int(tr.sum()),'evaluation_rows':int(ev.sum()),'seconds':time.monotonic()-start,'models':reports})
  v.save(folder/'complete.json',{'files':{p.name:v.sha(p) for p in folder.iterdir() if p.is_file()}})
  print(json.dumps({'protocol':protocol,'results':reports}),flush=True)
 v.save(OUT/'complete.json',{'new_count_models':4,'new_base_or_ebm_fits':0,'promoted':False,'configuration_sha256':v.sha(OUT/'configuration.json')})
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='fit':fit()
 else:raise ValueError('Use prepare or fit')
