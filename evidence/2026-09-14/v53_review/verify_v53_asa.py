"""Independently replay EBM bin tables and audit the ASA factorial experiment."""
import hashlib,json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v53_asa_factorial_r2_20260914'
OUT=ROOT/'evidence/2026-09-14/v53_review'
P=['p_benign','p_malicious','p_suspicious']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def stats(y,p):
 q=p.argmax(1);cm=np.zeros((3,3),dtype=np.int64);np.add.at(cm,(y,q),1)
 support=cm.sum(1);error=support-cm.diagonal()
 f=np.divide(2*cm.diagonal(),support+cm.sum(0),out=np.zeros(3),where=(support+cm.sum(0))>0)
 return {'rows':len(y),'errors':int(error.sum()),'class_errors_B_M_S':error.tolist(),'confusion_B_M_S':cm.tolist(),
   'recall_B_M_S':[float(cm[i,i]/support[i]) if support[i] else None for i in range(3)],
   'macro_f1_M_S':float(f[1:].mean()),'log_loss':float(-np.log(np.maximum(p[np.arange(len(y)),y],np.finfo(float).eps)).mean())}
def feature_frame(x,pairs):
 z=x.copy()
 for pair in pairs:
  columns=[x.columns[i] for i in pair]
  z['joint__'+'__'.join(columns)]=[np.nan if any(pd.isna(v) for v in row) else '|'.join(str(v) for v in row) for row in x[columns].itertuples(index=False,name=None)]
 return z
def manual(model,x):
 score=np.tile(np.asarray(model.intercept_,dtype=float),(len(x),1))
 assert all(len(t)==1 for t in model.term_features_)
 for term,cols in enumerate(model.term_features_):
  j=cols[0];lookup=model.bins_[j][0];table=model.term_scores_[term]
  b=np.array([0 if pd.isna(v) else lookup.get(str(v),len(table)-1) for v in x.iloc[:,j]],dtype=int)
  score+=table[b]
 return softmax(score,axis=1)
def paired(y,base,p,groups):
 old=base.argmax(1);new=p.argmax(1);fixed=(old!=y)&(new==y);regressed=(old==y)&(new!=y)
 c=pd.DataFrame({'g':groups,'fixed':fixed,'regressed':regressed}).groupby('g')[['fixed','regressed']].sum()
 c['net']=c.fixed-c.regressed
 return {'fixed':int(fixed.sum()),'regressed':int(regressed.sum()),'net':int(fixed.sum()-regressed.sum()),
 'changed':int((old!=new).sum()),'fix_body_groups':int(c.fixed.gt(0).sum()),'regress_body_groups':int(c.regressed.gt(0).sum()),
 'largest_net_bodies':c.sort_values('net',ascending=False).head(10).reset_index().to_dict('records')}

def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 cfg=read(RUN/'configuration.json');checks={}
 assert sha(RUN/'configuration.json')==read(RUN/'preregistered.json')['configuration_sha256']
 for path,h in cfg['local_bindings'].items():assert sha(RUN/path)==h,path
 for path,h in cfg['source_bindings'].items():assert sha(ROOT/path)==h,path
 audit=ROOT/'evidence/2026-09-14/v53_raw_audit'
 for name,h in read(audit/'receipt.json')['files'].items():assert sha(audit/name)==h,name
 for path,h in read(audit/'summary.json')['source_bindings'].items():assert sha(ROOT/path)==h,path
 checks['frozen_inputs_code_and_original_train_rehashed']=True
 r=pd.read_parquet(RUN/'rows.parquet');x=pd.read_parquet(RUN/'observations.parquet').astype(object);x=x.where(x.notna(),np.nan)
 projections=pd.read_parquet(ROOT/'artifacts/v48_information_repair_r2_20260914/projections.parquet')
 limits={'src_port_fixed':65535,'dst_port_fixed':65535,'icmp_type':255,'icmp_code':255}
 unique=r.projection_id.unique();actual={}
 for pid in unique:
  f=json.loads(projections.iloc[int(pid)].facts);a={}
  assert set(f)<=set(cfg['fields'])
  for key in cfg['fields']:
   value=f.get(key)
   if value is None or value=='':a[key]=np.nan
   elif key in limits and (type(value)!=int or value<0 or value>limits[key]):a[key]=np.nan
   else:a[key]=str(value)
  actual[pid]=a
 xx=pd.DataFrame([actual[i] for i in r.projection_id],columns=cfg['fields'],dtype=object)
 pd.testing.assert_frame_equal(x,xx)
 checks['all_observations_rebuilt_from_retained_facts']=len(x)
 diagnosis=pd.read_parquet(ROOT/'evidence/2026-09-14/v52_overall/asa_row_diagnosis.parquet').set_index('row_position')
 stored={};maxdiff=0.;rows_checked=0;manifests={};training=[]
 for protocol in cfg['protocols']:
  folder=RUN/protocol;receipt=read(folder/'complete.json')
  for name,h in receipt['bindings'].items():assert sha(folder/name)==h,(protocol,name)
  fit=(r.inner_role!=2).to_numpy() if protocol=='pressure' else (r.fold!=int(protocol[-1])).to_numpy();ev=~fit
  pd.testing.assert_frame_equal(pd.read_parquet(folder/'fit_manifest.parquet').reset_index(drop=True),r.loc[fit].reset_index(drop=True))
  assert set(r.loc[fit,'body_group']).isdisjoint(r.loc[ev,'body_group'])
  if protocol=='pressure':assert set(r.loc[fit,'union_group']).isdisjoint(r.loc[ev,'union_group'])
  manifests[protocol]={'fit':int(fit.sum()),'evaluation':int(ev.sum()),'body_isolated':True,'union_isolation_required':protocol=='pressure'}
  base=pd.read_parquet(ROOT/'artifacts/v51_fact_residual_20260914'/protocol/'evaluation.parquet')
  base=base[base.route.isin(['asa','asa_acl'])].sort_values('row_position').reset_index(drop=True)
  np.testing.assert_array_equal(base.row_position,r.loc[ev,'row_position'])
  stored[protocol]={'v51':base.assign(scenario='full',eligible_stress=True)}
  for name in cfg['candidates']:
   model=joblib.load(folder/(name+'.joblib'));d=pd.read_parquet(folder/(name+'_evaluation.parquet'))
   report=read(folder/(name+'_report.json'));assert sha(folder/(name+'.joblib'))==report['model_sha256']
   assert list(model.classes_)==[0,1,2]
   # Check the categories against fit-side views; evaluation values cannot enter the vocabulary.
   fitframes=[]
   for view,cols in cfg['views'].items():
    if name.startswith('N_') and view!='full':continue
    f=x.loc[fit].copy();f[cols]=np.nan;fitframes.append(feature_frame(f,cfg['interaction_terms'] if name.endswith('PAIR') else []))
   fx=pd.concat(fitframes,ignore_index=True)
   assert model.feature_names_in_==fx.columns.tolist()
   for j,col in enumerate(fx.columns):
    vocab=set(str(v) for v in fx[col].dropna().unique());mapping=model.bins_[j][0]
    assert vocab==set(mapping) and len(vocab)==len(set(mapping.values())),(name,col)
   del fx,fitframes
   for scenario,cols in cfg['views'].items():
    xe=x.loc[ev].copy();xe[cols]=np.nan;xe=feature_frame(xe,cfg['interaction_terms'] if name.endswith('PAIR') else [])
    dd=d[d.scenario==scenario];np.testing.assert_array_equal(dd.row_position,r.loc[ev,'row_position']);np.testing.assert_array_equal(dd.label_index,r.loc[ev,'label_index'])
    p=manual(model,xe);ref=dd[P].to_numpy();delta=float(np.max(abs(p-ref)));maxdiff=max(maxdiff,delta)
    np.testing.assert_allclose(p,ref,rtol=0,atol=1e-12);np.testing.assert_array_equal(p.argmax(1),ref.argmax(1));rows_checked+=len(p)
    eligible=np.ones(ev.sum(),dtype=bool) if not cols else x.loc[ev,cols].notna().any(axis=1).to_numpy()
    np.testing.assert_array_equal(eligible,dd.eligible_stress)
   stored[protocol][name]=d
   training.append({'protocol':protocol,'model':name,**report['fit']})
   del model
  print(json.dumps({'verified_protocol':protocol,'probability_rows':rows_checked,'max_difference':maxdiff}),flush=True)
 checks.update(manifests=manifests,probability_rows=rows_checked,probability_values=rows_checked*3,manual_table_probability_max_difference=maxdiff,
     saved_labels_and_decisions_equal=True,fit_only_vocabulary_verified=True,all_declared_features_retained=True)
 summaries={};changes=[]
 for scope,protocols in [('primary',['fold_0','fold_1','fold_2']),('pressure',['pressure'])]:
  entries={}
  for name in ['v51']+cfg['candidates']:
   ds=pd.concat([stored[p][name] for p in protocols],ignore_index=True)
   entries[name]={}
   for scenario,g in ds.groupby('scenario',sort=False):
    a=g[(g.route=='asa')&g.eligible_stress].sort_values('row_position');pa=a[P].to_numpy();y=a.label_index.to_numpy()
    entry=stats(y,pa)
    if scenario=='full':
     b=pd.concat([stored[p]['v51'] for p in protocols]);b=b[b.route=='asa'].sort_values('row_position');np.testing.assert_array_equal(b.row_position,a.row_position)
     entry['paired_v51']=paired(y,b[P].to_numpy(),pa,a.body_group.to_numpy())
     ac=g[g.route=='asa_acl'];entry['ACL_normal']=stats(ac.label_index.to_numpy(),ac[P].to_numpy()) if len(ac) else {'rows':0}
     if scope=='primary':
      support=diagnosis.loc[a.row_position,'support_category'].to_numpy();parts={}
      for k in sorted(set(support)):
       mask=support==k;parts[k]={**stats(y[mask],pa[mask]),'paired_v51':paired(y[mask],b[P].to_numpy()[mask],pa[mask],a.body_group.to_numpy()[mask])}
      entry['support_slices']=parts
     else:support=['not_primary_protocol']*len(a)
     z=a[['row_position','label_index','fold','body_group']].copy();z['scope']=scope;z['model']=name;z['pred']=pa.argmax(1);z['baseline_pred']=b[P].to_numpy().argmax(1);z['support_category']=support;changes.append(z)
    entries[name][scenario]=entry
  summaries[scope]=entries
 pd.concat(changes,ignore_index=True).to_parquet(OUT/'paired_decisions.parquet',index=False)
 gates={}
 for name in cfg['candidates']:
  a=summaries['primary'][name]['full'];b=summaries['primary']['v51']['full'];q=summaries['pressure'][name]['full'];z=summaries['pressure']['v51']['full']
  foldbetter=sum(read(RUN/p/(name+'_report.json'))['ASA']['errors']<read(RUN/p/'report.json')['baseline_ASA']['errors'] for p in ['fold_0','fold_1','fold_2'])
  tests={'primary_total_and_each_class':a['errors']<b['errors'] and all(c<=d for c,d in zip(a['class_errors_B_M_S'][1:],b['class_errors_B_M_S'][1:])),
   'at_least_two_folds_improve':foldbetter>=2,'no_added_ASA_to_normal':a['confusion_B_M_S'][1][0]+a['confusion_B_M_S'][2][0]==0 and q['confusion_B_M_S'][1][0]+q['confusion_B_M_S'][2][0]==0,
   'normal_ACL_no_worse':a['ACL_normal']['errors']<=b['ACL_normal']['errors'],
   'pressure_total_and_each_class':q['errors']<=z['errors'] and all(c<=d for c,d in zip(q['class_errors_B_M_S'][1:],z['class_errors_B_M_S'][1:]))}
  gates[name]={'checks':tests,'all_passed':all(tests.values()),'improved_primary_folds':foldbetter}
 save('summary.json',{'scope':'Descriptive official development comparison, no new blind test, no promotion.','metrics':summaries,'continuation_gates':gates,'fits':training})
 save('verification.json',{'all_checks_passed':True,'checks':checks,'scope':'Independent saved bin-table replay and source/manifest verification, not deployment or external generalization.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 save('receipt.json',{'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps({'complete':str(OUT),'gates':gates,'rows_checked':rows_checked}),flush=True)
if __name__=='__main__':main()
