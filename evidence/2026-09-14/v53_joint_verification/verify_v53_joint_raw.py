"""Independent joint-count reconstruction, stress replay and original-message checks."""
from collections import Counter
import hashlib,json,re,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.special import softmax
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v53_asa_factorial_r2_20260914'
JOINT=ROOT/'artifacts/v53_complete_joint_20260914'
OUT=ROOT/'evidence/2026-09-14/v53_joint_verification'
sys.path.insert(0,str(RUN/'input_runtime'))
import v48_input,v331_prepare
import verify_v53_asa as check
from verify_v53_asa import sha,read,stats,paired,manual,feature_frame
P=check.P

def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def normalize_tuple(row):return tuple(None if pd.isna(v) else str(v) for v in row)
def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 cfg=read(RUN/'configuration.json');jc=read(JOINT/'configuration.json')
 assert sha(JOINT/'configuration.json')==read(JOINT/'preregistered.json')['configuration_sha256']
 assert sha(JOINT/'probe_v53_joint.py')==jc['script_sha256']
 for path,h in jc['bindings'].items():assert sha(ROOT/path)==h,path
 r=pd.read_parquet(RUN/'rows.parquet');x=pd.read_parquet(RUN/'observations.parquet').astype(object);x=x.where(x.notna(),np.nan)
 tuples=[normalize_tuple(row) for row in x.itertuples(index=False,name=None)]
 prep=pd.read_parquet(ROOT/'artifacts/v48_information_repair_r2_20260914/projections.parquet');facts=[json.loads(prep.iloc[int(i)].facts) for i in r.projection_id]
 stored={};verified=0;maxdiff=0.;counts_checks=[]
 for protocol in cfg['protocols']:
  folder=JOINT/protocol
  for name,h in read(folder/'complete.json')['files'].items():assert sha(folder/name)==h,name
  fit=(r.inner_role!=2).to_numpy() if protocol=='pressure' else (r.fold!=int(protocol[-1])).to_numpy();ev=~fit
  pd.testing.assert_frame_equal(pd.read_parquet(folder/'fit_manifest.parquet').reset_index(drop=True),r.loc[fit].reset_index(drop=True))
  assert set(r.loc[fit,'body_group']).isdisjoint(r.loc[ev,'body_group'])
  if protocol=='pressure':assert set(r.loc[fit,'union_group']).isdisjoint(r.loc[ev,'union_group'])
  n={}
  for i in np.flatnonzero(fit):
   k=tuples[i]
   if k not in n:n[k]=[0,0,0]
   n[k][int(r.iloc[i].label_index)]+=1
  model=joblib.load(folder/'joint_counts.joblib');saved={tuple(json.loads(k)):value for k,value in model['counts'].items()};assert n==saved
  assert sum(sum(z) for z in n.values())==fit.sum()
  counts_checks.append({'protocol':protocol,'keys':len(n),'original_fit_rows_counted':int(fit.sum()),'class_counts':[int(sum(c[j] for c in n.values())) for j in range(3)]})
  base=joblib.load(ROOT/'artifacts/v51_fact_residual_20260914'/protocol/'model.joblib')['base'];ebm=joblib.load(RUN/protocol/'A_PAIR.joblib')
  ds={name:pd.read_parquet(folder/(name+'_evaluation.parquet')) for name in jc['models']+['V51_backoff']};stored[protocol]=ds
  for scenario,cols in cfg['views'].items():
   xx=x.loc[ev].copy();xx[cols]=np.nan
   keys=[normalize_tuple(row) for row in xx.itertuples(index=False,name=None)];ns=np.array([n.get(k,[0,0,0]) for k in keys]);total=ns.sum(1)
   ff=[{k:v for k,v in facts[i].items() if k not in cols} for i in np.flatnonzero(ev)]
   mx=sparse.hstack([base['text_encoder'].transform(['']*len(ff)),base['fact_encoder'].transform(ff)],format='csr')
   bp=softmax(mx@base['model'].coef_.T+base['model'].intercept_,axis=1);ap=manual(ebm,feature_frame(xx,cfg['interaction_terms']))
   for name,prior in [('JOINT_V51',bp),('JOINT_A_PAIR',ap),('V51_backoff',bp)]:
    q=prior.copy()
    if name!='V51_backoff':
     seen=total>0;q[seen]=(ns[seen]+prior[seen])/(total[seen,None]+1.)
    d=ds[name];d=d[d.scenario==scenario]
    np.testing.assert_array_equal(d.row_position,r.loc[ev,'row_position']);np.testing.assert_array_equal(d.label_index,r.loc[ev,'label_index'])
    if name!='V51_backoff':np.testing.assert_array_equal(d.fit_key_rows,total)
    ref=d[P].to_numpy();delta=float(abs(q-ref).max());maxdiff=max(maxdiff,delta)
    np.testing.assert_allclose(q,ref,atol=1e-12,rtol=0);np.testing.assert_array_equal(q.argmax(1),ref.argmax(1));verified+=len(q)
  del base,ebm
  print(json.dumps({'joint_verified':protocol,'probability_rows':verified}),flush=True)
 diagnosis=pd.read_parquet(ROOT/'evidence/2026-09-14/v52_overall/asa_row_diagnosis.parquet').set_index('row_position')
 summary={};decisions=[]
 for scope,protocols in [('primary',['fold_0','fold_1','fold_2']),('pressure',['pressure'])]:
  summary[scope]={};bd=pd.concat([stored[p]['V51_backoff'] for p in protocols],ignore_index=True)
  for name in ['V51_backoff']+jc['models']:
   d=pd.concat([stored[p][name] for p in protocols],ignore_index=True);entries={}
   for scenario,g in d.groupby('scenario',sort=False):
    a=g[(g.route=='asa')&g.eligible_stress].sort_values('row_position');q=a[P].to_numpy();y=a.label_index.to_numpy()
    b=bd[(bd.route=='asa')&(bd.scenario==scenario)&bd.eligible_stress].sort_values('row_position');np.testing.assert_array_equal(b.row_position,a.row_position)
    entries[scenario]={**stats(y,q),'paired_v51_same_view':paired(y,b[P].to_numpy(),q,a.body_group.to_numpy())}
    if scenario=='full':
     if scope=='primary':
      support=diagnosis.loc[a.row_position,'support_category'].to_numpy();pieces={}
      for k in sorted(set(support)):
       mask=support==k;pieces[k]={**stats(y[mask],q[mask]),'paired_v51':paired(y[mask],b[P].to_numpy()[mask],q[mask],a.body_group.to_numpy()[mask])}
      entries[scenario]['support_slices']=pieces
     else:support=['not_primary_protocol']*len(a)
     if name!='V51_backoff':entries[scenario]['seen_fit_input_rows']=int(a.fit_key_rows.gt(0).sum())
     acl=g[g.route=='asa_acl'];entries[scenario]['normal_ACL']=stats(acl.label_index.to_numpy(),acl[P].to_numpy()) if len(acl) else {'rows':0}
     z=a[['row_position','label_index','fold','body_group']].copy();z['scope']=scope;z['model']=name;z['pred']=q.argmax(1);z['baseline_pred']=b[P].to_numpy().argmax(1);z['support_category']=support;decisions.append(z)
   summary[scope][name]=entries
 pd.concat(decisions,ignore_index=True).to_parquet(OUT/'paired_decisions.parquet',index=False)
 # Exact-input frequency alone is retained as a negative control, with unknowns included.
 freq=pd.read_parquet(ROOT/'evidence/2026-09-14/v53_raw_audit/fit_frequency_control.parquet');frequency={}
 for scope,protos in [('primary',['fold_0','fold_1','fold_2']),('pressure',['pressure'])]:
  a=freq[freq.protocol.isin(protos)];q=np.c_[np.zeros(len(a)),a.qM,1-a.qM]
  frequency[scope]={**stats(a.label_index.to_numpy(),q),'seen_rows':int(a.seen_fit_input.sum())}
 # Read original bytes for two labels per mixed group, additional distinct error bodies, and every ACL normal control.
 raw_audit=pd.read_parquet(ROOT/'evidence/2026-09-14/v53_raw_audit/raw_fields.parquet')
 mixed=raw_audit.groupby('semantic_key').label_index.nunique();mixed=set(mixed[mixed>1].index)
 selected=set(raw_audit[raw_audit.semantic_key.isin(mixed)].groupby(['semantic_key','label_index']).head(1).row_position)
 selected.update(diagnosis[diagnosis.error].drop_duplicates('body_group').head(300).index)
 selected.update(r.loc[r.route=='asa_acl','row_position'])
 rows={};offset=0
 for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=16384,columns=['message_sanitized']):
  for i in sorted(p for p in selected if offset<=p<offset+batch.num_rows):rows[i]=batch.column(0)[int(i)-offset].as_py()
  offset+=batch.num_rows
 assert set(rows)==selected
 bypos=r.set_index('row_position');raw_cases=[];raw_variants=[];parsed_facts=[];positions=[]
 changed_headers=changed_endpoint_literals=0
 for pos,raw in sorted(rows.items()):
  parsed=v48_input.prepare_record({'message_sanitized':raw});expected=json.loads(prep.iloc[int(bypos.loc[pos,'projection_id'])].facts)
  assert parsed['facts']==expected and parsed['text']==''
  original={'message_sanitized':raw};variant={'message_sanitized':raw,'timestamp':'2099-01-01T00:00:00Z','product_name':None,'vendor_name':'UNSEEN','label_binary':'ignored','src_ip':'192.0.2.1','dst_ip':'198.51.100.2','src_port':7,'username':'other','event_id':'other'}
  if parsed['route']=='asa':
   m=v331_prepare.ASA_BODY.search(raw);assert m is not None
   body=raw[m.start():];m=v331_prepare.ASA_BODY.fullmatch(body);replacements=[]
   for side in ['src','dst']:
    value=m[side];zone,rest=value.split(':',1)
    if '/' in rest:address,port=rest.rsplit('/',1);tail='/'+port
    else:tail=''
    if re.fullmatch(r'dmz[-_]\d+',zone,re.I):zone='dmz-99999'
    replacements.append((*m.span(side),zone+':192.0.2.77'+tail))
   replacements.append((*m.span('acl'),'OTHER_ACL'))
   replacements.append((*m.span('hex'),'0xabcdef, 0x123456'))
   for a,b,s in sorted(replacements,reverse=True):body=body[:a]+s+body[b:]
   variant['message_sanitized']='<166>Sep 14 12:34:56 collector %ASA-4-106023: '+body
   changed_headers+=1;changed_endpoint_literals+=1
  changed=v48_input.prepare_record(variant)
  assert changed['route']==parsed['route'] and changed['facts']==expected and changed['text']==''
  raw_cases.append(original);raw_variants.append(variant);parsed_facts.append(expected);positions.append(pos)
 # Same parsed facts establish equal model inputs. Also exercise each saved EBM on the raw-derived inputs and compare its saved full predictions where evaluation applies.
 observed=pd.DataFrame([{k:(np.nan if v is None else v) for k,v in zip(cfg['fields'],normalize_tuple(x.iloc[int(np.flatnonzero(r.row_position.to_numpy()==pos)[0])]))} for pos in positions],columns=cfg['fields'],dtype=object)
 raw_replays=0
 for protocol in cfg['protocols']:
  keep=np.array([bypos.loc[pos,'inner_role']==2 if protocol=='pressure' else bypos.loc[pos,'fold']==int(protocol[-1]) for pos in positions]);sel=np.array(positions)[keep]
  for name in cfg['candidates']:
   model=joblib.load(RUN/protocol/(name+'.joblib'));q=manual(model,feature_frame(observed.loc[keep],cfg['interaction_terms'] if name.endswith('PAIR') else []))
   d=pd.read_parquet(RUN/protocol/(name+'_evaluation.parquet'));d=d[d.scenario=='full'].set_index('row_position').loc[sel]
   np.testing.assert_allclose(q,d[P],rtol=0,atol=1e-12);raw_replays+=len(q)
 raw_report={'original_records':len(rows),'mixed_groups_covered':len(mixed),'metadata_variants':len(rows),'embedded_header_variants':changed_headers,'endpoint_identity_acl_hash_variants':changed_endpoint_literals,'saved_EBM_raw_prediction_rows':raw_replays,
  'positions_sha256':hashlib.sha256(json.dumps(positions).encode()).hexdigest(),'all_route_fact_and_output_checks_passed':True,
  'scope':'Specified literal identity/header/outer-field changes preserve parsed facts. Actual roles, real port values and ICMP semantics were not changed. This is invariance verification, not proof these labels are true.'}
 save('summary.json',{'metrics':summary,'frequency_only_control':frequency,'scope':'Complete joint distribution is a capacity control, not unseen-input generalization. Original official labels and fit row multiplicities unchanged.'})
 save('verification.json',{'all_checks_passed':True,'independent_count_models':counts_checks,'probability_rows_recomputed':verified,'probability_values':verified*3,'max_probability_difference':maxdiff,'raw_checks':raw_report,'promoted':False})
 save('raw_case_positions.json',positions)
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 save('receipt.json',{'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps({'verified_rows':verified,'raw':raw_report}),flush=True)
if __name__=='__main__':main()
