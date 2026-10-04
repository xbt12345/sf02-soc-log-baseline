"""Read-only model comparison, support slices, raw input invariance and review."""
import json,re,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/v54_method_change_20260914';OUT=ROOT/'evidence/2026-09-14/v54_review'
sys.path.insert(0,str(RUN));sys.path.insert(0,str(ROOT/'artifacts/v53_asa_factorial_r2_20260914/input_runtime'))
import run_v54 as v
import v54_inference as inference
import v48_input,v331_prepare
def paired(y,p,b):return {'fixed':int(((b!=y)&(p==y)).sum()),'regressed':int(((b==y)&(p!=y)).sum()),'changed':int((p!=b).sum())}
def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 r=pd.read_parquet(RUN/'rows.parquet');obs=pd.read_parquet(RUN/'observations.parquet');diag=pd.read_parquet(ROOT/'evidence/2026-09-14/v52_overall/asa_row_diagnosis.parquet').set_index('row_position')
 info={};all_d=[];choices={};curves={}
 for protocol in v.PROTOCOLS:
  info[protocol]={};choices[protocol]={};curves[protocol]={}
  b=pd.read_parquet(ROOT/'artifacts/v51_fact_residual_20260914'/protocol/'evaluation.parquet');b=b[b.route.isin(['asa','asa_acl'])].sort_values('row_position').reset_index(drop=True);b['pred']=b[['p_benign','p_malicious','p_suspicious']].to_numpy().argmax(1);b['scenario']='full';b['eligible_stress']=True
  info[protocol]['V51']=b
  for family in ['kernel','neural']:
   folder=RUN/(protocol+'_'+family);receipt=v.read(folder/'evaluation_complete.json')
   for p,h in receipt['bindings'].items():assert v.sha(folder/p)==h,p
   sel=v.read(folder/'selection.json');choices[protocol][family]=sel
   if family=='kernel':
    for name,target in [('KERNEL_SELECTED',sel['selected']),('LINEAR_CONTROL','LINEAR')]:info[protocol][name]=pd.read_parquet(folder/(target+'_evaluation.parquet'))
   else:
    d=pd.read_parquet(folder/'NEURAL_evaluation.parquet');info[protocol]['NEURAL']=d
    for i in [0,1]:z=d.copy();z['pred']=z['seed_'+str(i)+'_pred'];info[protocol]['NEURAL_SEED_'+str(i)]=z
    curves[protocol]=v.read(folder/'inner_report.json')
  for name,d in info[protocol].items():
   z=d[d.scenario=='full'].copy();z['method']=name;z['protocol']=protocol;all_d.append(z[['row_position','label_index','body_group','fold','route','pred','method','protocol']])
 summaries={};gates={}
 for scope,protos in [('primary',['fold_0','fold_1','fold_2']),('pressure',['pressure'])]:
  summaries[scope]={}
  base=pd.concat([info[p]['V51'] for p in protos]);base=base[base.route=='asa'].sort_values('row_position');baseline=base.pred.to_numpy()
  for name in info['pressure']:
   d=pd.concat([info[p][name] for p in protos],ignore_index=True);byview={}
   for scenario,g in d.groupby('scenario',sort=False):
    a=g[(g.route=='asa')&g.eligible_stress].sort_values('row_position');m=v.metrics(a.label_index.to_numpy(),a.pred.to_numpy())
    if scenario=='full':
     np.testing.assert_array_equal(a.row_position,base.row_position);m['paired_v51']=paired(a.label_index.to_numpy(),a.pred.to_numpy(),baseline)
     normal=g[g.route=='asa_acl'];m['normal_ACL']=v.metrics(normal.label_index.to_numpy(),normal.pred.to_numpy()) if len(normal) else {'rows':0}
     if scope=='primary':
      support=diag.loc[a.row_position,'support_category'].to_numpy();ss={}
      for k in sorted(set(support)):
       sub=a[support==k];ss[k]={**v.metrics(sub.label_index.to_numpy(),sub.pred.to_numpy()),'paired_v51':paired(sub.label_index.to_numpy(),sub.pred.to_numpy(),baseline[support==k])}
      m['support_slices']=ss
     else:
      key=obs.set_index(r.row_position);type13=key.loc[a.row_position,'icmp_code'].eq('13').to_numpy();m['ICMP_code13']=v.metrics(a.label_index.to_numpy()[type13],a.pred.to_numpy()[type13])
    byview[scenario]=m
   summaries[scope][name]=byview
 for name in ['KERNEL_SELECTED','LINEAR_CONTROL','NEURAL']:
  a=summaries['primary'][name]['full'];b=summaries['primary']['V51']['full'];p=summaries['pressure'][name]['full'];q=summaries['pressure']['V51']['full']
  folds=0
  for f in ['fold_0','fold_1','fold_2']:
   d=info[f][name];d=d[(d.scenario=='full')&(d.route=='asa')];bb=info[f]['V51'];bb=bb[bb.route=='asa'];folds+=int((d.pred!=d.label_index).sum()<(bb.pred!=bb.label_index).sum())
  keys=['unseen_combination_known_values','unseen_fact_value'];new_errors=sum(a['support_slices'][k]['errors'] for k in keys);old_errors=sum(b['support_slices'][k]['errors'] for k in keys)
  tests={'primary_total_and_each_class':a['errors']<b['errors'] and all(x<=y for x,y in zip(a['class_errors_B_M_S'][1:],b['class_errors_B_M_S'][1:])),
   'pressure_total_and_each_class':p['errors']<=q['errors'] and all(x<=y for x,y in zip(p['class_errors_B_M_S'][1:],q['class_errors_B_M_S'][1:])),
   'no_new_ASA_to_normal':a['ASA_to_normal']==0 and p['ASA_to_normal']==0,'ACL_no_worse':a['normal_ACL']['errors']<=b['normal_ACL']['errors'],'two_folds_improve':folds>=2}
  gates[name]={'checks':tests,'all_passed':all(tests.values()),'improved_folds':folds,'primary_unseen_combination_errors':new_errors,'baseline_unseen_combination_errors':old_errors,
   'new_generalization_claim_supported':bool(new_errors<old_errors and tests['pressure_total_and_each_class'])}
 pd.concat(all_d,ignore_index=True).to_parquet(OUT/'paired_full_decisions.parquet',index=False)
 # Reuse the previously declared original-record sample, without selecting favorable examples.
 positions=v.read(ROOT/'evidence/2026-09-14/v53_joint_verification/raw_case_positions.json');wanted=set(positions);raw={};offset=0
 for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=16384,columns=['message_sanitized']):
  for pos in sorted(p for p in wanted if offset<=p<offset+batch.num_rows):raw[pos]=batch.column(0)[int(pos)-offset].as_py()
  offset+=batch.num_rows
 assert set(raw)==wanted;cases=[];variants=[];bypos=r.set_index('row_position')
 for pos in positions:
  message=raw[pos];rec={'message_sanitized':message};variant={'message_sanitized':message,'product_name':None,'timestamp':'2099-01-01T00:00:00Z','src_ip':'192.0.2.7','dst_ip':'198.51.100.7','label_binary':'not_a_feature','event_id':'other','vendor_name':'unseen'}
  parsed=v48_input.prepare_record(rec)
  if parsed['route']=='asa':
   m=v331_prepare.ASA_BODY.search(message);body=message[m.start():];m=v331_prepare.ASA_BODY.fullmatch(body);replacement=[]
   for side in ['src','dst']:
    zone,rest=m[side].split(':',1);tail='/'+rest.rsplit('/',1)[1] if '/' in rest else ''
    if re.fullmatch(r'dmz[-_]\d+',zone,re.I):zone='dmz-99999'
    replacement.append((*m.span(side),zone+':192.0.2.77'+tail))
   replacement.extend([(*m.span('acl'),'DIFFERENT_ACL'),(*m.span('hex'),'0xabcdef, 0x123456')])
   for a,b,s in sorted(replacement,reverse=True):body=body[:a]+s+body[b:]
   variant['message_sanitized']='<166>Sep 14 12:34:56 collector %ASA-4-106023: '+body
  assert v48_input.prepare_record(variant)['facts']==parsed['facts']
  expected=obs.loc[r.row_position.eq(pos)].iloc[0].to_dict();assert inference.observations(parsed['facts'])==expected
  cases.append(rec);variants.append(variant)
 rawchecks=[]
 for protocol in v.PROTOCOLS:
  for family in ['kernel','neural']:
   folder=RUN/(protocol+'_'+family)
   for path in folder.glob('final_*.joblib'):
    bundle=joblib.load(path);one=inference.classify_records(bundle,cases,v48_input);two=inference.classify_records(bundle,variants,v48_input)
    np.testing.assert_array_equal(one['labels'],two['labels']);np.testing.assert_array_equal(one['scores'],two['scores'])
    ev=np.array([bypos.loc[p,'inner_role']==2 if protocol=='pressure' else bypos.loc[p,'fold']==int(protocol[-1]) for p in positions]);d=pd.read_parquet(folder/(path.stem[6:]+'_evaluation.parquet'));d=d[d.scenario=='full'].set_index('row_position').loc[np.array(positions)[ev]]
    np.testing.assert_array_equal(one['labels'][ev],d.pred);np.testing.assert_allclose(one['scores'][ev],d[['score_0','score_1','score_2']],atol=3e-6 if family=='neural' else 1e-7,rtol=0)
    rawchecks.append({'protocol':protocol,'family':family,'model':path.stem,'raw_cases':len(cases),'outer_evaluation_cases':int(ev.sum()),'identical_metadata_variant_decisions':True})
 v.save(OUT/'summary.json',{'metrics':summaries,'gates':gates,'inner_choices':choices,'neural_learning_curve_checkpoints':curves,'scope':'Descriptive comparison on repeatedly inspected official development; no deployment or external transfer.'})
 v.save(OUT/'raw_inference_verification.json',{'all_checks_passed':True,'models':rawchecks,'case_positions_sha256':v.sha(ROOT/'evidence/2026-09-14/v53_joint_verification/raw_case_positions.json')})
 for p in [Path(__file__),ROOT/'training/v54_inference.py']: (OUT/p.name).write_bytes(p.read_bytes())
 v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps({'gates':gates,'raw_models':len(rawchecks),'metrics':{s:{n:{k:z['full'][k] for k in ['errors','class_errors_B_M_S','ASA_to_normal']} for n,z in a.items()} for s,a in summaries.items()}},indent=2),flush=True)
if __name__=='__main__':main()
