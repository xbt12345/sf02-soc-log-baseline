"""Paired capacity, conflict and coarsening analysis; never fit or select a threshold."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from verify_v53_asa import read,sha,stats,paired
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/2026-09-14/v53_findings'

def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 e=read(ROOT/'evidence/2026-09-14/v53_review/summary.json');j=read(ROOT/'evidence/2026-09-14/v53_joint_verification/summary.json')
 diagnosis=pd.read_parquet(ROOT/'evidence/2026-09-14/v52_overall/asa_row_diagnosis.parquet').set_index('row_position')
 changes=pd.concat([pd.read_parquet(ROOT/'evidence/2026-09-14/v53_review/paired_decisions.parquet'),pd.read_parquet(ROOT/'evidence/2026-09-14/v53_joint_verification/paired_decisions.parquet')],ignore_index=True)
 primary=changes[changes.scope=='primary'];summary={};by_support={}
 for name,g in primary.groupby('model'):
  z=diagnosis.loc[g.row_position].copy();z['current_prediction']=g.pred.to_numpy();z['current_error']=z.label_index!=z.current_prediction
  z['fit_same_label_rows']=np.where(z.label_index==1,z.fit_M,z.fit_S)
  z['fit_opposite_label_rows']=np.where(z.label_index==1,z.fit_S,z.fit_M)
  pure=z[z.support_category=='seen_pure_same_label'];err=pure[pure.current_error]
  by_support[name]={'pure_same_error_count':len(err),'pure_same_error_groups':int(err.body_group.nunique()),
   'remaining_fit_support_counts':{str(k):int(v) for k,v in err.groupby('fit_same_label_rows').size().items()},
   'fit_supported_error_labels_M_S':[int((err.label_index==k).sum()) for k in [1,2]]}
  examples=z[z.current_error].groupby(['support_category','input_key'],sort=False).agg(rows=('label_index','size'),M=('label_index',lambda x:int((x==1).sum())),S=('label_index',lambda x:int((x==2).sum())),fit_M=('fit_M','max'),fit_S=('fit_S','max')).reset_index()
  examples.sort_values('rows',ascending=False).head(40).to_json(OUT/(name+'_largest_error_inputs.json'),orient='records',force_ascii=False,indent=2)
 names=['v51','N_MAIN','N_PAIR','A_MAIN','A_PAIR','JOINT_V51','JOINT_A_PAIR']
 for name in names:
  source=e if name not in ['JOINT_V51','JOINT_A_PAIR'] else j
  a=source['metrics']['primary'][name]['full'];p=source['metrics']['pressure'][name]['full']
  summary[name]={'primary_errors':a['errors'],'primary_M_errors':a['class_errors_B_M_S'][1],'primary_S_errors':a['class_errors_B_M_S'][2],
   'primary_ASA_macro_f1_M_S':a['macro_f1_M_S'],'primary_log_loss':a['log_loss'],'pressure_errors':p['errors'],
   'pressure_M_errors':p['class_errors_B_M_S'][1],'pressure_S_errors':p['class_errors_B_M_S'][2],
   'pressure_S_correct':p['confusion_B_M_S'][2][2],'pressure_M_to_normal':p['confusion_B_M_S'][1][0],
   'pressure_log_loss':p['log_loss'],'original_labels_changed':False}
 # Whole-development metrics are a paired-table diagnostic, not a promoted runtime.
 base=pd.concat([pd.read_parquet(ROOT/'artifacts/v51_fact_residual_20260914'/f'fold_{f}'/'evaluation.parquet') for f in range(3)]).sort_values('row_position').reset_index(drop=True)
 byposition=pd.Series(np.arange(len(base)),index=base.row_position);pcols=['p_benign','p_malicious','p_suspicious'];overall={}
 for name in names:
  pred=base[pcols].to_numpy().argmax(1)
  if name!='v51':
   run='v53_asa_factorial_r2_20260914' if name not in ['JOINT_V51','JOINT_A_PAIR'] else 'v53_complete_joint_20260914'
   d=pd.concat([pd.read_parquet(ROOT/'artifacts'/run/f'fold_{f}'/(name+'_evaluation.parquet')) for f in range(3)])
   d=d[d.scenario=='full'];ii=byposition.loc[d.row_position].to_numpy();np.testing.assert_array_equal(base.iloc[ii].label_index,d.label_index)
   pred[ii]=d[pcols].to_numpy().argmax(1)
  y=base.label_index.to_numpy();cm=np.zeros((3,3),dtype=int);np.add.at(cm,(y,pred),1);den=cm.sum(0)+cm.sum(1)
  overall[name]={'rows':len(y),'errors':int((pred!=y).sum()),'macro_f1_B_M_S':float(np.mean(2*np.diag(cm)/den)),'confusion_B_M_S':cm.tolist(),'scope':'Only paired prediction table; ASA plus ACL predictions replaced diagnostically, not deployed.'}
 stress={}
 for pair in [('N_MAIN','A_MAIN'),('N_PAIR','A_PAIR')]:
  entries={}
  for scenario in ['full','no_src','no_dst','no_ports','no_src_role','no_dst_role']:
   a=e['metrics']['primary'][pair[0]][scenario];b=e['metrics']['primary'][pair[1]][scenario]
   entries[scenario]={'eligible_rows':a['rows'],'natural_errors':a['errors'],'augmented_errors':b['errors'],'natural_log_loss':a['log_loss'],'augmented_log_loss':b['log_loss'],
    'natural_M_S_errors':a['class_errors_B_M_S'][1:],'augmented_M_S_errors':b['class_errors_B_M_S'][1:]}
  stress['__'.join(pair)]=entries
 # Exact counts cannot generalize: explicitly retain the disagreement and no-support slices.
 joint_gates={}
 old=e['metrics']['primary']['v51']['full'];oldp=e['metrics']['pressure']['v51']['full']
 for name in ['JOINT_V51','JOINT_A_PAIR']:
  a=j['metrics']['primary'][name]['full'];p=j['metrics']['pressure'][name]['full']
  q={'primary_total_and_each_class':a['errors']<old['errors'] and all(x<=y for x,y in zip(a['class_errors_B_M_S'][1:],old['class_errors_B_M_S'][1:])),
     'pressure_total_and_each_class':p['errors']<=oldp['errors'] and all(x<=y for x,y in zip(p['class_errors_B_M_S'][1:],oldp['class_errors_B_M_S'][1:])),
     'normal_ACL_no_worse':a['normal_ACL']['errors']<=16,'no_added_ASA_to_normal':a['confusion_B_M_S'][1][0]+a['confusion_B_M_S'][2][0]+p['confusion_B_M_S'][1][0]+p['confusion_B_M_S'][2][0]==0}
  q['at_least_two_folds_improve']=sum(read(ROOT/'artifacts/v53_complete_joint_20260914'/f'fold_{f}'/'report.json')['models'][name]['ASA']['errors']<read(ROOT/'artifacts/v53_asa_factorial_r2_20260914'/f'fold_{f}'/'report.json')['baseline_ASA']['errors'] for f in range(3))>=2
  joint_gates[name]={'checks':q,'all_passed':all(q.values())}
 save('summary.json',{'model_comparison':summary,'fit_supported_capacity':by_support,'coarsening_effect':stress,'joint_continuation_gates':joint_gates,'overall_paired_diagnostic':overall,
  'conclusion':'No new candidate passed all fixed continuation gates. Pairwise capacity repairs many seen-label-consistent inputs; exact-count control repairs all those inputs but worsens contradictory-label cases. No claim ASA is solved.'})
 save('research_sources.json',[
  {'url':'https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html','accessed':'2026-09-14','use':'Nominal main bins, explicit interactions, additive learned-table replay, fixed rounds with early stopping off. Installed interpret-core 0.7.8. We used explicit observed tuple columns to avoid dense category products.'},
  {'url':'https://github.com/interpretml/interpret','accessed':'2026-09-14','use':'Official open-source project provenance; method API verified in linked official documentation and installed implementation.'},
  {'url':'https://www.cisco.com/c/en/us/td/docs/security/asa/syslog/asa-syslog/syslog-messages-101001-to-199021.html','accessed':'2026-09-14','use':'106023 describes ACL-denied traffic; this observation is not itself a malicious/suspicious truth rule. It does not provide the competition label policy.'}])
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 save('receipt.json',{'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps({'comparison':summary,'capacity':by_support,'joint_gates':joint_gates},indent=2),flush=True)
if __name__=='__main__':main()
