"""Frozen evaluation gates plus post-selection identifiability/support diagnosis."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import run_v56 as t
import run_v54 as v
import run_v55 as n
import v56_pooling as m
ROOT=t.ROOT;RUN=t.RUN;OUT=ROOT/'evidence/2026-09-14/v56_review'

def metrics(d):
 out={}
 for view in v.VIEWS:
  a=d[(d.scenario==view)&(d.route=='asa')&d.eligible_stress];b=d[(d.scenario==view)&(d.route=='asa_acl')]
  out[view]={'asa':v.metrics(a.label_index.to_numpy(),a.pred.to_numpy()),'normal':v.metrics(b.label_index.to_numpy(),b.pred.to_numpy()),'body_groups':int(a.body_group.nunique())}
 return out

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);c,r,x=t.load();stats={};allpred=[];diagnosis=[];fitcount=0
 for protocol in c['protocols']:
  folder=RUN/protocol;receipt=v.read(folder/'complete.json');fitcount+=receipt['new_base_fits']+receipt['residual_fits']
  for p,h in receipt['bindings'].items():assert v.sha(folder/p)==h
  split=pd.read_parquet(RUN/(protocol+'_split.parquet'));tr=split.role.eq('fit').to_numpy();ev=split.role.eq('evaluation').to_numpy();sel=v.read(folder/'selection.json');stats[protocol]={'selection':sel,'models':{}}
  for kind in ['base']+c['kinds']:
   path=folder/(kind+'_evaluation.parquet')
   if not path.exists():continue
   d=pd.read_parquet(path);stats[protocol]['models'][kind]=metrics(d);d['protocol']=protocol;d['kind']=kind;allpred.append(d)
  yy=r.loc[tr,'label_index'].to_numpy();fullfit=v.frame_keys(x.loc[tr]);fit_labels={k:set(g.label) for k,g in pd.DataFrame({'key':fullfit,'label':yy}).groupby('key',sort=False)};design=m.fit_design(x.loc[tr],r.loc[tr,'body_group'])
  for view,cols in v.VIEWS.items():
   xx=x.loc[ev].copy();xx[cols]=v.MISSING;key=v.frame_keys(xx);eligible=np.ones(ev.sum(),dtype=bool) if not cols else x.loc[ev,cols].ne(v.MISSING).any(axis=1).to_numpy();asa=(r.loc[ev,'route'].to_numpy()=='asa')&eligible;labels=r.loc[ev,'label_index'].to_numpy();g=pd.DataFrame({'key':key[asa],'label':labels[asa],'body':r.loc[ev,'body_group'].to_numpy()[asa]});counts=pd.crosstab(g.key,g.label);floor=int((counts.sum(1)-counts.max(1)).sum());mixed=counts.gt(0).sum(1)>1;mm=g.key.isin(set(counts.index[mixed]));known={level:np.array([k in design['maps'][level] for k in keys])[asa] for level,keys in m.keys(xx).items()}
   strata=np.array(['unseen_observation' if k not in fit_labels else ('seen_conflict' if len(fit_labels[k])>1 else ('seen_same_label' if labels[i] in fit_labels[k] else 'seen_opposite_label')) for i,k in enumerate(key)])[asa]
   item={'protocol':protocol,'scenario':view,'ASA_rows':int(asa.sum()),'empirical_same_observation_min_errors':floor,'mixed_observations':int(mixed.sum()),'rows_in_mixed_observations':int(mm.sum()),'body_groups_in_mixed_observations':int(g.loc[mm,'body'].nunique()),'level_activation_rows':{k:int(a.sum()) for k,a in known.items()},'models':{},'note':'Same-evaluation empirical deterministic lower bound, not future Bayes risk or label-truth diagnosis. Support strata refer to original full fit inputs; missing views never trained as new labeled rows.'}
   for kind in stats[protocol]['models']:
    d=pd.read_parquet(folder/(kind+'_evaluation.parquet'));a=d[d.scenario==view].reset_index(drop=True).loc[asa];err=(a.pred!=a.label_index).to_numpy();item['models'][kind]={'errors':int(err.sum()),'excess_over_empirical_floor':int(err.sum()-floor),'strata':{str(k):{'rows':int((strata==k).sum()),'errors':int(err[strata==k].sum())} for k in np.unique(strata)}}
   diagnosis.append(item)
 gates={}
 for kind in c['kinds']:
  present=all(kind in a['models'] for a in stats.values());tests={'all_protocols_available':present}
  if present:
   body_improved=stats['body']['models'][kind]['full']['asa']['errors']<stats['body']['models']['base']['full']['asa']['errors'];perclass=True;bad=True;normal=True;totals={k:0 for k in ['base',kind]};worst={k:0 for k in ['base',kind]}
   for protocol,z in stats.items():
    a=z['models'][kind]['full']['asa'];b=z['models']['base']['full']['asa'];perclass &= all(i<=j for i,j in zip(a['class_errors_B_M_S'][1:],b['class_errors_B_M_S'][1:]))
    if protocol!='body':
     for k in totals:totals[k]+=z['models'][k]['full']['asa']['errors']
    for view in v.VIEWS:
     a=z['models'][kind][view];b=z['models']['base'][view];bad &= a['asa']['ASA_to_normal']<=b['asa']['ASA_to_normal'];normal &= a['normal']['errors']<=b['normal']['errors']
     if view in n.VIEWS:
      for k in worst:worst[k]=max(worst[k],max(1-a for a in z['models'][k][view]['asa']['recall_B_M_S'][1:]))
   tests.update({'each_protocol_full_M_S_nonincrease':bool(perclass),'BODY_full_improved':body_improved,'pooled_behavior_full_improved':totals[kind]<totals['base'],'all_six_views_ASA_to_normal_nonincrease':bool(bad),'all_six_views_normal_errors_nonincrease':bool(normal),'worst_core_class_rate_nonincrease':worst[kind]<=worst['base']})
  gates[kind]={'checks':tests,'passed':all(tests.values())}
 dd=pd.concat(allpred,ignore_index=True);dd.to_parquet(OUT/'decisions.parquet',index=False)
 # Each ASA observation appears once in the behavior OOF pool per model/view.
 asa=dd[(dd.protocol!='body')&(dd.route=='asa')&(dd.scenario=='full')]
 for kind in asa.kind.unique():assert not asa[asa.kind==kind].row_position.duplicated().any();assert len(asa[asa.kind==kind])==99382
 pooled={kind:metrics(dd[(dd.protocol!='body')&(dd.kind==kind)]) for kind in dd.kind.unique()}
 result={'actual_new_fits':fitcount,'reused_base_models':2,'protocols':stats,'pooled_behavior':pooled,'gates':gates,'eligible_next_stage':[k for k,a in gates.items() if a['passed']],'scope':'Current fit-side adaptive development. Three behavior ASA buckets nonoverlapping; BODY overlaps. Pooled normal rows are repeated copies of the same four examples, not 12 independent normals. No old pressure or formal submission executed.'}
 v.save(OUT/'summary.json',result);v.save(OUT/'identifiability.json',diagnosis);(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}});print(json.dumps({'fits':fitcount,'gates':gates,'pooled_full':{k:a['full']['asa'] for k,a in pooled.items()}},indent=2),flush=True)
if __name__=='__main__':main()
