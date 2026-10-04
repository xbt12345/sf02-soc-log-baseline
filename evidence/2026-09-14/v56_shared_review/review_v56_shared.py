"""Apply unchanged main gates to the post-hoc shared-branch controls."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import run_v54 as v
import run_v55 as n
import v56_shared_control as s
from review_v56 import metrics
ROOT=s.ROOT;RUN=s.RUN;OUT=ROOT/'evidence/2026-09-14/v56_shared_review'

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);c,r,x=s.load();main=v.read(ROOT/'evidence/2026-09-14/v56_review/summary.json');stats={};total={'base':0,'shared':0};perclass=True;bad=True;normal=True;worst={'base':0.,'shared':0.};decisions=[]
 for protocol in c['protocols']:
  folder=RUN/protocol
  for p,h in v.read(folder/'complete.json')['bindings'].items():assert v.sha(folder/p)==h
  d=pd.read_parquet(folder/'evaluation.parquet');d['protocol']=protocol;decisions.append(d);new=metrics(d);base=main['protocols'][protocol]['models']['base'];selection=v.read(folder/'selection.json');stats[protocol]={'selection':selection['selected'],'metrics':new}
  perclass &= all(i<=j for i,j in zip(new['full']['asa']['class_errors_B_M_S'][1:],base['full']['asa']['class_errors_B_M_S'][1:]))
  if protocol!='body':total['base']+=base['full']['asa']['errors'];total['shared']+=new['full']['asa']['errors']
  for view in v.VIEWS:
   bad &= new[view]['asa']['ASA_to_normal']<=base[view]['asa']['ASA_to_normal'];normal &= new[view]['normal']['errors']<=base[view]['normal']['errors']
   if view in n.VIEWS:
    worst['base']=max(worst['base'],max(1-a for a in base[view]['asa']['recall_B_M_S'][1:]));worst['shared']=max(worst['shared'],max(1-a for a in new[view]['asa']['recall_B_M_S'][1:]))
 gates={'each_protocol_full_M_S_nonincrease':bool(perclass),'BODY_full_improved':stats['body']['metrics']['full']['asa']['errors']<main['protocols']['body']['models']['base']['full']['asa']['errors'],'pooled_behavior_full_improved':total['shared']<total['base'],'all_six_views_ASA_to_normal_nonincrease':bool(bad),'all_six_views_normal_errors_nonincrease':bool(normal),'worst_core_class_rate_nonincrease':worst['shared']<=worst['base']}
 dd=pd.concat(decisions,ignore_index=True);dd.to_parquet(OUT/'decisions.parquet',index=False);pool=dd[dd.protocol!='body'];assert not pool[(pool.scenario=='full')&(pool.route=='asa')].row_position.duplicated().any()
 result={'protocols':stats,'pooled_behavior':metrics(pool),'gates':gates,'passed':all(gates.values()),'new_residual_fits':12,'new_base_fits':0,'scope':'Supplement after main failure, same original evaluation gates; adaptive development, not blind testing. The same four normal evaluation rows are repeated. No threshold tuning or evaluation-based parameter replacement.'};v.save(OUT/'summary.json',result);(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}});print(json.dumps({'gates':gates,'passed':result['passed'],'full':{p:z['metrics']['full']['asa'] for p,z in stats.items()},'pool':result['pooled_behavior']['full']['asa']},indent=2),flush=True)
if __name__=='__main__':main()
