"""Joint fit-side checkpoint selection and prerequisite continuation gates."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/v55_risk_validation_20260914';OUT=ROOT/'evidence/2026-09-14/v55_review'
sys.path.insert(0,str(RUN));import run_v55 as n;import run_v54 as v

def score(d):
 result={};rates=[];worstgroup=0.
 for split in n.SPLITS:
  result[split]={}
  for view in n.VIEWS:
   a=d[(d.split==split)&(d.scenario==view)];asa=a[(a.route=='asa')&a.eligible_stress];normal=a[a.route=='asa_acl']
   m=v.metrics(asa.label_index.to_numpy(),asa.pred.to_numpy());b=v.metrics(normal.label_index.to_numpy(),normal.pred.to_numpy())
   p=asa[['p_0','p_1','p_2']].to_numpy();y=asa.label_index.to_numpy();loss=-np.log(np.maximum(p[np.arange(len(p)),y],1e-30))
   zz=asa[['behavior_group','label_index','body_group']].copy();zz['nll']=loss
   g=zz.groupby(['behavior_group','label_index']).agg(mean_nll=('nll','mean'),rows=('nll','size'),bodies=('body_group','nunique')).reset_index()
   # Normal counterexamples enter the same behavior-class criterion.
   pp=normal[['p_0','p_1','p_2']].to_numpy();by=normal.label_index.to_numpy();bb=normal[['behavior_group','label_index','body_group']].copy();bb['nll']=-np.log(np.maximum(pp[np.arange(len(pp)),by],1e-30))
   bg=bb.groupby(['behavior_group','label_index']).agg(mean_nll=('nll','mean'),rows=('nll','size'),bodies=('body_group','nunique')).reset_index();g=pd.concat([g,bg],ignore_index=True)
   component=float(g.mean_nll.max());worstgroup=max(worstgroup,component)
   m.update({'normal_ACL':b,'mean_nll':float(loss.mean()),'worst_observed_component_nll':component,'group_class_risks':g.to_dict('records')})
   result[split][view]=m;rates.append([m['ASA_to_normal']/len(asa),max(1-t for t in m['recall_B_M_S'][1:]),b['errors']/len(normal),m['errors']/len(asa)])
 rates=np.asarray(rates);rank=[float(rates[:,0].max()),float(rates[:,1].max()),float(rates[:,2].max()),worstgroup,float(rates[:,3].mean())]
 return rank,result

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);cfg,r,x=n.load(ROOT,RUN);allpred=[];scores={};selected={}
 for candidate in cfg['specs']:
  ds=[]
  for split in n.SPLITS:
   folder=RUN/(split+'_'+candidate)
   for p,h in v.read(folder/'complete.json')['bindings'].items():assert v.sha(folder/p)==h,p
   z=pd.read_parquet(folder/'validation.parquet');z['split']=split;z['candidate']=candidate;ds.append(z)
  d=pd.concat(ds,ignore_index=True);scores[candidate]={}
  for epoch in cfg['epochs']:
   a=d[d.epoch==epoch];rank,m=score(a);scores[candidate][str(epoch)]={'rank':rank,'metrics':m}
  epoch=min(cfg['epochs'],key=lambda e:scores[candidate][str(e)]['rank']);selected[candidate]={'epoch':epoch,**scores[candidate][str(epoch)]};allpred.append(d[d.epoch==epoch])
 baseline=selected['E0'];gates={}
 for candidate,s in selected.items():
  if candidate=='E0':continue
  perclass=True;strict=False;nonormal=True;normals=True
  for split in n.SPLITS:
   a=s['metrics'][split]['full'];b=baseline['metrics'][split]['full']
   perclass &= all(i<=j for i,j in zip(a['class_errors_B_M_S'][1:],b['class_errors_B_M_S'][1:]));strict |= a['errors']<b['errors']
   for view in n.VIEWS:
    a=s['metrics'][split][view];b=baseline['metrics'][split][view]
    nonormal &= a['ASA_to_normal']<=b['ASA_to_normal'];normals &= a['normal_ACL']['errors']<=b['normal_ACL']['errors']
  tests={'both_protocols_full_M_S_no_worse':bool(perclass),'full_strict_improvement':bool(strict),'all_views_ASA_to_normal_no_worse':bool(nonormal),'all_views_normal_ACL_no_worse':bool(normals),'worst_component_nll_no_worse':s['rank'][3]<=baseline['rank'][3]}
  gates[candidate]={'checks':tests,'passed':all(tests.values())}
 passed=[k for k,z in gates.items() if z['passed']];chosen=min(passed,key=lambda k:selected[k]['rank']) if passed else None
 decisions=pd.concat(allpred,ignore_index=True);decisions.to_parquet(OUT/'selected_inner_decisions.parquet',index=False)
 v.save(OUT/'summary.json',{'selected_checkpoints':selected,'all_checkpoint_scores':scores,'continuation_gates':gates,'outer_candidate':chosen,'outer_fitting_executed':False,'scope':'Two fit-side development protocols. Same four ACL validation rows reused across both protocols; not eight independent normals. No new blind, outer, or transfer result.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps({'outer_candidate':chosen,'gates':gates,'selected':{k:{'epoch':z['epoch'],'rank':z['rank'],'full':{s:{t:z['metrics'][s]['full'][t] for t in ['errors','class_errors_B_M_S','ASA_to_normal']} for s in n.SPLITS}} for k,z in selected.items()}},indent=2),flush=True)
if __name__=='__main__':main()
