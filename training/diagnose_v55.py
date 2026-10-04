"""Expose the posterior tradeoff from risk reweighting and update-budget confounds."""
import json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/v55_risk_validation_20260914';OUT=ROOT/'evidence/2026-09-14/v55_diagnosis'
sys.path.insert(0,str(RUN));import run_v55 as n;import run_v54 as v

def optimum(ids,y,w):
 c=np.zeros((int(ids.max())+1,3));np.add.at(c,(ids,y),w);z=c/c.sum(1,keepdims=True)
 return z,-float((w*np.log(np.maximum(z[ids,y],1e-30))).sum()/w.sum())

def decision_frontier(name,review):
 # Optimistic post-hoc inner diagnostic only, never used to override the frozen gate.
 epoch=review['selected_checkpoints'][name]['epoch'];events=[];target=[];initial=[];false_normal=[]
 for i,split in enumerate(n.SPLITS):
  d=pd.read_parquet(RUN/(split+'_'+name)/'validation.parquet');d=d[(d.epoch==epoch)&(d.scenario=='full')&(d.route=='asa')]
  p=d[['p_0','p_1','p_2']].to_numpy();y=d.label_index.to_numpy();ratio=p[:,1]/np.maximum(p[:,1]+p[:,2],1e-30);keep=d.pred.to_numpy()!=0
  initial.extend([int(((y==1)&~keep).sum()),int((y==2).sum())]);false_normal.append(int((~keep).sum()))
  target.extend(review['selected_checkpoints']['E0']['metrics'][split]['full']['class_errors_B_M_S'][1:])
  events.extend([(float(s),i,int(t)) for s,t in zip(ratio[keep],y[keep])])
 events.sort();s=np.array([a[0] for a in events]);increments=np.zeros((len(events),4),dtype=int)
 for j,(_,i,t) in enumerate(events):increments[j,2*i+(t==2)]=1 if t==1 else -1
 cumulative=np.vstack([np.zeros(4,dtype=int),np.cumsum(increments,axis=0)]);ends=np.r_[0,np.flatnonzero(np.r_[s[1:]!=s[:-1],True])+1];errors=cumulative[ends]+initial
 feasible=np.all(errors<=np.asarray(target),axis=1)&(errors.sum(1)<sum(target))
 return {'candidate':name,'epoch':epoch,'base_full_errors_body_M_S_behavior_M_S':target,'both_protocols_one_threshold_can_dominate_M_S':bool(feasible.any()),
  'best_feasible_full_error_sum':int(errors[feasible].sum(1).min()) if feasible.any() else None,'unchanged_ASA_to_normal':false_normal,
  'scope':'Exhaustive thresholds on already inspected inner scores, B argmax held fixed. Optimistic ranking diagnostic, not calibrated probabilities, frozen candidate selection or a newly validated classifier.'}

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);cfg,r,x=n.load(ROOT,RUN);review=v.read(ROOT/'evidence/2026-09-14/v55_review/summary.json');result=[];budget=[]
 for split in n.SPLITS:
  tr=pd.read_parquet(RUN/(split+'_split.parquet')).role.eq('fit').to_numpy();y=r.label_index.to_numpy()
  for name,s in cfg['specs'].items():
   folder=RUN/(split+'_'+name);epoch=review['selected_checkpoints'][name]['epoch'];item=v.read(folder/'learning.json')[epoch-1]
   xx,yy,vi,gi,keys,counts,gv,prior=n.expanded(x.loc[tr],y[tr],s['partial_views']);q=np.array(item['used_q']);w=n.risk_weights(s['risk'],gi,counts,gv,prior,q);base=n.risk_weights('erm',gi,counts,gv,prior,q)
   ids,_=pd.factorize(v.frame_keys(xx),sort=False);p,floor=optimum(ids,yy,w);p0,basefloor=optimum(ids,yy,base);changed=p.argmax(1)!=p0.argmax(1)
   top=np.argsort(q)[-5:][::-1];rawb=r.loc[tr,'body_group'].to_numpy();bodies=np.tile(rawb,3 if s['partial_views'] else 1)
   details=[{'component':json.loads(keys[i]),'objective_mass':float(q[i]),'rows':int(counts[i]),'bodies':len(set(bodies[gi==i])),'last_fit_nll':float(item['group_mean_nll'][i])} for i in top]
   result.append({'split':split,'candidate':name,'epoch':epoch,'fit_original_rows':int(tr.sum()),'expanded_rows':len(xx),'weighted_fit_nll':item['weighted_fit_nll'],'weighted_empirical_minimum_nll':floor,'ERM_same_views_empirical_minimum_nll':basefloor,
    'full_view_rows_empirical_optimal_label_changed_vs_ERM':int(changed[ids[vi==0]].sum()),'maximum_row_weight':float(w.max()),'top_risk_components':details})
  for base,part in [('E1','E1V'),('B1','B1V'),('D1','D1V')]:
   a=joblib.load(RUN/(split+'_'+base)/'epoch_30.joblib');b=joblib.load(RUN/(split+'_'+part)/'epoch_10.joblib')
   budget.append({'split':split,'full_candidate':base,'full_epoch':30,'full_adam_steps':int(a['model']._optimizer.t),'partial_candidate':part,'partial_epoch':10,'partial_adam_steps':int(b['model']._optimizer.t),
    'full_metrics':review['all_checkpoint_scores'][base]['30']['metrics'][split],'partial_metrics':review['all_checkpoint_scores'][part]['10']['metrics'][split],
    'note':'Post-hoc budget control, not used for selection. DRO has different numbers of adversary updates; view addition is not a perfectly isolated optimizer-budget comparison.'})
 frontier=[decision_frontier(name,review) for name in cfg['specs'] if name!='E0']
 v.save(OUT/'summary.json',{'risk_tradeoffs':result,'near_matched_update_controls':budget,'posthoc_decision_frontier':frontier,'scope':'Training empirical objectives and previously inspected inner diagnostics; no change to labels, selection or deployment.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps({'risk_cases':len(result),'budget_controls':len(budget)}),flush=True)
if __name__=='__main__':main()
