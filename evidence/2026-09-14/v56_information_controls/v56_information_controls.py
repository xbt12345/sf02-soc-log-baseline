"""Compare information loss on exactly the same eligible records."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import run_v56 as t
import run_v54 as v
ROOT=t.ROOT;OUT=ROOT/'evidence/2026-09-14/v56_information_controls'

def floor(keys,y):
 counts=defaultdict(lambda:np.zeros(3,dtype=int))
 for k,c in zip(keys,y):counts[k][int(c)]+=1
 return int(sum(a.sum()-a.max() for a in counts.values()))

def main():
 assert not OUT.exists();OUT.mkdir(parents=True);c,r,x=t.load();result=[]
 for protocol in c['protocols']:
  roles=pd.read_parquet(t.RUN/(protocol+'_split.parquet')).role;ev=roles.eq('evaluation').to_numpy();xx=x.loc[ev];keys=v.frame_keys(xx);y=r.loc[ev,'label_index'].to_numpy();asa=r.loc[ev,'route'].eq('asa').to_numpy();d=pd.read_parquet(t.RUN/protocol/'base_evaluation.parquet');full=d[d.scenario=='full'].reset_index(drop=True)
  for view,cols in v.VIEWS.items():
   a=d[d.scenario==view].reset_index(drop=True);mask=asa&a.eligible_stress.to_numpy();z=xx.copy();z[cols]=v.MISSING;k=v.frame_keys(z);result.append({'protocol':protocol,'view':view,'same_eligible_ASA_rows':int(mask.sum()),'full_input_empirical_floor':floor(keys[mask],y[mask]),'view_input_empirical_floor':floor(k[mask],y[mask]),'full_input_base_errors':int((full.pred.to_numpy()[mask]!=y[mask]).sum()),'view_input_base_errors':int((a.pred.to_numpy()[mask]!=y[mask]).sum())})
 v.save(OUT/'controls.json',{'comparisons':result,'scope':'Post-hoc exact same-record comparison. Empirical same-input deterministic error floor is not a future Bayes-risk estimate, and excess error is not proof that the present fit labels suffice to learn the solution.'});(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}})
if __name__=='__main__':main()
