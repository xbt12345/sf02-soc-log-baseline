"""Post-hoc failure controls. No fitting, label changes, or deployment changes."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/v54_method_change_20260914'
sys.path.insert(0,str(RUN));import run_v54 as v
OUT=ROOT/'evidence/2026-09-14/v54_failure_controls'

def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 r=pd.read_parquet(RUN/'rows.parquet');x=pd.read_parquet(RUN/'observations.parquet');cases=[];summary={}
 for protocol in v.PROTOCOLS:
  folder=RUN/(protocol+'_neural')
  for p,h in v.read(folder/'evaluation_complete.json')['bindings'].items():assert v.sha(folder/p)==h
  fit=v.select(r,protocol);d=pd.read_parquet(folder/'NEURAL_evaluation.parquet');d=d[d.scenario=='full'].reset_index(drop=True)
  np.testing.assert_array_equal(d.row_position,r.loc[~fit,'row_position'])
  facts=x.loc[~fit].reset_index(drop=True);q=d[['score_0','score_1','score_2']].to_numpy();y=d.label_index.to_numpy();pred=d.pred.to_numpy()
  asa=d.route.eq('asa').to_numpy();err=asa&(pred!=y);normal=asa&(pred==0);conf=q.max(1)
  # A diagnostic optimistic intervention, NOT a valid deployed three-class rule.
  # Even removing the normal head cannot supply missing malicious/suspicious evidence.
  restricted=q[:,1:].argmax(1)+1
  cold=[];seen=[set(x.loc[fit,k]) for k in v.FIELDS]
  for row in facts.itertuples(index=False,name=None):
   cold.append([k for k,value,values in zip(v.FIELDS,row,seen) if value not in values])
  d['confidence_uncalibrated']=conf;d['novel_fields']=list(map(json.dumps,cold))
  for k in v.FIELDS:d[k]=facts[k]
  cases.append(d.loc[err|normal].assign(protocol=protocol))
  summary[protocol]={
   'ASA_full':v.metrics(y[asa],pred[asa]),
   'forced_non_normal_diagnostic_only':v.metrics(y[asa],restricted[asa]),
   'normal_controls_harmed_by_forcing_non_normal':int(((y==0)&(pred==0)).sum()),
   'false_normal':{'rows':int(normal.sum()),'src_role_missing':int((normal&facts.src_role.eq(v.MISSING)).sum()),
     'protocol_counts':facts.loc[normal,'transport_protocol'].value_counts().to_dict(),
     'dst_role_counts':facts.loc[normal,'dst_role'].value_counts().to_dict()},
   'confidence_at_least_0_9':{'ASA_rows':int((asa&(conf>=.9)).sum()),'ASA_errors':int((err&(conf>=.9)).sum())},
   'ASA_external_nll':float(-np.log(np.maximum(q[asa,y[asa]],1e-30)).mean())}
  print(json.dumps({protocol:summary[protocol]}),flush=True)
 pd.concat(cases,ignore_index=True).to_parquet(OUT/'neural_error_cases.parquet',index=False)
 v.save(OUT/'summary.json',{'protocols':summary,'scope':'Post-hoc explanatory controls; forced non-normal is rejected as a deployment rule and never used for model selection.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 v.save(OUT/'receipt.json',{'files':{p.name:v.sha(p) for p in OUT.iterdir() if p.is_file()}})
if __name__=='__main__':main()
