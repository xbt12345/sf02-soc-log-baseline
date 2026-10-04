"""ASA-only research inference; does not install or promote a global classifier."""
import numpy as np
import pandas as pd
import run_v54 as v
LIMITS={'src_port_fixed':65536,'dst_port_fixed':65536,'icmp_type':256,'icmp_code':256}
def observations(facts):
 values={}
 for k in v.FIELDS:
  a=facts.get(k)
  if a is None or a=='':values[k]=v.MISSING
  elif k in LIMITS and (type(a)!=int or not 0<=a<LIMITS[k]):values[k]=v.MISSING
  else:values[k]=str(a)
 return values
def classify_records(bundle,records,adapter):
 parsed=[adapter.prepare_record(row) for row in records]
 if any(p['route'] not in ['asa','asa_acl'] for p in parsed):raise ValueError('This research classifier accepts ASA/ACL only; use the retained global runtime for other records.')
 x=pd.DataFrame([observations(p['facts']) for p in parsed],columns=v.FIELDS,dtype=object)
 pred,score=v.predict(bundle,x)
 return {'labels':pred,'scores':score,'score_type':'uncalibrated_probabilities' if bundle['family']=='neural' else 'margins_not_probabilities',
  'novel_fields':[[k for j,k in enumerate(v.FIELDS) if x.iloc[i,j] not in bundle['input']['seen'][j] and x.iloc[i,j]!=v.MISSING] for i in range(len(x))]}
