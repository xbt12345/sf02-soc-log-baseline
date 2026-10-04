"""Exhaustive ASA raw grammar / representation collision audit. No fitting."""
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/2026-09-14/v53_raw_audit'
PAT=re.compile(r'^(?P<header>.*?)(?P<action>Deny)\s+(?P<protocol>tcp|udp|icmp6?|[0-9]+)\s+src\s+(?P<src>\S+)\s+dst\s+(?P<dst>\S+)(?:\s+\(type\s+(?P<icmp_type>\S+),\s*code\s+(?P<icmp_code>\S+)\))?\s+by\s+(?P<aclword>[A-Za-z0-9_-]+[-_]group)\s+"(?P<acl>[^"]*)"\s+\[(?P<hashes>0x[0-9a-f]+,\s*0x[0-9a-f]+)\]\s*$',re.I|re.S)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def integer(s,upper):return int(s) if s is not None and re.fullmatch(r'[0-9]+',s) and 0<=int(s)<upper else None
def endpoint(s):
 zone,rest=s.split(':',1);port=None
 if '/' in rest:addr,port=rest.rsplit('/',1)
 else:addr=rest
 return zone.casefold(),addr,port
def role(s):return 'dmz' if re.fullmatch(r'dmz(?:[-_]\d+)?',s) else s if s in {'inside','outside','identity','internet'} else None
def floor(d,keys):
 c=d.groupby(['fold']+keys+['label_index'],dropna=False).size().unstack(fill_value=0)
 return int((c.sum(axis=1)-c.max(axis=1)).sum())
def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 prep=ROOT/'artifacts/v48_information_repair_r2_20260914';r=pd.read_parquet(prep/'rows.parquet');r=r[r.route=='asa'].set_index('row_position');p=pd.read_parquet(prep/'projections.parquet')
 records=[];unmatched=[];disagreements=[];offset=0
 for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=16384,columns=['message_sanitized']):
  positions=r.index[(r.index>=offset)&(r.index<offset+batch.num_rows)]
  for pos in positions:
   raw=batch.column(0)[int(pos)-offset].as_py();m=PAT.fullmatch(raw.strip())
   if not m:unmatched.append(int(pos));continue
   z=m.groupdict();row=r.loc[pos];f=json.loads(p.iloc[int(row.projection_id)].facts)
   safe={'action':'deny','outcome':'blocked','transport_protocol':z['protocol'].casefold()}
   if safe['transport_protocol'].isdigit():safe['transport_protocol']='ipproto_'+str(int(safe['transport_protocol']))
   for side in ['src','dst']:
    zone,addr,pt=endpoint(z[side]);z[side+'_zone']=zone;z[side+'_address']=addr;z[side+'_port_token']=pt or ''
    if role(zone):safe[side+'_role']=role(zone)
    n=integer(pt,65536);safe[side+'_port_fixed']=65536 if n is None else n
   for k in ['icmp_type','icmp_code']:
    n=integer(z[k],256)
    if n is not None:safe[k]=n
   mismatch={k:[v,f.get(k)] for k,v in safe.items() if f.get(k)!=v}
   if mismatch:disagreements.append({'row_position':int(pos),'mismatch':mismatch})
   z.update(row_position=int(pos),fold=int(row.fold),inner_role=int(row.inner_role),projection_id=int(row.projection_id),label_index=int(row.label_index),body_group=str(row.body_group),
       raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),body_sha256=hashlib.sha256(raw.strip()[m.start('action'):].encode()).hexdigest(),
       semantic_key=json.dumps(f,sort_keys=True,separators=(',',':')),independent_safe_key=json.dumps(safe,sort_keys=True,separators=(',',':')))
   records.append(z)
  offset+=batch.num_rows
 d=pd.DataFrame(records);d.to_parquet(OUT/'raw_fields.parquet',index=False)
 assert len(d)+len(unmatched)==len(r)==106953
 fields=['src_zone','dst_zone','src_address','dst_address','src_port_token','dst_port_token','aclword','acl','hashes','header']
 collision=d.groupby('semantic_key').label_index.nunique();mixed=set(collision[collision>1].index)
 groups=[]
 for key,g in d[d.semantic_key.isin(mixed)].groupby('semantic_key'):
  varying=[k for k in fields if g[k].fillna('').nunique()>1]
  groups.append({'semantic_key':key,'rows':len(g),'labels_M_S':[int((g.label_index==k).sum()) for k in [1,2]],'body_groups':int(g.body_group.nunique()),'varying_raw_fields':varying,
      'safe_observation_versions':int(g.independent_safe_key.nunique()),'raw_body_versions':int(g.body_sha256.nunique())})
 ablation={'current_semantics':floor(d,['semantic_key']),'independent_safe_observations':floor(d,['independent_safe_key']),
    'literal_body_diagnostic_not_model_input':floor(d,['body_sha256']),'literal_full_raw_diagnostic_not_model_input':floor(d,['raw_sha256'])}
 for field in fields:ablation['semantics_plus_'+field+'_diagnostic_only']=floor(d,['semantic_key',field])
 # Fit-only input-frequency control on current semantic facts. No smoothing search.
 controls=[]
 for name in ['pressure','fold_0','fold_1','fold_2']:
  tr=d.inner_role!=2 if name=='pressure' else d.fold!=int(name[-1]);ev=~tr
  c=pd.crosstab(d.loc[tr,'semantic_key'],d.loc[tr,'label_index']).reindex(columns=[1,2],fill_value=0)
  n=c.reindex(d.loc[ev,'semantic_key'],fill_value=0).to_numpy();prior=float((d.loc[tr,'label_index']==1).mean())
  q=(n[:,0]+prior)/(n.sum(axis=1)+1.0)
  dd=d.loc[ev,['row_position','fold','inner_role','semantic_key','label_index']].copy();dd['qM']=q;dd['seen_fit_input']=n.sum(axis=1)>0;dd['pred']=np.where(q>=.5,1,2);dd['protocol']=name
  controls.append(dd)
 pd.concat(controls).to_parquet(OUT/'fit_frequency_control.parquet',index=False)
 save('collision_groups.json',groups)
 save('summary.json',{'scope':'Exhaustive allowed-development ASA grammar audit; literal identifiers are diagnostic only, never new training features.','rows':len(d),'unmatched_positions':unmatched,'observable_field_disagreements':disagreements,
     'mixed_semantic_groups':len(groups),'mixed_group_rows':sum(g['rows'] for g in groups),'safe_observation_versions_in_mixed_groups':sorted(set(g['safe_observation_versions'] for g in groups)),
     'finite_fold_error_floors':ablation,'varying_field_mixed_group_counts':{k:sum(k in g['varying_raw_fields'] for g in groups) for k in fields},
     'source_bindings':{str(x.relative_to(ROOT)):sha(x) for x in [prep/'rows.parquet',prep/'projections.parquet',ROOT/'data/official/train.parquet',Path(__file__)]},
     'limits':'No raw-identity/time association is treated as attack truth. No conclusion that all possible safe semantics or outside context is exhausted. All evaluations already inspected.'})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 save('receipt.json',{'files':{x.name:sha(x) for x in OUT.iterdir() if x.is_file()}})
 print(json.dumps({'rows':len(d),'unmatched':len(unmatched),'field_disagreements':len(disagreements),'mixed_groups':len(groups),'floors':ablation}),flush=True)
if __name__=='__main__':main()
