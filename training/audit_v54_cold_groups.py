"""Concrete cold-concept and opposite-body supervision audit. No new fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_v54 import sha,save,frame_keys,metrics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/2026-09-14/v54_cold_groups'
def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 run=ROOT/'artifacts/v54_method_change_20260914';r=pd.read_parquet(run/'rows.parquet');x=pd.read_parquet(run/'observations.parquet')
 d=r.copy();d['key']=frame_keys(x)
 for k in ['transport_protocol','src_role','dst_role','icmp_type','icmp_code','icmp_message','icmp_unreachable']:d[k]=x[k]
 d['split']=np.where(r.inner_role==2,'pressure','fit')
 selected=d[(d.transport_protocol=='icmp')&(d.icmp_type=='3')&(d.icmp_code=='13')]
 assert len(selected)==865 and selected.body_group.nunique()==2 and selected.key.nunique()==1 and set(selected.inner_role)=={2}
 assert sorted(selected.groupby('label_index').size().to_dict().items())==[(1,180),(2,685)]
 groups=selected.groupby(['body_group','fold','label_index']).size().reset_index(name='rows')
 groups.to_json(OUT/'icmp_3_13_body_groups.json',orient='records',indent=2)
 records=[]
 for protocol in ['pressure','fold_0','fold_1','fold_2']:
  fit=r.inner_role!=2 if protocol=='pressure' else r.fold!=int(protocol[-1]);target=d.key==selected.key.iloc[0]
  records.append({'protocol':protocol,'fit_same_key_labels_B_M_S':[int((fit&target&(r.label_index==k)).sum()) for k in range(3)],'evaluation_same_key_labels_B_M_S':[int((~fit&target&(r.label_index==k)).sum()) for k in range(3)],
   'fit_code13_rows':int((fit&(d.icmp_code=='13')).sum()),'fit_same_key_bodies':int(d.loc[fit&target,'body_group'].nunique())})
 base=pd.concat([pd.read_parquet(ROOT/'artifacts/v51_fact_residual_20260914'/f'fold_{f}'/'evaluation.parquet') for f in range(3)])
 base=base[base.row_position.isin(selected.row_position)].sort_values('row_position');q=base[['p_benign','p_malicious','p_suspicious']].to_numpy();baseline=metrics(base.label_index.to_numpy(),q.argmax(1));assert baseline['errors']==865
 # Audit every conflicting complete key by independent body support, not just this chosen example.
 mixed=d[d.route=='asa'].groupby('key').label_index.nunique();mixed=set(mixed[mixed>1].index)
 info=[]
 for key,g in d[d.key.isin(mixed)].groupby('key'):
  info.append({'key':key,'rows':len(g),'body_groups':int(g.body_group.nunique()),'bodies_M_S':[int(g.loc[g.label_index==k,'body_group'].nunique()) for k in [1,2]],'row_labels_M_S':[int((g.label_index==k).sum()) for k in [1,2]]})
 save(OUT/'mixed_key_support.json',info)
 d.groupby(['split','transport_protocol','src_role','dst_role','icmp_type','icmp_code','label_index']).size().reset_index(name='rows').to_parquet(OUT/'protocol_role_code_labels.parquet',index=False)
 save(OUT/'summary.json',{'scope':'Post-hoc diagnosis of previously inspected official development; not an input rule, relabeling or new selection criterion.',
   'icmp_3_13':{'observed_key':json.loads(selected.key.iloc[0]),'rows':865,'body_groups':2,'class_rows_B_M_S':[0,180,685],'protocol_support':records,'v51_primary':baseline,
    'pressure_suspicious_share':685/733,'same_observation_nonidentifiability':'One single deterministic classifier for these same observed inputs cannot classify both labels correctly. Per-fold training here exposes the opposite-label body; the pressure fit exposes neither body. This does not prove which official label is objectively right.'},
   'mixed_keys':len(info),'mixed_keys_at_most_2_bodies':sum(g['body_groups']<=2 for g in info),
   'source_bindings':{str(p.relative_to(ROOT)):sha(p) for p in [run/'rows.parquet',run/'observations.parquet']}})
 (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());save(OUT/'receipt.json',{'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps({'rows':865,'bodies':2,'baseline_primary_errors':865,'pressure_S_share':685/733,'mixed_keys':len(info)}),flush=True)
if __name__=='__main__':main()
