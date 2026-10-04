"""Inspect the two already-seen dominant residual inputs, without fitting."""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
import pandas as pd

PATTERN=re.compile(r'^(?P<header>.*?)Deny icmp src (?P<src_zone>[^:\s]+):(?P<src_address>\S+) dst (?P<dst_zone>[^:\s]+):(?P<dst_address>\S+) \(type (?P<type>\d+), code (?P<code>\d+)\)(?P<tail>.*)$',re.S)

def main(root):
    sys.path.insert(0,str(root/'artifacts/v50_context_training_r2_20260914/runtime'))
    from run_v48 import raw_rows,save,sha
    rows=pd.read_parquet(root/'artifacts/v48_information_repair_r2_20260914/rows.parquet')
    evidence=root/'evidence/2026-09-14/v50_review'
    diagnosed=pd.read_parquet(evidence/'asa_fit_support_rows.parquet')
    keys=diagnosed[diagnosed.error].encoded_key.value_counts().head(2).index
    selected=diagnosed[diagnosed.encoded_key.isin(keys)].set_index('row_position')
    records=[];unmatched=[]
    for pos,raw in raw_rows(root/'data/official/train.parquet',selected.index):
        match=PATTERN.fullmatch(raw.strip())
        if match is None:unmatched.append(pos);continue
        r=match.groupdict();r.update(row_position=pos,label_index=int(selected.loc[pos,'label_index']),
            projection_id=int(selected.loc[pos,'projection_id']),raw_sha256=hashlib.sha256(raw.encode()).hexdigest())
        records.append(r)
    d=pd.DataFrame(records);groups=[]
    for pid,g in d.groupby('projection_id'):
        groups.append({'projection_id':int(pid),'rows':len(g),'labels_B_M_S':[int((g.label_index==i).sum()) for i in range(3)],
            'tails':g['tail'].value_counts().to_dict(),'src_zone_by_label':[dict(zone=str(z),label=int(y),rows=int(n)) for (z,y),n in g.groupby(['src_zone','label_index']).size().items()],
            'dst_zone_by_label':[dict(zone=str(z),label=int(y),rows=int(n)) for (z,y),n in g.groupby(['dst_zone','label_index']).size().items()],
            'type_values':g['type'].unique().tolist(),'code_values':g['code'].unique().tolist()})
    assert len(d)+len(unmatched)==len(selected)==993
    d.to_parquet(evidence/'dominant_icmp_raw_fields.parquet',index=False)
    save(evidence/'dominant_icmp_raw_audit.json',{'selected_rows':len(selected),'matched_rows':len(d),'unmatched_positions':unmatched,
        'groups':groups,'scope':'Already-inspected development errors only. Full grammar partitions header, action, endpoints, ICMP type/code and remainder. Remaining identity/date/zone suffix differences have no provided topology or incident truth; not training features or new label rules.',
        'source_sha256':sha(Path(__file__))})
    (evidence/'audit_v50_icmp.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'matched':len(d),'unmatched':len(unmatched),'groups':groups}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
