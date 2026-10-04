"""Audit all official ASA ICMP records and independent support, no fitting."""
import hashlib
import json
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from v89_common import ROOT, DEST as V89, read, save, sha, data
from v96_icmp_parser import parse, self_check

DEST = ROOT / 'artifacts/v96_protocol_and_experiment_audit_20260928'
V95 = ROOT / 'artifacts/v95_cross_component_training_20260928'


def main():
    assert not (ROOT/'evidence/2026-09-28/v96_root_cause_audit/delivery.json').exists()
    DEST.mkdir(exist_ok=True)
    checks = self_check(); save(DEST/'parser_checks.json', checks)
    r, y, fid, _, old, _, fit = data()
    roles = pd.read_parquet(V95/'full_format_roles.parquet', columns=['role']).role.to_numpy().astype(object)
    for name, fold in [('inner',1),('C',2),('H',0)]: roles[r.fold.eq(fold).to_numpy()] = name
    obs = np.load(V89/'row_fact_code.npy')
    dictionary = pd.read_parquet(V89/'row_fact_dictionary.parquet').observation_json
    oldfacts = [json.loads(x)['facts'] for x in dictionary]
    records = []; rejects = []; off = 0; matched_candidate = 0; asa_total = 0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=32768, columns=['message_sanitized','label_binary'], use_threads=False):
        df = batch.to_pandas(); end = off + len(df)
        assert np.array_equal(df.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(), y[off:end])
        for j in np.flatnonzero(r.route.iloc[off:end].eq('asa').to_numpy()):
            i = int(off+j); raw = df.message_sanitized.iloc[j]; asa_total += 1
            if not isinstance(raw, str) or re.search(r'\bDeny\s+icmp\b', raw, re.I) is None: continue
            matched_candidate += 1; parsed = parse(raw)
            f = parsed['facts']; complete = all(k in f for k in ['icmp_type','icmp_code','src_role','dst_role'])
            if not parsed['format_matched']:
                rejects.append({'row_position':i,'label':int(y[i]),'role':str(roles[i]),'reason':'header_or_body_not_recognized',
                                'raw_prefix':raw[:130] if len(rejects)<10 else None})
                continue
            records.append({'row_position':i, 'label':int(y[i]), 'role':str(roles[i]), 'component':int(r.component.iloc[i]),
                'R0':int(fid[i]), 'complete_known_tuple':complete, 'auxiliary_parser_known_fact_count':len(oldfacts[obs[i]]),
                'src_role':f.get('src_role','unknown'), 'dst_role':f.get('dst_role','unknown'),
                'icmp_type':f.get('icmp_type',-1), 'icmp_code':f.get('icmp_code',-1),
                'src_zone_audit':parsed['audit_zones'].get('src_role'), 'dst_zone_audit':parsed['audit_zones'].get('dst_role'),
                'spans_json':json.dumps(parsed['observations']), 'raw_sha256':hashlib.sha256(raw.encode()).hexdigest()})
        off = end
    assert off == len(y) and asa_total == 112807
    frame = pd.DataFrame(records); frame.to_parquet(DEST/'icmp_rows.parquet',index=False)
    save(DEST/'unmatched_icmp_candidates.json',rejects)
    keys = ['src_role','dst_role','icmp_type','icmp_code']
    counts = frame.groupby(keys+['role','label']).agg(rows=('row_position','size'),components=('component','nunique'),inputs=('R0','nunique')).reset_index()
    counts.to_csv(DEST/'icmp_semantic_support.csv',index=False)
    coarse = frame.groupby(['icmp_type','icmp_code','role','label']).agg(rows=('row_position','size'),components=('component','nunique')).reset_index()
    coarse.to_csv(DEST/'icmp_type_code_support.csv',index=False)
    target = frame[(frame.src_role=='outside') & (frame.dst_role=='dmz') & (frame.icmp_type==3) & (frame.icmp_code==13)]
    exactzone=frame[(frame.src_zone_audit=='outside')&(frame.dst_zone_audit=='dmz-2')&(frame.icmp_type==3)&(frame.icmp_code==13)]
    def summary(x):
        return x.groupby(['role','label']).agg(rows=('row_position','size'),components=('component','nunique'),inputs=('R0','nunique')).reset_index().to_dict('records')
    # A conflict in this reduced tuple is not a conflict in complete observable input.
    ab=frame[frame.role.isin(['A','B']) & frame.complete_known_tuple]
    n=ab.groupby(keys).label.nunique()
    result={'status':'all_official_ASA_ICMP_audited_no_fit','official_label_rows_checked':len(y),'ASA_rows_checked':asa_total,
        'ICMP_candidates':matched_candidate,'parsed_ICMP_rows':len(frame),'unmatched_candidates':len(rejects),
        'complete_known_tuple_rows':int(frame.complete_known_tuple.sum()),'auxiliary_parser_empty_known_facts_rows':int(frame.auxiliary_parser_known_fact_count.eq(0).sum()),
        'ICMP_role_class_counts':summary(frame),'target_direction_type_code_support':summary(target),
        'target_exact_zone_type_code_support_audit_only':summary(exactzone),
        'type3_code13_all_directions_support':summary(frame[(frame.icmp_type==3)&(frame.icmp_code==13)]),
        'AB_behavior_tuples':len(n),'AB_mixed_label_behavior_tuples':int((n>1).sum()),
        'new_classifier_fits':0,'new_calibration_fits':0,'source_sha256':sha(__file__),'parser_sha256':sha(ROOT/'training/v96_icmp_parser.py'),
        'scope':'Observed fields and support only. Empty facts refer to v89 auxiliary grouping, NOT the R0 model facts. No inferred attack label, input removal, label cleaning or model quality claim.'}
    save(DEST/'protocol_audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
