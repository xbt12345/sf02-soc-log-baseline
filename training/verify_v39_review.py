"""Cross-check review inputs, row-accounting and document links, without fit."""
import hashlib,json,re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/2026-09-13/v39_review';RUN=ROOT/'artifacts/v38_local_r1_20260913'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def main():
    target=OUT/'verification.json'
    if target.exists():raise FileExistsError('Preserve prior verification')
    checks={}
    checks['official_file_identity']=sha(ROOT/'data/official/train.parquet')=='6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
    old=ROOT/'artifacts/v37_prepared_r13_20260913';r=json.loads((old/'result.json').read_text(encoding='utf-8'))
    checks['legacy_prepared_identity']=sha(old/'prepared.parquet')==r['prepared_sha256']
    binding=json.loads((RUN/'equal_classifiers_attempt2/binding.json').read_text(encoding='utf-8'))
    checks['frozen_training_source_closure']=all(sha(RUN/'frozen_training_runtime'/n)==h for n,h in binding['sources'].items())
    for branch in ['A_SEMANTIC','B_DESTINATION','C_BOTH']:
        f=RUN/'equal_classifiers_attempt2'/branch;c=json.loads((f/'complete.json').read_text(encoding='utf-8'))
        checks['unchanged_model_'+branch]=sha(f/'model.joblib')==c['model_sha256']
    d=pq.read_table(RUN/'equal_classifiers_attempt2/C_BOTH/evaluation.parquet').to_pandas()
    pred=d.pred_label.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();y=d.label_index.to_numpy();wrong=pred!=y
    checks['4008_errors_recounted']=int(wrong.sum())==4008
    checks['3156_flow_errors_recounted']=int((wrong&(d.route.to_numpy()=='native_flow')).sum())==3156
    checks['827_ASA_errors_recounted']=int((wrong&(d.route.to_numpy()=='asa')).sum())==827
    checks['normal_errors_zero']=int((wrong&(y==0)).sum())==0
    checks['case_raw_duplicate_WAF']=d.loc[(d.route=='cef_fields')&(d.label_index==2),'row_position'].tolist()==[1477712,1477722]
    # Independently count the raw original values for the two WAF positions.
    column=pq.read_table(ROOT/'data/official/train.parquet',columns=['message_sanitized'])['message_sanitized']
    checks['WAF_same_original_bytes']=column[1477712].as_py()==column[1477722].as_py()
    conflicts=json.loads((OUT/'asa_conflicts.json').read_text(encoding='utf-8'))
    checks['conflict_minimum_204']=sum(sum(z['audit_counts'])-max(z['audit_counts']) for z in conflicts)==204
    checks['code13_180_685_same_vector']=any(z['facts'].get('icmp_code')==13 and z['audit_counts']==[0,180,685] for z in conflicts)
    missing=[]
    for filename in ['docs/V39_NEXT_PLAN.md','docs/TRAINING_PLAN.md','README.md']:
        f=ROOT/filename
        for link in re.findall(r'\]\(([^)]+)\)',f.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):continue
            if not (f.parent/link.split('#')[0]).exists():missing.append([filename,link])
    checks['local_document_links_exist']=not missing
    files={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*.json')}
    for name in ['docs/V39_NEXT_PLAN.md','docs/TRAINING_PLAN.md','README.md','training/audit_v39_remaining.py','training/audit_v39_support.py','training/audit_v39_conditional.py','training/verify_v39_review.py']:
        files[name]=sha(ROOT/name)
    result={'all_evidence_checks_passed':all(checks.values()),'checks':checks,'missing_links':missing,'files_sha256':files,
        'new_model_trained':False,'new_quality_claim':False,'platform_used':False,'source_and_plan_review_not_training_acceptance':True}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},ensure_ascii=False))
    if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
