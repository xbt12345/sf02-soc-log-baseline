"""Frozen-model metamorphic tests on actual WAF collector headers; no training."""
from pathlib import Path
import sys,json,re,warnings
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import joblib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'training'))
import v36_representation as rep
from v36_infer import predict
from diagnose_inputs import selected,save
HEADER=re.compile(r'^<\d{1,3}>[A-Za-z]{3}\s+\d{1,2}\s+(?P<clock>\S+)\s+\S+\s+CEF:')

def rewrite(raw,mode):
    m=HEADER.match(raw)
    if not m:return raw
    clock='00:00:00' if mode=='complete_clock' else 'USER-9999'
    return raw[:m.start('clock')]+clock+raw[m.end('clock'):]

def main():
    root=ROOT/'artifacts/v36_cloud_20260912T191112Z/work';prepared=ROOT/'artifacts/v36_prepared_r13_20260912/prepared.parquet'
    ev=pq.read_table(root/'models/source_waf_B2/evaluation.parquet');pos=ev['row_position'].to_numpy();y=ev['label_index'].to_numpy()
    raw=selected(ROOT/'data/official/train.parquet',pos,['message_sanitized']);records=[raw[int(p)] for p in pos]
    route=[rep.prepare_record(v)['route'] for v in records]
    variants={'original':records}
    for mode in ['complete_clock','redacted_clock']:variants[mode]=[{'message_sanitized':rewrite(v['message_sanitized'],mode)} for v in records]
    result={'scope':'Collector-time-slot transformations only; no payload alteration, training, threshold tuning or label changes. Existing models and calibration thresholds frozen. This probes parsing invariance; not future deployment quality.','header_matches':sum(bool(HEADER.match(v['message_sanitized'])) for v in records),'models':[]}
    for suffix in ['B2','B2_W']:
        folder=root/'models'/('source_waf_'+suffix);bundle=joblib.load(folder/'model.joblib');a=json.loads((folder/'report.json').read_text())
        rp=json.loads((root/'model_replay.json').read_text());rr=next(v for v in rp['models'] if v['task']=='source_waf' and v['view']=='B2' and v['weighted']==(suffix=='B2_W'))
        threshold=rr['worst_calibration_source_threshold_diagnostic'][1]['threshold'];base=None
        for mode,v in variants.items():
            p=predict(bundle,v)
            if base is None:base=p
            pred=p.argmax(1);risk=1-p[:,0];r=[rep.prepare_record(row)['route'] for row in v]
            info={'model':suffix,'mode':mode,'confusion':np.bincount(y*3+pred,minlength=9).reshape(3,3).tolist(),
                  'risk001_counts':[int(((1-p[:,0]>=a['risk'][1]['threshold'])&(y==c)).sum()) for c in range(3)],
                  'worst001_counts':[int(((risk>=threshold)&(y==c)).sum()) for c in range(3)],
                  'probability_max_change':float(abs(p-base).max()),'classification_flips':int((pred!=base.argmax(1)).sum()),
                  'route_counts':{k:[int(sum(q==k and yy==c for q,yy in zip(r,y))) for c in range(3)] for k in set(r)}}
            result['models'].append(info);print(json.dumps(info,ensure_ascii=False),flush=True)
            pq.write_table(pa.table({'row_position':pos,'label_index':y,'p_benign':p[:,0],'p_malicious':p[:,1],'p_suspicious':p[:,2]}),OUT/('waf_'+suffix+'_'+mode+'.parquet'),compression='zstd')
    # A separate scalar spelling consistency probe. It is not alleged to be the
    # cause of original observed WAF errors; the local parser explicitly accepts quotes.
    sample=next(v['message_sanitized'] for v,yy in zip(records,y) if yy==2)
    quoted=re.sub(r'\bact=DENY\b','act="DENY"',sample)
    q1=rep.prepare_message(sample);q2=rep.prepare_message(quoted)
    result['quoted_action_probe']={'same_text':q1['b1']==q2['b1'],'facts_before':q1['facts'],'facts_after':q2['facts'],'same_facts':q1['facts']==q2['facts'],'reason':'Decoded allowed text scalar is not reused by CEF fact derivation; separate consistency defect, not verified training-error cause.'}
    save('collector_time_stress.json',result)
if __name__=='__main__':main()
