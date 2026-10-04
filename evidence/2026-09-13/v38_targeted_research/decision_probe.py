"""Explore joint 3-class decisions with calibration-only offsets, no model fits.

Offsets are frozen before loading evaluation scores. Evaluation slices were
already inspected in prior research: this is development evidence, not blind.
"""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
WORK=ROOT/'artifacts/v37_cloud_20260913T072309Z/work'
sys.path.insert(0,str(ROOT/'artifacts/v37_cloud_20260913T072309Z/runtime'))
import v37_learning as learning
GRID=[-8.,-6.,-4.,-2.,0.,2.,4.,6.,8.]
CONFIG={'kind':'posthoc_development_probe','grid':GRID,'benign_offset':0,'choice':'maximum calibration macro F1; tied within 1e-12 prefer smallest squared offsets, then sorted offsets',
        'uses_target_labels_for_choice':False,'new_model_fit':False,'same_calibration_reused_for_choice':True,'independent_evaluation_claim':False}
(OUT/'decision_probe_config.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
result=[]
for task in ['known_dev','source_ad','source_duo','source_waf','asa_hard']:
    folder=WORK/'models'/(task+'_B2_REPAIRED');cal=pq.read_table(folder/'calibration.parquet').to_pandas()
    p=cal[['p_benign','p_malicious','p_suspicious']].to_numpy();y=cal.label_index.to_numpy()
    unique,back=np.unique(p,axis=0,return_inverse=True);count=np.bincount(back*3+y,minlength=len(unique)*3).reshape(-1,3)
    lp=np.log(np.clip(unique,1e-300,1));best=None
    for bm in GRID:
        for bs in GRID:
            pred=(lp+np.array([0,bm,bs])).argmax(1)
            cm=np.zeros((3,3),np.int64)
            for c in range(3):np.add.at(cm[c],pred,count[:,c])
            f1=np.divide(2*cm.diagonal(),cm.sum(0)+cm.sum(1),out=np.zeros(3),where=(cm.sum(0)+cm.sum(1))>0).mean()
            order=(bm*bm+bs*bs,bm,bs)
            if best is None or f1>best[0]+1e-12 or abs(f1-best[0])<=1e-12 and order<best[1]:best=(float(f1),order,[0,bm,bs])
    frozen={'task':task,'calibration_sha256':hashlib.sha256((folder/'calibration.parquet').read_bytes()).hexdigest(),'offsets':best[2],'calibration_macro_f1_in_sample_selection':best[0]}
    (OUT/(task+'_frozen_offsets.json')).write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    ev=pq.read_table(folder/'evaluation.parquet').to_pandas();p=ev[['p_benign','p_malicious','p_suspicious']].to_numpy();y=ev.label_index.to_numpy()
    joint=(np.log(np.clip(p,1e-300,1))+np.asarray(best[2])).argmax(1)
    report=json.loads((folder/'report.json').read_text());primary=next(t for t in report['policies'] if t['alpha']==.001 and t['name']=='empirical_worst_source_global')
    result.append({'task':task,'frozen':frozen,'raw_argmax':learning.cm_metrics(y,p.argmax(1)),
        'joint_offsets':learning.cm_metrics(y,joint),'previous_alarm_gate':primary['gated_classification'],
        'joint_normal_false_positives':int(((y==0)&(joint!=0)).sum()),'joint_risk_class_alarm_counts':[int(((y==c)&(joint!=0)).sum()) for c in [1,2]],'scope':'calibration-chosen constant class offsets; no binary alarm gate; development only'})
(OUT/'decision_probe_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
