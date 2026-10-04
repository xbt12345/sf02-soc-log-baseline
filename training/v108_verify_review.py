"""Independent counting checks for the no-fit V108 review artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from run_v75 import ROOT, save, sha
from v108_root_evidence_audit import DEST


def main():
    target=DEST/'verification.json'; assert not target.exists()
    j=json.loads((DEST/'root_evidence.json').read_text(encoding='utf-8'))
    raw=json.loads((DEST/'raw_trace_review.json').read_text(encoding='utf-8'))
    meta=json.loads((DEST/'resolution_and_metadata_audit.json').read_text(encoding='utf-8'))
    d=pd.read_parquet(DEST/'support_and_error_ledger.parquet')
    trace=pd.read_parquet(DEST/'ASA_raw_fact_prediction_trace.parquet')
    checks={}
    for name,report in [('root_evidence_audit',j),('raw_trace_review',raw),('resolution_audit',meta)]:
        checks[name+'_source_bound']=sha(ROOT/f'training/v108_{name}.py')==report['source_sha256']
        checks[name+'_no_fits']=report['classifier_fits']==0
    checks['all_input_receipts_match']=all(sha(ROOT/p)==h for p,h in j['input_sha256'].items())
    checks['raw_row_alignment']=bool(np.array_equal(d.row_position,trace.row_position))
    positions=d.row_position.to_numpy();labels=[];offset=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=65536,columns=['label_binary']):
        a=np.searchsorted(positions,offset);b=np.searchsorted(positions,offset+len(batch))
        if a!=b:labels.extend(batch.take(positions[a:b]-offset).column(0).to_pylist())
        offset+=len(batch)
    checks['ASA_truth_matches_current_official_rows']=bool(np.array_equal(pd.Series(labels).map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(),d.truth))
    y=d.truth.to_numpy();a=d.N1_TabM25.to_numpy();b=d.N2_TabM25.to_numpy()
    checks['paired_real_class_counts']=bool(((y==1)&(a==1)).sum()==78430 and ((y==2)&(a==2)).sum()==31965 and ((y==1)&(b==1)).sum()==78452 and ((y==2)&(b==2)).sum()==28865)
    lost=(y==2)&(a==2)&(b!=2)
    checks['regression_support_partition']=bool(lost.sum()==3102 and np.sum(lost&(d.behavior_same_roots==-1))==652 and np.sum(lost&(d.behavior_same_roots==0))==2232 and np.sum(lost&(d.behavior_same_roots==1))==166 and np.sum(lost&(d.behavior_same_roots>=2))==52)
    checks['coarse_known_despite_port_redaction']=bool((d.loc[lost,'coarse_same_roots']>=2).all() and (d.loc[lost,'coarse_other_roots']>=2).all())
    eligible=(d.behavior_same_roots>=2)&(d.behavior_other_roots>=2)
    checks['strict_dual_class_source_support_52']=int((lost&eligible).sum())==52
    for view in ('N1','N2'):
        # Alternate implementation of score-tie oracle using grouped histograms.
        t=pd.DataFrame({'score':d[f'{view}_S_probability'],'M':y==1,'S':y==2})
        counts=t.groupby('score')[['M','S']].sum().sort_index(ascending=False).cumsum()
        maximum=int(counts.loc[counts.M<=318,'S'].max())
        minimum=int(counts.loc[counts.S>=31965,'M'].min())
        want=j['ranking_diagnostic'][view]
        checks[view+'_oracle_histogram_reproduced']=maximum==want['max_S_correct_with_M_budget'] and minimum==want['min_M_false_S_to_reach_target']
    checks['metadata_rebuilt_from_raw']=meta['metadata_reconstruction_matches_all_ASA']
    checks['unknown_ports_not_lost_numeric']=raw['unknown_port_plain_numeric_in_range']==0
    checks['all_original_models_unchanged']=True
    prev=ROOT/'artifacts/v107_matched_training_20260928'
    for p,v in j['model_prediction_receipts'].items():
        checks['all_original_models_unchanged'] &= sha(prev/p/'model.pt')==v['model_sha256'] and sha(prev/p/'ASA_input_prob.npy')==v['prob_sha256']
    assert all(checks.values()), checks
    paths=[p for p in DEST.iterdir() if p.is_file()]
    out={'all_checks_passed':True,'checks':checks,'source_sha256':sha(__file__),
         'artifact_sha256':{p.name:sha(p) for p in paths},
         'scope':'Audit integrity/counting only. No classifier trained, no model-quality acceptance and no external blind test.'}
    save(target,out);print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
