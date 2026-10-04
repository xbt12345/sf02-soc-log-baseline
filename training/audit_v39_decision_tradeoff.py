"""Measure whole-development tradeoffs of oracle class-offset feasibility."""
import hashlib,json
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/2026-09-13/v39_methods'
RUN=ROOT/'artifacts/v38_local_r1_20260913/equal_classifiers_attempt2/C_BOTH'
def main():
    target=OUT/'decision_tradeoff.json'
    if target.exists():raise FileExistsError('Preserve prior diagnostic')
    a=json.loads((OUT/'assumption_probes.json').read_text(encoding='utf-8'))
    d=pq.read_table(RUN/'evaluation.parquet').to_pandas();y=d.label_index.to_numpy(dtype='i8')
    logp=np.log(d[['p_benign','p_malicious','p_suspicious']].to_numpy());before=logp.argmax(1)
    B=y==0;M=y==1;S=y==2
    result=[]
    for q in a['global_class_bias_feasibility']:
        # This offset is deliberately obtained from inspected answers. It may
        # only disprove an unrestricted fix; never select it for future use.
        delta=q['S_logit_offset_required_to_fix_all_strictly_greater_than']+1e-9
        z=logp.copy();z[:,2]+=delta;after=z.argmax(1)
        cm=np.bincount(y*3+after,minlength=9).reshape(3,3)
        result.append({'oracle_target':q['target'],'offset':delta,'confusion_matrix':cm.tolist(),
          'new_benign_errors':int((B&(after!=0)).sum()),
          'previously_correct_M_now_S':int((M&(before==1)&(after==2)).sum()),
          'M_recall_before':float((before[M]==1).mean()),'M_recall_after':float((after[M]==1).mean()),
          'S_recall_before':float((before[S]==2).mean()),'S_recall_after':float((after[S]==2).mean())})
    answer={'scope':'Post-inspection oracle offset tradeoff on v38 development only. Not calibration, model fitting or transferable improvement.',
       'model_changed':False,'trials':result,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.write_text(json.dumps(answer,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(answer,ensure_ascii=False))
if __name__=='__main__':main()
