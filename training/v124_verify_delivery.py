"""Recheck V124 delivery identity and official-label outcomes without fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v124_header_trial_20260929'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    target=OUT/'verification.json'
    if target.exists():raise FileExistsError(target)
    delivery=read(OUT/'delivery.json')
    for rel,h in delivery['artifact_sha256'].items():assert sha(ROOT/rel)==h,rel
    for name,manifest in [('v121',ROOT/'artifacts/v121_paired_batch_training_20260929/delivery.json'),
                          ('v122',ROOT/'artifacts/v122_evidence_review_20260929/verification.json')]:
        for rel,h in read(manifest)['artifact_sha256'].items():assert sha(ROOT/rel)==h,(name,rel)
    expert=pd.read_parquet(OUT/'expert_ASA_predictions.parquet')
    full=pd.read_parquet(OUT/'full_prediction_ledger.parquet')
    official=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['event_id','label_binary'])
    y=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(dtype=np.int8)
    assert len(full)==len(y)==2056871 and np.array_equal(full.row_position,np.arange(len(y)))
    assert len(expert)==112807 and not expert.row_position.duplicated().any()
    assert np.array_equal(y[expert.row_position],expert.truth)
    val={}
    for arm in ('A','B'):
        p=expert[f'expert_pred_{arm}'].to_numpy()
        f=full[f'final_pred_{arm}'].to_numpy()
        val[arm]={'M_ASA_errors':int(((p!=expert.truth)&(expert.truth==1)).sum()),
                  'S_ASA_errors':int(((p!=expert.truth)&(expert.truth==2)).sum()),
                  'full_errors':int((f!=y).sum())}
    assert [val[z]['M_ASA_errors'] for z in ('A','B')]==delivery['ASA_M_errors_A_B']
    assert [val[z]['S_ASA_errors'] for z in ('A','B')]==delivery['ASA_S_errors_A_B']
    assert delivery['classifier_fits_new']==6 and delivery['optimizer_steps']==8900
    assert delivery['confirmation_fits']==0 and not delivery['model_promoted']
    result={'status':'v124_delivery_integrity_and_independent_counts_verified',
        'delivery_sha256':sha(OUT/'delivery.json'),
        'bound_files':len(delivery['artifact_sha256']),
        'historical_V121_files':len(read(ROOT/'artifacts/v121_paired_batch_training_20260929/delivery.json')['artifact_sha256']),
        'historical_V122_files':len(read(ROOT/'artifacts/v122_evidence_review_20260929/verification.json')['artifact_sha256']),
        'independent_counts':val,'new_fits':6,'model_promoted':False,
        'scope':'Hash identity and label counts, not an external or new blind test.'}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
