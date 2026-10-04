"""Frozen-model missing-code intervention; diagnostic only, never deployment."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/v39_local_r2_20260913'
sys.path.insert(0,str(BASE/'frozen_training_runtime'))
import joblib
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from run_v39_prepare import sha,save
from run_v39_train import metric
from v38_learning import BIT_FIELDS


def main():
    dest=ROOT/'evidence/2026-09-13/v40_mechanisms/missing_ablation.json'
    assert not dest.exists()
    folder=BASE/'old_protocol_stress'; prep=BASE/'prepared'
    done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    assert sha(folder/'model.joblib')==done['model_sha256']
    receipt=json.loads((prep/'complete.json').read_text(encoding='utf-8'))
    assert sha(prep/'projections.parquet')==receipt['files']['projections.parquet']
    assert sha(folder/'evaluation.parquet')==done['predictions_sha256']
    d=pq.read_table(folder/'evaluation.parquet').to_pandas()
    pr=pq.read_table(prep/'projections.parquet').to_pandas()
    active,ids=np.unique(d.projection_id.to_numpy(),return_inverse=True)
    pr=pr.iloc[active].reset_index(drop=True)
    facts=[json.loads(f) for f in pr.facts]
    b=joblib.load(folder/'model.joblib')
    tx=b['text_encoder'].transform(pr.text.tolist()); fx=b['fact_encoder'].transform(facts)
    x=sparse.hstack([tx,fx],format='csr')
    old=b['model'].predict_proba(x)
    assert np.max(np.abs(old[ids]-d[['p_benign','p_malicious','p_suspicious']].to_numpy()))<1e-10
    names=b['fact_encoder'].names()
    field_by_col={i:str(n).split(':bit')[0] for i,n in enumerate(names) if ':bit' in str(n)}
    coo=fx.tocoo(copy=True); remove=np.zeros(coo.nnz,dtype=bool)
    for i,(r,c) in enumerate(zip(coo.row,coo.col)):
        key=field_by_col.get(int(c))
        if key is not None and facts[r].get(key,BIT_FIELDS[key][1])==BIT_FIELDS[key][1]:
            remove[i]=True
    coo.data[remove]=0
    edited=coo.tocsr();edited.eliminate_zeros()
    new=b['model'].predict_proba(sparse.hstack([tx,edited],format='csr'))
    yy=d.label_index.to_numpy(dtype=int); routes=d.route.to_numpy()
    origpred=old[ids].argmax(1); pred=new[ids].argmax(1)
    changed={}
    for r in sorted(set(routes)):
        m=routes==r
        changed[str(r)]={'fixed':int((m&(origpred!=yy)&(pred==yy)).sum()),
            'regressed':int((m&(origpred==yy)&(pred!=yy)).sum()),
            'new_normal_errors':int((m&(yy==0)&(pred!=0)).sum()),
            'new_errors':int((m&(pred!=yy)).sum())}
    save(dest,{'scope':'Post-inspection diagnostic input intervention on frozen model; distribution mismatch, not a trained candidate',
        'script_sha256':sha(__file__),'model_sha256':done['model_sha256'],
        'predictions_sha256':done['predictions_sha256'], 'projections_sha256':receipt['files']['projections.parquet'],
        'baseline':metric(yy,old[ids]),'intervention':metric(yy,new[ids]),'routes':changed,
        'authentication_probabilities_after_intervention':[{'text':pr.iloc[k].text,'probabilities':new[k].tolist()} for k in np.unique(ids[routes=='authentication'])],
        'new_training_executed':False,'deployment_allowed':False,'fresh_blind_test':False})
    print(json.dumps({'old_errors':int((origpred!=yy).sum()),'new_errors':int((pred!=yy).sum()),'new_normal_errors':int(((yy==0)&(pred!=0)).sum()),'auth':changed['authentication']}),flush=True)


if __name__=='__main__': main()
