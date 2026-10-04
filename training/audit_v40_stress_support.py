"""Frozen stress attribution and support; no fitting or threshold selection."""
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
from run_v39_prepare import sha, save


def main():
    out=ROOT/'evidence/2026-09-13/v40_mechanisms/stress_support.json'
    assert not out.exists()
    prep=BASE/'prepared'; folder=BASE/'old_protocol_stress'
    receipt=json.loads((prep/'complete.json').read_text(encoding='utf-8'))
    for name in ['rows.parquet','projections.parquet']:
        assert sha(prep/name)==receipt['files'][name]
    done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    assert sha(folder/'model.joblib')==done['model_sha256']
    assert sha(folder/'evaluation.parquet')==done['predictions_sha256']
    rows=pq.read_table(prep/'rows.parquet',columns=['row_position','projection_id','inner_role','body_group','route','label_index']).to_pandas()
    rows=rows[rows.inner_role>=0].reset_index(drop=True)
    active,ids=np.unique(rows.projection_id.to_numpy(),return_inverse=True)
    pr=pq.read_table(prep/'projections.parquet').to_pandas().iloc[active].reset_index(drop=True)
    facts=[json.loads(f) for f in pr.facts]
    b=joblib.load(folder/'model.joblib')
    x=sparse.hstack([b['text_encoder'].transform(pr.text.tolist()),b['fact_encoder'].transform(facts)],format='csr')
    names=np.concatenate([np.array(['text:'+v for v in b['text_encoder'].names()],dtype=object),b['fact_encoder'].names()])
    assert len(names)==x.shape[1]
    fit=rows.inner_role.to_numpy()!=2; ev=~fit
    y=rows.label_index.to_numpy(dtype=int); routes=rows.route.to_numpy(); body=rows.body_group.to_numpy()
    prob=b['model'].predict_proba(x)
    old=pq.read_table(folder/'evaluation.parquet').to_pandas()
    assert np.array_equal(old.row_position,rows.row_position.to_numpy()[ev])
    diff=float(np.abs(old[['p_benign','p_malicious','p_suspicious']].to_numpy()-prob[ids[ev]]).max())
    assert diff<1e-10
    examples=[]
    for k in np.unique(ids[(routes=='authentication') & ev]):
        z=x.getrow(k)
        coef=b['model'].coef_[2]-b['model'].coef_[0]
        terms=coef[z.indices]*z.data
        order=np.argsort(np.abs(terms))[::-1]
        examples.append({'projection_id':int(active[k]),'text':pr.iloc[k].text,'facts':facts[k],
            'rows':int(((ids==k)&ev).sum()),'probabilities':prob[k].tolist(),
            'suspicious_minus_benign_intercept':float(b['model'].intercept_[2]-b['model'].intercept_[0]),
            'suspicious_minus_benign_margin':float(z.dot(coef)[0]+b['model'].intercept_[2]-b['model'].intercept_[0]),
            'active_terms':[{'name':str(names[z.indices[j]]),'value':float(z.data[j]),'margin_contribution':float(terms[j])} for j in order],
            'fit_counts_same_projection':np.bincount(y[fit & (ids==k)],minlength=3).tolist()})
    support={}
    for key,value in [('auth_result','failure'),('response','missing'),('authentication_interaction','no_response'),('credential_check','invalid'),('outcome','failure')]:
        have=np.array([f.get(key)==value for f in facts])[ids]
        support[key+'='+value]={role:{str(route):{'counts':np.bincount(y[pop & have & (routes==route)],minlength=3).tolist(),
               'body_keys_by_class':[int(len(np.unique(body[pop & have & (routes==route) & (y==c)]))) for c in range(3)]}
               for route in sorted(set(routes[pop & have]))} for role,pop in [('fit',fit),('evaluation',ev)]}
    # Confidence check is descriptive. A threshold was neither selected nor applied.
    wrong=prob[ids[ev]].argmax(1)!=y[ev]
    confidence=prob[ids[ev]].max(1)
    save(out,{'scope':'Previously inspected stress; exact linear score decomposition, not causal attribution',
         'inputs':{n:receipt['files'][n] for n in ['rows.parquet','projections.parquet']},
         'model_sha256':done['model_sha256'],'script_sha256':sha(__file__),'prediction_replay_max_difference':diff,
         'authentication_examples':examples,'fact_support':support,
         'wrong_prediction_confidence_quantiles':dict(zip(['minimum','median','p90','maximum'],np.quantile(confidence[wrong],[0,.5,.9,1]).tolist())),
         'wrong_predictions_over_0_95':int((wrong & (confidence>.95)).sum()),'errors':int(wrong.sum()),
         'new_training_executed':False,'fresh_blind_test':False})
    print(json.dumps({'replay_difference':diff,'examples':len(examples),'errors':int(wrong.sum()),'over_0_95_wrong':int((wrong&(confidence>.95)).sum())}),flush=True)


if __name__=='__main__': main()
