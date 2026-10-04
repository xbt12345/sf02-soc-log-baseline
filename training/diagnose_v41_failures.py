"""Read-only feature/margin attribution for observed residual failures."""
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pyarrow.parquet as pq


def main(a):
    root=Path(a.root).resolve();run=Path(a.run).resolve();sys.path.insert(0,str(run/'frozen_training_runtime'))
    import v41_core as core
    from run_v39_prepare import save,sha
    old=root/'artifacts/v39_local_r2_20260913';pr=pq.read_table(old/'prepared/projections.parquet').to_pandas()
    rows=pq.read_table(old/'prepared/rows.parquet',columns=['row_position','projection_id','fold','label_index']).to_pandas()
    native=json.loads((run/'primary_review/candidate_A.json').read_text(encoding='utf-8'))['critical_regressed_rows']
    audit=[]
    for pos in native:
        row=rows.iloc[pos];fold=int(row.fold);pid=int(row.projection_id);b=joblib.load(old/('primary/fold_%s/SEMANTIC/model.joblib'%fold));a1=joblib.load(run/('primary/fold_%s/A/model.joblib'%fold))
        text=pr.text.iloc[pid];facts=json.loads(pr.facts.iloc[pid]);xb=core.matrix(b,[text],[facts]);xa=core.matrix(a1,[text],[facts])
        def named(bundle,x):
            names=['text:'+str(n) for n in bundle['text_encoder'].names()]+['fact:'+str(n) for n in bundle['fact_encoder'].names()]
            z=x.tocoo();return {names[int(j)]:float(v) for j,v in zip(z.col,z.data)}
        nb,na=named(b,xb),named(a1,xa);diff=max(abs(nb.get(k,0)-na.get(k,0)) for k in set(nb)|set(na))
        zb=b['model'].decision_function(xb)[0];za=a1['model'].decision_function(xa)[0]
        audit.append({'row_position':pos,'fold':fold,'facts':facts,'text':text,'same_named_feature_values_max_difference':diff,
            'B_probability':b['model'].predict_proba(xb)[0].tolist(),'A_probability':a1['model'].predict_proba(xa)[0].tolist(),
            'B_suspicious_minus_benign_logit':float(zb[2]-zb[0]),'A_suspicious_minus_benign_logit':float(za[2]-za[0])})
    parameter=[]
    for fold in range(3):
        changes=pq.read_table(run/'primary_review/changes_P.parquet').to_pandas()
        ev=pq.read_table(run/('primary/fold_%s/P/evaluation.parquet'%fold)).to_pandas()
        take=ev[(ev.route=='asa')&ev.unseen_parameter&ev.row_position.isin(changes.loc[changes.change=='regressed','row_position'])].drop_duplicates('projection_id')
        if take.empty:continue
        b=joblib.load(old/('primary/fold_%s/SEMANTIC/model.joblib'%fold));p=joblib.load(run/('primary/fold_%s/P/model.joblib'%fold))
        for r in take.itertuples(index=False):
            pid=int(r.projection_id);text=pr.text.iloc[pid];facts=json.loads(pr.facts.iloc[pid]);xb=core.matrix(b,[text],[facts]);xp=core.matrix(p,[text],[facts]);n=xb.shape[1]
            oldz=b['model'].decision_function(xb)[0];retuned=(xb@p['model'].coef_[:,:n].T+p['model'].intercept_)[0]
            added=(xp[:,n:]@p['model'].coef_[:,n:].T)[0];assert np.allclose(retuned+added,p['model'].decision_function(xp)[0])
            truth=int(r.label_index);other=int(p['model'].predict_proba(xp)[0].argmax())
            z=xp[:,n:].tocoo();effects=[]
            for j,value in zip(z.col,z.data):
                effects.append({'term':list(p['parameter_encoder'].term_names[int(j)]),
                    'true_minus_competitor_logit_contribution':float(value*(p['model'].coef_[truth,n+j]-p['model'].coef_[other,n+j]))})
            parameter.append({'fold':fold,'projection_id':pid,'facts':facts,'true_class':truth,'P_predicted_class':other,
                'B_true_minus_competitor_margin':float(oldz[truth]-oldz[other]),
                'change_from_refitted_original_coefficients':float((retuned[truth]-retuned[other])-(oldz[truth]-oldz[other])),
                'added_parameter_term_margin':float(added[truth]-added[other]),
                'P_true_minus_competitor_margin':float((retuned+added)[truth]-(retuned+added)[other]),'active_parameter_terms':effects})
    save(run/'failure_mechanism_review.json',{'A_cross_family_regressions':audit,'P_unseen_parameter_regression_margins':parameter,
        'A_inputs_identical_by_named_feature_values':all(r['same_named_feature_values_max_difference']==0 for r in audit),
        'scope':'Observed saved-model scores only; margin blocks are exact linear accounting, not causal threat explanations.',
        'script_sha256':sha(__file__)})
    print(json.dumps({'A':audit,'P':parameter}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True);main(p.parse_args())
