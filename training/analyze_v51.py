"""Post-fit attribution, support and saved-objective reconstruction."""
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp

def probability_metrics(y,p):
    return {'log_loss':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-300,1)).mean()),
        'multiclass_brier_sum':float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1)))}

def main(root,out):
    sys.path.insert(0,str(out/'runtime'))
    from run_v48 import save,read,sha
    from run_v49 import data,metrics
    import v51_residual as v
    import v50_views
    rows,ids,texts,facts=data(root);summary=[];audit=[]
    names=['pressure']+(['fold_'+str(f) for f in range(3)] if read(out/'execution.json')['primary_executed'] else [])
    for name in names:
        folder=out/name;b=joblib.load(folder/'model.joblib');base=b['base'];fold=b['fold'];fit=rows.inner_role.to_numpy()!=2 if fold is None else rows.fold.to_numpy()!=fold
        te=base['text_encoder'];fe=base['fact_encoder'];x=sparse.hstack([te.transform(texts),fe.transform(facts)],format='csr')
        partial,indices,targets,weights,origin,eligible=v50_views.training_views(facts,ids[fit],rows.label_index.to_numpy()[fit])
        xx=sparse.vstack([x,sparse.hstack([te.transform(['']*len(partial)),fe.transform(partial)],format='csr')],format='csr')
        gate=np.r_[v.applicable(facts),v.applicable(partial)];active=gate[indices]
        supported=np.flatnonzero(np.asarray((xx[np.unique(indices[active])]!=0).sum(axis=0)).ravel()>0)
        np.testing.assert_array_equal(supported,b['residual']['columns'])
        count=np.bincount(indices[active]*3+targets[active],weights=weights[active],minlength=xx.shape[0]*3).reshape(-1,3)
        # Reconstruct the likelihood from class-count sufficient statistics,
        # independently of optimizer's separate input/label rows.
        where=np.flatnonzero(count.sum(axis=1)>0);a=sparse.hstack([xx[where][:,supported],np.ones((len(where),1))],format='csr')
        offset=base['model'].decision_function(xx[where]);coef=b['residual']['coef'];z=offset+a.dot(coef);logp=z-logsumexp(z,axis=1,keepdims=True)
        normalizer=count.sum();C=read(out/'configuration.json')['C']
        objective=float((-np.sum(count[where]*logp)+.5*np.sum(coef**2)/C)/normalizer)
        gradient=(np.asarray(a.T.dot(np.exp(logp)*count[where].sum(axis=1,keepdims=True)-count[where]))+coef/C)/normalizer
        opt=read(folder/'report.json')['optimizer'];assert abs(objective-opt['objective_after'])<1e-12
        assert abs(float(abs(gradient).max())-opt['gradient_max_abs'])<1e-12
        d=pd.read_parquet(folder/'evaluation.parquet');y=d.label_index.to_numpy();new=d[['p_benign','p_malicious','p_suspicious']].to_numpy();old=d[['b_benign','b_malicious','b_suspicious']].to_numpy();selected=d.residual_applicable.to_numpy()
        summary.append({'model':name,'classification':metrics(d),'applicable_eval_rows':int(selected.sum()),
            'applicable_eval_class_counts_B_M_S':[int(((y==k)&selected).sum()) for k in range(3)],
            'full_probabilities_before':probability_metrics(y,old),'full_probabilities_after':probability_metrics(y,new),
            'applicable_probabilities_before':probability_metrics(y[selected],old[selected]),'applicable_probabilities_after':probability_metrics(y[selected],new[selected]),
            'exactly_equal_nonapplicable_probabilities':bool(np.array_equal(old[~selected],new[~selected])),
            'optimizer_objective_recomputed':objective,'optimizer_gradient_max_recomputed':float(abs(gradient).max()),
            'residual_columns_from_applicable_fit_views_only':True})
        if name=='pressure':
            fullq=v.predict_matrix(base['model'],b['residual'],x,v.applicable(facts));before=base['model'].predict_proba(x)
            for k in np.unique(ids[rows.route.to_numpy()=='authentication']):
                audit.append({'text':texts[k],'facts':facts[k],'rows':int(((ids==k)&(~fit)).sum()),'before':before[k].tolist(),'after':fullq[k].tolist()})
            reduced=v.predict_matrix(base['model'],b['residual'],xx[-len(partial):],v.applicable(partial))
            old_reduced=base['model'].predict_proba(xx[-len(partial):]);probe=[]
            for f,p,q in zip(partial,old_reduced,reduced):
                if f.get('credential_check')=='invalid':probe.append({'facts':f,'before':p.tolist(),'after':q.tolist()})
            invalid_probe={'fit_rows':int(sum(facts[k].get('credential_check')=='invalid' for k in ids[fit])),'cases':probe,'scope':'Fit-side partial-view diagnostic, not unseen classification.'}
    e=root/'evidence/2026-09-14/v51_review';e.mkdir(parents=True,exist_ok=True)
    save(e/'summary.json',{'models':summary,'pressure_authentication':audit,'invalid_credential_probe':invalid_probe,
        'scope':'Previously inspected official development data; probability losses do not establish calibration or new-environment accuracy.',
        'analysis_source_sha256':sha(Path(__file__))})
    (e/'analyze_v51.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'models_checked':len(names),'pressure_auth':audit,'saved_objectives_reconstructed':True}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root.resolve(),a.out.resolve())
