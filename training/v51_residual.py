"""Frozen multiclass base plus fact-gated, independent linear residual."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from v50_views import partial_auth

VERSION = 'v51-fact-gated-residual-1.0'

def applicable(facts):
    return np.asarray([partial_auth(f) is not None for f in facts], dtype=bool)

def loss_gradient(flat, x, offset, labels, weights, C):
    w=flat.reshape(x.shape[1],3);z=offset+x.dot(w)
    logp=z-logsumexp(z,axis=1,keepdims=True);denom=float(weights.sum())
    loss=(-np.dot(weights,logp[np.arange(len(labels)),labels])+.5*np.dot(flat,flat)/C)/denom
    residual=np.exp(logp);residual[np.arange(len(labels)),labels]-=1
    grad=(np.asarray(x.T.dot(residual*weights[:,None]))+w/C)/denom
    return float(loss),grad.ravel()

def fit_residual(base_model,x,indices,labels,weights,gate,C):
    # Nonapplicable rows contribute constant base loss and zero gradients.
    active=gate[indices];ii=indices[active];yy=labels[active];ww=weights[active]
    counts=np.bincount(ii*3+yy,weights=ww,minlength=x.shape[0]*3).reshape(-1,3)
    positions,targets=np.nonzero(counts);sample_weight=counts[positions,targets]
    columns=np.flatnonzero(np.asarray((x[np.unique(ii)]!=0).sum(axis=0)).ravel()>0)
    xx=sparse.hstack([x[positions][:,columns],np.ones((len(positions),1))],format='csr')
    offset=base_model.decision_function(x[positions]);zero=np.zeros(xx.shape[1]*3)
    before,_=loss_gradient(zero,xx,offset,targets,sample_weight,C)
    result=minimize(loss_gradient,zero,args=(xx,offset,targets,sample_weight,C),jac=True,method='L-BFGS-B',
        options={'maxiter':1000,'maxls':40,'gtol':1e-9,'ftol':1e-12})
    if not result.success or not np.isfinite(result.fun):raise RuntimeError(str(result.message))
    assert result.fun<=before+1e-12
    return {'columns':columns,'coef':result.x.reshape(xx.shape[1],3)}, {
        'converged':True,'iterations':int(result.nit),'objective_before':before,'objective_after':float(result.fun),
        'gradient_max_abs':float(abs(result.jac).max()),'message':str(result.message),
        'original_total_weight':float(weights.sum()),'active_weight':float(ww.sum()),
        'constant_base_only_weight':float(weights[~active].sum()),'aggregated_active_rows':len(positions),
        'active_class_weights_B_M_S':[float(ww[yy==k].sum()) for k in range(3)],'trainable_parameters':len(result.x)}

def predict_matrix(base_model,residual,x,gate):
    probabilities=base_model.predict_proba(x).copy()
    selected=np.flatnonzero(gate)
    if len(selected):
        xx=sparse.hstack([x[selected][:,residual['columns']],np.ones((len(selected),1))],format='csr')
        probabilities[selected]=softmax(base_model.decision_function(x[selected])+xx.dot(residual['coef']),axis=1)
    return probabilities

def classify_records(bundle,records):
    if bundle.get('version')!=VERSION:raise ValueError('Wrong residual model version')
    import v48_input
    parsed=[v48_input.prepare_record(r) for r in records]
    facts=[p['facts'] for p in parsed];texts=[p['text'] for p in parsed];base=bundle['base']
    tx=base['text_encoder'].transform(texts);fx=base['fact_encoder'].transform(facts)
    x=sparse.hstack([tx,fx],format='csr');gate=applicable(facts)
    q=predict_matrix(base['model'],bundle['residual'],x,gate)
    cold=np.asarray(bundle['original_fit_cold_fact_columns'],dtype=int);names=base['fact_encoder'].names();audits=[]
    for i,f in enumerate(facts):
        active=cold[(fx[i,cold].toarray().ravel()!=0)]
        audits.append({'residual_applied':bool(gate[i]),'text_encoded_zero':bool(texts[i]) and tx[i].nnz==0,
            'facts_without_original_fit_activation':[str(names[c]) for c in active],
            'auth_result_fit_class_counts':bundle['auth_support'].get(f.get('auth_result'),[0,0,0]),
            'scope':'Coverage, not calibrated confidence or a new class; official labels may lack counterexamples.'})
    return q,audits
