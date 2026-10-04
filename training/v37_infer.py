"""Message-only inference for the two explicitly named paired models."""
from scipy import sparse
import v36_representation as old
import v37_representation as new
import v37_learning as learning

def predict(bundle,records):
    rep=old if bundle['view']=='B2_CONTROL' else new
    if bundle['view'] not in ('B2_CONTROL','B2_REPAIRED'):raise ValueError('Unknown view')
    if bundle['representation_version']!=new.VERSION:raise ValueError('Paired version differs')
    rows=[rep.prepare_record(r) for r in records]
    x=sparse.hstack([bundle['tfidf'].transform([r['b1'] for r in rows]),
                    bundle['facts'].transform([r['facts'] for r in rows])],format='csr')
    return bundle['model'].predict_proba(x)

def classify(bundle,records):
    """Frozen primary three-class decision; argmax is a separate diagnostic."""
    p=predict(bundle,records)
    policy=next(x for x in bundle['policies'] if x['alpha']==.001)
    threshold=policy['empirical_worst_source_global']
    return {'probabilities':p,'prediction':learning.gated_predictions(p,threshold),
            'raw_argmax':p.argmax(1),'alarm':1-p[:,0]>=threshold,'threshold':threshold}
