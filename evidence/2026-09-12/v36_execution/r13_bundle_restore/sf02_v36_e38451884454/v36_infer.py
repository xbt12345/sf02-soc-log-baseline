"""Single frozen message-only inference route; outer fields cannot route models."""
import json
from scipy import sparse
import numpy as np
import v331_prepare as old
import v36_representation as rep


def predict(bundle,records):
    view=bundle['view']
    if view=='B0':
        text=[old.prepare_record({'message_sanitized':r.get('message_sanitized')})['text'] for r in records]
        facts=None
    else:
        if bundle['representation_version']!=rep.VERSION:
            raise ValueError('Representation version mismatch')
        prepared=[rep.prepare_record(r) for r in records]
        text=[r['b1'] for r in prepared];facts=[r['facts'] for r in prepared]
    x=bundle['tfidf'].transform(text)
    if view=='B2':
        x=sparse.hstack([x,bundle['facts'].transform(facts)],format='csr')
    return bundle['model'].predict_proba(x)
