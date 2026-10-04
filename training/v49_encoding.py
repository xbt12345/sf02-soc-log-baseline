"""Observed-value encoding and visible evidence coverage; frozen v48 parsing.

An observation flag distinguishes observed zero from unavailable. This still
reveals availability and is not claimed to eliminate missingness shortcuts.
Absent cached slots do not establish whether a raw slot was absent or redacted.
"""
import json
import numpy as np
from scipy import sparse
import v48_input as input_adapter

VERSION = 'v49-observed-value-encoding-1.0'
FINITE = input_adapter.old.learning.BIT_FIELDS


def finite_observation(f, key):
    limit = FINITE[key][1]
    if key not in f:
        return None
    value = f[key]
    if key in ('status', 'substatus') and isinstance(value, str):
        value = int(value, 16)
    if isinstance(value, bool) or int(value) != value or not 0 <= value <= limit:
        raise ValueError('Invalid finite observation: ' + key)
    return None if int(value) == limit else int(value)


class EvidenceFacts:
    """No finite missing sentinel contributes as an observed value."""
    def __init__(self):
        self.base = input_adapter.old.SemanticFacts()
        self.keys = list(FINITE)

    def fit(self, facts):
        self.base.fit(facts)
        self.original_names = self.base.names()
        self.blocks = {k: np.flatnonzero(np.char.startswith(self.original_names.astype(str), k + ':bit')) for k in self.keys}
        return self

    def transform(self, facts):
        facts = [json.loads(f) if isinstance(f, str) else f for f in facts]
        for f in facts:
            if any(s.startswith('unencoded') for s in input_adapter.fact_dispositions(f).values()):
                raise ValueError('Unaccounted fact cannot silently disappear')
        x = self.base.transform(facts).tocsc()
        observed = np.array([[finite_observation(f, k) is not None for k in self.keys] for f in facts], dtype=float)
        for j, key in enumerate(self.keys):
            for col in self.blocks[key]:
                start, end = x.indptr[col:col + 2]
                x.data[start:end] *= observed[x.indices[start:end], j]
        x.eliminate_zeros()
        return sparse.hstack([x, sparse.csr_matrix(observed)], format='csr')

    def names(self):
        return np.concatenate([self.original_names, np.array(['observed:' + k for k in self.keys])])


def classify_records(bundle, rows):
    if bundle.get('version') != VERSION:
        raise ValueError('Wrong model/encoder version')
    records = [input_adapter.prepare_record(row) for row in rows]
    texts = [r['text'] for r in records]; facts = [r['facts'] for r in records]
    tx = bundle['text_encoder'].transform(texts); fx = bundle['fact_encoder'].transform(facts)
    x = sparse.hstack([tx, fx], format='csr')
    probabilities = bundle['model'].predict_proba(x)
    cold = np.asarray(bundle['unseen_fit_fact_columns'], dtype=int)
    active = (fx[:, cold] != 0).tocsr()
    names = bundle['fact_encoder'].names()
    audits = []
    for i, r in enumerate(records):
        missing = cold[active.indices[active.indptr[i]:active.indptr[i+1]]]
        audits.append({'input_audit': r['information_audit'],
            'nonempty_text_encoded_as_zero': bool(texts[i]) and tx[i].nnz == 0,
            'facts_without_fit_activation': [str(names[j]) for j in missing],
            'scope': 'Coverage warnings do not change the required three-class prediction or establish error probability.'})
    return probabilities, audits
