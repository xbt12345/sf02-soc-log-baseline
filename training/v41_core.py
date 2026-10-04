"""Single-factor contrasts over the exact v39 semantic projection."""
import json
import numpy as np
from scipy import sparse
import v40_core as previous

prior = previous.prior
VERSION = 'v41-isolated-contrasts-1.0'
ParameterEffects = previous.ParameterEffects
FINITE = previous.FINITE
MAIN_FIELDS = previous.MAIN_FIELDS
PAIR_FIELDS = previous.PAIR_FIELDS
canonicalize = previous.canonicalize  # Audit/coverage only, never a P/F/D parser.
matrix = previous.matrix


def deduplicate(value):
    f = dict(json.loads(value) if isinstance(value, str) else value)
    if f.get('auth_result') in ('success', 'failure') and f.get('outcome') == f['auth_result']:
        f.pop('outcome')
    if f.get('authentication_interaction') == 'no_response' and f.get('response') == 'missing':
        f.pop('response')
    return f


class FiniteOnlyFacts(previous.NeutralFacts):
    def fit(self, values):
        self.legacy.fit(values)
        return self

    def transform(self, values):
        # The legacy transform receives ORIGINAL facts, including every alias.
        original = [json.loads(v) if isinstance(v, str) else v for v in values]
        finite = [previous.canonicalize(v)[0] for v in original]
        old = self.legacy.transform(original).tocoo()
        keep = ~np.isin(old.col, list(self.finite_columns))
        rr, cc, dd = [old.row[keep]], [old.col[keep]], [old.data[keep]]
        for k, cols in self.columns.items():
            bits = previous.observed_bits([f.get(k) for f in finite], k)
            r, b = np.nonzero(bits)
            rr.append(r); cc.append(cols[b]); dd.append(np.ones(len(r)))
        return sparse.csr_matrix((np.concatenate(dd), (np.concatenate(rr), np.concatenate(cc))), shape=old.shape)


class DeduplicatedFacts:
    def __init__(self):
        self.legacy = prior.SemanticFacts()

    def fit(self, values):
        self.legacy.fit([deduplicate(v) for v in values])
        return self

    def transform(self, values):
        return self.legacy.transform([deduplicate(v) for v in values])

    def names(self):
        return self.legacy.names()


def isolation_check(view, baseline, replacement, baseline_fact_names, text_columns):
    """Check ALL active projections, not just a synthetic example."""
    changed = replacement[:, :baseline.shape[1]] - baseline
    cols = set(int(c) for c in changed.tocoo().col)
    if view == 'P':
        allowed = set()
    elif view == 'F':
        allowed = {text_columns+i for i, n in enumerate(baseline_fact_names)
                   if any(str(n).startswith(k+':bit') for k in FINITE)}
    elif view == 'D':
        allowed = {text_columns+i for i, n in enumerate(baseline_fact_names)
                   if n in ('outcome=failure', 'outcome=success', 'response=missing')}
    else:
        raise ValueError(view)
    assert cols <= allowed, (view, cols-allowed)
    return {'all_active_projections': baseline.shape[0], 'changed_original_columns': len(cols),
            'unauthorized_changed_columns': 0, 'added_columns': replacement.shape[1]-baseline.shape[1]}


def predict_records(bundle, records):
    if bundle['view'] == 'A':
        from v41_native_auth import prepare_record
    else:
        prepare_record = prior.prepare_record
    items = [prepare_record(r) for r in records]
    x = matrix(bundle, [v['text'] for v in items], [v['facts'] for v in items])
    return bundle['model'].predict_proba(x)
