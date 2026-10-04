"""Observed-value encoding and bounded, label-free categorical interactions."""
import json
import math
import numpy as np
from scipy import sparse
import v39_core as prior

VERSION = 'v40-observed-interactions-1.0'
FINITE = prior.learning.BIT_FIELDS
MAIN_FIELDS = ('src_port_fixed', 'dst_port_fixed', 'icmp_type', 'icmp_code')
PAIR_FIELDS = (('transport_protocol','dst_port_fixed'), ('transport_protocol','src_port_fixed'),
               ('icmp_type','icmp_code'), ('action','transport_protocol'))


def canonicalize(value):
    f = dict(json.loads(value) if isinstance(value, str) else value)
    audit = {'merged_aliases': [], 'unobserved_codes': []}
    if f.get('auth_result') in ('success','failure') and f.get('outcome') == f['auth_result']:
        f.pop('outcome'); audit['merged_aliases'].append('outcome->auth_result')
    # Both fields here describe the same observed lack of user interaction.
    # They are not missing input metadata. Keep the specific observation once.
    if f.get('authentication_interaction') == 'no_response' and f.get('response') == 'missing':
        f.pop('response'); audit['merged_aliases'].append('response->authentication_interaction')
    for k, (width, sentinel) in FINITE.items():
        if k not in f:
            audit['unobserved_codes'].append(k)
            continue
        n = f[k]
        if k in ('status','substatus') and isinstance(n,str):
            n = int(n,16)
        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,float,np.integer,np.floating)) or not math.isfinite(n) or int(n) != n or not 0 <= int(n) <= sentinel:
            raise ValueError('Invalid finite observation '+k+': '+repr(n))
        if int(n) == sentinel:
            f.pop(k); audit['unobserved_codes'].append(k)
        else:
            f[k] = int(n)
    return f, audit


def observed_bits(values, field):
    width, sentinel = FINITE[field]
    result = np.zeros((len(values),width),dtype=np.float64)
    for i,n in enumerate(values):
        if n is None or (not isinstance(n,(bool,np.bool_)) and n == sentinel):
            continue
        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,float,np.integer,np.floating)) or not math.isfinite(n) or int(n)!=n or not 0<=int(n)<sentinel:
            raise ValueError('Invalid finite observation')
        result[i] = ((int(n)+1) >> np.arange(width,dtype=np.int64)) & 1
    return result


class NeutralFacts:
    def __init__(self):
        self.legacy = prior.SemanticFacts()
        oldnames = self.legacy.names()
        self.columns = {k:np.array([int(np.flatnonzero(oldnames == k+':bit'+str(b))[0]) for b in range(width)]) for k,(width,_) in FINITE.items()}
        self.finite_columns = set(int(c) for cols in self.columns.values() for c in cols)
        self.feature_names = oldnames.copy()
        for k,cols in self.columns.items():
            for bit,c in enumerate(cols): self.feature_names[c] = 'observed_code:'+k+':bit'+str(bit)

    def fit(self, values):
        self.legacy.fit([canonicalize(v)[0] for v in values])
        return self

    def transform(self, values):
        values = [canonicalize(v)[0] for v in values]
        old = self.legacy.transform(values).tocoo()
        keep = ~np.isin(old.col, list(self.finite_columns))
        rr, cc, dd = [old.row[keep]], [old.col[keep]], [old.data[keep]]
        for k,cols in self.columns.items():
            bits = observed_bits([f.get(k) for f in values], k)
            r,b = np.nonzero(bits)
            rr.append(r); cc.append(cols[b]); dd.append(np.ones(len(r)))
        return sparse.csr_matrix((np.concatenate(dd),(np.concatenate(rr),np.concatenate(cc))),shape=old.shape)

    def names(self): return self.feature_names


def terms(value):
    f = canonicalize(value)[0]
    mains = [('main',k,str(f[k])) for k in MAIN_FIELDS if k in f]
    pairs = []
    for a,b in PAIR_FIELDS:
        if a not in f or b not in f: continue
        if a == 'icmp_type' and f.get('transport_protocol') not in ('icmp','icmp6'): continue
        if a == 'icmp_type':
            pair = ('pair',a,b,str(f[a]),str(f[b]),f['transport_protocol'])
        else:
            pair = ('pair',a,b,str(f[a]),str(f[b]))
        pairs.append(pair)
    return mains, pairs


class ParameterEffects:
    def fit(self, facts, fit_projection_ids, fit_body_groups, minimum_bodies=3):
        if len(fit_projection_ids)!=len(fit_body_groups): raise ValueError('Support alignment')
        self.minimum_bodies = minimum_bodies
        mains = set(); support = {}
        cached = {int(i):terms(facts[i]) for i in np.unique(fit_projection_ids)}
        for m, pairs in cached.values(): mains.update(m)
        # Saturate support after threshold: records the required lower bound,
        # never pretends to count all independent events.
        for pid,body in zip(fit_projection_ids,fit_body_groups):
            for pair in cached[int(pid)][1]:
                s = support.setdefault(pair,set())
                if len(s)<minimum_bodies: s.add(int(body))
        eligible = {k for k,v in support.items() if len(v)>=minimum_bodies}
        self.support_lower_bounds = {k:len(v) for k,v in support.items()}
        self.term_names = sorted(mains|eligible)
        self.index = {k:i for i,k in enumerate(self.term_names)}
        self.main_count = len(mains); self.pair_count = len(eligible)
        return self

    def transform(self, facts):
        rr=[];cc=[]
        for i,f in enumerate(facts):
            mains,pairs = terms(f)
            for t in mains+pairs:
                j = self.index.get(t)
                if j is not None: rr.append(i);cc.append(j)
        return sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(facts),len(self.index)))

    def coverage(self, facts):
        result=[]
        for f in facts:
            mains,pairs=terms(f)
            result.append({'has_parameter':bool(mains),'unseen_parameter':any(t not in self.index for t in mains),
                           'has_joint_value':bool(pairs),'unseen_joint_value':any(t not in self.support_lower_bounds for t in pairs),
                           'unsupported_joint_value':any(t not in self.index for t in pairs)})
        return result

    def names(self):
        return np.array([json.dumps(t,separators=(',',':')) for t in self.term_names],dtype=object)


def prepare_message(raw):
    p = prior.prepare_message(raw)
    f,audit = canonicalize(p['facts'])
    return {'text':p['text'],'facts':f,'audit':{'v39':p['audit'],'v40':audit}}


def matrix(bundle, texts, facts):
    pieces=[bundle['text_encoder'].transform(texts),bundle['fact_encoder'].transform(facts)]
    if bundle.get('parameter_encoder') is not None:pieces.append(bundle['parameter_encoder'].transform(facts))
    return sparse.hstack(pieces,format='csr')


def predict_records(bundle, records):
    # R must replay the original projection, including its original aliases.
    parser=prior.prepare_record if bundle['view']=='R' else lambda r:prepare_message(prior.previous.prior.base.string(r.get('message_sanitized')))
    items=[parser(r) for r in records]
    x=matrix(bundle,[v['text'] for v in items],[v['facts'] for v in items])
    return bundle['model'].predict_proba(x)
