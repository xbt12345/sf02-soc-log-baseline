"""Frozen-base conditional subtype offsets with exact context fallback."""
import hashlib
import json
import math
import numpy as np
from scipy.special import expit
import v39_core as prior
import v40_core as old_parameters

VERSION='v42-frozen-conditional-offset-1.0'
MINIMUM_BODIES=3
L2=10.0
MAX_OFFSET=math.log(2.0)


def complete_context(text,facts):
    f=json.loads(facts) if isinstance(facts,str) else facts
    if f.get('action')!='deny' or f.get('outcome')!='blocked':return None,'outside_semantic_scope'
    protocol=f.get('transport_protocol')
    if protocol not in ('tcp','udp','icmp','icmp6'):return None,'unsupported_protocol'
    for field in ('src_role','dst_role'):
        if f.get(field) not in prior.learning.ENUMS[field]:return None,'missing_endpoint_role'
    required=(('src_port_fixed',65536),('dst_port_fixed',65536)) if protocol in ('tcp','udp') else (('icmp_type',256),('icmp_code',256))
    for name,stop in required:
        v=f.get(name)
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or not 0<=int(v)<stop:
            return None,'missing_or_invalid_'+name
    # Include ALL current observed semantic content. A common protocol, or a
    # known source port, cannot activate a correction on a new full context.
    raw=prior.canonical([text,f]);return hashlib.sha256(raw.encode('utf-8')).hexdigest(),'complete'


class ContextSupport:
    def fit(self,texts,facts,fit_ids,body_groups,minimum_bodies=MINIMUM_BODIES):
        if len(fit_ids)!=len(body_groups):raise ValueError('Support alignment')
        keys=[complete_context(t,f)[0] for t,f in zip(texts,facts)];support={}
        for i,body in zip(fit_ids,body_groups):
            key=keys[int(i)]
            if key is None:continue
            s=support.setdefault(key,set())
            if len(s)<minimum_bodies:s.add(int(body))
        self.minimum_bodies=minimum_bodies
        self.support_lower_bound={k:len(v) for k,v in support.items()}
        self.context_keys=sorted(k for k,v in support.items() if len(v)>=minimum_bodies)
        self.index={k:i for i,k in enumerate(self.context_keys)}
        return self

    def encode(self,texts,facts):
        ids=[];reasons=[]
        for t,f in zip(texts,facts):
            key,reason=complete_context(t,f)
            if key is None:ids.append(-1);reasons.append(reason);continue
            i=self.index.get(key,-1);ids.append(i)
            reasons.append('supported' if i>=0 else ('unseen_full_context' if key not in self.support_lower_bound else 'insufficient_distinct_bodies'))
        return np.array(ids,dtype=np.int64),reasons


def conditional_margin(prob):
    return np.log(np.maximum(prob[:,1],1e-300))-np.log(np.maximum(prob[:,2],1e-300))


def apply_offsets(base_prob,delta,eligible):
    p=np.asarray(base_prob,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):
        raise ValueError('Invalid baseline probabilities')
    d=np.asarray(delta,dtype=float);mask=np.asarray(eligible,dtype=bool)
    if d.shape!=(len(p),) or mask.shape!=(len(p),) or not np.isfinite(d).all():raise ValueError('Invalid correction')
    if (np.abs(d)>MAX_OFFSET+1e-12).any():raise ValueError('Unbounded correction')
    # Copy baseline bytes for inactive rows, zero offsets and normal decisions.
    out=p.copy();mask=mask&(d!=0)&(p.argmax(1)!=0)
    ix=np.flatnonzero(mask)
    if len(ix):
        q=expit(conditional_margin(p[ix])+d[ix]);mass=p[ix,1]+p[ix,2]
        out[ix,1]=mass*q;out[ix,2]=mass-out[ix,1]
        flip=(out[ix].argmax(1)==0)
        out[ix[flip]]=p[ix[flip]]
    assert np.array_equal(out[:,0],p[:,0])
    assert np.array_equal(out.argmax(1)==0,p.argmax(1)==0)
    assert np.array_equal(out[~np.asarray(eligible,dtype=bool)],p[~np.asarray(eligible,dtype=bool)])
    return out


class FrozenSubtypeOffset:
    def fit(self,support,texts,facts,base_prob,fit_ids,labels,l2=L2,max_offset=MAX_OFFSET):
        if l2<=0 or max_offset!=MAX_OFFSET:raise ValueError('Frozen penalty contract')
        self.support=support;self.l2=l2;self.max_offset=max_offset
        ctx,_=support.encode(texts,facts);fit_ids=np.asarray(fit_ids);labels=np.asarray(labels)
        if len(fit_ids)!=len(labels):raise ValueError('Label alignment')
        if not np.isin(labels,[0,1,2]).all():raise ValueError('Invalid labels')
        use=(ctx[fit_ids]>=0)&(labels!=0)&(np.asarray(base_prob).argmax(1)[fit_ids]!=0)
        # Exact row multiplicities; excluded normals supply no subtype target.
        units=fit_ids[use]*2+(labels[use]==1)
        unique,counts=np.unique(units,return_counts=True)
        pid=unique//2;y=unique%2;g=ctx[pid];m=conditional_margin(base_prob)[pid]
        n=len(support.context_keys)
        def grad(delta):return np.bincount(g,weights=counts*(expit(m+delta[g])-y),minlength=n)+l2*delta
        lo=np.full(n,-max_offset);hi=np.full(n,max_offset)
        for _ in range(64):
            mid=(lo+hi)/2;z=grad(mid);lo=np.where(z<0,mid,lo);hi=np.where(z>=0,mid,hi)
        self.delta=(lo+hi)/2
        empty=np.bincount(g,weights=counts,minlength=n)==0;self.delta[empty]=0.
        gradient=grad(self.delta)
        projected=np.where(self.delta<=-max_offset+1e-10,np.minimum(gradient,0),
                   np.where(self.delta>=max_offset-1e-10,np.maximum(gradient,0),gradient))
        kkt=float(np.max(np.abs(projected))) if n else 0.;assert kkt<1e-7
        def loss(d):return float(np.sum(counts*(np.logaddexp(0,m+d[g])-y*(m+d[g])))+.5*l2*np.dot(d,d))
        before=loss(np.zeros(n));after=loss(self.delta);assert after<=before+1e-8
        self.optimization={'context_parameters':n,'original_fit_rows':len(fit_ids),'subtype_fit_rows':int(use.sum()),
            'excluded_normal_label_rows':int(((ctx[fit_ids]>=0)&(labels==0)).sum()),'exact_aggregated_row_weight':int(counts.sum()),
            'objective_before':before,'objective_after':after,'projected_gradient_max':kkt,
            'offsets_at_bound':int((np.abs(self.delta)>=max_offset-1e-10).sum()),'iterations':64,
            'fit_offset_scope':'Same fitted-side data as fixed B; supervised residual fit, NOT independent probability calibration.'}
        return self

    def predict(self,texts,facts,base_prob):
        ctx,reasons=self.support.encode(texts,facts);eligible=ctx>=0;delta=np.zeros(len(ctx));delta[eligible]=self.delta[ctx[eligible]]
        return apply_offsets(base_prob,delta,eligible),eligible,reasons


def gated_reference(base_prob,reference_prob,eligible):
    # Fixed control: the old P model supplies a proposal, with identical scope,
    # maximum odds change and preserved benign probability. No new fitting.
    d=np.clip(conditional_margin(reference_prob)-conditional_margin(base_prob),-MAX_OFFSET,MAX_OFFSET)
    return apply_offsets(base_prob,d,eligible)


def predict_records(bundle,records):
    items=[prior.prepare_record(r) for r in records];texts=[v['text'] for v in items];facts=[v['facts'] for v in items]
    base=bundle['baseline'];x=old_parameters.matrix(base,texts,facts);p=base['model'].predict_proba(x)
    if bundle['view']=='G':
        reference=bundle['reference'];xp=old_parameters.matrix(reference,texts,facts)
        other=reference['model'].predict_proba(xp);ctx,_=bundle['support'].encode(texts,facts)
        return gated_reference(p,other,ctx>=0)
    return bundle['offset'].predict(texts,facts,p)[0]
