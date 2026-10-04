"""A fixed prior; B a single shared exp(beta), complete unchanged residual."""
import numpy as np
import torch
from torch import nn
from v159_current_input_boundary_v3 import CurrentInputBoundary as Original,FLOOR
from v159_float64_repeat_policy_v2 import EPS,REPEAT_EPS,SEGMENTS,repeat_gradient as original_repeat

class PriorPairBoundary(Original):
    def __init__(self,arm,width=66287,hidden=16,seed=15901):
        if arm not in ('A','B'):raise ValueError('Fixed declared arm required')
        super().__init__(width,hidden,seed);self.arm=arm
        if arm=='B':self.beta=nn.Parameter(torch.zeros((),dtype=torch.float64))

    def forward(self,x,p):
        if p.ndim!=3 or p.shape[1:]!=(16,3) or x.ndim!=2 or x.shape!=(len(p),self.width):raise ValueError('Complete input required')
        if x.dtype!=torch.float64 or p.dtype!=torch.float64:raise ValueError('Explicit float64 required')
        values=x if x.layout==torch.strided else x.values()
        if not torch.isfinite(values).all() or not torch.isfinite(p).all() or (p<0).any() or (p>1).any() or (p.sum(-1)-1).abs().max()>1e-12:raise ValueError('Invalid complete data')
        mean,features=self.opinions(p)
        obs=x@self.observation_weight if x.layout==torch.strided else torch.sparse.mm(x,self.observation_weight)
        hidden=torch.tanh(obs[:,None,:]+features@self.opinion_weight+self.bias)
        delta=(hidden@self.output_weight).mean(1)
        prior=mean.clamp_min(FLOOR).log()
        if self.arm=='B':
            coefficient=self.beta.exp()
            if not torch.isfinite(coefficient) or not coefficient>0:raise FloatingPointError('Unrepresentable exp(beta), no clipping or fallback')
            prior=coefficient*prior
        z=prior+delta;lp=torch.log_softmax(z,dim=-1);q=torch.softmax(z,dim=-1)
        if not torch.isfinite(lp).all() or not torch.isfinite(q).all():raise FloatingPointError('Nonfinite pair output')
        return q,lp,delta

def schema(arm):
    if arm not in ('A','B'):raise ValueError('Unknown arm')
    return list(SEGMENTS)+([('beta',1060832,1060833)] if arm=='B' else [])

def width(arm):return schema(arm)[-1][2]

def validate_schema(model):
    expected=schema(model.arm);observed=[];offset=0
    for name,p in model.named_parameters():observed.append((name,offset,offset+p.numel()));offset+=p.numel()
    if observed!=expected:raise ValueError('Complete parameter names/order/width mismatch')
    return expected

def gradient_repeat(a,b,arm):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    if a.shape!=(width(arm),) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():return dict(passed=False,reason='complete_pair_parameter_shape_or_nonfinite')
    old=original_repeat(a[:1060832],b[:1060832])
    if arm=='A':return old
    x,y=float(a[-1]),float(b[-1]);scale=max(abs(x),abs(y));gap=abs(x-y);limit=REPEAT_EPS*EPS*scale
    sign_change=scale>limit and np.sign(x)!=np.sign(y)
    passed=gap<=limit and not sign_change
    extra=dict(segment='beta',passed=bool(passed),maximum_absolute=gap,own_scale=scale,maximum_scaled_eps=gap/(EPS*scale) if scale else 0.,relative_L2=gap/scale if scale else 0.,resolved_sign_changes=int(sign_change),exact_zero=scale==0)
    return dict(passed=bool(old['passed'] and passed),segments=[*old['segments'],extra])

def load_origin(model,state):
    old={k:v for k,v in state.items()}
    if model.arm=='B':
        if 'beta' in old:raise ValueError('Original baseline must have no learned prior parameter')
        old['beta']=torch.zeros((),dtype=torch.float64)
    model.load_state_dict(old,strict=True);validate_schema(model)
    if model.arm=='B' and float(model.beta.detach())!=0.:raise ValueError('Exact zero beta required')
