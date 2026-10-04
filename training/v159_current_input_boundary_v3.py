"""Single refined pre-registration candidate, fixed 1e-12 origin floor.

Current complete input and 16 current role-legal probabilities only. Earlier
member-logit prototype stays historical; no official comparison selected this.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from v159_current_input_boundary import original_class_risk,clipped_probability_class_ce,flattened_class_gradient

FLOOR=1e-12
ORIGIN_TOLERANCE=3e-12

class CurrentOpinionFeatures(nn.Module):
    def forward(self,p):
        entropy=-(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1,keepdim=True)
        mean=p.mean(1)
        common=torch.cat([mean,p.var(1,unbiased=False),entropy.mean(1)],-1)
        return mean,torch.cat([p,common[:,None,:].expand(-1,16,-1),entropy],-1)

class CurrentInputBoundary(nn.Module):
    def __init__(self,width=66287,hidden=16,seed=15901):
        super().__init__()
        if width<=0 or hidden<=0:raise ValueError('Positive widths required')
        self.width,self.hidden=width,hidden
        self.opinions=CurrentOpinionFeatures()
        rng=torch.Generator(device='cpu').manual_seed(seed)
        self.observation_weight=nn.Parameter(torch.randn(width,hidden,generator=rng,dtype=torch.float64)/math.sqrt(width))
        self.opinion_weight=nn.Parameter(torch.randn(11,hidden,generator=rng,dtype=torch.float64)/math.sqrt(11))
        self.bias=nn.Parameter(torch.zeros(hidden,dtype=torch.float64))
        self.output_weight=nn.Parameter(torch.zeros(hidden,3,dtype=torch.float64))

    def forward(self,x,p):
        if p.ndim!=3 or p.shape[1:]!=(16,3) or x.ndim!=2 or x.shape!=(len(p),self.width):raise ValueError('Complete input and exactly16 current members required')
        if x.dtype!=torch.float64 or p.dtype!=torch.float64:raise ValueError('Explicit float64 conversion required')
        values=x if x.layout==torch.strided else x.values()
        if not torch.isfinite(values).all() or not torch.isfinite(p).all() or (p<0).any() or (p>1).any() or (p.sum(-1)-1).abs().max()>1e-12:raise ValueError('Invalid data may not be repaired by flooring')
        mean,features=self.opinions(p)
        obs=x@self.observation_weight if x.layout==torch.strided else torch.sparse.mm(x,self.observation_weight)
        hidden=torch.tanh(obs[:,None,:]+features@self.opinion_weight+self.bias)
        delta=(hidden@self.output_weight).mean(1)
        # Stable log_softmax is the training output, never log(clamped new q).
        z=mean.clamp_min(FLOOR).log()+delta
        lp=torch.log_softmax(z,dim=-1);q=torch.softmax(z,dim=-1)
        if not torch.isfinite(lp).all() or not torch.isfinite(q).all():raise ValueError('Nonfinite boundary output')
        return q,lp,delta
