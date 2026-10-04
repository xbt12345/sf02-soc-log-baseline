"""Mathematical fusion prototype only; no official training/runtime registration."""
import torch
from torch import nn

class SharedProbabilityScorer(nn.Module):
    def __init__(self,condition_dim=0,hidden=16):
        super().__init__();self.condition_dim=condition_dim
        self.hidden=nn.Linear(11+condition_dim,hidden,dtype=torch.float64)
        self.output=nn.Linear(hidden,1,bias=False,dtype=torch.float64)
        nn.init.zeros_(self.output.weight)

    def forward(self,p,condition=None):
        if p.ndim!=3 or p.shape[-1]!=3 or not torch.isfinite(p).all():raise ValueError('Complete member three-class probabilities required')
        if (p<0).any() or not torch.allclose(p.sum(-1),torch.ones_like(p[:,:,0]),atol=1e-12,rtol=0):raise ValueError('Invalid probabilities')
        entropy=-(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1,keepdim=True)
        common=torch.cat([p.mean(1),p.var(1,unbiased=False),entropy.mean(1)],-1)
        features=torch.cat([p,common[:,None,:].expand(-1,p.shape[1],-1),entropy],-1)
        if self.condition_dim:
            if condition is None or condition.shape!=(len(p),self.condition_dim):raise ValueError('Condition shape changed')
            features=torch.cat([features,condition[:,None,:].expand(-1,p.shape[1],-1)],-1)
        elif condition is not None:raise ValueError('A scorer cannot consume observation conditions')
        scores=self.output(torch.tanh(self.hidden(features))).squeeze(-1)
        weights=torch.softmax(scores,1)
        return (weights[:,:,None]*p).sum(1),weights
