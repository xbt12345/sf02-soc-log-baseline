"""V125 SparseTabM plus an optional small ordered-body byte branch."""
import math

import torch
from torch import nn

from v104_phase_b import SparseTabM, csr_tensor
from v75_views import BYTE_FEATURES


class BodyBranch(nn.Module):
    def __init__(self, maxlen=176):
        super().__init__()
        width=64
        self.embed=nn.Embedding(257,width,padding_idx=0)
        layer=nn.TransformerEncoderLayer(d_model=width,nhead=4,dim_feedforward=128,
                                          dropout=0.0,activation='gelu',batch_first=True,
                                          norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        pos=torch.arange(maxlen,dtype=torch.float32).unsqueeze(1)
        k=torch.arange(0,width,2,dtype=torch.float32)
        angle=pos*torch.exp(-math.log(10000)*k/width)
        pe=torch.zeros(maxlen,width,dtype=torch.float32)
        pe[:,0::2]=angle.sin();pe[:,1::2]=angle.cos()
        self.register_buffer('position',pe,persistent=True)
        self.head=nn.Linear(width,3)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self, raw_bytes, lengths):
        n=raw_bytes.shape[1]
        pad=torch.arange(n,device=raw_bytes.device)[None,:]>=lengths[:,None]
        ids=raw_bytes.long()+1
        ids=ids.masked_fill(pad,0)
        h=self.embed(ids)+self.position[:n]
        h=self.encoder(h,src_key_padding_mask=pad)
        pooled=h.masked_fill(pad[:,:,None],0).sum(1)/lengths[:,None]
        return self.head(pooled)


class Expert(nn.Module):
    def __init__(self,arm,seed):
        super().__init__()
        if arm not in ('A','B','C'):raise ValueError(arm)
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        self.base=SparseTabM()
        self.arm=arm
        if arm!='A':
            # Base state is identical before allocating the additional branch.
            torch.manual_seed(seed+981)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(seed+981)
            self.branch=BodyBranch()
        else:self.branch=None

    def forward(self,x,raw_bytes=None,lengths=None):
        device=next(self.base.parameters()).device
        fact=torch.as_tensor(x[:,BYTE_FEATURES:].toarray(),device=device,dtype=torch.float32)
        logits=self.base(csr_tensor(x,device),fact)
        if self.branch is not None:
            if raw_bytes is None or lengths is None:raise ValueError('Missing ordered-body input')
            extra=self.branch(raw_bytes,lengths)
            logits=logits+extra[:,None,:]
        return logits


def probabilities(model,x,body=None,lengths=None,device='cuda',batch_size=256):
    model.eval();result=torch.empty((x.shape[0],3),dtype=torch.float32,device='cpu')
    with torch.no_grad():
        for start in range(0,x.shape[0],batch_size):
            end=min(start+batch_size,x.shape[0]);ids=slice(start,end)
            if model.branch is not None:
                width=int(lengths[ids].max())
                b=torch.as_tensor(body[ids,:width],device=device)
                l=torch.as_tensor(lengths[ids],device=device,dtype=torch.int64)
                z=model(x[ids],b,l)
            else:z=model(x[ids])
            result[ids]=torch.softmax(z,dim=-1).mean(1).cpu()
    return result.numpy()
