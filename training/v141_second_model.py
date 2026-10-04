"""Adapt existing second layer only, retain actual cached zero-step numerics."""
import torch
from torch import nn


class SecondRepresentation(nn.Module):
    def __init__(self, backbone_state, readout_state):
        super().__init__()
        for key in ['weight','r','s','bias']:
            value=backbone_state['second.'+key].detach().clone().double()
            self.register_parameter(key,nn.Parameter(value.clone()))
            self.register_buffer('reference_'+key,value)
        for key in ['weight','bias','facts']:
            self.register_buffer('head_'+key,readout_state[key].detach().clone().double())

    def features(self, packed):
        h1,h2=packed.split(128,dim=-1)
        changed=torch.relu(((h1*self.r)@self.weight.T)*self.s+self.bias)
        with torch.no_grad():
            reference=torch.relu(((h1*self.reference_r)@self.reference_weight.T)*self.reference_s+self.reference_bias)
        # The baseline h2 is original float32 GEMM cached then cast. Anchoring
        # preserves its rounding exactly while learning an existing-layer delta.
        return h2+(changed-reference)

    def forward(self,packed,facts):
        h=self.features(packed)
        return (h.transpose(0,1)@self.head_weight).transpose(0,1)+self.head_bias+(facts@self.head_facts.T)[:,None,:]
