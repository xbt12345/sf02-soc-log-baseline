"""Exact current function components; no alternate classifier or ablations."""
import numpy as np
import torch

def groups(names):
    assert len(names)==477
    result={}
    for i,name in enumerate(names):
        field=name.split('=')[0].split(':')[0]
        result.setdefault(field,[]).append(i)
    result['record_src_port_bits_state_conflict']=list(range(477,495))
    assert sorted(i for indices in result.values() for i in indices)==list(range(495))
    return result

def components(h2,facts,model,partition):
    body=(h2.transpose(0,1)@model.head_weight).transpose(0,1)
    direct=facts@model.head_facts.T
    bias=model.head_bias
    fields={k:facts[:,v]@model.head_facts[:,v].T for k,v in partition.items()}
    return body,direct,bias,fields

def ensemble_summary(z,body,direct,bias):
    probability=torch.softmax(z,-1).mean(1)
    margin=z[:,:,2]-z[:,:,1]
    b=body[:,:,2]-body[:,:,1];f=direct[:,2]-direct[:,1];bi=bias[:,2]-bias[:,1]
    return probability,dict(body_S_minus_M_mean=b.mean(1),body_S_minus_M_min=b.min(1).values,
        body_S_minus_M_max=b.max(1).values,facts_S_minus_M=f,
        bias_S_minus_M_mean=bi.mean().expand_as(f),total_S_minus_M_mean=margin.mean(1),
        total_S_minus_M_min=margin.min(1).values,total_S_minus_M_max=margin.max(1).values,
        positive_S_minus_M_members=(margin>0).sum(1),negative_S_minus_M_members=(margin<0).sum(1),
        mean_logit_pred=z.mean(1).argmax(1),probability_mean_pred=probability.argmax(1),
        member_N_predictions=(z.argmax(-1)==0).sum(1),member_M_predictions=(z.argmax(-1)==1).sum(1),
        member_S_predictions=(z.argmax(-1)==2).sum(1))
