"""Inference from any registered numerical input, no row IDs or cache lookup."""
import numpy as np
import torch
from v135_model import Classifier,batch
from v75_views import BYTE_FEATURES
from v141_second_model import SecondRepresentation


@torch.no_grad()
def infer_numeric_input(x,original_backbone_state,initial_readout_state,candidate_state,device='cuda',chunk=256):
    if x.shape[1]!=66287:raise ValueError('Use registered text/fact preprocessing and feature width')
    body=Classifier(128).to(device);body.load_state_dict(original_backbone_state);body.eval()
    candidate=SecondRepresentation(original_backbone_state,initial_readout_state).to(device);candidate.load_state_dict(candidate_state);candidate.eval()
    result=[]
    for start in range(0,x.shape[0],chunk):
        ids=np.arange(start,min(start+chunk,x.shape[0]));_,h1,h2=batch(body,x,ids,True)
        packed=torch.cat([h1,h2],dim=-1).double()
        facts=torch.as_tensor(x[ids,BYTE_FEATURES:].toarray(),device=device,dtype=torch.float64)
        result.append(torch.softmax(candidate(packed,facts),-1).mean(1).cpu().numpy())
    return np.concatenate(result)
