"""V130 planned direct SparseTabM classifiers; exact active-column algebra."""
import hashlib
import math

import numpy as np
import torch
from torch import nn
import tabm

from v104_phase_b import SparseFirstBatchEnsemble, SparseTabM, csr_tensor
from v75_views import BYTE_FEATURES

WIDTH = 66287
FACTS = 495


class ActiveFirst(SparseFirstBatchEnsemble):
    def forward(self, x):
        columns, inverse = torch.unique(x.col_indices(), sorted=True, return_inverse=True)
        compact = torch.sparse_csr_tensor(x.crow_indices(), inverse, x.values(),
            size=(x.shape[0], len(columns)), device=x.device)
        effective = (self.weight[:, columns].T[:, None, :] *
            self.r[:, columns].T[:, :, None]).reshape(len(columns), self.r.shape[0]*self.weight.shape[0])
        y = torch.sparse.mm(compact, effective).reshape(x.shape[0], self.r.shape[0], self.weight.shape[0])
        return y * self.s + self.bias


class Classifier(nn.Module):
    def __init__(self, hidden=128, din=WIDTH, facts=FACTS, members=16):
        super().__init__()
        self.first = ActiveFirst(din, hidden, members)
        self.second = tabm.LinearBatchEnsemble(hidden, hidden, k=members, scaling_init='ones')
        self.head = tabm.LinearEnsemble(hidden, 3, k=members)
        self.facts_direct = nn.Linear(facts, 3, bias=False)

    def forward(self, x, facts, return_hidden=False):
        h1 = torch.relu(self.first(x))
        h2 = torch.relu(self.second(h1))
        z = self.head(h2) + self.facts_direct(facts)[:, None, :]
        return (z, h1, h2) if return_hidden else z


def extend(small, wide):
    n = small.first.weight.shape[0]
    with torch.no_grad():
        wide.first.weight[:n].copy_(small.first.weight)
        wide.first.r.copy_(small.first.r)
        wide.first.s[:, :n].copy_(small.first.s)
        wide.first.bias[:, :n].copy_(small.first.bias)
        wide.second.weight[:n, :n].copy_(small.second.weight)
        # nn.Linear weight is (out, in): extra -> common is top-right.
        wide.second.weight[:n, n:].zero_()
        wide.second.r[:, :n].copy_(small.second.r)
        wide.second.s[:, :n].copy_(small.second.s)
        wide.second.bias[:, :n].copy_(small.second.bias)
        wide.head.weight[:, :n].copy_(small.head.weight)
        wide.head.weight[:, n:].zero_()
        wide.head.bias.copy_(small.head.bias)
        wide.facts_direct.weight.copy_(small.facts_direct.weight)
    return wide


def make_model(hidden, device='cuda'):
    torch.manual_seed(10201)
    small = Classifier(128)
    if hidden == 128:
        return small.to(device)
    if hidden != 256:
        raise ValueError('Unregistered width')
    torch.manual_seed(13017)
    wide = Classifier(256)
    return extend(small, wide).to(device)


def batch(model, x, ids, hidden=False):
    device = next(model.parameters()).device
    block = x[ids]
    facts = torch.as_tensor(block[:, BYTE_FEATURES:].toarray(), device=device, dtype=torch.float32)
    return model(csr_tensor(block, device), facts, return_hidden=hidden)


def objective(logits, mass, pure, n, pure_totals, batches=1, focused=False):
    ce = -torch.log_softmax(logits, -1).mean(1)
    row = (ce*mass).sum()/n
    auxiliary = .5 * sum((ce[:, c]*mass[:, c]*pure).sum()/pure_totals[c] for c in (1, 2))
    loss = .5*(row+auxiliary) if focused else row
    return loss*batches, row, auxiliary


def tensor_hash(state):
    h = hashlib.sha256()
    for key, val in state.items():
        h.update(key.encode()); h.update(val.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def infer(model, x, ids, counts=None):
    model.eval()
    probs=[]; margins=[]; no_member=[]
    member_sum=np.zeros(3); hidden_zero=np.zeros(2); hidden_total=np.zeros(2)
    with torch.no_grad():
        for start in range(0, len(ids), 256):
            take=ids[start:start+256]
            z,h1,h2=batch(model,x,take,True)
            q=torch.softmax(z,-1).mean(1).cpu().numpy()
            lp=torch.log_softmax(z,-1).mean(1).cpu().numpy()
            member=z.argmax(-1).cpu().numpy()
            avg=z.mean(1).cpu().numpy()
            probs.append(q)
            no_member.append(np.stack([(member!=c).all(1) for c in range(3)],1))
            margins.append(np.stack([avg[:,c]-np.max(np.delete(avg,c,axis=1),axis=1) for c in range(3)],1))
            for j,h in enumerate([h1,h2]):
                hidden_zero[j]+=int((h==0).sum()); hidden_total[j]+=h.numel()
            if counts is not None: member_sum-=np.sum(counts[take]*lp,axis=0)
    return (np.concatenate(probs),np.concatenate(no_member),np.concatenate(margins),
            member_sum,(hidden_zero/hidden_total).tolist())


def module_norms(model, gradient=False, before=None):
    result={}
    for name,module in model.named_children():
        total=torch.zeros((),device=next(model.parameters()).device)
        for key,p in module.named_parameters():
            val=p.grad if gradient else p.detach()-before[name+'.'+key]
            if val is not None: total+=val.square().sum()
        result[name]=float(total.sqrt())
    return result
