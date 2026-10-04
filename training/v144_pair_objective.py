"""Restricted same-observed-context cross-root supervised contrast, not label rules."""
import json
import numpy as np
import pandas as pd
import torch
from v143_pair_geometry_probe import observed_key
from v135_runtime import fit_context

TEMPERATURE=0.1
AUX_GRADIENT_RATIO=0.1


def build_pairs(d,facts,fold):
    frame,c,pure,_,ids=fit_context(d,fold)
    frame['pair_key']=facts.map(json.loads).map(observed_key).to_numpy()[frame.index]
    eligible=frame[frame.pair_key.notna() & pure[frame.local].astype(bool)].copy()
    valid=[]
    for key,g in eligible.groupby('pair_key'):
        sources=g.groupby('truth').root.nunique()
        actual=g.groupby('truth').canonical_key.nunique()
        if all(sources.get(y,0)>=2 and actual.get(y,0)>=2 for y in [1,2]):valid.append(key)
    selected=eligible[eligible.pair_key.isin(valid)].copy()
    if selected.empty:raise ValueError('No observed dual-class multiroot contrast; no auxiliary gradient qualification')
    # Repeated original rows determine each real root prototype, not synthetic input.
    cells=selected.groupby(['pair_key','truth','root'],sort=True).size().reset_index(name='mass')
    cells['cell']=np.arange(len(cells))
    mapping=selected[['row_position','local','pair_key','truth','root']].merge(cells[['pair_key','truth','root','cell']],on=['pair_key','truth','root'],validate='many_to_one')
    counts=mapping.groupby(['local','cell']).size().reset_index(name='mass')
    locals_=np.sort(counts.local.unique());lookup=dict(zip(locals_,range(len(locals_))))
    groups=[]
    for _,g in cells.groupby('pair_key',sort=True):groups.append(g.cell.to_numpy(np.int64))
    roots=set(d[d.fold.eq(fold)].root)
    if roots&set(selected.root):raise ValueError('Held root in legal pair supervision')
    if selected.groupby('canonical_key').truth.nunique().max()!=1:raise ValueError('Identical numeric inputs cannot be negative pairs')
    p=dict(ids=locals_,source_index=counts.local.map(lookup).to_numpy(np.int64),cell_index=counts.cell.to_numpy(np.int64),
           edge_mass=counts.mass.to_numpy(np.float64),cell_mass=cells.mass.to_numpy(np.float64),labels=cells.truth.to_numpy(np.int64),groups=groups,cells=cells)
    report=dict(fold=fold,selected_original_rows=len(selected),unique_numeric_inputs=len(locals_),keys=len(groups),root_class_cells=len(cells),
        original_M=int(selected.truth.eq(1).sum()),original_S=int(selected.truth.eq(2).sum()),
        retained_classification_original_rows=int(c.sum()),classification_original_class_mass=c.sum(0).tolist(),
        pair_facts='Full body facts except exact observed source port; source range/mask and every other fact retained.',
        root_prototypes='Each cell frequency-weighted; anchors averaged equally within class and each class equally within key. This is an explicit auxiliary weighting, not a change to original-frequency CE.',
        minimum_sources_per_class=min(int(g.groupby('truth').root.nunique().min()) for _,g in selected.groupby('pair_key')))
    return frame,c,pure,ids,p,report


def contrast_from_features(features,p):
    device=features.device
    unit=torch.nn.functional.normalize(features,dim=-1,eps=1e-12)
    src=torch.as_tensor(p['source_index'],device=device);cell=torch.as_tensor(p['cell_index'],device=device)
    mass=torch.as_tensor(p['edge_mass'],device=device,dtype=features.dtype)
    total=torch.as_tensor(p['cell_mass'],device=device,dtype=features.dtype)
    centers=torch.zeros((len(total),features.shape[1],features.shape[2]),device=device,dtype=features.dtype)
    centers.index_add_(0,cell,unit[src]*mass[:,None,None])
    centers=torch.nn.functional.normalize(centers/total[:,None,None],dim=-1,eps=1e-12)
    labels=torch.as_tensor(p['labels'],device=device);losses=[]
    for ids in p['groups']:
        take=torch.as_tensor(ids,device=device);a=centers[take].transpose(0,1);y=labels[take];n=len(ids)
        logits=(a@a.transpose(1,2))/TEMPERATURE
        self_mask=torch.eye(n,device=device,dtype=torch.bool)
        positive=y[:,None].eq(y[None,:])&~self_mask
        if not positive.any(1).all():raise ValueError('Anchor without a legal cross-root same-class positive')
        logden=torch.logsumexp(logits.masked_fill(self_mask[None,:,:],-torch.inf),-1)
        peranchor=-(logits.masked_fill(~positive[None,:,:],0).sum(-1)/positive.sum(-1)[None,:]-logden)
        losses.append(0.5*(peranchor[:,y.eq(1)].mean()+peranchor[:,y.eq(2)].mean()))
    return torch.stack(losses).mean()


def auxiliary(model,h,p):
    return contrast_from_features(model.features(h[p['ids']]),p)


def gradients(model,h,p):
    model.zero_grad(set_to_none=True);value=auxiliary(model,h,p);value.backward()
    if not torch.isfinite(value) or any(not torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError('Nonfinite restricted contrastive gradient')
    return float(value.detach()),tuple(v.grad.detach().clone() for v in model.parameters())
