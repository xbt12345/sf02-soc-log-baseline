"""Chunked observed-fact relation objective, without a learned auxiliary head."""
import numpy as np
import torch
from torch import nn


class FeatureView(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, packed):
        return self.model.features(packed)


def prepare(edges):
    result = {name: edges[name].to_numpy(copy=True) for name in
        ['left_local', 'right_local', 'target_distance', 'weight']}
    assert len(edges) and np.isfinite(result['target_distance']).all() and np.isfinite(result['weight']).all()
    assert (result['weight'] > 0).all() and ((result['target_distance'] >= 0) & (result['target_distance'] <= 1)).all()
    result['total_weight'] = float(result['weight'].sum())
    return result


def relation(model, packed, graph, parameters=None, backward_scale=None, chunk=4096):
    """The caller owns gradient clearing. Each edge is evaluated exactly once.

    With backward_scale supplied, backward occurs per chunk and adds to existing
    CE gradients. With parameters supplied, a functional finite-state probe is
    used; the registered model state remains unchanged.
    """
    if parameters is not None and backward_scale is not None:
        raise ValueError('Finite probes do not request parameter gradients')
    if backward_scale is not None and backward_scale < 0:
        raise ValueError('Unregistered negative relation coefficient')
    value = 0.0; seen = 0; weight_seen = 0.0
    view = FeatureView(model) if parameters is not None else None
    for start in range(0, len(graph['weight']), chunk):
        stop = min(start + chunk, len(graph['weight']))
        left, right = graph['left_local'][start:stop], graph['right_local'][start:stop]
        unique, inverse = np.unique(np.concatenate([left, right]), return_inverse=True)
        if parameters is None:
            features = model.features(packed[unique])
        else:
            features = torch.func.functional_call(view, {'model.'+k:v for k,v in parameters.items()}, (packed[unique],), strict=False)
        unit = torch.nn.functional.normalize(features, dim=-1, eps=1e-12)
        a, b = unit[inverse[:len(left)]], unit[inverse[len(left):]]
        actual = 0.5 * (1 - (a*b).sum(-1))
        target = torch.as_tensor(graph['target_distance'][start:stop], device=packed.device, dtype=features.dtype)
        weight = torch.as_tensor(graph['weight'][start:stop], device=packed.device, dtype=features.dtype)
        loss = ((actual-target[:, None]).square().mean(1)*weight).sum()/graph['total_weight']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite observed relation objective')
        if backward_scale is not None:
            (backward_scale*loss).backward()
        value += float(loss.detach()); seen += len(left); weight_seen += float(weight.sum())
    assert seen == len(graph['weight'])
    assert abs(weight_seen-graph['total_weight']) < 1e-8
    return value
