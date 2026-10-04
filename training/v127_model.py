"""V127 matched Post-LN and Pre-LN byte residuals on frozen V125 member logits."""
import math

import torch
from torch import nn


class BodyResidual(nn.Module):
    def __init__(self, pre_norm=False, maxlen=176):
        super().__init__()
        width = 64
        self.embed = nn.Embedding(257, width, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=128,
            dropout=0.0, activation='gelu', batch_first=True,
            norm_first=pre_norm)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(width) if pre_norm else nn.Identity()
        pos = torch.arange(maxlen, dtype=torch.float32).unsqueeze(1)
        k = torch.arange(0, width, 2, dtype=torch.float32)
        angle = pos * torch.exp(-math.log(10000) * k / width)
        pe = torch.zeros(maxlen, width, dtype=torch.float32)
        pe[:, 0::2] = angle.sin()
        pe[:, 1::2] = angle.cos()
        self.register_buffer('position', pe, persistent=True)
        self.head = nn.Linear(width, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def encode(self, raw_bytes, lengths):
        n = raw_bytes.shape[1]
        pad = torch.arange(n, device=raw_bytes.device)[None, :] >= lengths[:, None]
        ids = (raw_bytes.long() + 1).masked_fill(pad, 0)
        h = self.embed(ids) + self.position[:n]
        h = self.encoder(h, src_key_padding_mask=pad)
        h = self.final_norm(h)
        pooled = h.masked_fill(pad[:, :, None], 0).sum(1) / lengths[:, None]
        return pooled

    def forward(self, raw_bytes, lengths):
        return self.head(self.encode(raw_bytes, lengths))


def make_branch(arm, seed, device='cuda'):
    if arm not in ('W', 'P'):
        raise ValueError(arm)
    torch.manual_seed(seed + 981)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 981)
    return BodyResidual(pre_norm=(arm == 'P')).to(device)


def branch_batch(model, body, lengths, ids, device='cuda'):
    length = torch.as_tensor(lengths[ids], device=device, dtype=torch.int64)
    width = int(length.max())
    raw = torch.as_tensor(body[ids, :width], device=device, dtype=torch.uint8)
    return model(raw, length)


def residuals(model, body, lengths, device='cuda', batch_size=256):
    model.eval()
    out = torch.empty((len(lengths), 3), dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(lengths), batch_size):
            ids = np.arange(start, min(start + batch_size, len(lengths)))
            out[ids] = branch_batch(model, body, lengths, ids, device).cpu()
    return out.numpy()


import numpy as np
