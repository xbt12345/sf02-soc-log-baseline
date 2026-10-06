"""Fixed P_IS checkpoint inference from the registered numerical inputs.

The three fold models are fixed evaluation roles, not a newly validated
three-fold averaging ensemble. Raw-log preprocessing is a separate contract.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from scipy.special import softmax
import torch
from torch import nn

WIDTH = 66287
BYTE_FEATURES = 65792
FACTS = 495
MEMBERS = 16
HIDDEN = 128
MAX_BODY_BYTES = 176
CLASS_NAMES = ("benign", "malicious", "suspicious")

WEIGHT_SHA256 = {
    0: {
        "base.pt": "0c4ee3716574d618f42d2224681910c6b54332c19fc870c11f8a641bcc74202a",
        "residual.pt": "686be81ac70946066c1a71f291ae56f55fa34c11264249220c884d0a71e32bfd",
        "router.joblib": "ddf0f02f97f8f8d399986554273cab8070d2a0ac401e6158d5d4f6a6665ec893",
    },
    1: {
        "base.pt": "3e3f5d7ab4ba4bf2050fc9a2e8fc28fdafa393471c05ebcecdd473ec7ef7f728",
        "residual.pt": "5bf09bab3354da65c8ff5578822865ea709825f95460a8a124d3e8df07d4c936",
        "router.joblib": "1e0c4adb44942b07cc772fcecb6377aca984cc2bce7f28414e7a7dd7c4668831",
    },
    2: {
        "base.pt": "7bf3874d1c3c63fb2b213b06f8804202caeab7604f6636346dfe2210db2f0e69",
        "residual.pt": "c8820009d80fff1b740b3539fe798835bc96b41adcc42ad16a23512422012cc0",
        "router.joblib": "dca4ed992841a36efd49d822f2399d84b33462824c651e2853aae435128fd548",
    },
}


def _verify_file(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path.name}")


def _features(x):
    if not sparse.issparse(x) or x.ndim != 2 or x.shape[1] != WIDTH:
        raise ValueError(f"Expected a sparse numerical matrix with {WIDTH} columns")
    x = x.tocsr()
    if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x.data).all():
        raise ValueError("Numerical feature values must be finite floating-point values")
    return x


def csr_tensor(x, device):
    x = x.tocsr()
    return torch.sparse_csr_tensor(
        torch.as_tensor(x.indptr.astype(np.int64), device=device),
        torch.as_tensor(x.indices.astype(np.int64), device=device),
        torch.as_tensor(x.data.astype(np.float32), device=device),
        size=x.shape,
        device=device,
    )


class SparseFirstBatchEnsemble(nn.Module):
    """Registered sparse first-layer equation and checkpoint parameter names."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(HIDDEN, WIDTH))
        self.r = nn.Parameter(torch.empty(MEMBERS, WIDTH))
        self.s = nn.Parameter(torch.empty(MEMBERS, HIDDEN))
        self.bias = nn.Parameter(torch.empty(MEMBERS, HIDDEN))

    def forward(self, x):
        effective = (self.weight.T[:, None, :] * self.r.T[:, :, None]).reshape(
            WIDTH, MEMBERS * HIDDEN
        )
        y = torch.sparse.mm(x, effective).reshape(x.shape[0], MEMBERS, HIDDEN)
        return y * self.s + self.bias


class SecondBatchEnsemble(nn.Module):
    """Inference equation corresponding to the fitted BatchEnsemble layer."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN))
        self.r = nn.Parameter(torch.empty(MEMBERS, HIDDEN))
        self.s = nn.Parameter(torch.empty(MEMBERS, HIDDEN))
        self.bias = nn.Parameter(torch.empty(MEMBERS, HIDDEN))

    def forward(self, x):
        x = x * self.r
        x = x @ self.weight.T
        x = x * self.s
        return x + self.bias


class EnsembleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(MEMBERS, HIDDEN, 3))
        self.bias = nn.Parameter(torch.empty(MEMBERS, 3))

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        return x + self.bias


class SparseTabM(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = SparseFirstBatchEnsemble()
        self.second = SecondBatchEnsemble()
        self.head = EnsembleHead()
        self.facts_direct = nn.Linear(FACTS, 3, bias=False)

    def forward(self, x, facts):
        h = torch.relu(self.first(x))
        h = torch.relu(self.second(h))
        return self.head(h) + self.facts_direct(facts)[:, None, :]


class BodyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        width = 64
        self.embed = nn.Embedding(257, width, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(width)
        pos = torch.arange(MAX_BODY_BYTES, dtype=torch.float32).unsqueeze(1)
        k = torch.arange(0, width, 2, dtype=torch.float32)
        angle = pos * torch.exp(-math.log(10000) * k / width)
        pe = torch.zeros(MAX_BODY_BYTES, width, dtype=torch.float32)
        pe[:, 0::2] = angle.sin()
        pe[:, 1::2] = angle.cos()
        self.register_buffer("position", pe, persistent=True)
        self.head = nn.Linear(width, 3)

    def forward(self, raw_bytes, lengths):
        n = raw_bytes.shape[1]
        pad = torch.arange(n, device=raw_bytes.device)[None, :] >= lengths[:, None]
        ids = (raw_bytes.long() + 1).masked_fill(pad, 0)
        h = self.embed(ids) + self.position[:n]
        h = self.encoder(h, src_key_padding_mask=pad)
        h = self.final_norm(h)
        pooled = h.masked_fill(pad[:, :, None], 0).sum(1) / lengths[:, None]
        return self.head(pooled)


class FoldModel:
    """One fixed fold role, loaded from its three necessary fitted components."""

    def __init__(self, weights_root, fold: int, device="cpu"):
        if fold not in WEIGHT_SHA256:
            raise ValueError("fold must be one of the fixed roles 0, 1, 2")
        self.fold = fold
        self.device = torch.device(device)
        folder = Path(weights_root) / f"fold{fold}"
        for name, digest in WEIGHT_SHA256[fold].items():
            _verify_file(folder / name, digest)
        base = torch.load(folder / "base.pt", map_location="cpu", weights_only=True)
        residual = torch.load(folder / "residual.pt", map_location="cpu", weights_only=True)
        if (base["fold"], base["arm"], base["seed"], base["epoch"]) != (fold, "A", 10201, 25):
            raise ValueError("Backbone checkpoint provenance mismatch")
        if (residual["fold"], residual["arm"], residual["seed"], residual["epoch"]) != (fold, "P_IS", 12701, 50):
            raise ValueError("Residual checkpoint provenance mismatch")
        self.base = SparseTabM().to(self.device)
        self.base.load_state_dict(base["base"], strict=True)
        self.base.eval().requires_grad_(False)
        self.residual = BodyResidual().to(self.device)
        self.residual.load_state_dict(residual["branch"], strict=True)
        self.residual.eval().requires_grad_(False)
        teacher = joblib.load(folder / "router.joblib")
        self.coef = np.asarray(teacher["coef"])
        self.intercept = np.asarray(teacher["intercept"])
        if self.coef.shape != (WIDTH, 3) or self.intercept.shape != (3,):
            raise ValueError("Linear component shape mismatch")
        if not np.isfinite(self.coef).all() or not np.isfinite(self.intercept).all():
            raise ValueError("Linear component contains nonfinite values")

    @torch.no_grad()
    def predict_asa(self, header_features, body_bytes, lengths, batch_size=256):
        """ASA inputs: canonical-header features and the complete registered body."""
        x = _features(header_features)
        body = np.asarray(body_bytes)
        lengths = np.asarray(lengths)
        n = x.shape[0]
        if body.ndim != 2 or body.shape[0] != n or body.dtype != np.uint8:
            raise ValueError("body_bytes must be an aligned uint8 matrix")
        if lengths.shape != (n,) or not np.issubdtype(lengths.dtype, np.integer):
            raise ValueError("lengths must contain one integer byte length per row")
        if np.any(lengths < 1) or np.any(lengths > MAX_BODY_BYTES) or np.any(lengths > body.shape[1]):
            raise ValueError("Body length falls outside the registered 1..176-byte input range")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        result = np.empty((n, 3), dtype=np.float32)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            block = x[start:stop]
            facts = torch.as_tensor(block[:, BYTE_FEATURES:].toarray(), device=self.device, dtype=torch.float32)
            logits = self.base(csr_tensor(block, self.device), facts).cpu().numpy()
            length = torch.as_tensor(lengths[start:stop], device=self.device, dtype=torch.int64)
            body_width = int(length.max())
            raw = torch.as_tensor(body[start:stop, :body_width], device=self.device, dtype=torch.uint8)
            extra = self.residual(raw, length).cpu().numpy()
            # Preserve the original scipy float32 softmax/mean operation order.
            result[start:stop] = softmax(logits + extra[:, None, :], axis=-1).mean(1).astype(np.float32)
        return result

    def predict_router(self, n1_features):
        """Non-ASA inputs retain their registered N1 numerical representation."""
        x = _features(n1_features)
        logits = np.asarray(x @ self.coef) + self.intercept
        return softmax(logits, axis=-1)

    def predict_routed_proba(self, n1_features, asa_mask, header_features=None, body_bytes=None, lengths=None):
        """Route all ASA rows to P_IS and all other rows to the linear component.

        ASA-specific inputs are ordered as n1_features[np.flatnonzero(asa_mask)].
        No label, original row ID, source-group score or prediction lookup is used.
        """
        x = _features(n1_features)
        mask = np.asarray(asa_mask)
        if mask.shape != (x.shape[0],) or mask.dtype != np.bool_:
            raise ValueError("asa_mask must be an aligned boolean routing vector")
        result = self.predict_router(x)
        if mask.any():
            if header_features is None or body_bytes is None or lengths is None:
                raise ValueError("ASA routes require aligned header and body inputs")
            if header_features.shape[0] != int(mask.sum()):
                raise ValueError("ASA-specific input count disagrees with routing")
            result[mask] = self.predict_asa(header_features, body_bytes, lengths)
        return result

    def predict(self, n1_features, asa_mask, header_features=None, body_bytes=None, lengths=None):
        return self.predict_routed_proba(n1_features, asa_mask, header_features, body_bytes, lengths).argmax(1)
