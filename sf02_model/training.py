"""Fresh, fixed-endpoint three-class training without private experiment files.

This is a new single-fold recipe using in-fit scores from its own frozen base.
It is not a replay of the published nested-teacher experiment or its metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Sequence

import joblib
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn

from .model import (BYTE_FEATURES, CLASS_NAMES, HIDDEN, MAX_BODY_BYTES, MEMBERS,
                    WIDTH, BodyResidual, SparseTabM, _features, csr_tensor)

SCHEMA = "sf02-trained-bundle-v1"


@dataclass(frozen=True)
class TrainingConfig:
    base_epochs: int = 25
    residual_epochs: int = 50
    batch_size: int = 256
    base_seed: int = 10201
    residual_seed: int = 12701
    base_lr: float = 0.002
    residual_lr: float = 0.0003
    weight_decay: float = 0.0003
    router_l2: float = 1e-6
    router_max_iterations: int = 1000
    fold: int = 0
    device: str = "cpu"
    cpu_threads: int = 2
    unsupported_policy: str = "error"

    def validate(self):
        for name in ("base_epochs", "residual_epochs", "batch_size",
                     "router_max_iterations", "cpu_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.fold not in (0, 1, 2):
            raise ValueError("fold must be 0, 1 or 2")
        for name in ("base_lr", "residual_lr", "router_l2"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.unsupported_policy not in ("error", "router"):
            raise ValueError("unsupported_policy must be error or router")


@dataclass
class TrainingDataset:
    prepared: Any
    labels: np.ndarray
    event_ids: Sequence[str]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_base(seed=10201):
    """Explicit finite initialization for the inference-only empty parameters."""
    _seed(seed)
    model = SparseTabM()
    # The inference constructor only initializes facts_direct. Restore the RNG
    # before reproducing first -> second -> head -> facts initialization order.
    _seed(seed)
    nn.init.kaiming_uniform_(model.first.weight, a=math.sqrt(5))
    nn.init.normal_(model.first.r)
    nn.init.ones_(model.first.s)
    nn.init.uniform_(model.first.bias, -1 / math.sqrt(WIDTH), 1 / math.sqrt(WIDTH))
    nn.init.kaiming_uniform_(model.second.weight, a=math.sqrt(5))
    nn.init.ones_(model.second.r)
    nn.init.ones_(model.second.s)
    nn.init.uniform_(model.second.bias, -1 / math.sqrt(HIDDEN), 1 / math.sqrt(HIDDEN))
    nn.init.uniform_(model.head.weight, -1 / math.sqrt(HIDDEN), 1 / math.sqrt(HIDDEN))
    nn.init.uniform_(model.head.bias, -1 / math.sqrt(HIDDEN), 1 / math.sqrt(HIDDEN))
    model.facts_direct.reset_parameters()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite base initialization")
    return model


def initialize_residual(seed=12701):
    _seed(seed + 981)
    model = BodyResidual()
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    return model


def _state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _csr_hash(digest, x):
    x = x.tocsr(copy=True)
    x.sort_indices()
    x.eliminate_zeros()
    digest.update(np.asarray(x.shape, dtype="<i8").tobytes())
    digest.update(x.indptr.astype("<i8").tobytes())
    digest.update(x.indices.astype("<i4").tobytes())
    digest.update(x.data.astype("<f4").tobytes())


def _dataset_hash(data):
    digest = hashlib.sha256()
    p = data.prepared
    _csr_hash(digest, p.full_features)
    _csr_hash(digest, p.header_features)
    digest.update(np.asarray(p.asa_mask, dtype=np.bool_).tobytes())
    digest.update(np.asarray(p.body_bytes, dtype=np.uint8).tobytes())
    digest.update(np.asarray(p.lengths, dtype="<i8").tobytes())
    digest.update(np.asarray(data.labels, dtype=np.int8).tobytes())
    digest.update(json.dumps(list(data.event_ids), ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _validate_data(data, policy, require_all_classes=False):
    p = data.prepared
    x = _features(p.full_features)
    n = x.shape[0]
    y = np.asarray(data.labels)
    if n < 1 or y.shape != (n,) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("One integer training label is required for every input row")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Labels must be benign=0, malicious=1 or suspicious=2")
    if require_all_classes and set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Complete three-class training requires all three classes")
    if len(data.event_ids) != n or any(not isinstance(i, str) or not i for i in data.event_ids):
        raise ValueError("One nonempty string event_id is required per row")
    if len(set(data.event_ids)) != n:
        raise ValueError("Duplicate event_id in input dataset")
    mask = np.asarray(p.asa_mask)
    if mask.shape != (n,) or mask.dtype != np.bool_:
        raise ValueError("asa_mask must be an aligned boolean vector")
    count = int(mask.sum())
    if require_all_classes and count == 0:
        raise ValueError("Complete P_IS training requires at least one supported ASA training row")
    headers = _features(p.header_features)
    body, lengths = np.asarray(p.body_bytes), np.asarray(p.lengths)
    if headers.shape[0] != count or body.ndim != 2 or body.shape[0] != count or body.dtype != np.uint8:
        raise ValueError("ASA input rows are misaligned")
    if lengths.shape != (count,) or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("ASA lengths must be aligned integers")
    if np.any(lengths < 1) or np.any(lengths > MAX_BODY_BYTES) or np.any(lengths > body.shape[1]):
        raise ValueError("ASA body lengths must lie in 1..176 without truncation")
    unsupported = list(getattr(p, "unsupported_rows", ()))
    if unsupported and policy == "error":
        raise ValueError(f"{len(unsupported)} unsupported raw rows; use explicit router policy to retain them")
    return {"rows": n, "class_support": np.bincount(y, minlength=3).tolist(),
            "supported_ASA_rows": count, "unsupported_router_rows": len(unsupported),
            "dataset_sha256": _dataset_hash(data)}


def _row_keys(data):
    """Exact numerical-input identities, excluding IDs and labels."""
    p = data.prepared
    x = p.full_features.tocsr()
    headers = p.header_features.tocsr()
    asa_lookup = np.full(x.shape[0], -1, dtype=np.int64)
    asa_lookup[np.flatnonzero(p.asa_mask)] = np.arange(headers.shape[0])
    keys = []
    for row in range(x.shape[0]):
        digest = hashlib.sha256()
        _csr_hash(digest, x[row:row + 1])
        k = asa_lookup[row]
        digest.update(bytes([int(k >= 0)]))
        if k >= 0:
            _csr_hash(digest, headers[k:k + 1])
            digest.update(np.asarray(p.body_bytes[k, :int(p.lengths[k])]).tobytes())
        keys.append(digest.hexdigest())
    return keys


def _source_bindings():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "sf02_model").rglob("*.py"))
    paths += [root / "train.py", root / "requirements.txt"]
    return {str(p): _sha(p) for p in paths if p.is_file()}


def _require_bindings(plan, data, validation, verify_data=True):
    for path, digest in plan["source_sha256"].items():
        if _sha(path) != digest:
            raise ValueError("Training source changed after plan binding: " + path)
    if not verify_data:
        return
    for path, digest in plan["input_file_sha256"].items():
        if _sha(path) != digest:
            raise ValueError("Training input file changed after plan binding: " + path)
    if _dataset_hash(data) != plan["train"]["dataset_sha256"]:
        raise ValueError("Training numerical inputs/labels changed after binding")
    if validation is not None and _dataset_hash(validation) != plan["validation"]["dataset_sha256"]:
        raise ValueError("Validation numerical inputs/labels changed after binding")


def _base_logits(model, x, device):
    facts = torch.as_tensor(x[:, BYTE_FEATURES:].toarray(), device=device, dtype=torch.float32)
    return model(csr_tensor(x, device), facts)


def _body_logits(model, prepared, ids, device):
    length = torch.as_tensor(np.asarray(prepared.lengths)[ids], device=device, dtype=torch.int64)
    raw = torch.as_tensor(np.asarray(prepared.body_bytes)[ids, :int(length.max())],
                          device=device, dtype=torch.uint8)
    return model(raw, length)


def _fit_neural(model, prepared, labels, config, stage, log, before_epoch, base=None):
    n = len(labels)
    if not n:
        return {"epochs": 0, "optimizer_steps": 0, "reason": "no_supported_ASA_rows"}
    epochs = config.base_epochs if stage == "base" else config.residual_epochs
    seed = config.base_seed if stage == "base" else config.residual_seed
    lr = config.base_lr if stage == "base" else config.residual_lr
    rng = np.random.default_rng(seed + config.fold)
    batches = math.ceil(n / config.batch_size)
    total_steps = epochs * batches
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    initial = _state_sha(model)
    steps = 0
    for epoch in range(1, epochs + 1):
        before_epoch()
        model.train()
        order = rng.permutation(n)
        numerator = 0.0
        seen = np.zeros(3, np.int64)
        for start in range(0, n, config.batch_size):
            ids = order[start:start + config.batch_size]
            x = prepared.header_features[ids]
            if base is None:
                logits = _base_logits(model, x, config.device)
            else:
                with torch.no_grad():
                    logits = _base_logits(base, x, config.device)
                logits = logits + _body_logits(model, prepared, ids, config.device)[:, None, :]
            target = torch.as_tensor(labels[ids], device=config.device, dtype=torch.int64)
            lp = torch.log_softmax(logits, -1).mean(1)
            loss_sum = -lp.gather(1, target[:, None]).sum()
            loss = loss_sum / (n / batches)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            steps += 1
            if stage == "residual":
                warmup = max(1, math.ceil(0.1 * total_steps))
                for group in optimizer.param_groups:
                    group["lr"] = lr * min(steps / warmup, 1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("Nonfinite training gradient")
            optimizer.step()
            numerator += float(loss_sum.detach())
            seen += np.bincount(labels[ids], minlength=3)
        if not np.array_equal(seen, np.bincount(labels, minlength=3)):
            raise ValueError("Original row frequency or class exposure changed")
        item = {"stage": stage, "epoch": epoch, "optimizer_steps": steps,
                "class_rows_seen": seen.tolist(), "online_mean_member_CE": numerator / n}
        log.write(json.dumps(item) + "\n")
        log.flush()
        print(json.dumps(item), flush=True)
    model.eval()
    final = _state_sha(model)
    return {"epochs": epochs, "optimizer_steps": steps, "initial_parameter_sha256": initial,
            "final_parameter_sha256": final, "parameter_identity_changed": initial != final}


def fit_router(x, labels, config, log=None):
    """Three one-vs-rest logistic scores with the original sparse full-row objective."""
    n = x.shape[0]
    target = np.eye(3, dtype=np.float64)[labels]
    initial = np.zeros((WIDTH + 1, 3), dtype=np.float64)
    initial[-1] = np.log(np.maximum(target.mean(0), 1e-9))
    evaluations = 0

    def objective(flat):
        nonlocal evaluations
        evaluations += 1
        w = flat.reshape(WIDTH + 1, 3)
        z = np.asarray(x @ w[:-1]) + w[-1]
        value = (np.logaddexp(0, z).sum() - (target * z).sum()) / n
        value += config.router_l2 / 2 * np.square(w[:-1]).sum()
        residual = (expit(z) - target) / n
        grad = np.vstack([np.asarray(x.T @ residual) + config.router_l2 * w[:-1], residual.sum(0)])
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite sparse router objective/gradient")
        return float(value), grad.ravel()

    iterations = 0

    def callback(flat):
        nonlocal iterations
        iterations += 1
        if log is not None:
            log.write(json.dumps({"stage": "router", "accepted_iteration": iterations}) + "\n")
            log.flush()

    result = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B", callback=callback,
                      options={"maxiter": config.router_max_iterations, "maxcor": 10,
                               "gtol": 1e-6, "ftol": 1e-12, "maxls": 30})
    w = result.x.reshape(WIDTH + 1, 3)
    receipt = {"iterations": int(result.nit), "objective_evaluations": evaluations,
               "optimizer_success": bool(result.success), "termination": str(result.message),
               "objective": float(result.fun), "gradient_inf": float(np.abs(result.jac).max()),
               "parameter_identity_changed": not np.array_equal(w, initial)}
    return {"coef": w[:-1].copy(), "intercept": w[-1].copy()}, receipt


def classification_metrics(labels, predictions):
    cm = np.bincount(np.asarray(labels, np.int64) * 3 + predictions, minlength=9).reshape(3, 3)
    classes = {}
    for k, name in enumerate(CLASS_NAMES):
        support, called, correct = int(cm[k].sum()), int(cm[:, k].sum()), int(cm[k, k])
        classes[name] = {"support": support, "correct": correct, "missed": support - correct,
                         "false_called": called - correct,
                         "precision": correct / called if called else None,
                         "recall": correct / support if support else None,
                         "f1": 2 * correct / (support + called) if support else None}
    complete = all(c["support"] > 0 for c in classes.values())
    return {"rows": int(cm.sum()), "confusion_matrix": cm.tolist(), "classes": classes,
            "all_classes_present": complete,
            "macro_F1": float(np.mean([c["f1"] for c in classes.values()])) if complete else None,
            "errors": int(cm.sum() - np.trace(cm))}


def train_bundle(data: TrainingDataset, output, config=None, validation=None, input_files=()):
    """Train fresh parameters, seal inputs first, save a separate loadable bundle."""
    config = config or TrainingConfig()
    config.validate()
    train_info = _validate_data(data, config.unsupported_policy, require_all_classes=True)
    valid_info = None
    if validation is not None:
        valid_info = _validate_data(validation, config.unsupported_policy)
        if validation is data or valid_info["dataset_sha256"] == train_info["dataset_sha256"]:
            raise ValueError("Training and validation are the same complete dataset")
        train_keys, valid_keys = _row_keys(data), _row_keys(validation)
        known_inputs = set(train_keys)
        valid_info["shared_numerical_input_rows"] = sum(k in known_inputs for k in valid_keys)
        valid_info["shared_event_id_values"] = len(set(data.event_ids) & set(validation.event_ids))
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Training output already exists; choose a new directory")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    torch.set_num_threads(config.cpu_threads)
    plan = {"schema": SCHEMA, "recipe": "fresh_same_architecture_single_fold_IS",
            "configuration": asdict(config), "train": train_info, "validation": valid_info,
            "validation_used_for_gradients_or_checkpoint_selection": False,
            "fixed_terminal_epochs": True, "published_historical_metrics_applicable": False,
            "source_group_independence_established": False,
            "source_sha256": _source_bindings(),
            "input_file_sha256": {str(Path(p).resolve()): _sha(p) for p in input_files},
            "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                        "scipy": importlib.metadata.version("scipy"), "device": str(device)}}
    raw_inputs = list(input_files)
    if validation is not None and len(raw_inputs) == 2 and _sha(raw_inputs[0]) == _sha(raw_inputs[1]):
        raise ValueError("Training and validation raw files are byte-identical")
    output.mkdir(parents=True)
    _json(output / "training_plan.json", plan)
    seal = _sha(output / "training_plan.json")

    def require(verify_data=True):
        if _sha(output / "training_plan.json") != seal:
            raise ValueError("Bound training plan changed")
        _require_bindings(plan, data, validation, verify_data=verify_data)

    try:
        require()
        base = initialize_base(config.base_seed).to(device)
        residual = initialize_residual(config.residual_seed).to(device)
        asa_labels = np.asarray(data.labels, dtype=np.int64)[data.prepared.asa_mask]
        with (output / "training_history.jsonl").open("x", encoding="utf-8") as log:
            base_fit = _fit_neural(base, data.prepared, asa_labels, config, "base", log,
                                    lambda: require(False))
            base.eval().requires_grad_(False)
            frozen_base_sha = _state_sha(base)
            require()
            residual_fit = _fit_neural(residual, data.prepared, asa_labels, config,
                                       "residual", log, lambda: require(False), base=base)
            if _state_sha(base) != frozen_base_sha:
                raise ValueError("Frozen base changed during residual training")
            require()
            router, router_fit = fit_router(data.prepared.full_features,
                                            np.asarray(data.labels, dtype=np.int64), config, log)
        require()
        folder = output / f"fold{config.fold}"
        folder.mkdir()
        base_meta = {"fold": config.fold, "arm": "TRAINED_BASE", "seed": config.base_seed,
                     "epoch": base_fit["epochs"]}
        residual_meta = {"fold": config.fold, "arm": "TRAINED_RESIDUAL", "seed": config.residual_seed,
                         "epoch": residual_fit["epochs"]}
        torch.save({"base": {k: v.detach().cpu() for k, v in base.state_dict().items()},
                    **base_meta, "schema": SCHEMA, "training_plan_sha256": seal}, folder / "base.pt")
        torch.save({"branch": {k: v.detach().cpu() for k, v in residual.state_dict().items()},
                    **residual_meta, "schema": SCHEMA, "training_plan_sha256": seal}, folder / "residual.pt")
        joblib.dump({**router, "schema": SCHEMA, "fold": config.fold,
                     "training_plan_sha256": seal}, folder / "router.joblib")
        files = {p.relative_to(output).as_posix(): {"sha256": _sha(p), "bytes": p.stat().st_size}
                 for p in sorted(folder.iterdir())}
        manifest = {"schema": SCHEMA, "recipe": plan["recipe"], "folds": [config.fold],
                    "files": files, "fold_metadata": {str(config.fold): {"base": base_meta,
                                                                          "residual": residual_meta}},
                    "configuration": asdict(config), "training_plan_sha256": seal,
                    "published_historical_metrics_applicable": False}
        _json(output / "manifest.json", manifest)
        # Use the same public loader and routing function as deployment.
        from .model import FoldModel
        predictor = FoldModel(output, config.fold, device=config.device)
        def evaluate(dataset):
            p = dataset.prepared
            pred = predictor.predict(p.full_features, p.asa_mask, p.header_features,
                                     p.body_bytes, p.lengths)
            return classification_metrics(dataset.labels, pred)
        training_metrics = evaluate(data)
        validation_metrics = evaluate(validation) if validation is not None else None
        outcome = {"status": "fixed_endpoint_training_completed", "train": train_info,
                   "validation": valid_info, "base": base_fit, "residual": residual_fit,
                   "router": router_fit, "frozen_base_retained": True,
                   "training_metrics": training_metrics, "validation_metrics": validation_metrics,
                   "validation_evaluated": validation is not None,
                   "validation_used_for_gradients_or_checkpoint_selection": False,
                   "quality_acceptance": False, "published_historical_metrics_applicable": False,
                   "manifest_sha256": _sha(output / "manifest.json")}
        _json(output / "training_result.json", outcome)
        return outcome
    except Exception as exc:
        _json(output / "training_failure.json", {"status": "training_or_verification_failed",
                                                "error": f"{type(exc).__name__}: {exc}",
                                                "quality_acceptance": False})
        raise


def read_training_data(path, *, batch_size=2048, limit_rows=None, unsupported_policy="error"):
    """Read raw CSV/parquet in batches; retain every selected row and label."""
    from .io import iter_input_batches
    from .preprocess import prepare_batch
    chunks, labels, event_ids, unsupported, offset = [], [], [], [], 0
    for raw in iter_input_batches(path, batch_size=batch_size, require_labels=True):
        count = len(raw.event_ids)
        if limit_rows is not None:
            count = min(count, limit_rows - offset)
        if count <= 0:
            break
        batch = prepare_batch(raw.messages[:count], raw.src_ports[:count])
        if batch.unsupported_rows and unsupported_policy == "error":
            raise ValueError(f"Unsupported training rows in batch at {offset}; explicit router policy required")
        chunks.append(batch)
        labels.append(raw.labels[:count])
        event_ids.extend(raw.event_ids[:count])
        unsupported.extend({**row, "index": int(row["index"]) + offset} for row in batch.unsupported_rows)
        offset += count
        if limit_rows is not None and offset >= limit_rows:
            break
    if not chunks:
        raise ValueError("Training input is empty")
    lengths = np.concatenate([p.lengths for p in chunks])
    width = int(lengths.max()) if lengths.size else 0
    body = np.zeros((len(lengths), width), dtype=np.uint8)
    cursor = 0
    for p in chunks:
        n = len(p.lengths)
        body[cursor:cursor + n, :p.body_bytes.shape[1]] = p.body_bytes
        cursor += n
    mask = np.concatenate([p.asa_mask for p in chunks])
    prepared = SimpleNamespace(full_features=sparse.vstack([p.full_features for p in chunks], format="csr"),
                               asa_mask=mask, asa_indices=np.flatnonzero(mask),
                               header_features=sparse.vstack([p.header_features for p in chunks], format="csr"),
                               body_bytes=body, lengths=lengths, unsupported_rows=tuple(unsupported))
    return TrainingDataset(prepared, np.concatenate(labels).astype(np.int64), event_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw labelled CSV or parquet")
    parser.add_argument("--validation", type=Path, help="Separate labelled validation file; never used for fitting")
    parser.add_argument("--output", type=Path, required=True, help="New model-bundle directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--base-epochs", type=int, default=25)
    parser.add_argument("--residual-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--router-max-iterations", type=int, default=1000)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--limit-rows", type=int, help="Explicit smoke-test row limit; default reads all rows")
    parser.add_argument("--unsupported-policy", choices=("error", "router"), default="error")
    args = parser.parse_args(argv)
    if args.limit_rows is not None and args.limit_rows < 1:
        parser.error("--limit-rows must be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    config = TrainingConfig(base_epochs=args.base_epochs, residual_epochs=args.residual_epochs,
                            batch_size=args.batch_size, router_max_iterations=args.router_max_iterations,
                            fold=args.fold, device=args.device, cpu_threads=args.cpu_threads,
                            unsupported_policy=args.unsupported_policy)
    config.validate()
    data = read_training_data(args.input, limit_rows=args.limit_rows,
                              unsupported_policy=args.unsupported_policy)
    validation = (read_training_data(args.validation, limit_rows=args.limit_rows,
                                    unsupported_policy=args.unsupported_policy)
                  if args.validation is not None else None)
    files = [args.input] + ([args.validation] if args.validation is not None else [])
    outcome = train_bundle(data, args.output, config, validation, input_files=files)
    print(json.dumps({"status": outcome["status"], "bundle": str(args.output),
                      "training_rows": outcome["train"]["rows"], "limited_smoke_run": args.limit_rows is not None,
                      "validation_evaluated": outcome["validation_evaluated"],
                      "quality_acceptance": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
