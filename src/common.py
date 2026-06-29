from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config["__config_dir__"] = str(config_file.parent)
    return config


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def resolve_artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["paths"]["artifacts_dir"])
    if not root.is_absolute():
        root = Path(config["__config_dir__"]) / root
    root = ensure_dir(root)
    return {
        "root": root,
        "reports": ensure_dir(root / "reports"),
        "models": ensure_dir(root / "models"),
        "submissions": ensure_dir(root / "submissions"),
    }


def read_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path_obj, columns=columns)
    if suffix == ".csv":
        return pd.read_csv(path_obj, usecols=columns)
    raise ValueError(f"Unsupported file type: {suffix}")


def fill_categorical(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def add_time_features(df: pd.DataFrame, datetime_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    created: list[str] = []
    for col in datetime_cols:
        if col not in out.columns:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        out[col] = numeric.fillna(numeric.median())
        dt = pd.to_datetime(out[col], unit="s", errors="coerce", utc=True)
        for suffix, series in {
            "hour": dt.dt.hour.fillna(-1).astype("int16"),
            "weekday": dt.dt.weekday.fillna(-1).astype("int16"),
            "month": dt.dt.month.fillna(-1).astype("int16"),
            "is_weekend": dt.dt.weekday.isin([5, 6]).astype("int8"),
        }.items():
            name = f"{col}_{suffix}"
            out[name] = series
            created.append(name)
    return out, created


def prepare_route_a_features(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    feature_cfg = config["features"]
    categorical_cols = [c for c in feature_cfg["categorical_cols"] if c in df.columns]
    numeric_cols = [c for c in feature_cfg["numeric_cols"] if c in df.columns]
    datetime_cols = [c for c in feature_cfg["datetime_cols"] if c in df.columns]
    keep_cols = categorical_cols + numeric_cols + datetime_cols
    out = df[keep_cols].copy()
    out = fill_categorical(out, categorical_cols)
    out, created_time_cols = add_time_features(out, datetime_cols)
    return out, categorical_cols


def build_group_key(df: pd.DataFrame, entity_cols: list[str]) -> pd.Series:
    parts: list[pd.Series] = []
    for col in entity_cols:
        if col not in df.columns:
            continue
        candidate = df[col].fillna("").astype(str).str.strip()
        candidate = candidate.mask(candidate.eq(""), "__MISSING__")
        parts.append(col + "=" + candidate)
    if not parts:
        return ("row=" + df.index.astype(str)).astype(str)
    group_key = parts[0]
    for extra in parts[1:]:
        group_key = group_key + "|" + extra
    return group_key.astype(str)


def split_train_valid(
    df: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.Series | None]:
    split_cfg = config["split"]
    test_size = float(split_cfg["test_size"])
    random_seed = int(split_cfg["random_seed"])
    strategy = split_cfg["strategy"]
    if strategy == "stratified_group":
        groups = build_group_key(df, config["features"]["entity_cols"])
        splitter = StratifiedGroupKFold(n_splits=max(int(round(1 / test_size)), 5), shuffle=True, random_state=random_seed)
        train_idx, valid_idx = next(splitter.split(df, y, groups))
        return train_idx, valid_idx, groups
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, valid_idx = next(splitter.split(df, y))
    return train_idx, valid_idx, None


def dummy_predictions(y_train: pd.Series, n_samples: int, strategy: str, random_seed: int) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    labels = y_train.to_numpy()
    if strategy == "majority":
        majority = y_train.value_counts().idxmax()
        return np.repeat(majority, n_samples)
    priors = y_train.value_counts(normalize=True)
    return rng.choice(priors.index.to_list(), size=n_samples, p=priors.values)


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    proba: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted")),
        "per_class_recall": {
            label: float(score)
            for label, score in zip(labels, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if proba is not None:
        y_true_bin = label_binarize(y_true, classes=labels)
        try:
            metrics["auc_ovr_macro"] = float(roc_auc_score(y_true_bin, proba, average="macro", multi_class="ovr"))
        except ValueError:
            metrics["auc_ovr_macro"] = None
    return metrics


def append_run_record(path: str | Path, record: dict[str, Any]) -> None:
    path_obj = Path(path)
    row = pd.DataFrame([record])
    if path_obj.exists():
        prev = pd.read_csv(path_obj)
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(path_obj, index=False)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def validate_submission(df: pd.DataFrame, id_col: str, pred_col: str) -> None:
    if id_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"Submission must contain {id_col} and {pred_col}")
    if df[id_col].isna().any():
        raise ValueError(f"{id_col} contains null values")
    if df[pred_col].isna().any():
        raise ValueError(f"{pred_col} contains null values")
    if df[id_col].duplicated().any():
        raise ValueError(f"{id_col} contains duplicates")
