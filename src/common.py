from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    with config_file.open('r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)
    config['__config_dir__'] = str(config_file.parent)
    return config


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def resolve_artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config['paths']['artifacts_dir'])
    if not root.is_absolute():
        root = Path(config['__config_dir__']) / root
    root = ensure_dir(root)
    return {
        'root': root,
        'reports': ensure_dir(root / 'reports'),
        'models': ensure_dir(root / 'models'),
        'submissions': ensure_dir(root / 'submissions'),
    }


def read_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == '.parquet':
        return pd.read_parquet(path_obj, columns=columns)
    if suffix == '.csv':
        return pd.read_csv(path_obj, usecols=columns)
    raise ValueError(f'Unsupported file type: {suffix}')


def fill_categorical(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].fillna('__MISSING__').astype(str)
    return out


def add_time_features(df: pd.DataFrame, datetime_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    created: list[str] = []
    for col in datetime_cols:
        if col not in out.columns:
            continue
        numeric = pd.to_numeric(out[col], errors='coerce')
        out[col] = numeric.fillna(numeric.median())
        dt = pd.to_datetime(out[col], unit='s', errors='coerce', utc=True)
        features = {
            'hour': dt.dt.hour.fillna(-1).astype('int16'),
            'weekday': dt.dt.weekday.fillna(-1).astype('int16'),
            'month': dt.dt.month.fillna(-1).astype('int16'),
            'is_weekend': dt.dt.weekday.isin([5, 6]).astype('int8'),
        }
        for suffix, series in features.items():
            name = f'{col}_{suffix}'
            out[name] = series
            created.append(name)
    return out, created


def prepare_route_a_features(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    feature_cfg = config['features']
    categorical_cols = [c for c in feature_cfg['categorical_cols'] if c in df.columns]
    numeric_cols = [c for c in feature_cfg['numeric_cols'] if c in df.columns]
    datetime_cols = [c for c in feature_cfg['datetime_cols'] if c in df.columns]
    keep_cols = categorical_cols + numeric_cols + datetime_cols
    out = df[keep_cols].copy()
    out = fill_categorical(out, categorical_cols)
    out, _ = add_time_features(out, datetime_cols)
    return out, categorical_cols


def build_group_key(df: pd.DataFrame, entity_cols: list[str]) -> pd.Series:
    parts: list[pd.Series] = []
    for col in entity_cols:
        if col not in df.columns:
            continue
        candidate = df[col].fillna('').astype(str).str.strip()
        candidate = candidate.mask(candidate.eq(''), '__MISSING__')
        parts.append(col + '=' + candidate)
    if not parts:
        return ('row=' + df.index.astype(str)).astype(str)
    group_key = parts[0]
    for extra in parts[1:]:
        group_key = group_key + '|' + extra
    return group_key.astype(str)


def _split_random(
    df: pd.DataFrame,
    y: pd.Series,
    val_size: float,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_idx, test_idx = next(outer.split(df, y))
    train_val_df = df.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    inner_val_ratio = val_size / (1.0 - test_size)
    inner = StratifiedShuffleSplit(n_splits=1, test_size=inner_val_ratio, random_state=random_seed)
    train_idx_rel, val_idx_rel = next(inner.split(train_val_df, y_train_val))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    return train_idx, val_idx, test_idx, None


def _split_group(
    df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    val_size: float,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_idx, test_idx = next(outer.split(df, y, groups))
    train_val_df = df.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    train_val_groups = groups.iloc[train_val_idx]
    inner_val_ratio = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=inner_val_ratio, random_state=random_seed)
    train_idx_rel, val_idx_rel = next(inner.split(train_val_df, y_train_val, train_val_groups))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    return train_idx, val_idx, test_idx, groups


def _split_time(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    if 'timestamp' not in df.columns:
        raise ValueError('chronological split requires timestamp column')
    order = np.argsort(pd.to_numeric(df['timestamp'], errors='coerce').fillna(0).to_numpy())
    n = len(order)
    test_n = max(1, int(round(n * test_size)))
    val_n = max(1, int(round(n * val_size)))
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError('Not enough rows for chronological train/val/test split')
    train_idx = order[:train_n]
    val_idx = order[train_n : train_n + val_n]
    test_idx = order[train_n + val_n :]
    return train_idx, val_idx, test_idx, None


def split_train_val_test(
    df: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
    strategy: str,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series | None]:
    split_cfg = config['split']
    val_size = float(split_cfg['val_size'])
    test_size = float(split_cfg['test_size'])
    seed = int(random_seed if random_seed is not None else split_cfg['random_seed'])
    if strategy == 'stratified_random':
        return _split_random(df, y, val_size, test_size, seed)
    if strategy == 'stratified_group':
        groups = build_group_key(df, config['features']['entity_cols'])
        return _split_group(df, y, groups, val_size, test_size, seed)
    if strategy == 'chronological':
        return _split_time(df, val_size, test_size)
    raise ValueError(f'Unsupported split strategy: {strategy}')


def dummy_predictions(y_train: pd.Series, n_samples: int, strategy: str, random_seed: int) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    if strategy == 'majority':
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
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'per_class_recall': {
            label: float(score)
            for label, score in zip(labels, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0))
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if proba is not None:
        y_true_bin = label_binarize(y_true, classes=labels)
        try:
            metrics['auc_ovr_macro'] = float(roc_auc_score(y_true_bin, proba, average='macro', multi_class='ovr'))
        except ValueError:
            metrics['auc_ovr_macro'] = None
    return metrics


def summarize_split_labels(y: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in y.value_counts(dropna=False).to_dict().items()}


def append_run_record(path: str | Path, record: dict[str, Any]) -> None:
    path_obj = Path(path)
    row = pd.DataFrame([record])
    if path_obj.exists():
        prev = pd.read_csv(path_obj)
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(path_obj, index=False)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def validate_submission(df: pd.DataFrame, id_col: str, pred_col: str) -> None:
    if id_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f'Submission must contain {id_col} and {pred_col}')
    if df[id_col].isna().any():
        raise ValueError(f'{id_col} contains null values')
    if df[pred_col].isna().any():
        raise ValueError(f'{pred_col} contains null values')
    if df[id_col].duplicated().any():
        raise ValueError(f'{id_col} contains duplicates')
