from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import (
    append_run_record,
    compute_metrics,
    dummy_predictions,
    load_config,
    prepare_route_a_features,
    read_table,
    resolve_artifact_paths,
    save_json,
    split_train_valid,
)


def stratified_sample(df: pd.DataFrame, label_col: str, n_rows: int, random_seed: int) -> pd.DataFrame:
    if n_rows >= len(df):
        return df
    frac = n_rows / len(df)
    rng = np.random.default_rng(random_seed)
    sampled_indices: list[np.ndarray] = []
    value_counts = df[label_col].value_counts()
    for label, count in value_counts.items():
        target_n = max(1, int(round(count * frac)))
        label_indices = df.index[df[label_col] == label].to_numpy()
        take_n = min(target_n, len(label_indices))
        sampled_indices.append(rng.choice(label_indices, size=take_n, replace=False))
    sampled_idx = np.concatenate(sampled_indices)
    sampled = df.loc[sampled_idx].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    if len(sampled) > n_rows:
        sampled = sampled.sample(n_rows, random_state=random_seed).reset_index(drop=True)
    return sampled


def class_weight_map(y: pd.Series) -> dict[str, float]:
    counts = y.value_counts()
    n_classes = len(counts)
    total = len(y)
    return {label: float(total / (n_classes * count)) for label, count in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample-rows", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_paths = resolve_artifact_paths(config)

    train_df = read_table(config["paths"]["train_path"])
    label_col = config["features"]["label_col"]
    id_col = config["features"]["id_col"]
    random_seed = int(config["split"]["random_seed"])

    if args.sample_rows:
        train_df = stratified_sample(train_df, label_col, args.sample_rows, random_seed)

    y = train_df[label_col].astype(str)
    labels = sorted(y.unique().tolist())

    train_idx, valid_idx, groups = split_train_valid(train_df, y, config)
    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    valid_split = train_df.iloc[valid_idx].reset_index(drop=True)

    x_train, cat_cols = prepare_route_a_features(train_split, config)
    x_valid, _ = prepare_route_a_features(valid_split, config)
    y_train = train_split[label_col].astype(str)
    y_valid = valid_split[label_col].astype(str)

    dummy_majority = dummy_predictions(y_train, len(y_valid), strategy="majority", random_seed=random_seed)
    dummy_random = dummy_predictions(y_train, len(y_valid), strategy="random", random_seed=random_seed)
    dummy_metrics = {
        "majority": compute_metrics(y_valid, dummy_majority, labels),
        "random": compute_metrics(y_valid, dummy_random, labels),
    }

    model_cfg = config["model"]
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=int(model_cfg["iterations"]),
        depth=int(model_cfg["depth"]),
        learning_rate=float(model_cfg["learning_rate"]),
        l2_leaf_reg=float(model_cfg["l2_leaf_reg"]),
        random_seed=int(model_cfg["random_seed"]),
        verbose=int(model_cfg["verbose"]),
        class_weights=class_weight_map(y_train),
    )

    model.fit(x_train, y_train, cat_features=cat_cols, eval_set=(x_valid, y_valid), use_best_model=True)

    valid_pred = model.predict(x_valid).reshape(-1)
    valid_proba = model.predict_proba(x_valid)
    model_metrics = compute_metrics(y_valid, valid_pred, labels, valid_proba)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = artifact_paths["models"] / "route_a_catboost.cbm"
    report_path = artifact_paths["reports"] / f"baseline_metrics_{timestamp}.json"
    valid_pred_path = artifact_paths["reports"] / f"valid_predictions_{timestamp}.csv"
    runs_path = artifact_paths["root"] / "runs.csv"

    model.save_model(str(model_path))
    save_json(
        report_path,
        {
            "sample_rows": args.sample_rows,
            "split_strategy": config["split"]["strategy"],
            "train_rows": int(len(train_split)),
            "valid_rows": int(len(valid_split)),
            "labels": labels,
            "dummy_metrics": dummy_metrics,
            "model_metrics": model_metrics,
            "class_weights": class_weight_map(y_train),
            "group_split_enabled": groups is not None,
        },
    )

    pd.DataFrame(
        {
            id_col: valid_split[id_col],
            "true_label": y_valid,
            "pred_label": valid_pred,
        }
    ).to_csv(valid_pred_path, index=False)

    append_run_record(
        runs_path,
        {
            "timestamp": timestamp,
            "sample_rows": args.sample_rows if args.sample_rows else len(train_df),
            "split_strategy": config["split"]["strategy"],
            "macro_f1": model_metrics["macro_f1"],
            "weighted_f1": model_metrics["weighted_f1"],
            "auc_ovr_macro": model_metrics.get("auc_ovr_macro"),
            "dummy_majority_macro_f1": dummy_metrics["majority"]["macro_f1"],
            "dummy_random_macro_f1": dummy_metrics["random"]["macro_f1"],
        },
    )

    print("Dummy baselines:")
    for name, metrics in dummy_metrics.items():
        print(name, metrics["macro_f1"], metrics["weighted_f1"])
    print("Model metrics:")
    print(model_metrics)
    print(f"Saved model to {model_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
