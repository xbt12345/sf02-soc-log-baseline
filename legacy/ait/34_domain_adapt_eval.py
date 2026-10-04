from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common import (
    compute_metrics,
    load_config,
    prepare_route_a_features,
    read_table,
    resolve_artifact_paths,
    save_json,
)


def stratified_sample(df: pd.DataFrame, label_col: str, n_rows: int, random_seed: int) -> pd.DataFrame:
    if n_rows is None or n_rows <= 0 or n_rows >= len(df):
        return df.reset_index(drop=True)
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


def split_target_domain(
    df: pd.DataFrame,
    label_col: str,
    adapt_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[label_col].astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - adapt_size, random_state=random_seed)
    adapt_idx, test_idx = next(splitter.split(df, y))
    adapt_df = df.iloc[adapt_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return adapt_df, test_df


def fit_model(
    train_df: pd.DataFrame,
    config: dict,
    random_seed: int,
    sample_weights: np.ndarray | None = None,
) -> CatBoostClassifier:
    label_col = config['features']['label_col']
    x_train, cat_cols = prepare_route_a_features(train_df, config)
    y_train = train_df[label_col].astype(str)
    model_cfg = config['model']
    model = CatBoostClassifier(
        loss_function='MultiClass',
        iterations=int(model_cfg['iterations']),
        depth=int(model_cfg['depth']),
        learning_rate=float(model_cfg['learning_rate']),
        l2_leaf_reg=float(model_cfg['l2_leaf_reg']),
        random_seed=int(random_seed),
        verbose=int(model_cfg['verbose']),
        class_weights=class_weight_map(y_train),
    )
    fit_kwargs = {'cat_features': cat_cols}
    if sample_weights is not None:
        fit_kwargs['sample_weight'] = sample_weights
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def evaluate_model(model: CatBoostClassifier, eval_df: pd.DataFrame, config: dict) -> dict:
    label_col = config['features']['label_col']
    x_eval, _ = prepare_route_a_features(eval_df, config)
    y_true = eval_df[label_col].astype(str)
    labels = [str(x) for x in model.classes_]
    pred = model.predict(x_eval).reshape(-1)
    proba = model.predict_proba(x_eval)
    metrics = compute_metrics(y_true, pred, labels=labels, proba=proba)
    return {
        'rows': int(len(eval_df)),
        'label_distribution': {str(k): int(v) for k, v in y_true.value_counts().to_dict().items()},
        'metrics': metrics,
    }


def evaluate_existing_model(model_path: Path, eval_df: pd.DataFrame, config: dict) -> dict:
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return evaluate_model(model, eval_df, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_ait_external_eval.yaml')
    parser.add_argument('--external-data', required=True, help='AIT labeled parquet built by 32_ait_build_eval_dataset.py')
    parser.add_argument('--model-path', required=True, help='Competition-trained CatBoost model path')
    parser.add_argument('--adapt-size', type=float, default=0.2, help='Fraction of target-domain labeled data used for adaptation')
    parser.add_argument('--source-sample-rows', type=int, default=300000, help='Optional stratified sample from competition train set')
    parser.add_argument('--target-weight', type=float, default=5.0, help='Upweight target-domain adaptation samples in joint training')
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_paths = resolve_artifact_paths(config)
    label_col = config['features']['label_col']

    source_df = read_table(config['paths']['train_path'])
    source_df = stratified_sample(source_df, label_col, args.source_sample_rows, args.random_seed)
    target_df = read_table(args.external_data)
    adapt_df, test_df = split_target_domain(target_df, label_col, args.adapt_size, args.random_seed)

    results: dict[str, object] = {
        'config': {
            'adapt_size': args.adapt_size,
            'source_sample_rows': args.source_sample_rows,
            'target_weight': args.target_weight,
            'random_seed': args.random_seed,
        },
        'source_rows': int(len(source_df)),
        'target_rows': int(len(target_df)),
        'target_adapt_rows': int(len(adapt_df)),
        'target_test_rows': int(len(test_df)),
        'target_adapt_label_distribution': {str(k): int(v) for k, v in adapt_df[label_col].value_counts().to_dict().items()},
        'target_test_label_distribution': {str(k): int(v) for k, v in test_df[label_col].value_counts().to_dict().items()},
    }

    model_path = Path(args.model_path).resolve()
    if model_path.exists():
        results['source_only_zero_shot'] = evaluate_existing_model(model_path, test_df, config)

    target_only_model = fit_model(adapt_df, config, random_seed=args.random_seed)
    results['target_only'] = evaluate_model(target_only_model, test_df, config)

    source_train = source_df.copy()
    target_train = adapt_df.copy()
    combined_df = pd.concat([source_train, target_train], ignore_index=True)
    source_weights = np.ones(len(source_train), dtype='float32')
    target_weights = np.full(len(target_train), float(args.target_weight), dtype='float32')
    combined_weights = np.concatenate([source_weights, target_weights])
    adapted_model = fit_model(combined_df, config, random_seed=args.random_seed, sample_weights=combined_weights)
    results['source_plus_target_adapted'] = evaluate_model(adapted_model, test_df, config)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = artifact_paths['reports'] / f'domain_adapt_eval_{timestamp}.json'
    save_json(report_path, results)

    print(f'Saved domain adaptation report to {report_path}')
    for key in ['source_only_zero_shot', 'target_only', 'source_plus_target_adapted']:
        if key not in results:
            continue
        block = results[key]
        metrics = block['metrics']
        print(f'[{key}]')
        print(f"rows={block['rows']}")
        print(f"macro_f1={metrics['macro_f1']:.6f}")
        print(f"weighted_f1={metrics['weighted_f1']:.6f}")
        print(f"accuracy={metrics['accuracy']:.6f}")
        print(f"per_class_recall={metrics['per_class_recall']}")


if __name__ == '__main__':
    main()
