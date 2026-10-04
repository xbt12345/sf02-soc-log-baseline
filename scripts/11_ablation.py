from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import (
    compute_metrics,
    dummy_predictions,
    load_config,
    prepare_route_a_features,
    read_table,
    resolve_artifact_paths,
    save_json,
    split_train_val_test,
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


def run_one(train_df: pd.DataFrame, config: dict, strategy: str, ablate_col: str | None, random_seed: int) -> dict:
    local_config = deepcopy(config)
    if ablate_col:
        for key in ['categorical_cols', 'numeric_cols', 'datetime_cols', 'entity_cols']:
            local_config['features'][key] = [c for c in local_config['features'][key] if c != ablate_col]
    label_col = local_config['features']['label_col']
    y = train_df[label_col].astype(str)
    labels = sorted(y.unique().tolist())
    train_idx, val_idx, test_idx, _ = split_train_val_test(train_df, y, local_config, strategy=strategy, random_seed=random_seed)
    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_split = train_df.iloc[val_idx].reset_index(drop=True)
    test_split = train_df.iloc[test_idx].reset_index(drop=True)

    x_train, cat_cols = prepare_route_a_features(train_split, local_config)
    x_val, _ = prepare_route_a_features(val_split, local_config)
    x_test, _ = prepare_route_a_features(test_split, local_config)
    y_train = train_split[label_col].astype(str)
    y_val = val_split[label_col].astype(str)
    y_test = test_split[label_col].astype(str)

    model_cfg = local_config['model']
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
    model.fit(x_train, y_train, cat_features=cat_cols, eval_set=(x_val, y_val), use_best_model=True)

    test_pred = model.predict(x_test).reshape(-1)
    test_proba = model.predict_proba(x_test)
    dummy_majority = dummy_predictions(y_train, len(y_test), strategy='majority', random_seed=random_seed)

    return {
        'ablate_col': ablate_col or '__none__',
        'strategy': strategy,
        'test_macro_f1': compute_metrics(y_test, test_pred, labels, test_proba)['macro_f1'],
        'test_weighted_f1': compute_metrics(y_test, test_pred, labels, test_proba)['weighted_f1'],
        'dummy_majority_test_macro_f1': compute_metrics(y_test, dummy_majority, labels)['macro_f1'],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--sample-rows', type=int, default=50000)
    parser.add_argument('--strategy', default='stratified_random')
    parser.add_argument('--columns', nargs='*', default=['pipeline', 'src_ip', 'src_host', 'username', 'product_name', 'vendor_name'])
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_paths = resolve_artifact_paths(config)
    train_df = read_table(config['paths']['train_path'])
    label_col = config['features']['label_col']
    seed = int(config['split']['random_seed'])
    train_df = stratified_sample(train_df, label_col, args.sample_rows, seed)

    results = [run_one(train_df, config, args.strategy, None, seed)]
    for col in args.columns:
        results.append(run_one(train_df, config, args.strategy, col, seed))

    out_df = pd.DataFrame(results).sort_values(by='test_macro_f1', ascending=False).reset_index(drop=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = artifact_paths['reports'] / f'ablation_summary_{ts}.csv'
    json_path = artifact_paths['reports'] / f'ablation_summary_{ts}.json'
    out_df.to_csv(csv_path, index=False)
    save_json(json_path, {'results': out_df.to_dict(orient='records')})
    print(out_df.to_string(index=False))
    print(f'Saved ablation report to {csv_path}')


if __name__ == '__main__':
    main()
