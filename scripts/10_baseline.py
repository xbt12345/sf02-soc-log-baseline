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
    split_train_val_test,
    summarize_split_labels,
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


def run_single_experiment(train_df: pd.DataFrame, config: dict, strategy: str, random_seed: int) -> tuple[dict, CatBoostClassifier]:
    label_col = config['features']['label_col']
    id_col = config['features']['id_col']
    y = train_df[label_col].astype(str)
    labels = sorted(y.unique().tolist())
    train_idx, val_idx, test_idx, groups = split_train_val_test(train_df, y, config, strategy=strategy, random_seed=random_seed)

    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_split = train_df.iloc[val_idx].reset_index(drop=True)
    test_split = train_df.iloc[test_idx].reset_index(drop=True)

    x_train, cat_cols = prepare_route_a_features(train_split, config)
    x_val, _ = prepare_route_a_features(val_split, config)
    x_test, _ = prepare_route_a_features(test_split, config)
    y_train = train_split[label_col].astype(str)
    y_val = val_split[label_col].astype(str)
    y_test = test_split[label_col].astype(str)

    dummy_metrics = {}
    for split_name, y_target in [('val', y_val), ('test', y_test)]:
        dummy_majority = dummy_predictions(y_train, len(y_target), strategy='majority', random_seed=random_seed)
        dummy_random = dummy_predictions(y_train, len(y_target), strategy='random', random_seed=random_seed)
        dummy_metrics[split_name] = {
            'majority': compute_metrics(y_target, dummy_majority, labels),
            'random': compute_metrics(y_target, dummy_random, labels),
        }

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
    model.fit(x_train, y_train, cat_features=cat_cols, eval_set=(x_val, y_val), use_best_model=True)

    val_pred = model.predict(x_val).reshape(-1)
    test_pred = model.predict(x_test).reshape(-1)
    val_proba = model.predict_proba(x_val)
    test_proba = model.predict_proba(x_test)

    result = {
        'strategy': strategy,
        'random_seed': random_seed,
        'group_split_enabled': groups is not None,
        'missing_labels': {
            'val': [label for label in labels if label not in set(y_val)],
            'test': [label for label in labels if label not in set(y_test)],
        },
        'split_rows': {
            'train': int(len(train_split)),
            'val': int(len(val_split)),
            'test': int(len(test_split)),
        },
        'split_label_counts': {
            'train': summarize_split_labels(y_train),
            'val': summarize_split_labels(y_val),
            'test': summarize_split_labels(y_test),
        },
        'class_weights': class_weight_map(y_train),
        'best_iteration': int(model.get_best_iteration()) if model.get_best_iteration() is not None else None,
        'dummy_metrics': dummy_metrics,
        'val_metrics': compute_metrics(y_val, val_pred, labels, val_proba),
        'test_metrics': compute_metrics(y_test, test_pred, labels, test_proba),
        'test_predictions_preview': pd.DataFrame(
            {
                id_col: test_split[id_col].head(10),
                'true_label': y_test.head(10),
                'pred_label': test_pred[:10],
            }
        ).to_dict(orient='records'),
    }
    return result, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--sample-rows', type=int, default=None)
    parser.add_argument('--num-runs', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_paths = resolve_artifact_paths(config)

    train_df = read_table(config['paths']['train_path'])
    label_col = config['features']['label_col']
    random_seed = int(config['split']['random_seed'])
    num_runs = int(args.num_runs or config['split'].get('num_runs', 1))

    if args.sample_rows:
        train_df = stratified_sample(train_df, label_col, args.sample_rows, random_seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = artifact_paths['models'] / 'route_a_catboost.cbm'
    report_path = artifact_paths['reports'] / f'baseline_metrics_{timestamp}.json'
    summary_path = artifact_paths['reports'] / f'baseline_summary_{timestamp}.csv'
    runs_path = artifact_paths['root'] / 'runs.csv'

    strategies = config['split'].get('comparison_strategies') or [config['split']['primary_strategy']]
    primary_strategy = config['split']['primary_strategy']

    all_results = []
    primary_model = None
    primary_result = None

    for strategy in strategies:
        for run_idx in range(num_runs):
            run_seed = random_seed + run_idx
            result, model = run_single_experiment(train_df, config, strategy=strategy, random_seed=run_seed)
            result['run_index'] = run_idx + 1
            all_results.append(result)
            if strategy == primary_strategy and run_idx == 0:
                primary_model = model
                primary_result = result

    if primary_model is None or primary_result is None:
        raise RuntimeError('Primary strategy model was not produced')

    y_full = train_df[label_col].astype(str)
    train_idx, val_idx, _, _ = split_train_val_test(train_df, y_full, config, strategy=primary_strategy, random_seed=random_seed)
    train_val_df = pd.concat([train_df.iloc[train_idx], train_df.iloc[val_idx]], ignore_index=True)
    x_train_val, cat_cols = prepare_route_a_features(train_val_df, config)
    y_train_val = train_val_df[label_col].astype(str)
    best_iteration = primary_result['best_iteration'] if primary_result['best_iteration'] is not None else int(config['model']['iterations']) - 1

    final_model = CatBoostClassifier(
        loss_function='MultiClass',
        iterations=int(best_iteration) + 1,
        depth=int(config['model']['depth']),
        learning_rate=float(config['model']['learning_rate']),
        l2_leaf_reg=float(config['model']['l2_leaf_reg']),
        random_seed=int(config['model']['random_seed']),
        verbose=int(config['model']['verbose']),
        class_weights=class_weight_map(y_train_val),
    )
    final_model.fit(x_train_val, y_train_val, cat_features=cat_cols)
    final_model.save_model(str(model_path))

    summary_rows = []
    for result in all_results:
        summary_rows.append(
            {
                'strategy': result['strategy'],
                'run_index': result['run_index'],
                'random_seed': result['random_seed'],
                'val_missing_labels': '|'.join(result['missing_labels']['val']),
                'test_missing_labels': '|'.join(result['missing_labels']['test']),
                'val_macro_f1': result['val_metrics']['macro_f1'],
                'test_macro_f1': result['test_metrics']['macro_f1'],
                'val_weighted_f1': result['val_metrics']['weighted_f1'],
                'test_weighted_f1': result['test_metrics']['weighted_f1'],
                'test_auc_ovr_macro': result['test_metrics'].get('auc_ovr_macro'),
                'dummy_majority_test_macro_f1': result['dummy_metrics']['test']['majority']['macro_f1'],
                'dummy_random_test_macro_f1': result['dummy_metrics']['test']['random']['macro_f1'],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    save_json(
        report_path,
        {
            'sample_rows': args.sample_rows,
            'num_runs': num_runs,
            'primary_strategy': primary_strategy,
            'results': all_results,
        },
    )

    append_run_record(
        runs_path,
        {
            'timestamp': timestamp,
            'sample_rows': args.sample_rows if args.sample_rows else len(train_df),
            'primary_strategy': primary_strategy,
            'primary_test_macro_f1': primary_result['test_metrics']['macro_f1'],
            'primary_test_weighted_f1': primary_result['test_metrics']['weighted_f1'],
            'primary_test_auc_ovr_macro': primary_result['test_metrics'].get('auc_ovr_macro'),
            'primary_dummy_majority_test_macro_f1': primary_result['dummy_metrics']['test']['majority']['macro_f1'],
            'primary_dummy_random_test_macro_f1': primary_result['dummy_metrics']['test']['random']['macro_f1'],
        },
    )

    print(summary_df.to_string(index=False))
    print(f'Saved final model to {model_path}')
    print(f'Saved report to {report_path}')


if __name__ == '__main__':
    main()
