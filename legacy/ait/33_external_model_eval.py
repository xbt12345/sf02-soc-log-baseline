from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from catboost import CatBoostClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common import compute_metrics, load_config, prepare_route_a_features, read_table, resolve_artifact_paths, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--external-data', required=True, help='Parquet built from external labeled dataset')
    parser.add_argument('--model-path', required=True, help='Competition-trained CatBoost model path')
    parser.add_argument('--label-col', default='label_binary')
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_paths = resolve_artifact_paths(config)
    df = read_table(args.external_data)
    label_col = args.label_col
    if label_col not in df.columns:
        raise ValueError(f'Label column not found: {label_col}')

    x_eval, _ = prepare_route_a_features(df, config)
    y_true = df[label_col].astype(str)

    model = CatBoostClassifier()
    model.load_model(args.model_path)
    labels = [str(x) for x in model.classes_]

    pred = model.predict(x_eval).reshape(-1)
    proba = model.predict_proba(x_eval)
    metrics = compute_metrics(y_true, pred, labels=labels, proba=proba)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = artifact_paths['reports'] / f'external_eval_{timestamp}.json'
    preview_cols = [c for c in ['event_id', 'pipeline', 'src_host', 'username', 'message_sanitized', label_col] if c in df.columns]
    preview = df[preview_cols].copy().head(20)
    preview['pred_label'] = pred[: len(preview)]

    save_json(
        report_path,
        {
            'external_data': str(Path(args.external_data).resolve()),
            'model_path': str(Path(args.model_path).resolve()),
            'rows': int(len(df)),
            'label_distribution': {str(k): int(v) for k, v in y_true.value_counts().to_dict().items()},
            'metrics': metrics,
            'preview': preview.to_dict(orient='records'),
        },
    )

    print(f'Saved external eval report to {report_path}')
    print(pd.Series(metrics).to_string())


if __name__ == '__main__':
    main()
