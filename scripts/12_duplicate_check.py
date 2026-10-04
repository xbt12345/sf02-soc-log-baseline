from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import load_config, read_table, resolve_artifact_paths, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--sample-rows', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_paths = resolve_artifact_paths(config)
    df = read_table(config['paths']['train_path'])
    if args.sample_rows and args.sample_rows < len(df):
        df = df.sample(args.sample_rows, random_state=int(config['split']['random_seed'])).reset_index(drop=True)

    label_col = config['features']['label_col']
    id_col = config['features']['id_col']
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]

    exact_dup_mask = df.duplicated(subset=feature_cols, keep=False)
    exact_dup_count = int(exact_dup_mask.sum())
    exact_dup_unique_groups = int(df.loc[exact_dup_mask, feature_cols].drop_duplicates().shape[0]) if exact_dup_count else 0

    exact_conflict = (
        df.groupby(feature_cols, dropna=False)[label_col]
        .nunique()
        .reset_index(name='label_nunique')
    )
    exact_conflict_rows = int((exact_conflict['label_nunique'] > 1).sum())

    message_col = 'message_sanitized' if 'message_sanitized' in df.columns else None
    message_dup_count = 0
    message_conflict_rows = 0
    if message_col:
        msg_conflict = (
            df.groupby([message_col], dropna=False)[label_col]
            .nunique()
            .reset_index(name='label_nunique')
        )
        message_conflict_rows = int((msg_conflict['label_nunique'] > 1).sum())
        message_dup_count = int(df.duplicated(subset=[message_col], keep=False).sum())

    key_cols = [c for c in ['pipeline', 'src_ip', 'dst_ip', 'src_port', 'src_host', 'dst_host', 'username', 'product_name', 'vendor_name'] if c in df.columns]
    struct_dup_count = int(df.duplicated(subset=key_cols, keep=False).sum()) if key_cols else 0

    result = {
        'rows_checked': int(len(df)),
        'feature_cols_checked': feature_cols,
        'exact_duplicate_rows': exact_dup_count,
        'exact_duplicate_unique_patterns': exact_dup_unique_groups,
        'exact_duplicate_label_conflict_patterns': exact_conflict_rows,
        'message_duplicate_rows': message_dup_count,
        'message_label_conflict_patterns': message_conflict_rows,
        'structured_key_duplicate_rows': struct_dup_count,
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = artifact_paths['reports'] / f'duplicate_check_{ts}.json'
    save_json(json_path, result)
    print(result)
    print(f'Saved duplicate report to {json_path}')


if __name__ == '__main__':
    main()
