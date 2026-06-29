from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import load_config, resolve_artifact_paths, save_json


def series_length_stats(series: pd.Series) -> dict[str, float | int]:
    text = series.fillna("").astype(str)
    lengths = text.str.len()
    return {
        "mean": float(lengths.mean()),
        "p50": float(lengths.quantile(0.5)),
        "p90": float(lengths.quantile(0.9)),
        "max": int(lengths.max()),
        "nunique": int(text.nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_paths = resolve_artifact_paths(config)

    train = pd.read_parquet(config["paths"]["train_path"])
    predict_df = pd.read_parquet(config["paths"]["predict_path"])

    id_col = config["features"]["id_col"]
    label_col = config["features"]["label_col"]
    entity_cols = [col for col in config["features"]["entity_cols"] if col in train.columns]
    text_cols = [col for col in config["features"]["text_cols"] if col in train.columns]

    summary: dict[str, object] = {
        "train_shape": list(train.shape),
        "predict_shape": list(predict_df.shape),
        "train_columns": train.columns.tolist(),
        "predict_columns": predict_df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in train.dtypes.items()},
        "id_unique": bool(train[id_col].nunique() == len(train)),
        "label_counts": train[label_col].value_counts(dropna=False).to_dict(),
        "null_rate_top20": train.isna().mean().sort_values(ascending=False).head(20).to_dict(),
        "nunique_top20": train.nunique(dropna=False).sort_values(ascending=False).head(20).to_dict(),
    }

    if "timestamp" in train.columns:
        summary["timestamp_range"] = {
            "min": float(train["timestamp"].min()),
            "max": float(train["timestamp"].max()),
            "quantiles": {str(k): float(v) for k, v in train["timestamp"].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict().items()},
        }

    text_stats = {}
    for col in text_cols:
        text_stats[col] = series_length_stats(train[col])
    summary["text_stats"] = text_stats

    entity_stats = {}
    for col in entity_cols:
        counts = train[col].fillna("__NULL__").astype(str).value_counts()
        entity_stats[col] = {
            "nunique": int(counts.shape[0]),
            "repeat_ratio": float((counts > 1).sum() / max(counts.shape[0], 1)),
            "top10": counts.head(10).to_dict(),
        }
    summary["entity_stats"] = entity_stats

    summary_path = artifact_paths["reports"] / "inspect_summary.json"
    save_json(summary_path, summary)

    md_lines = [
        "# Inspect Summary",
        "",
        f"- train shape: {train.shape}",
        f"- predict shape: {predict_df.shape}",
        f"- id unique: {summary['id_unique']}",
        f"- label counts: {summary['label_counts']}",
        "",
        "## Columns",
    ]
    md_lines.extend([f"- {col}: {dtype}" for col, dtype in summary["dtypes"].items()])
    md_lines.extend(["", "## Entity Repeat Stats"])
    for col, stats in entity_stats.items():
        md_lines.append(f"- {col}: repeat_ratio={stats['repeat_ratio']:.4f}, nunique={stats['nunique']}")
    md_lines.extend(["", "## Text Stats"])
    for col, stats in text_stats.items():
        md_lines.append(f"- {col}: mean={stats['mean']:.2f}, p50={stats['p50']:.2f}, p90={stats['p90']:.2f}, max={stats['max']}, nunique={stats['nunique']}")

    (artifact_paths["reports"] / "inspect_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved inspect summary to {summary_path}")
    print("Label counts:")
    print(train[label_col].value_counts(dropna=False).to_string())
    print("Top entity repeat ratios:")
    for col, stats in entity_stats.items():
        print(f"- {col}: repeat_ratio={stats['repeat_ratio']:.4f}")


if __name__ == "__main__":
    main()
