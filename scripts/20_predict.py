from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from catboost import CatBoostClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import load_config, prepare_route_a_features, read_table, resolve_artifact_paths, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_paths = resolve_artifact_paths(config)

    input_path = args.input_path or config["paths"]["predict_path"]
    model_path = args.model_path or str(artifact_paths["models"] / "route_a_catboost.cbm")
    output_path = args.output_path or str(artifact_paths["submissions"] / "res.csv")

    df = read_table(input_path)
    id_col = config["features"]["id_col"]
    x_pred, _ = prepare_route_a_features(df, config)

    model = CatBoostClassifier()
    model.load_model(model_path)
    pred = model.predict(x_pred).reshape(-1)

    submission = pd.DataFrame({id_col: df[id_col], "pred_label": pred})
    validate_submission(submission, id_col=id_col, pred_col="pred_label")
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
