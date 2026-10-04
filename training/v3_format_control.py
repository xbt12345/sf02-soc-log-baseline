"""Official-only development negative control: learn class frequencies by format.

No text classifier. Uses frozen V0/V1 roles and never trains on an outer fold.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

LABELS = ["benign", "malicious", "suspicious"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for block in iter(lambda: s.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def metrics(y, pred):
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {"rows": len(y), "macro_f1": float(f1.mean()) if (support > 0).all() else None,
            "confusion_matrix_true_rows_pred_columns": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            "classes": {name: {"support": int(support[i]), "precision": float(precision[i]) if (pred == i).any() else None,
                       "recall": float(recall[i]) if support[i] else None, "f1": float(f1[i]) if support[i] else None} for i, name in enumerate(LABELS)}}


def threshold_for_budget(normal_risks, budget):
    if len(normal_risks) == 0:
        return None
    allowed = int(np.floor(budget * len(normal_risks)))
    # Above the first score which cannot all be admitted at this budget.
    # Strictly handles ties and the zero-alert candidate.
    ordered = np.sort(normal_risks)[::-1]
    return float(np.nextafter(ordered[min(allowed, len(ordered) - 1)], np.inf))


def alert_metrics(y, risk, threshold):
    if threshold is None:
        return {"status": "no_benign_calibration_support"}
    alerts = risk >= threshold
    rates = {name: float(alerts[y == i].mean()) if (y == i).any() else None for i, name in enumerate(LABELS)}
    return {"threshold": threshold, "alert_rows": int(alerts.sum()), "class_alert_rates": rates,
            "false_alerts_per_10000_benign": rates["benign"] * 10000 if rates["benign"] is not None else None,
            "alert_precision_nonbenign": float((y[alerts] != 0).mean()) if alerts.any() else None}


def main(args):
    run = Path(args.run_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "v3_format_control.py").write_bytes(Path(__file__).read_bytes())
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if digest(run / "split_manifest.parquet") != result["split_sha256"] or digest(run / "prepared_corpus.parquet") != result["corpus_sha256"]:
        raise ValueError("Frozen input identity mismatch")
    df = pd.read_parquet(run / "prepared_corpus.parquet", columns=["row_position", "group_id", "label", "format", "product", "original_empty", "filtered_empty"])
    sm = pd.read_parquet(run / "split_manifest.parquet")
    if not np.array_equal(df.row_position, sm.row_position):
        raise ValueError("Unaligned role manifest")
    y = df.label.map({label: i for i, label in enumerate(LABELS)}).to_numpy(dtype=np.uint8)
    fmt_codes, fmt_values = pd.factorize(df["format"], sort=True)
    # The factorization is only an index; all learned probabilities use fit+selection.
    oof = np.full((len(df), 3), np.nan)
    folds = []
    config = {"scope": "N format-only negative control and C0 all-benign; official-train development only, not a main candidate or transfer test",
              "code_sha256": digest(__file__), "input_corpus_sha256": result["corpus_sha256"], "input_split_sha256": result["split_sha256"],
              "training_roles": "fit plus selection (inner 0..3); calibration inner 4; outer excluded",
              "format_features": "Only prepare format category; no product, message tokens, dates, IPs or missing-product flag",
              "probabilities": "Unsmoothed empirical class frequencies by seen format; unseen format falls back to fit class frequencies. No probability calibration.",
              "risk": "1-P(benign), evaluated separately from three-class argmax", "fpr_budgets": [0.0001, 0.001, 0.01]}
    (out / "configuration.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for fold in range(3):
        inner = sm["inner_for_outer_" + str(fold)].to_numpy()
        fit = (inner >= 0) & (inner <= 3)
        cal = inner == 4
        test = sm.outer_fold.to_numpy() == fold
        if not ((fit.astype(int) + cal.astype(int) + test.astype(int)) == 1).all():
            raise AssertionError("Role coverage mismatch")
        table = np.bincount(fmt_codes[fit]*3 + y[fit], minlength=len(fmt_values)*3).reshape(-1, 3).astype(float)
        totals = table.sum(axis=1)
        prior = table.sum(axis=0) / table.sum()
        table[totals > 0] /= totals[totals > 0, None]
        table[totals == 0] = prior
        scores = table[fmt_codes]
        risk = 1.0 - scores[:, 0]
        pred = scores.argmax(axis=1).astype(np.uint8)
        oof[test] = scores[test]
        budgets = []
        for budget in config["fpr_budgets"]:
            threshold = threshold_for_budget(risk[cal & (y == 0)], budget)
            budgets.append({"budget": budget, "calibration": alert_metrics(y[cal], risk[cal], threshold),
                            "outer_development": alert_metrics(y[test], risk[test], threshold)})
        by_source = {str(source): metrics(y[test & (df["product"] == source).to_numpy()], pred[test & (df["product"] == source).to_numpy()]) for source in sorted(df["product"].unique())}
        largest_removed = {}
        for index, label in enumerate(LABELS):
            mask = test & (y == index) & ~(df.original_empty | df.filtered_empty).to_numpy()
            sizes = df.loc[mask].groupby("group_id").size()
            if len(sizes):
                largest = int(sizes.idxmax())
                retained = test & (df.group_id.to_numpy() != largest)
                largest_removed[label] = {"removed_group": largest, "removed_rows_all_classes": int((test & ~retained).sum()),
                                          "metrics": metrics(y[retained], pred[retained])}
        folds.append({"fold": fold, "fit_rows": int(fit.sum()), "calibration_rows": int(cal.sum()), "outer_rows": int(test.sum()),
                      "learned_format_distributions": {str(name): dict(zip(LABELS, table[i].tolist())) for i, name in enumerate(fmt_values)},
                      "format_only": metrics(y[test], pred[test]), "all_benign": metrics(y[test], np.zeros(test.sum(), dtype=np.uint8)),
                      "risk_threshold_diagnostics": budgets, "by_source": by_source, "largest_group_removal_sensitivity": largest_removed})
    if not np.isfinite(oof).all() or not np.allclose(oof.sum(axis=1), 1):
        raise AssertionError("Incomplete outer predictions")
    pred = oof.argmax(axis=1)
    group_weight = 1.0 / df.groupby("group_id").group_id.transform("size").to_numpy()
    report = {"scope": config["scope"], "main_text_model_trained": False, "transfer_validated": False,
              "pooled_outer_format_only": metrics(y, pred), "pooled_outer_all_benign": metrics(y, np.zeros(len(y), dtype=np.uint8)),
              "pooled_candidate_group_equal_macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], sample_weight=group_weight, zero_division=0)),
              "group_equal_warning": "Candidate text groups are not confirmed independent events; original-empty rows are individual groups.",
              "folds": folds, "limitations": ["Outer folds are development evidence; format distribution shortcuts can persist across these folds.",
                  "Only N and all-benign controls run; no independent fact-text model or source holdout performed.",
                  "No probability calibration, no statistical low-FPR guarantee, no business operating threshold."]}
    pd.DataFrame({"row_position": df.row_position, "outer_fold": sm.outer_fold, "true_label_index": y, "pred_label_index": pred,
                  **{"p_" + label: oof[:, i] for i, label in enumerate(LABELS)}}).to_parquet(out / "outer_predictions.parquet", index=False)
    report["outer_predictions_sha256"] = digest(out / "outer_predictions.parquet")
    (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["scope", "pooled_outer_format_only", "pooled_outer_all_benign", "main_text_model_trained"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    main(parser.parse_args())
