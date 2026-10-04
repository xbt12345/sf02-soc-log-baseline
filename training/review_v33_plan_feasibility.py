"""Read-only model diagnostics for planning v3.3; no new fit or deployed thresholds."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "artifacts/v32_ready_20260912"
EXP = ROOT / "artifacts/v32_cloud_return_20260912T040052Z/round_20260912T040052Z"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    meta = pq.read_table(PREP / "prepared_corpus.parquet", columns=["product", "format"])
    gm = pq.read_table(PREP / "group_manifest.parquet")
    y, groups, informative = [gm[k].to_numpy() for k in ["label_index", "group_id", "informative"]]
    sp = pq.read_table(PREP / "split_manifest.parquet")
    product = meta["product"].combine_chunks().dictionary_encode()
    names, codes = product.dictionary.to_pylist(), product.indices.to_numpy()
    budgets = [0.0001, 0.001]
    rows = []; fit_seconds = 0.0; views = []
    per_fold_score_limits = []
    for fold in range(3):
        folder = EXP / ("fold_" + str(fold))
        report = load(folder / "report.json")
        for fname in ["C_0_1_fit.json", "C_1_0_fit.json", "refit_fit.json"]:
            fit_seconds += load(folder / fname)["seconds"]
        cal = pq.read_table(folder / "calibration_predictions.parquet")
        out = pq.read_table(folder / "outer_predictions.parquet")
        for table, fname, key in [(cal, "calibration_predictions.parquet", "calibration_prediction_sha256"), (out, "outer_predictions.parquet", "outer_prediction_sha256")]:
            assert hashlib.sha256((folder / fname).read_bytes()).hexdigest() == report[key]
        cpos, opos = cal["row_position"].to_numpy(), out["row_position"].to_numpy()
        crisk = 1 - cal["C1_W_p_benign"].to_numpy().astype(float)
        orisk = 1 - out["C1_W_p_benign"].to_numpy().astype(float)
        inner = sp["inner_for_outer_" + str(fold)].to_numpy()
        role_counts = {}
        for key, mask in [("fit", (inner >= 0) & (inner <= 2)), ("fit_selection", (inner >= 0) & (inner <= 3)), ("calibration", inner == 4), ("outer", sp["outer_fold"].to_numpy() == fold)]:
            role_counts[key] = {"rows": int(mask.sum()), "nonempty_groups": len(np.unique(groups[mask & informative])), "blank_rows": int((mask & ~informative).sum())}
        views.append({"fold": fold, "roles": role_counts})
        for code, name in enumerate(names):
            cm = codes[cpos] == code; om = codes[opos] == code
            normal = crisk[cm & (y[cpos] == 0)]
            for budget in budgets:
                global_threshold = next(t["C1_W_calibration"]["threshold"] for t in report["risk_thresholds"] if t["requested_calibration_fpr_budget"] == budget)
                threshold = None
                if len(normal):
                    descending = np.sort(normal)[::-1]
                    threshold = float(np.nextafter(descending[int(math.floor(len(normal) * budget))], np.inf))
                support = np.bincount(y[opos[om]], minlength=3)
                def summarize(threshold):
                    if threshold is None: return None
                    alerts = orisk[om] >= threshold
                    counts = np.bincount(y[opos[om]][alerts], minlength=3)
                    return {"alerts": counts.tolist(), "rates": [float(counts[c] / support[c]) if support[c] else None for c in range(3)]}
                rows.append({"fold": fold, "product": name, "budget": budget,
                    "calibration_support": np.bincount(y[cpos[cm]], minlength=3).tolist(),
                    "calibration_normal_nonempty_groups": len(np.unique(groups[cpos[cm & (y[cpos] == 0) & informative[cpos]]])),
                    "outer_support": support.tolist(), "source_threshold": threshold,
                    "global": summarize(global_threshold), "source_conditional": summarize(threshold)})
                # Deliberately optimistic, label-informed outer bound. Never a proposed threshold.
                outer_normal = orisk[om & (y[opos] == 0)]
                if len(outer_normal):
                    allowed = int(math.floor(len(outer_normal) * budget))
                    bound_threshold = float(np.nextafter(np.sort(outer_normal)[::-1][allowed], np.inf))
                    per_fold_score_limits.append({"fold": fold, "product": name, "budget": budget,
                        "outer_support": support.tolist(), "best_observed_threshold_bound": summarize(bound_threshold)})
    merged = []
    for name in names:
        for budget in budgets:
            cells = [r for r in rows if r["product"] == name and r["budget"] == budget]
            support = np.sum([r["outer_support"] for r in cells], axis=0)
            all_available = all(r["source_conditional"] is not None for r in cells)
            def merge(field):
                if any(r[field] is None for r in cells): return None
                alerts = np.sum([r[field]["alerts"] for r in cells], axis=0)
                return {"alerts": alerts.tolist(), "rates": [float(alerts[c] / support[c]) if support[c] else None for c in range(3)]}
            merged.append({"product": name, "budget": budget, "outer_support": support.tolist(),
                "calibration_normal_rows_each_fold": [r["calibration_support"][0] for r in cells],
                "calibration_normal_groups_each_fold": [r["calibration_normal_nonempty_groups"] for r in cells],
                "all_folds_supported": all_available, "global": merge("global"), "source_conditional": merge("source_conditional")})
    ci = []
    for budget in budgets:
        minimum = math.ceil(math.log(0.05) / math.log1p(-budget))
        high = binomtest(0, minimum, alternative="less").proportion_ci(confidence_level=0.95, method="exact").high
        assert high <= budget
        ci.append({"target_fpr": budget, "minimum_zero_error_iid_validation_trials": minimum, "one_sided_95_upper_at_minimum": high})
    # Feasibility only; product groups removed as a whole, including cross-product duplicates.
    holdout_support = []
    for selected in [["Windows Active Directory"], ["Duo"], ["Barracuda WAF"]]:
        selected_codes = [names.index(name) for name in selected]
        target = np.isin(codes, selected_codes)
        held_groups = np.unique(groups[target])
        excluded = np.isin(groups, held_groups)
        training = ~excluded
        holdout_support.append({"sources": selected, "target_rows": int(target.sum()), "linked_other_rows_excluded": int((excluded & ~target).sum()),
            "remaining_rows_by_class": np.bincount(y[training], minlength=3).tolist(),
            "remaining_nonempty_groups_by_class": [len(np.unique(groups[training & informative & (y == c)])) for c in range(3)],
            "held_rows_by_class": np.bincount(y[target], minlength=3).tolist(),
            "held_nonempty_groups_by_class": [len(np.unique(groups[target & informative & (y == c)])) for c in range(3)]})
    total = load(EXP / "result.json")["elapsed_seconds"]
    score_limits = []
    for name in names:
        for budget in budgets:
            cells = [r for r in per_fold_score_limits if r["product"] == name and r["budget"] == budget]
            if len(cells) != 3: continue
            support = np.sum([r["outer_support"] for r in cells], axis=0)
            alerts = np.sum([r["best_observed_threshold_bound"]["alerts"] for r in cells], axis=0)
            score_limits.append({"product": name, "budget": budget, "support": support.tolist(), "alerts": alerts.tolist(),
                "rates": [float(alerts[c] / support[c]) if support[c] else None for c in range(3)]})
    result = {"status": "plan_diagnostics_only_no_new_model", "scope": "Post-hoc feasibility on already inspected v3.2 calibration/OOF outputs. Conditional cutoffs are diagnostic only, not validated or deployed v3.3 policy.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"cloud_round_seconds": total, "classifier_fit_seconds": fit_seconds, "other_seconds": total-fit_seconds, "fit_fraction": fit_seconds/total, "note": "Residual is not a profiler breakdown; includes I/O, transforms, selection inference, serialization, diagnostics and other overhead."},
        "roles": views, "conditional_thresholds": merged, "per_fold_conditional": rows, "zero_error_iid_requirements": ci,
        "source_holdout_support_current_groups": holdout_support,
        "optimistic_outer_score_limits": {"scope": "Uses each already-seen outer fold's labels to bound attainable recall of its frozen score at an empirical FPR budget. This is deliberately label-informed and is NOT a calibrated performance estimate or selectable policy.", "results": score_limits},
        "limitations": ["No next-round source policy was selected by these outer results; all are already seen development diagnostics.", "Product identity is diagnostic routing metadata, not a classifier feature or proof of an independent organization.", "IID binomial bounds require independent trials and a fixed threshold tested on separate data; neither row count nor approximate group count establishes these assumptions.", "Source holdout counts must be repeated after repaired grouping; unsupported classes cannot be evaluated as full three-class Macro-F1."],
        "elapsed_seconds": time.perf_counter()-started}
    path = ROOT / "evidence/2026-09-12/v33_plan_feasibility.json"
    with path.open("w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    critical = [r for r in merged if r["product"] in ["Duo", "Barracuda WAF", "Windows Active Directory", "ASA Firewall"] and r["budget"] == 0.001]
    print(json.dumps({"runtime": result["runtime"], "threshold_probe": critical, "optimistic_score_limits": [r for r in score_limits if r["product"] in ["Duo", "Barracuda WAF", "Windows Active Directory"] and r["budget"] == 0.001], "holdout_support": holdout_support, "confidence_requirement": ci, "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__": main()
