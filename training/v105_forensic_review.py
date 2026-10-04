"""Read-only v10.4 error audit for planning the next experiment.

This script does not fit a model or read the private answer file. Root-local
prefix statistics are descriptive: root IDs must not become model features.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v105_review_20260928"
LEDGER = ROOT / "artifacts/v104_phase_b_20260928/phase_B_ASA_OOF_ledger.parquet"
POSTFIT = ROOT / "artifacts/v104_phase_b_20260928/postfit_error_mechanism.json"
SUPPORT = ROOT / "artifacts/v103_plan_preflight_20260928/ASA_support_preflight.parquet"
ROWS = ROOT / "artifacts/v75_four_arm_20260921_r2/rows.parquet"
PROJECTIONS = ROOT / "artifacts/v75_four_arm_20260921_r2/projections.parquet"
TRAIN = ROOT / "data/official/train.parquet"
N1_GROUPS = ROOT / "artifacts/v101_full_input_group_n1_20260928/N1_ASA_group_map.parquet"
N1_MATRIX = ROOT / "artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz"
TEXTS = ROOT / "artifacts/v75_four_arm_20260921_r2/text_dictionary.parquet"
PROFILE_ROOTS = (2868, 637660, 27221, 3929, 46091)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def performance(d):
    return {
        name: {
            "errors": int((d[name] != d.truth).sum()),
            "M_correct": int(((d.truth == 1) & (d[name] == 1)).sum()),
            "S_correct": int(((d.truth == 2) & (d[name] == 2)).sum()),
        }
        for name in ("N1_teacher", "C_TabM_epoch25")
    }


def prior_distinct_ports(times, ports):
    """Count distinct valid ports strictly before each timestamp, within a root."""
    order = np.argsort(times, kind="stable")
    result = np.zeros(len(times), np.int32)
    seen = set()
    start = 0
    while start < len(times):
        end = start + 1
        while end < len(times) and times[order[end]] == times[order[start]]:
            end += 1
        result[order[start:end]] = len(seen)
        for ix in order[start:end]:
            if isinstance(ports[ix], (int, np.integer)) and 0 <= ports[ix] < 65536:
                seen.add(int(ports[ix]))
        start = end
    return result


def main():
    ledger = pd.read_parquet(LEDGER, columns=[
        "row_position", "root", "fold", "truth", "N1_teacher", "C_TabM_epoch25",
        "C_TabM_epoch25_M_prob", "C_TabM_epoch25_S_prob"
    ])
    support = pd.read_parquet(SUPPORT, columns=["row_position", "behavior"])
    assert len(ledger) == len(support) == 112807
    assert ledger.row_position.is_unique and support.row_position.is_unique
    merged = ledger.merge(support, on="row_position", validate="one_to_one")
    sgroups = ledger[ledger.truth == 2].groupby("root").agg(
        rows=("truth", "size"), correct=("C_TabM_epoch25", lambda x: (x == 2).sum())
    )
    result = {
        "status": "read_only_postfit_diagnosis_not_model_selection",
        "source_sha256": digest(Path(__file__)),
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                         (LEDGER, POSTFIT, SUPPORT, ROWS, PROJECTIONS, TRAIN,
                          N1_GROUPS, N1_MATRIX, TEXTS)},
        "global": {
            "all": performance(ledger),
            "excluding_two_dominant_roots": performance(ledger[~ledger.root.isin((2868, 637660))]),
            "by_fold": {str(k): performance(ledger[ledger.fold == k]) for k in (0, 1, 2)},
            "S_roots": int(len(sgroups)),
            "S_zero_recall_roots": int((sgroups.correct == 0).sum()),
            "S_zero_recall_rows": int(sgroups.loc[sgroups.correct == 0, "rows"].sum()),
            "S_roots_at_most_two_rows": int((sgroups.rows <= 2).sum()),
            "S_zero_recall_roots_at_most_two_rows": int(((sgroups.rows <= 2) & (sgroups.correct == 0)).sum()),
        },
        "root_profiles": {},
        "limitations": [
            "All labels and OOF outputs were already inspected; this is development data.",
            "Connected roots, source symbols, and original structured IPs do not prove real actor identity.",
            "Root-local temporal statistics are descriptive and cannot be used as a prediction feature.",
            "Same coarse behavior does not imply identical event semantics.",
            "Empty projections.text does not mean the actual N1 normalized text branch is empty.",
            "Sanitization-pattern strata were found after reviewing errors and are descriptive, not selected model features.",
            "No classifier, threshold or causal context model was fitted in this audit.",
        ],
    }
    groups = pd.read_parquet(N1_GROUPS, columns=["row_position", "N1_group"])
    grouped = ledger[["row_position", "truth", "C_TabM_epoch25"]].merge(
        groups, on="row_position", validate="one_to_one"
    )
    label_counts = grouped.groupby(["N1_group", "truth"]).size().unstack(fill_value=0)
    mixed = label_counts.gt(0).sum(axis=1).gt(1)
    mixed_rows = grouped.N1_group.isin(label_counts.index[mixed])
    result["exact_N1_input_conflicts"] = {
        "unique_inputs": int(len(label_counts)),
        "mixed_label_inputs": int(mixed.sum()),
        "rows_in_mixed_inputs": int(label_counts.loc[mixed].sum().sum()),
        "empirical_minimum_errors": int((label_counts.sum(axis=1) - label_counts.max(axis=1)).sum()),
        "TabM_errors_in_mixed_inputs": int((grouped.loc[mixed_rows, "truth"] != grouped.loc[mixed_rows, "C_TabM_epoch25"]).sum()),
        "TabM_errors_outside_mixed_inputs": int((grouped.loc[~mixed_rows, "truth"] != grouped.loc[~mixed_rows, "C_TabM_epoch25"]).sum()),
        "not_a_generalization_upper_bound": True,
    }
    # Audit a previously reported label-assisted upper bound. The threshold
    # was chosen on these same OOF truth labels in v10.4 and is never a model.
    postfit = json.loads(POSTFIT.read_text(encoding="utf-8"))
    threshold = postfit["label_assisted_uniform_boundary_upper_bound"]["threshold_selected_with_evaluation_labels"]
    historical = np.where(
        ledger.C_TabM_epoch25_S_prob.to_numpy() - ledger.C_TabM_epoch25_M_prob.to_numpy() > threshold,
        2, 1,
    )
    def threshold_metrics(mask):
        labels = ledger.truth.to_numpy()[mask]
        preds = historical[mask]
        return {"M_wrong": int(((labels == 1) & (preds == 2)).sum()),
                "S_correct": int(((labels == 2) & (preds == 2)).sum())}
    result["historical_label_assisted_threshold_diagnostic"] = {
        "threshold": threshold,
        "all": threshold_metrics(np.ones(len(ledger), dtype=bool)),
        "excluding_two_dominant_roots": threshold_metrics(
            ~ledger.root.isin((2868, 637660)).to_numpy()),
        "root_2868": threshold_metrics(ledger.root.eq(2868).to_numpy()),
        "root_637660": threshold_metrics(ledger.root.eq(637660).to_numpy()),
        "not_a_valid_calibrator": True,
    }
    rowmeta = pd.read_parquet(ROWS, columns=[
        "row_position", "route", "projection_id", "new_text_id", "source_symbol"
    ])
    projections = pd.read_parquet(PROJECTIONS, columns=["facts"])
    asa = rowmeta[rowmeta.route == "asa"]
    assert len(asa) == len(ledger)
    dictionary = pd.read_parquet(TEXTS, columns=["text_id", "text"]).set_index("text_id").text
    n1 = sparse.load_npz(N1_MATRIX)
    assert n1.shape == (22546, 66287)
    result["ASA_N1_text_input"] = {
        "rows": int(len(asa)),
        "nonempty_normalized_text_rows": int(sum(
            bool(dictionary.loc[int(i)]) for i in asa.new_text_id.to_numpy()
        )),
        "unique_normalized_text_ids": int(asa.new_text_id.nunique()),
        "N1_byte_branch_nonzero_values": int(n1[:, :65792].nnz),
        "interpretation": "N1 includes normalized single-event text and parsed facts; projection.text is a different intermediate field.",
    }
    raw = pd.read_parquet(TRAIN, columns=[
        "timestamp", "src_ip", "dst_ip", "product_name", "message_sanitized"
    ])
    wrapper_pattern = raw.message_sanitized.iloc[ledger.row_position.to_numpy()].fillna("").str.contains(
        r"\dCRED-\dCRED-", regex=True
    ).to_numpy()
    repaired_s = (ledger.truth.to_numpy() == 2) & (
        ledger.N1_teacher.to_numpy() != 2
    ) & (ledger.C_TabM_epoch25.to_numpy() == 2)
    result["synthetic_nested_wrapper_stratum"] = {
        "raw_regex": r"\dCRED-\dCRED-",
        "rows": int(wrapper_pattern.sum()),
        "label_rows": {
            str(k): int(v) for k, v in ledger.loc[wrapper_pattern, "truth"].value_counts().items()
        },
        "performance_with_pattern": performance(ledger[wrapper_pattern]),
        "performance_without_pattern": performance(ledger[~wrapper_pattern]),
        "repaired_S_with_pattern": int((repaired_s & wrapper_pattern).sum()),
        "all_repaired_S": int(repaired_s.sum()),
        "pattern_rows_in_root_637660": int((wrapper_pattern & ledger.root.eq(637660).to_numpy()).sum()),
        "roots": int(ledger.loc[wrapper_pattern, "root"].nunique()),
        "by_fold": {
            str(fold): {
                "rows": int((wrapper_pattern & ledger.fold.eq(fold).to_numpy()).sum()),
                "roots": int(ledger.loc[wrapper_pattern & ledger.fold.eq(fold).to_numpy(), "root"].nunique()),
            }
            for fold in (0, 1, 2)
        },
        "warning": "Label correlation plus model success is not causal attribution; require fixed-model legal wrapper counterfactual and matched retrain.",
    }
    pattern_positions = set(ledger.row_position.to_numpy()[wrapper_pattern])
    result["synthetic_nested_wrapper_stratum"]["body_source_symbols"] = int(
        rowmeta.loc[rowmeta.row_position.isin(pattern_positions), "source_symbol"].nunique()
    )
    source_rows = ledger[["row_position", "truth"]].merge(
        rowmeta[["row_position", "source_symbol", "projection_id"]],
        on="row_position", validate="one_to_one"
    )
    used = set(source_rows.projection_id)
    port_map = {
        i: json.loads(facts).get("dst_port_fixed")
        for i, facts in enumerate(projections.facts) if i in used
    }
    source_rows["dst_port"] = [port_map[int(i)] for i in source_rows.projection_id]
    source_rows["timestamp"] = pd.to_numeric(
        raw.timestamp.iloc[source_rows.row_position.to_numpy()].to_numpy(), errors="coerce"
    )
    assert np.isfinite(source_rows.timestamp.to_numpy(dtype=float)).all()
    source_summary = []
    for _, block in source_rows.groupby("source_symbol", sort=False):
        block = block.sort_values("timestamp")
        times = block.timestamp.to_numpy(dtype=float)
        ports = block.dst_port.to_numpy()
        begin = 0
        counts = Counter()
        maximum = 0
        for end, (time, port) in enumerate(zip(times, ports)):
            while times[begin] < time - 300:
                old = ports[begin]
                if isinstance(old, (int, np.integer)) and 0 <= old < 65536:
                    counts[old] -= 1
                    if counts[old] == 0:
                        del counts[old]
                begin += 1
            if isinstance(port, (int, np.integer)) and 0 <= port < 65536:
                counts[port] += 1
            maximum = max(maximum, len(counts))
        m = int((block.truth == 1).sum())
        s = int((block.truth == 2).sum())
        kind = "mixed" if m and s else "M_only" if m else "S_only"
        source_summary.append((kind, len(block), maximum))
    source_summary = pd.DataFrame(source_summary, columns=["kind", "rows", "max_distinct_dst_ports_5m"])
    result["source_local_port_diversity_descriptive"] = {
        kind: {
            "sources": int(len(z)),
            "rows": int(z.rows.sum()),
            "sources_at_least_10_ports": int((z.max_distinct_dst_ports_5m >= 10).sum()),
            "sources_at_least_50_ports": int((z.max_distinct_dst_ports_5m >= 50).sum()),
            "median_max_ports": float(z.max_distinct_dst_ports_5m.median()),
        }
        for kind, z in source_summary.groupby("kind")
    }
    result["source_local_port_diversity_descriptive"]["warning"] = (
        "Source symbol may not identify a real actor; thresholds 10/50 and observed labels are descriptive, not rules."
    )
    for root in PROFILE_ROOTS:
        r = merged[merged.root == root].merge(rowmeta, on="row_position", validate="one_to_one")
        assert len(r) and r.fold.nunique() == 1
        orig = raw.iloc[r.row_position.to_numpy()]
        facts = [json.loads(projections.iloc[int(i)].facts) for i in r.projection_id]
        ports = [x.get("dst_port_fixed") for x in facts]
        valid_ports = {int(x) for x in ports if isinstance(x, int) and 0 <= x < 65536}
        times = pd.to_numeric(orig.timestamp, errors="coerce").to_numpy(dtype=float)
        assert np.isfinite(times).all()
        prior = prior_distinct_ports(times, ports)
        behavior = r.behavior.value_counts().index[0]
        train = merged[merged.fold != int(r.fold.iloc[0])]
        same = train[train.behavior == behavior]
        result["root_profiles"][str(root)] = {
            "rows": int(len(r)), "fold": int(r.fold.iloc[0]),
            "label_rows": {str(k): int(v) for k, v in r.truth.value_counts().items()},
            "teacher_correct": int((r.N1_teacher == r.truth).sum()),
            "tabm_correct": int((r.C_TabM_epoch25 == r.truth).sum()),
            "duration_seconds": float(times.max() - times.min()),
            "structured_source_count": int(orig.src_ip.nunique(dropna=False)),
            "structured_destination_count": int(orig.dst_ip.nunique(dropna=False)),
            "body_source_symbol_count": int(r.source_symbol.nunique()),
            "valid_destination_port_count": len(valid_ports),
            "unknown_destination_port_rows": int(sum(x not in valid_ports for x in ports)),
            "prior_distinct_valid_ports_quantiles": {
                str(k): float(v) for k, v in zip((0, .1, .5, .9, 1), np.quantile(prior, (0, .1, .5, .9, 1)))
            },
            "product_name_counts": {str(k): int(v) for k, v in orig.product_name.value_counts(dropna=False).items()},
            "largest_strict_behavior": behavior,
            "largest_strict_behavior_rows": int((r.behavior == behavior).sum()),
            "other_fold_same_behavior_rows_by_label": {
                str(k): int(v) for k, v in same.truth.value_counts().items()
            },
            "other_fold_same_behavior_roots_by_label": {
                str(k): int(v) for k, v in same.groupby("truth").root.nunique().items()
            },
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnosis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(OUT / "diagnosis.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
