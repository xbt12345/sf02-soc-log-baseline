"""Independent local replay of the returned frozen v3.2 run; never fits a model."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "artifacts/v32_ready_20260912"
EXP = ROOT / "artifacts/v32_cloud_return_20260912T040052Z/round_20260912T040052Z"
FROZEN = ROOT / "platform/v32_bundle_20260912/training"
OUTPUT = ROOT / "evidence/2026-09-12/v32_cloud_full_audit.json"
LABELS = ["benign", "malicious", "suspicious"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def measure(y, p, weights=None):
    cm = np.bincount(y.astype(np.int32) * 3 + p, weights=weights, minlength=9).reshape(3, 3)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    f1 = np.divide(2 * cm.diagonal(), support + predicted, out=np.zeros(3), where=(support + predicted) > 0)
    return {"confusion_matrix": cm.tolist(), "macro_f1": float(f1.mean()) if (support > 0).all() else None,
            "recall": [float(cm[c, c] / support[c]) if support[c] else None for c in range(3)]}


def main():
    started = time.perf_counter()
    config, result, prior_review = [read(EXP / name) for name in ["configuration.json", "result.json", "independent_round_review.json"]]
    checks = {}
    for name, expected in config["source_sha256"].items():
        checks["source_" + name] = sha(EXP / name) == expected == sha(FROZEN / name)
    for name, expected in [("prepared_corpus.parquet", config["corpus_sha256"]),
                           ("split_manifest.parquet", config["split_sha256"]),
                           ("model_stress_fixtures.json", config["stress_fixtures_sha256"])]:
        checks[name + "_identity"] = sha(PREP / name) == expected
    checks["original_train_identity"] = sha(ROOT / "data/official/train.parquet") == config["official_train_sha256"]
    checks["group_manifest_identity"] = sha(PREP / "group_manifest.parquet") == read(PREP / "result.json")["group_manifest_sha256"]
    if not all(checks.values()):
        raise ValueError("Identity failed: " + repr(checks))
    sys.path.insert(0, str(FROZEN))
    from v32_features import word_tokens
    from v32_finalize import prepare_record

    gm = pq.read_table(PREP / "group_manifest.parquet")
    y, groups, informative = [gm[c].to_numpy() for c in ["label_index", "group_id", "informative"]]
    sp = pq.read_table(PREP / "split_manifest.parquet")
    outer = sp["outer_fold"].to_numpy()
    meta = pq.read_table(PREP / "prepared_corpus.parquet", columns=["format", "product", "original_empty", "filtered_empty", "unknown_format", "final_quarantined_fragments", "invalid_string_values", "unverified_field_count"])
    formats_col = meta["format"].combine_chunks().dictionary_encode()
    formats, format_names = formats_col.indices.to_numpy(), formats_col.dictionary.to_pylist()
    product_col = meta["product"].combine_chunks().dictionary_encode()
    products, product_names = product_col.indices.to_numpy(), product_col.dictionary.to_pylist()
    n = len(y)
    counts = np.zeros(n, dtype=np.uint8)
    cpred, npred = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    scores = np.zeros((n, 3), dtype=np.float32)
    alarms = np.zeros((3, n), dtype=bool)
    models, fold_reports, stress_rows, replay_warnings = [], [], [], []
    fixtures = read(PREP / "model_stress_fixtures.json")

    def predict(model, strings):
        counter, diag, coefficients, intercept = model
        matrix = counter.transform(strings).astype(np.float32)
        matrix = matrix @ diag
        matrix = normalize(matrix, norm="l2", copy=False)
        logits = np.asarray(matrix @ coefficients.T) + intercept
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=1, keepdims=True)
        return logits, np.diff(matrix.indptr) == 0

    for f in range(3):
        folder = EXP / ("fold_" + str(f))
        report = read(folder / "report.json")
        fold_reports.append(report)
        for name, field in [("outer_predictions.parquet", "outer_prediction_sha256"), ("calibration_predictions.parquet", "calibration_prediction_sha256"), ("model.joblib", "model_sha256"), ("vectorizer.joblib", "vectorizer_sha256")]:
            checks["fold{}_{}".format(f, name)] = sha(folder / name) == report[field]
        if not all(checks.values()):
            raise ValueError("Fold file hash failure")
        table = pq.read_table(folder / "outer_predictions.parquet")
        pos = table["row_position"].to_numpy()
        cal = pq.read_table(folder / "calibration_predictions.parquet")
        cpos = cal["row_position"].to_numpy()
        inner = sp["inner_for_outer_" + str(f)].to_numpy()
        checks["fold{}_positions_labels".format(f)] = bool(np.array_equal(pos, np.flatnonzero(outer == f)) and np.array_equal(table["label_index"].to_numpy(), y[pos]) and np.array_equal(cpos, np.flatnonzero(inner == 4)) and np.array_equal(cal["label_index"].to_numpy(), y[cpos]))
        counts[pos] += 1
        npred[pos], cpred[pos] = table["N_pred"].to_numpy(), table["C1_W_pred"].to_numpy()
        p = np.column_stack([table["C1_W_p_" + k].to_numpy() for k in LABELS])
        scores[pos] = p
        checks["fold{}_C1_score_decisions".format(f)] = bool(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1) and np.array_equal(p.argmax(axis=1), cpred[pos]))
        joined = (inner >= 0) & (inner <= 3)
        freq = np.bincount(formats[joined] * 3 + y[joined], minlength=len(format_names) * 3).reshape(-1, 3)
        prior = np.bincount(y[joined], minlength=3) / joined.sum()
        nt = np.tile(prior, (len(format_names), 1))
        seen = freq.sum(axis=1) > 0
        nt[seen] = freq[seen] / freq[seen].sum(axis=1, keepdims=True)
        checks["fold{}_N_relearned_frequencies".format(f)] = bool(np.array_equal(nt[formats[pos]].argmax(axis=1), npred[pos]) and np.allclose(nt[formats[pos]], np.column_stack([table["N_p_" + k].to_numpy() for k in LABELS])))
        checks["fold{}_refit_members".format(f)] = hashlib.sha256(np.flatnonzero(joined).astype("<i4").tobytes()).hexdigest() == report["refit_row_positions_sha256"]
        group_role = np.full(int(groups.max()) + 1, -99, dtype=np.int8)
        for role in np.unique(inner):
            ids = np.unique(groups[(inner == role) & informative])
            checks["fold{}_group_role_{}".format(f, role)] = bool((group_role[ids] == -99).all())
            group_role[ids] = role
        for name in ["C_0_1_fit.json", "C_1_0_fit.json", "refit_fit.json"]:
            fit = read(folder / name)
            checks["fold{}_{}".format(f, name)] = fit["converged"] and not fit["warnings"]
        chosen = 0.1 if report["selection_macro_f1"][0] >= report["selection_macro_f1"][1] - 0.001 else 1.0
        checks["fold{}_C_selection".format(f)] = chosen == report["selected_C"]
        for bi, item in enumerate(report["risk_thresholds"]):
            for name in ["N", "C1_W"]:
                crisk = 1 - np.column_stack([cal[name + "_p_" + k].to_numpy() for k in LABELS])[:, 0].astype(float)
                normal = np.sort(crisk[y[cpos] == 0])
                permitted = int(len(normal) * item["requested_calibration_fpr_budget"])
                threshold = float(np.nextafter(normal[len(normal) - permitted - 1], np.inf))
                checks["fold{}_{}_threshold{}".format(f, name, bi)] = threshold == item[name + "_calibration"]["threshold"]
                for scope, positions, risk in [("calibration", cpos, crisk), ("outer", pos, 1 - table[name + "_p_benign"].to_numpy().astype(float))]:
                    alert = risk >= threshold
                    expected = item[name + "_" + scope]
                    ok = int(alert.sum()) == expected["alert_rows"]
                    for c, label in enumerate(LABELS):
                        ok &= abs(float(alert[y[positions] == c].mean()) - expected["class_alert_rates"][label]) < 1e-12
                    checks["fold{}_{}_{}_rates{}".format(f, name, scope, bi)] = bool(ok)
                    if name == "C1_W" and scope == "outer":
                        alarms[bi, pos] = alert
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vectorizer = joblib.load(folder / "vectorizer.joblib")
            classifier = joblib.load(folder / "model.joblib")
        replay_warnings.extend(str(w.message) for w in caught)
        counter = CountVectorizer(vocabulary=vectorizer.vocabulary_, tokenizer=word_tokens, token_pattern=None, lowercase=False, ngram_range=(1, 2), dtype=np.float32)
        model = (counter, vectorizer._tfidf._idf_diag, classifier.coef_, classifier.intercept_)
        models.append(model)
        stress = read(folder / "model_stress.json")
        for pair, returned in zip(fixtures, stress["pairs"]):
            strings = [prepare_record({"message_sanitized": pair[k]})["text"] for k in ["raw_a", "raw_b"]]
            probs, zeros = predict(model, strings)
            stress_rows.append({"fold": f, "name": pair["name"], "kind": pair["kind"], "max_replay_score_difference": float(np.max(np.abs(probs - returned["scores"]))), "both_predicted_benign": bool((probs.argmax(axis=1) == 0).all()), "vectors_equal_reported": returned["vectors_equal"]})
        del table, cal, p, vectorizer, classifier
        gc.collect()

    checks["all_rows_once"] = bool((counts == 1).all())
    metrics = {"C1_W": measure(y, cpred), "N": measure(y, npred)}
    for name in metrics:
        checks[name + "_confusion"] = metrics[name]["confusion_matrix"] == result[name]["confusion_matrix"]
    group_count = np.bincount(groups[informative])
    weights = 1.0 / group_count[groups[informative]]
    equal = {name: measure(y[informative], pred[informative], weights) for name, pred in [("C1_W", cpred), ("N", npred)]}
    for name in equal:
        checks[name + "_equal_groups"] = abs(equal[name]["macro_f1"] - prior_review["nonempty_group_equal"][name]["macro_f1"]) < 1e-10
    largest = []
    for item in prior_review["without_largest_class_group"]:
        pos = np.flatnonzero(outer == item["fold"])
        c = LABELS.index(item["class_selecting_largest_group"])
        ids, sizes = np.unique(groups[pos[(y[pos] == c) & informative[pos]]], return_counts=True)
        gid = int(ids[np.argmax(sizes)])
        keep = pos[groups[pos] != gid]
        actual = measure(y[keep], cpred[keep]); control = measure(y[keep], npred[keep])
        checks["largest_{}_{}".format(item["fold"], c)] = gid == item["group_id"] and actual["confusion_matrix"] == item["result"]["C1_W"]["confusion_matrix"] and control["confusion_matrix"] == item["result"]["N"]["confusion_matrix"]
        largest.append({"fold": item["fold"], "class": LABELS[c], "C1": actual["macro_f1"], "N": control["macro_f1"]})
    slices = {}; returned_slices = read(EXP / "slices.json")
    for field, codes, names in [("format", formats, format_names), ("product", products, product_names)]:
        slices[field] = {}
        for code, name in enumerate(names):
            mask = codes == code
            actual = measure(y[mask], cpred[mask]); control = measure(y[mask], npred[mask])
            checks["slice_{}_{}".format(field, name)] = actual["confusion_matrix"] == returned_slices[field][name]["C1_W"]["confusion_matrix"] and control["confusion_matrix"] == returned_slices[field][name]["N"]["confusion_matrix"]
            corrected = int(((cpred == y) & (npred != y) & mask).sum()); broken = int(((cpred != y) & (npred == y) & mask).sum())
            slices[field][name] = {"C1_W": actual, "N": control, "corrected": corrected, "broken": broken,
                                    "supports": np.bincount(y[mask], minlength=3).tolist(),
                                    "nonempty_groups_by_class": [len(np.unique(groups[mask & informative & (y == c)])) for c in range(3)]}
    risk_report = []
    for bi, budget in enumerate(config["risk_budgets"]):
        item = {"budget": budget, "all_rows": {}, "nonempty": {}, "by_format": {}, "by_product": {}}
        def risk_slice(mask):
            supports = np.bincount(y[mask], minlength=3)
            alerts = np.bincount(y[mask & alarms[bi]], minlength=3)
            return {"supports": supports.tolist(), "alerts": alerts.tolist(), "rates": [float(alerts[c] / supports[c]) if supports[c] else None for c in range(3)], "false_per_10000": float(alerts[0] / supports[0] * 10000) if supports[0] else None}
        item["all_rows"] = risk_slice(np.ones(n, dtype=bool)); item["nonempty"] = risk_slice(informative)
        for code, name in enumerate(format_names): item["by_format"][name] = risk_slice(formats == code)
        for code, name in enumerate(product_names): item["by_product"][name] = risk_slice(products == code)
        risk_report.append(item)
    print(json.dumps({"phase": "independent_counts_complete", "checks": len(checks), "failures": [k for k, v in checks.items() if not v], "group_equal": equal, "risk_summary": [{k:v for k,v in r.items() if k in ["budget", "all_rows", "nonempty"]} for r in risk_report]}), flush=True)

    # Transform each normalized text group once per assigned outer model. All rows are compared.
    cache = np.full((int(groups.max()) + 1, 3), np.nan, dtype=np.float32)
    zero_cache = np.zeros(len(cache), dtype=bool)
    maximum = 0.0; mismatches = 0; comparison_rows = 0; unique_transforms = 0
    for bi, batch in enumerate(pq.ParquetFile(PREP / "prepared_corpus.parquet").iter_batches(batch_size=4096, columns=["row_position", "text"], use_threads=False)):
        pos = batch.column(0).to_numpy(); texts = batch.column(1).to_pylist()
        gids, first = np.unique(groups[pos], return_index=True)
        missing = first[np.isnan(cache[gids, 0])]
        for f in range(3):
            selected = missing[outer[pos[missing]] == f]
            if not len(selected): continue
            prob, zero = predict(models[f], [texts[int(i)] for i in selected])
            cache[groups[pos[selected]]] = prob; zero_cache[groups[pos[selected]]] = zero
            unique_transforms += len(selected)
        actual = cache[groups[pos]]
        maximum = max(maximum, float(np.max(np.abs(actual - scores[pos]))))
        mismatches += int((actual.argmax(axis=1) != cpred[pos]).sum())
        comparison_rows += len(pos)
        if bi % 100 == 0: print(json.dumps({"phase": "frozen_model_replay", "rows": comparison_rows, "decision_mismatches": mismatches, "max_probability_difference": maximum}), flush=True)
    expected_zero = np.asarray(read(EXP / "zero_vector_rows.json")["rows"], dtype=np.int64)
    checks["zero_vector_positions_replay"] = np.array_equal(np.sort(expected_zero), np.flatnonzero(zero_cache[groups]))
    checks["model_replay_all_rows"] = comparison_rows == n and mismatches == 0 and maximum < 1e-5
    checks["stress_score_replay"] = all(r["max_replay_score_difference"] < 1e-5 for r in stress_rows)
    output = {"status": "returned_run_locally_replayed_not_transfer_validated", "archive_sha256": "cc3ef531763615704820025673aed4db8966e02658ccb3148b888b76f4b28270", "source_file_sha256": sha(Path(__file__)),
        "checks": checks, "all_checks_passed": all(checks.values()), "rows": n, "metrics": metrics, "group_equal": equal, "largest_group_removal": largest, "slices": slices, "risk": risk_report,
        "frozen_model_replay": {"rows": comparison_rows, "group_transforms": unique_transforms, "decision_mismatches": mismatches, "max_probability_difference": maximum,
        "method": "Read frozen vocabulary, IDF diagonal, coefficients and intercept; count words/bigrams with frozen tokenizer; sparse TF-IDF/L2 then multinomial softmax. No fitting or parameter changes.", "tolerance": 1e-5, "environment_warnings": sorted(set(replay_warnings))},
        "stress_replay": stress_rows, "zero_vector_rows": int(zero_cache[groups].sum()), "nonempty_zero_vector_rows": int((zero_cache[groups] & informative).sum()),
        "elapsed_seconds": time.perf_counter() - started, "transfer_validated": False, "model_fitted_locally": False,
        "limitations": ["No new source/template holdout, no independent environment, and no probability calibration.", "Grouped OOF model replay does not turn seen development data into an unseen test.", "Model arrays were read under newer local libraries; numerical replay tolerance is reported, not binary runtime compatibility.", "Source hashes and frozen artifacts establish consistency, not trustworthy labels or absence of semantic shortcuts."]}
    with OUTPUT.open("x", encoding="utf-8") as stream: json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": output["status"], "all_checks_passed": output["all_checks_passed"], "checks": len(checks), "failed": [k for k,v in checks.items() if not v], "replay": output["frozen_model_replay"], "seconds": output["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
