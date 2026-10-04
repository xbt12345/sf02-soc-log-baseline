"""Audit risks of model-driven label filtering on frozen OOF results; never train or relabel."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "artifacts/v32_ready_20260912"
EXP = ROOT / "artifacts/v32_cloud_return_20260912T040052Z/round_20260912T040052Z"
OUT = ROOT / "evidence/2026-09-12/v33_label_influence_audit.json"
LABELS = ["benign", "malicious", "suspicious"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode(column):
    d = column.combine_chunks().dictionary_encode()
    return d.indices.to_numpy(), d.dictionary


def main():
    start = time.perf_counter()
    config = read(EXP / "configuration.json")
    identities = {}
    for name, expected in [
        ("prepared_corpus.parquet", config["corpus_sha256"]),
        ("group_manifest.parquet", read(PREP / "result.json")["group_manifest_sha256"]),
        ("split_manifest.parquet", config["split_sha256"]),
    ]:
        actual = sha(PREP / name)
        if actual != expected:
            raise ValueError("Identity mismatch: " + name)
        identities[name] = actual
    gm = pq.read_table(PREP / "group_manifest.parquet")
    y = gm["label_index"].to_numpy().astype(np.int32)
    groups = gm["group_id"].to_numpy()
    meta = pq.read_table(PREP / "prepared_corpus.parquet",
                         columns=["row_position", "label", "product", "text", "raw_hash"])
    n = len(y)
    assert np.array_equal(meta["row_position"].to_numpy(), np.arange(n))
    assert meta["label"].to_pylist() == [LABELS[int(c)] for c in y]
    product, product_names = encode(meta["product"])
    product_names = product_names.to_pylist()
    text_code, text_values = encode(meta["text"])
    raw_code, raw_values = encode(meta["raw_hash"])
    folds = pq.read_table(PREP / "split_manifest.parquet", columns=["outer_fold"])["outer_fold"].to_numpy()
    scores = np.full((n, 3), np.nan, dtype=np.float32)
    pred = np.full(n, -1, dtype=np.int8)
    seen = np.zeros(n, dtype=np.uint8)
    for fold in range(3):
        folder = EXP / ("fold_" + str(fold))
        report = read(folder / "report.json")
        path = folder / "outer_predictions.parquet"
        actual = sha(path)
        assert actual == report["outer_prediction_sha256"]
        identities[str(path.relative_to(ROOT))] = actual
        t = pq.read_table(path)
        pos = t["row_position"].to_numpy()
        assert np.array_equal(pos, np.flatnonzero(folds == fold))
        assert np.array_equal(t["label_index"].to_numpy(), y[pos])
        seen[pos] += 1
        p = np.column_stack([t["C1_W_p_" + c].to_numpy() for c in LABELS])
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        assert np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0)
        assert np.array_equal(p.argmax(axis=1), t["C1_W_pred"].to_numpy())
        scores[pos] = p
        pred[pos] = p.argmax(axis=1)
    assert (seen == 1).all()
    true_score = scores[np.arange(n), y]

    def summarize(mask, affected):
        out = {}
        for c, name in enumerate(LABELS):
            m = mask & (y == c)
            support, count = int(m.sum()), int((m & affected).sum())
            out[name] = {
                "support": support, "affected_rows": count,
                "affected_fraction": count / support if support else None,
                "support_groups": int(len(np.unique(groups[m]))),
                "groups_with_affected_rows": int(len(np.unique(groups[m & affected]))),
            }
        return out

    all_rows = np.ones(n, dtype=bool)
    policies = {
        "drop_or_downweight_if_argmax_disagrees": pred != y,
        "drop_or_downweight_if_true_label_score_below_0_5": true_score < 0.5,
        "drop_or_downweight_if_true_label_score_below_0_9": true_score < 0.9,
        "high_confidence_disagreement_at_least_0_9": (pred != y) & (scores.max(axis=1) >= 0.9),
    }
    filtering = {}
    for name, affected in policies.items():
        filtering[name] = {
            "all": summarize(all_rows, affected),
            "by_product": {p: summarize(product == i, affected) for i, p in enumerate(product_names)},
        }

    def conflict_report(code, count, name):
        counts = np.bincount(code.astype(np.int64) * 3 + y,
                             minlength=count * 3).reshape(count, 3)
        mixed = (counts > 0).sum(axis=1) > 1
        sizes = counts.sum(axis=1)
        majority = counts.argmax(axis=1)
        changed = mixed[code] & (majority[code] != y)
        assert int(changed.sum()) == int((sizes[mixed] - counts[mixed].max(axis=1)).sum())
        return {
            "representation": name, "unique_values": int(count),
            "mixed_label_values": int(mixed.sum()), "rows_in_mixed_values": int(sizes[mixed].sum()),
            "observed_minimum_errors_for_one_deterministic_class_per_value": int(changed.sum()),
            "majority_vote_hypothetical_label_changes": summarize(all_rows, changed),
            "majority_vote_tie_rule": "benign before malicious before suspicious; diagnostic only",
            "mixed_rows_by_class": np.bincount(y[mixed[code]], minlength=3).tolist(),
            "note": "Same observable text/hash is not proof of the same incident, true label, or mislabel. Other context may differ.",
        }

    result = {
        "scope": "Post-hoc audit of frozen v3.2 OOF scores and official prepared representations. No fit, deletion, weight change, label correction, or new holdout evaluation.",
        "labels": LABELS, "rows": n, "identity_sha256": identities,
        "script_sha256": sha(Path(__file__)),
        "filtering_scenarios": filtering,
        "conflicts": [
            conflict_report(text_code, len(text_values), "v3.2 exact prepared text"),
            conflict_report(raw_code, len(raw_values), "raw_hash stored by frozen preparation"),
        ],
        "checks": {
            "input_identity_matches_frozen_receipts": True,
            "all_outer_rows_once_with_correct_roles_labels": True,
            "probabilities_valid_and_argmax_matches": True,
            "majority_change_count_equals_observed_error_bound": True,
        },
        "limitations": [
            "The supplied labels are the audit reference, not independently verified attack truth.",
            "The scores are uncalibrated; 0.5 and 0.9 are illustrative filtering thresholds, not selected strategies.",
            "Noisy-label algorithms were not run; these scenarios expose risks of simplistic model-agreement filtering only.",
            "The source model already has observed shortcut sensitivity; its agreement is not independent evidence.",
            "Neither OOF predictions nor exact-message groups establish independence of incidents or environments.",
            "Original data were not rehashed by this script; prepared inputs and prediction files were rehashed.",
        ],
        "elapsed_seconds": time.perf_counter() - start,
    }
    if OUT.exists():
        raise FileExistsError("Preserve existing audit: " + str(OUT))
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": n, "filtering": {k: v["all"] for k, v in filtering.items()},
        "conflicts": result["conflicts"], "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

