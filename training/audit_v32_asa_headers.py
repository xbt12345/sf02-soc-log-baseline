"""Post-hoc frozen-model sensitivity to observed ASA header representations; no fitting."""
import hashlib
import json
from pathlib import Path
import sys
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
HEADERS = ["time: entity ", "time entity time: entity ", "time entity entity: entity ", "time entity: entity "]


def main():
    config = json.loads((EXP / "configuration.json").read_text())
    for name, expected in config["source_sha256"].items():
        assert hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() == expected
    sys.path.insert(0, str(FROZEN))
    from v32_features import word_tokens
    rows = []
    for batch in pq.ParquetFile(PREP / "prepared_corpus.parquet").iter_batches(batch_size=8192, columns=["row_position", "format", "text", "group_id", "label"], use_threads=False):
        rows.extend(r for r in batch.to_pylist() if r["format"] == "asa_like")
    positions = np.array([r["row_position"] for r in rows])
    labels = np.array([["benign", "malicious", "suspicious"].index(r["label"]) for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    ids, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    strings = [rows[int(i)]["text"] for i in first]
    original_headers = [s.partition("deny ")[0] for s in strings]
    assert all(h in HEADERS for h in original_headers)
    outer = pq.read_table(PREP / "split_manifest.parquet", columns=["outer_fold"])["outer_fold"].to_numpy()
    fold = outer[positions[first]]
    original = np.empty((len(ids), 3), dtype=np.float32)
    variants = [np.empty_like(original) for _ in HEADERS]
    returned = np.full(len(outer), -1, dtype=np.int8)
    for f in range(3):
        folder = EXP / ("fold_" + str(f))
        report = json.loads((folder / "report.json").read_text())
        for name, field in [("model.joblib", "model_sha256"), ("vectorizer.joblib", "vectorizer_sha256")]:
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == report[field]
        predictions = pq.read_table(folder / "outer_predictions.parquet", columns=["row_position", "C1_W_pred"])
        returned[predictions["row_position"].to_numpy()] = predictions["C1_W_pred"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vectorizer = joblib.load(folder / "vectorizer.joblib")
            model = joblib.load(folder / "model.joblib")
        cv = CountVectorizer(vocabulary=vectorizer.vocabulary_, tokenizer=word_tokens, token_pattern=None, lowercase=False, ngram_range=(1, 2), dtype=np.float32)
        selected = np.flatnonzero(fold == f)
        for start in range(0, len(selected), 1024):
            index = selected[start:start+1024]
            text = [strings[int(i)] for i in index]
            for j, header in enumerate([None] + HEADERS):
                current = text if header is None else [header + "deny " + s.partition("deny ")[2] for s in text]
                assert all(a.partition("deny ")[2] == b.partition("deny ")[2] for a, b in zip(text, current))
                x = normalize(cv.transform(current) @ vectorizer._tfidf._idf_diag, copy=False)
                p = np.asarray(x @ model.coef_.T) + model.intercept_
                p -= p.max(axis=1, keepdims=True); np.exp(p, out=p); p /= p.sum(axis=1, keepdims=True)
                (original if j == 0 else variants[j-1])[index] = p
    base = original[inverse].argmax(axis=1)
    assert np.array_equal(base, returned[positions])
    summaries = []
    for header, group_prob in zip(HEADERS, variants):
        pred = group_prob[inverse].argmax(axis=1)
        flips = pred != base
        changed = np.array([HEADERS.index(h) != HEADERS.index(header) for h in original_headers])[inverse]
        lost = (base == labels) & (pred != labels)
        examples = []
        for i in np.flatnonzero(lost)[:5]:
            examples.append({"row_position": int(positions[i]), "group_id": int(groups[i]), "label": int(labels[i]), "before": rows[int(i)]["text"], "after": header + "deny " + rows[int(i)]["text"].partition("deny ")[2], "before_scores": original[inverse[i]].tolist(), "after_scores": group_prob[inverse[i]].tolist()})
        summaries.append({"target_header": header, "changed_rows": int(changed.sum()), "class_flips": int(flips.sum()), "unique_groups_flipped": int((group_prob.argmax(axis=1) != original.argmax(axis=1)).sum()),
            "broke_previously_correct": int(lost.sum()), "corrected_previously_wrong": int(((base != labels) & (pred == labels)).sum()), "before_confusion": np.bincount(labels * 3 + base, minlength=9).reshape(3, 3).tolist(),
            "after_confusion": np.bincount(labels * 3 + pred, minlength=9).reshape(3, 3).tolist(), "max_score_change": float(np.max(np.abs(group_prob-original))), "examples": examples})
    output = {"scope": "Post-hoc input intervention on known ASA metadata prefix representations; entire substring from deny onward is unchanged. These are development counterexamples, not new labeled environment data or a retrained holdout.",
              "rows": len(rows), "unique_normalized_groups": len(ids), "all_action_bodies_byte_equal": True,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "variants": summaries, "models_retrained": False,
              "limitations": "Metadata header can proxy source/time/annotation process; intervention tests input dependence and does not establish the true attack label of another environment. No port/interface/ACL values or action bodies changed."}
    with (ROOT / "evidence/2026-09-12/v32_asa_header_sensitivity.json").open("x", encoding="utf-8") as stream: json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"rows": len(rows), "groups": len(ids), "variants": [{k:v for k,v in s.items() if k != "examples"} for s in summaries]}), flush=True)


if __name__ == "__main__":
    main()
