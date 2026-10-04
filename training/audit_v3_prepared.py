"""Read-only checks of frozen V0/V1 output against official train; write a new audit.

No classifier, vocabulary or external data. Python 3.8 compatible.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def main(args):
    out = Path(args.run_dir)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    config = json.loads((out / "configuration.json").read_text(encoding="utf-8"))
    train = Path(args.train)
    checks = {"official_input_sha": sha(train) == config["input_sha256"],
              "corpus_sha": sha(out / "prepared_corpus.parquet") == result["corpus_sha256"],
              "split_sha": sha(out / "split_manifest.parquet") == result["split_sha256"],
              "frozen_source_sha": sha(out / "soc_v3_prepare.py") == config["code_sha256"]}
    if not all(checks.values()):
        raise ValueError("Frozen artifact identity mismatch")
    df = pq.read_table(out / "prepared_corpus.parquet", columns=["row_position", "group_id", "label", "product", "format",
                       "original_empty", "filtered_empty", "unknown_format"]).to_pandas()
    gm = pd.read_parquet(out / "group_manifest.parquet")
    sm = pd.read_parquet(out / "split_manifest.parquet")
    n = len(df)
    checks["all_row_positions_once_in_order"] = all(np.array_equal(t.row_position.to_numpy(), np.arange(n)) for t in [df, gm, sm])
    checks["group_manifest_matches_corpus"] = np.array_equal(df.group_id, gm.group_id)
    checks["label_manifest_matches_corpus"] = np.array_equal(df.label.map({"benign": 0, "malicious": 1, "suspicious": 2}), gm.label_index)
    checks["no_empty_state_overlap"] = not bool((df.original_empty & df.filtered_empty).any())
    checks["empty_state_matches_informative"] = np.array_equal(~(df.original_empty | df.filtered_empty), gm.informative)
    group_column = df.group_id.to_numpy()
    for f in range(3):
        inner = sm["inner_for_outer_" + str(f)].to_numpy()
        outer = sm.outer_fold.to_numpy()
        roles = np.where(outer == f, 3, np.where(inner <= 2, 0, np.where(inner == 3, 1, 2)))
        checks["groups_do_not_cross_roles_fold_" + str(f)] = bool(pd.DataFrame({"g": group_column, "r": roles}).groupby("g").r.nunique().max() == 1)
        checks["valid_inner_roles_fold_" + str(f)] = bool(np.isin(inner[outer != f], range(5)).all() and (inner[outer == f] == -1).all())
    checks["valid_outer_roles"] = bool(np.isin(sm.outer_fold, range(3)).all())
    collision = json.loads((out / "text_collisions.json").read_text(encoding="utf-8"))
    focus = {g["group_id"] for g in collision["largest_conflict_groups"][:8]}
    selected = collections.Counter()
    conflict_examples = []
    raw_groups = {}
    text_groups = {}
    group_texts = {}
    normalized_duplicate_split = 0
    distinct_texts_merged = 0
    duplicate_split = 0
    raw_nonempty = 0
    input_mismatches = 0
    exact_output_label_mismatches = 0
    empty_status_mismatches = 0
    informative_status_mismatches = 0
    resource_samples = []
    length_bins = collections.Counter()
    official = pq.ParquetFile(train).iter_batches(batch_size=512, columns=["event_id", "message_sanitized", "label_binary"])
    prepared = pq.ParquetFile(out / "prepared_corpus.parquet").iter_batches(batch_size=512, columns=["row_position", "event_id", "text", "label", "diagnostic_text"])
    rows_seen = 0
    for rb, pb in zip(official, prepared):
        original_rows, prepared_rows = rb.to_pylist(), pb.to_pylist()
        if len(original_rows) != len(prepared_rows):
            raise AssertionError("Batch alignment mismatch")
        for original, current in zip(original_rows, prepared_rows):
            pos = rows_seen
            gid = int(group_column[pos])
            raw = original["message_sanitized"] or ""
            txt = current["text"]
            input_mismatches += current["row_position"] != pos or current["event_id"] != original["event_id"]
            exact_output_label_mismatches += current["label"] != original["label_binary"]
            empty_status_mismatches += bool(df.original_empty.iloc[pos]) != (not bool(raw.strip()))
            informative_status_mismatches += bool(gm.informative.iloc[pos]) != bool(txt)
            if txt:
                text_digest = hashlib.sha256(txt.encode("utf-8")).digest()
                normalized_duplicate_split += text_groups.setdefault(text_digest, gid) != gid
                distinct_texts_merged += group_texts.setdefault(gid, text_digest) != text_digest
            if raw.strip():
                raw_nonempty += 1
                digest = hashlib.sha256(raw.encode("utf-8")).digest()
                previous = raw_groups.setdefault(digest, gid)
                duplicate_split += previous != gid
            key = (gid, original["label_binary"])
            if gid in focus and selected[key] < 2:
                selected[key] += 1
                conflict_examples.append({"row_position": pos, "group_id": gid, "label": original["label_binary"],
                    "raw_message": raw, "text": txt, "diagnostic_text": current["diagnostic_text"]})
            length_bins["0" if not txt else "1-256" if len(txt) <= 256 else "257-2048" if len(txt) <= 2048 else "2049-8192" if len(txt) <= 8192 else ">8192"] += 1
            # Fixed row-position hash sample, no label/score selection. This is
            # only a rough character-gram resource estimate, not a fitted vocab.
            if int.from_bytes(hashlib.blake2b(str(pos).encode(), digest_size=4).digest(), "little") % 1024 == 0:
                unique = sum(len({txt[i:i+k] for i in range(max(0, len(txt)-k+1))}) for k in (3, 4, 5))
                resource_samples.append({"row_position": pos, "text_length": len(txt), "unique_char_3_5": unique})
            rows_seen += 1
    checks["all_original_rows_compared"] = rows_seen == n == pq.ParquetFile(train).metadata.num_rows
    checks["exact_event_id_order_matches_original"] = input_mismatches == 0
    checks["labels_unchanged_from_original"] = exact_output_label_mismatches == 0
    checks["original_empty_matches_original"] = empty_status_mismatches == 0
    checks["informative_matches_actual_text"] = informative_status_mismatches == 0
    checks["nonempty_raw_duplicates_never_split"] = duplicate_split == 0
    checks["identical_nonempty_texts_never_split"] = normalized_duplicate_split == 0
    checks["distinct_nonempty_texts_never_merged"] = distinct_texts_merged == 0
    labels = df.label.map({"benign": 0, "malicious": 1, "suspicious": 2}).to_numpy()
    info = gm.informative.to_numpy()
    counts = np.bincount(group_column[info] * 3 + labels[info], minlength=(int(group_column.max()) + 1) * 3).reshape(-1, 3)
    mixed = (counts > 0).sum(axis=1) > 1
    recalculated = {"informative_unique_texts": len(text_groups), "mixed_label_text_groups": int(mixed.sum()),
                    "rows_in_mixed_label_text_groups": int(counts[mixed].sum()),
                    "minimum_observed_mistakes_deterministic_text_only": int((counts.sum(axis=1) - counts.max(axis=1)).sum())}
    checks["collision_totals_independently_recomputed"] = all(collision[k] == v for k, v in recalculated.items())
    raw_unique_count = len(raw_groups)
    del counts, raw_groups, text_groups, group_texts
    cells = []
    for keys, part in df.groupby(["product", "format", "label"], observed=True):
        informative = part[~(part.original_empty | part.filtered_empty)]
        sizes = informative.groupby("group_id").size()
        cells.append({"product": keys[0], "format": keys[1], "label": keys[2], "rows": len(part),
                      "informative_groups": len(sizes), "largest_group_rows": int(sizes.max()) if len(sizes) else 0,
                      "original_empty": int(part.original_empty.sum()), "filtered_empty": int(part.filtered_empty.sum()),
                      "unknown_format": int(part.unknown_format.sum())})
    estimate = float(np.mean([r["unique_char_3_5"] for r in resource_samples])) * n if resource_samples else None
    audit = {"scope": "Official train and frozen preparation only; no main model fitting, no external data. Integrity checks are not semantic or model acceptance.",
             "checks": checks, "all_integrity_checks_passed": all(checks.values()), "rows": n,
             "nonempty_raw_unique": raw_unique_count, "nonempty_raw_duplicate_rows_after_first": raw_nonempty - raw_unique_count,
             "class_counts": {str(k): int(v) for k, v in df.label.value_counts().items()},
             "original_empty_rows": int(df.original_empty.sum()), "filtered_empty_rows": int(df.filtered_empty.sum()),
             "unknown_format_rows": int(df.unknown_format.sum()), "text_length_bins": dict(length_bins),
             "source_format_class_support": cells, "collision_summary": collision,
             "collision_totals_independently_recomputed": recalculated,
             "resource_estimate": {"method": "Fixed position-hash sample, about 1/1024 rows, no fitting; full-document unique raw char 3-5 grams before vocab cap", "sample_rows": len(resource_samples),
                 "estimated_full_corpus_char_nnz_before_vocab_cap": estimate,
                 "estimated_float32_int32_char_csr_bytes_before_vocab_cap": estimate * 8 + (n+1)*4 if estimate is not None else None,
                 "estimated_float32_int64_char_csr_bytes_before_vocab_cap": estimate * 12 + (n+1)*8 if estimate is not None else None,
                 "int32_offset_assumption_valid_for_estimate": estimate < 2147483647 if estimate is not None else None,
                 "limitations": "Estimate only; excludes word features, vocabulary, optimizer copies and workers; vocab cap changes nnz; not a peak-memory bound."},
             "code_sha256": sha(__file__), "configuration_sha256": sha(out / "configuration.json")}
    target = Path(args.output)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with target.with_name(target.stem + "_collision_examples.json").open("x", encoding="utf-8") as stream:
        json.dump({"scope": "Up to two raw examples per label of eight largest mixed-label normalized-text groups; deliberate diagnostic sample", "examples": conflict_examples}, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"all_integrity_checks_passed": audit["all_integrity_checks_passed"], "rows": n,
          "class_counts": audit["class_counts"], "resource_estimate": audit["resource_estimate"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
