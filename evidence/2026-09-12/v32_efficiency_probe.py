"""Read-only design probes on frozen v0.3 artifacts, never a replacement split.

Outputs a new report only. No parser repair, training, vocabulary fit or external data.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/v3_prepare_parallel_20260912"
LABELS = ["benign", "malicious", "suspicious"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for chunk in iter(lambda: s.read(8388608), b""):
            h.update(chunk)
    return h.hexdigest()


def main(output):
    started = time.perf_counter()
    frozen = json.loads((RUN / "result.json").read_text(encoding="utf-8"))
    if sha(RUN / "split_manifest.parquet") != frozen["split_sha256"]:
        raise ValueError("Split identity changed")
    gm = pd.read_parquet(RUN / "group_manifest.parquet")
    sm = pd.read_parquet(RUN / "split_manifest.parquet")
    if not np.array_equal(gm.row_position, sm.row_position):
        raise AssertionError("Row alignment changed")
    g = gm.group_id.to_numpy()
    y = gm.label_index.to_numpy()
    info = gm.informative.to_numpy()
    n = len(g)
    ng = int(g.max()) + 1
    counts = np.bincount(g * 3 + y, minlength=ng*3).reshape(-1, 3)
    icounts = np.bincount(g[info] * 3 + y[info], minlength=ng*3).reshape(-1, 3)
    inner = sm.inner_for_outer_0.to_numpy()
    roles = np.where(sm.outer_fold.to_numpy() == 0, 3, np.where(inner <= 2, 0, np.where(inner == 3, 1, 2)))
    lo = np.full(ng, 4, dtype=np.int8)
    hi = np.full(ng, -1, dtype=np.int8)
    np.minimum.at(lo, g, roles)
    np.maximum.at(hi, g, roles)
    if not np.array_equal(lo, hi):
        raise AssertionError("Original groups cross roles")
    gr = lo.copy()

    def support():
        answer = {}
        for role, name in enumerate(["fit", "selection", "calibration", "outer"]):
            selected = gr == role
            answer[name] = {}
            for label, text in enumerate(LABELS):
                sizes = icounts[selected, label]
                total = int(sizes.sum())
                answer[name][text] = {"rows": int(counts[selected, label].sum()), "nonempty_groups": int((sizes > 0).sum()),
                                     "largest_group_fraction": float(sizes.max()/total) if total else None}
        return answer

    before = support()
    selection_sus = np.flatnonzero((gr == 1) & (icounts[:, 2] > 0))
    if len(selection_sus) != 1:
        raise ValueError("This specific old-split feasibility probe no longer applies")
    giant = int(selection_sus[0])
    if counts[giant, :2].sum() != 0:
        raise ValueError("Specific pure-class witness assumption does not apply")
    gr[giant] = 0
    moved = [{"group_id": giant, "from": "selection", "to": "fit", "rows": int(counts[giant].sum())}]
    total_available_sus = int(counts[gr != 3, 2].sum())
    target = round(total_available_sus * 0.2)
    candidates = np.flatnonzero((gr == 0) & (icounts[:, 2] > 0) & (counts[:, :2].sum(axis=1) == 0))
    candidates = [int(v) for v in candidates if v != giant]
    candidates.sort(key=lambda v: (-int(counts[v, 2]), hashlib.sha256(("20260912:"+str(v)).encode()).digest()))
    selected_rows = 0
    selected_groups = 0
    for v in candidates[:10000]:
        if selected_rows >= target and selected_groups >= 30:
            break
        gr[v] = 1
        selected_rows += int(counts[v, 2])
        selected_groups += 1
        moved.append({"group_id": v, "from": "fit", "to": "selection", "rows": int(counts[v].sum())})
    after = support()
    constraint_ok = all(v["nonempty_groups"] >= 30 for role in ["fit", "selection", "calibration"] for v in after[role].values())
    constraint_ok &= all(v["largest_group_fraction"] is not None and v["largest_group_fraction"] <= 0.5
                         for role in ["selection", "calibration"] for v in after[role].values())
    checks = {"all_rows_retained": sum(v["rows"] for role in after.values() for v in role.values()) == n,
              "whole_groups_only": True, "outer_membership_unchanged": bool(np.array_equal(gr[g] == 3, roles == 3)),
              "all_inner_constraints_satisfied_in_simulation": bool(constraint_ok)}
    split_seconds = time.perf_counter() - started
    # Use the prior resource probe's fixed hash sampling rule. Labels are not read
    # for this resource estimate; there is no learned vocabulary or IDF.
    selected_positions = {i for i in range(n) if int.from_bytes(hashlib.blake2b(str(i).encode(), digest_size=4).digest(), "little") % 1024 == 0}
    word_nnz, char_nnz = [], []
    token = re.compile(r"(?u)\b\w+\b")  # Preserve single-character digits/tokens.
    for batch in pq.ParquetFile(RUN / "prepared_corpus.parquet").iter_batches(batch_size=4096, columns=["row_position", "text"]):
        positions = batch.column(0).to_pylist()
        for k, pos in enumerate(positions):
            if pos not in selected_positions:
                continue
            text = batch.column(1)[k].as_py()
            terms = token.findall(text)
            word_nnz.append(len(set(terms)) + len(set(zip(terms, terms[1:]))))
            char_nnz.append(sum(len({text[i:i+j] for i in range(max(0, len(text)-j+1))}) for j in [3, 4, 5]))
    old_audit = json.loads((ROOT / "evidence/2026-09-12/v3_full_preparation_audit.json").read_text(encoding="utf-8"))
    total_unique = old_audit["nonempty_raw_unique"] + 1
    report = {"scope": "Design feasibility/resource probes only on already rejected v0.3 text and frozen group metadata; not a parser fix, production split, model fit or acceptance.",
              "input_split_sha256": frozen["split_sha256"], "code_sha256": sha(__file__),
              "deduplicated_parsing": {"rows": n, "distinct_parser_input_strings_including_empty": total_unique,
                  "distinct_to_row_ratio": total_unique/n, "fewer_unique_calls_vs_one_call_per_row_fraction": 1-total_unique/n,
                  "scope": "Counts from previous full audit. Existing LRU already reuses some calls; this is not measured speedup or a claim of equivalent deduplicated IDF."},
              "split_feasibility_witness": {"before": before, "after": after, "moves": moved, "checks": checks,
                  "seconds_including_metadata_load": split_seconds,
                  "scope": "Only this existing fold and pure-class witness; no files or original roles changed. New normalized groups must be rebuilt and evaluated separately; this is not a general solver."},
              "resource_sample": {"rows": len(word_nnz), "word_1_2_unique_nnz_mean": float(np.mean(word_nnz)),
                  "char_3_5_unique_nnz_mean": float(np.mean(char_nnz)),
                  "estimated_whole_corpus_word_nnz_uncapped": float(np.mean(word_nnz)*n),
                  "estimated_whole_corpus_char_nnz_uncapped": float(np.mean(char_nnz)*n),
                  "word_to_char_nnz_ratio": float(np.sum(word_nnz)/np.sum(char_nnz)),
                  "scope": "Fixed 1/1024 position-hash sample, pre-vocabulary unique tokens; no IDF fit, tokenizer binding validation, peak-memory guarantee or performance evidence. v0.3 text remains invalid for main training."},
              "elapsed_seconds": time.perf_counter()-started}
    with Path(output).open("x", encoding="utf-8") as s:
        json.dump(report, s, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"checks": checks, "before_selection_suspicious": before["selection"]["suspicious"],
                      "simulated_selection_suspicious": after["selection"]["suspicious"],
                      "split_probe_seconds": split_seconds, "resource_sample": report["resource_sample"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    main(p.parse_args().output)
