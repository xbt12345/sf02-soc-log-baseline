"""Read-only task, label-support and context audit; no fitting or relabeling."""
from pathlib import Path
import collections
import hashlib
import json
import re
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/2026-09-12/v35_direction_review"
LABELS = ["benign", "malicious", "suspicious"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")


def run():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/official/train.parquet"
    assert digest(path) == "6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742"
    cols = ["label_binary", "product_name", "pipeline", "timestamp",
            "src_ip", "dst_ip", "src_port", "src_host", "dst_host", "username"]
    df = pq.read_table(path, columns=cols).to_pandas()
    y = df["label_binary"]
    assert len(df) == 2056871
    source = df["product_name"].fillna("").replace("", "<missing>")
    counts = pd.crosstab(source, y).reindex(columns=LABELS, fill_value=0)
    source_rows = []
    for s, row in counts.iterrows():
        source_rows.append({"source": s, "class_counts": row.tolist(),
                            "observed_classes": int((row > 0).sum())})
    context = {}
    for lab in LABELS:
        sub = df[y == lab]
        fields = {}
        for col in cols[3:]:
            v = sub[col]
            if col == "timestamp":
                ok = v.notna() & np.isfinite(v)
                fields[col] = {"nonmissing_rows": int(ok.sum()),
                               "unique_values": int(v[ok].nunique()),
                               "min": float(v[ok].min()) if ok.any() else None,
                               "max": float(v[ok].max()) if ok.any() else None}
            else:
                ok = v.notna() & v.ne("")
                fields[col] = {"nonmissing_rows": int(ok.sum()),
                               "unique_values": int(v[ok].nunique())}
        context[lab] = {"rows": len(sub), "fields": fields,
                        "rows_with_time_and_src_ip": int((sub.timestamp.notna() & sub.src_ip.notna() & sub.src_ip.ne("")).sum()),
                        "rows_with_time_and_username": int((sub.timestamp.notna() & sub.username.notna() & sub.username.ne("")).sum())}
    # Syntax/overlap can locate limitations, never prove identity or clock fidelity.
    overlaps = {}
    for col in ["src_ip", "username"]:
        pairs = pd.DataFrame({"source": source, "entity": df[col].replace("", None)}).dropna().drop_duplicates()
        per_entity = pairs.groupby("entity").size()
        overlaps[col] = {"distinct_nonmissing_values": len(per_entity),
                         "values_in_multiple_product_buckets": int((per_entity > 1).sum()),
                         "maximum_product_buckets_per_value": int(per_entity.max()) if len(per_entity) else 0}
    print("Metadata availability counted", flush=True)
    prepared = pq.read_table(
        ROOT / "artifacts/v331_ready_20260912_r2/prepared_corpus.parquet",
        columns=["row_position", "label", "format", "repair_route", "text",
                 "original_empty", "no_observable_auth_facts"]).to_pandas()
    assert np.array_equal(prepared["label"].to_numpy(), y.to_numpy())
    family = prepared.groupby(["format", "repair_route", "label"]).size()
    family_rows = [{"format": idx[0], "route": idx[1], "label": idx[2], "rows": int(n)}
                   for idx, n in family.items()]
    mixed = prepared.loc[prepared.text.ne("")].groupby(["text", "label"]).size().unstack(fill_value=0)
    mixed = mixed.reindex(columns=LABELS, fill_value=0)
    mixed = mixed[(mixed > 0).sum(axis=1) > 1]
    representation = {"nonempty_mixed_text_groups": len(mixed),
                      "rows_in_mixed_text_groups": int(mixed.to_numpy().sum()),
                      "fixed_text_deterministic_classifier_empirical_min_errors":
                          int((mixed.sum(axis=1) - mixed.max(axis=1)).sum()),
                      "original_empty_rows": int(prepared.original_empty.sum()),
                      "auth_no_observable_fact_rows": int(prepared.no_observable_auth_facts.sum()),
                      "scope": "An empirical limitation of this fixed representation, not a Bayes error or proof of incorrect labels."}
    print("Representation collisions counted", flush=True)
    # Literal occurrence probes are diagnosis only; quotations/negation still require review.
    patterns = {
        "windows_failed_logon": re.compile(r"an account failed to log on", re.I),
        "pam_auth_failure": re.compile(r"authentication failure", re.I),
        "ssh_failed_password": re.compile(r"failed password", re.I),
        "duo_invalid_passcode": re.compile(r"invalid_passcode", re.I),
    }
    probes = {name: collections.Counter() for name in patterns}
    examples = {name: [] for name in patterns}
    seen_strata = {name: collections.Counter() for name in patterns}
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=8192, columns=["message_sanitized", "product_name", "label_binary"],
            use_threads=False):
        for j, row in enumerate(batch.to_pylist()):
            raw = row["message_sanitized"] or ""
            for name, pattern in patterns.items():
                if pattern.search(raw):
                    key = (row["product_name"] or "<missing>", row["label_binary"])
                    probes[name][key] += 1
                    if seen_strata[name][key] < 2:
                        examples[name].append({"row_position": offset + j,
                            "source": key[0], "label": key[1], "raw": raw})
                        seen_strata[name][key] += 1
        offset += len(batch)
    assert offset == len(df)
    probe_rows = {name: [{"source": s, "label": lab, "rows": n}
                        for (s, lab), n in sorted(c.items())]
                  for name, c in probes.items()}
    # Read the actual source document, not instructions from derived notes.
    doc = next((ROOT / "docs/official").glob("*.docx"))
    with zipfile.ZipFile(doc) as z:
        xml = ET.fromstring(z.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    pars = ["".join(n.text or "" for n in p.findall(".//w:t", ns))
            for p in xml.findall(".//w:p", ns)]
    start = next(i for i, p in enumerate(pars) if "SF-2026-02" in p) - 1
    end = next(i for i, p in enumerate(pars) if "SF-2026-03" in p) - 1
    task = pars[start:end]
    task_text = "\n".join(task)
    (OUT / "official_task2_verified.txt").write_text(task_text + "\n", encoding="utf-8")
    named = counts.drop(index="<missing>")
    result = {
        "scope": "Read-only official task and full training-data support audit. No training, new labels, external data, or established causal claims.",
        "source_file_sha256": digest(path),
        "official_doc_sha256": digest(doc),
        "official_task_extraction_sha256": digest(OUT / "official_task2_verified.txt"),
        "rows": len(df), "source_class_counts": source_rows,
        "all_malicious_in_missing_product_bucket": bool(counts.loc["<missing>", "malicious"] == int((y == "malicious").sum())),
        "named_product_buckets": len(named),
        "named_product_buckets_with_malicious": int((named["malicious"] > 0).sum()),
        "named_product_buckets_with_one_observed_class": int(((named > 0).sum(axis=1) == 1).sum()),
        "context_availability_by_class": context,
        "cross_product_identifier_overlap": overlaps,
        "format_route_class_counts": family_rows,
        "representation": representation,
        "literal_authentication_probes": probe_rows,
        "literal_probe_scope": "Phrase occurrences, not attack facts or comparable label-policy evidence. See raw cases before interpreting.",
        "context_identity_and_clock_semantics_verified": False,
        "elapsed_seconds": time.perf_counter() - started,
        "source_code_sha256": digest(Path(__file__)),
    }
    save("task_assumptions_audit.json", result)
    save("literal_authentication_cases.json", examples)
    print(json.dumps({k: result[k] for k in [
        "rows", "all_malicious_in_missing_product_bucket", "named_product_buckets",
        "named_product_buckets_with_malicious", "named_product_buckets_with_one_observed_class",
        "context_availability_by_class", "representation", "literal_authentication_probes",
        "elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
