"""Reviewed, path-bounded cleanup. Does not import or execute any model."""
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\xiabutian\Desktop\人工智能算法挑战杯\SF02").resolve()
OUT = ROOT / "evidence/2026-10-02/project_cleanup"
SEAL = ROOT / "artifacts/v169_prior_pair_training/run_seal.json"
SUFFIXES = {".pt", ".joblib", ".npy", ".npz", ".parquet"}
NUM = re.compile(r"(?i)(epoch|step|accepted|checkpoint)[_-]?(\d+)")


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(relative):
    p = ROOT / relative
    if p.is_symlink() or p.is_junction():
        raise ValueError("Link target refused: " + relative)
    resolved = p.resolve(strict=True)
    if not resolved.is_relative_to(ROOT) or resolved == ROOT:
        raise ValueError("Out-of-root target refused: " + relative)
    for parent in p.parents:
        if parent == ROOT:
            break
        if parent.is_symlink() or parent.is_junction():
            raise ValueError("Linked ancestor refused: " + relative)
    return p


def protection():
    seal = load(SEAL)
    protected = {}
    for p, digest in seal["source_sha256"].items():
        pp = Path(p)
        if not pp.is_absolute():
            pp = ROOT / pp
        if pp.is_relative_to(ROOT):
            protected[pp.relative_to(ROOT).as_posix()] = digest
    for f in (ROOT / "mcp_readonly").rglob("*.json"):
        if "catalog" not in f.as_posix():
            continue
        obj = load(f)
        if isinstance(obj, dict):
            for doc in obj.get("documents", []):
                p = doc.get("path")
                if p:
                    protected.setdefault(Path(p).as_posix(), doc.get("sha256"))
    inventory = load(ROOT / "artifacts/v169_platform_dependency_inventory_v1_20261002/review.json")
    for p, record in inventory["files"].items():
        protected.setdefault(Path(p).as_posix(), record.get("sha256"))
    return protected


def selected_checkpoints(rows):
    selected = set()
    # Selected historical checkpoints are useful model results, not disposable steps.
    complete = load(ROOT / "artifacts/v63_factorial_20260920/training_complete.json")
    for arm, record in complete.get("arms", {}).items():
        step = record.get("checkpoint", {}).get("step")
        if step is not None:
            selected.add(f"artifacts/v63_factorial_20260920/{arm}/step_{step:04d}.pt")
    for row in rows:
        p = ROOT / row["relative_path"]
        if p.suffix != ".json" or p.name not in {"fit.json", "summary.json", "selection.json", "selected.json", "analysis.json"}:
            continue
        if int(row["bytes"]) > 2_000_000:
            continue
        try:
            obj = load(p)
        except (ValueError, OSError):
            continue
        def visit(value, context=""):
            if isinstance(value, dict):
                for key, v in value.items():
                    if isinstance(v, str) and re.search(r"selected|best|endpoint|final|checkpoint|model_path|model_file", context + "/" + key, re.I) and Path(v).suffix in {".pt", ".joblib"}:
                        for candidate in (ROOT / v, p.parent / v):
                            if candidate.is_file() and candidate.is_relative_to(ROOT):
                                selected.add(candidate.relative_to(ROOT).as_posix())
                    elif isinstance(v, int) and key in {"best_epoch", "selected_epoch", "selected_step", "best_step"}:
                        for candidate in p.parent.glob(f"*{v}*.pt"):
                            match = NUM.search(candidate.stem)
                            if match and int(match.group(2)) == v:
                                selected.add(candidate.relative_to(ROOT).as_posix())
                    if key not in {"source_sha256", "files", "bindings"}:
                        visit(v, context + "/" + key)
            elif isinstance(value, list):
                for v in value:
                    visit(v, context)
        visit(obj)
    return selected


def plan():
    rows = list(csv.DictReader((OUT / "inventory.csv").open(encoding="utf-8")))
    protected = protection()
    chosen = selected_checkpoints(rows)
    series = defaultdict(list)
    for row in rows:
        p = Path(row["relative_path"])
        match = NUM.search(p.stem)
        if p.parts[0] == "artifacts" and p.suffix in SUFFIXES and match:
            signature = NUM.sub(lambda m: m.group(1).lower() + "#", p.name)
            series[(p.parent.as_posix(), signature)].append((int(match.group(2)), p.as_posix()))
    keep_numeric = set(chosen)
    for (parent, signature), values in series.items():
        numbers = sorted(set(v[0] for v in values))
        keep_numbers = {numbers[-1]}
        if 0 in numbers:
            keep_numbers.add(0)
        version = re.search(r"artifacts/v(\d+)", parent)
        if version and int(version.group(1)) >= 131:
            # Preserve the actual final five-state window used by mastery audits.
            keep_numbers.update(numbers[-5:])
        keep_numeric.update(p for number, p in values if number in keep_numbers)
    files = []
    for row in rows:
        p = Path(row["relative_path"])
        rel = p.as_posix()
        if rel in protected or rel in keep_numeric:
            continue
        if p.parts[0] in {"data", "docs", "mcp_readonly", "training", ".git", ".runtime", ".venv", ".venv-mcp", ".venv-v61"}:
            continue
        if (p.parts[0] == "artifacts" and re.match(r"v(?:159|164|168|169)(?:_|$)", p.parts[1])) or rel.startswith("platform/v169_cpu_v1/"):
            continue
        reason = None
        if "__pycache__" in p.parts and p.suffix == ".pyc":
            reason = "Unbound regenerable project bytecode"
        elif p.parts[0] == "artifacts":
            if p.suffix in {".data", ".indices", ".indptr"}:
                reason = "Closed historical sparse-matrix cache; raw data and source retained"
            elif p.suffix == ".whl":
                reason = "Installer cache; installed active runtime retained"
            elif rel == "artifacts/v32_prepare_20260912/input_cache.sqlite":
                reason = "Derived V32 parsing cache; official parquet and preparation source retained"
            elif p.suffix in SUFFIXES and NUM.search(p.stem):
                reason = "Superseded historical step; selected/terminal/initial and required final windows retained"
            elif p.name == "latest_training_state.pt" and (ROOT / p.parent / "model.pt").is_file():
                reason = "Closed V61 optimizer resume state; standalone final model and reports retained"
            elif p.suffix == ".npy" and (re.search(r"(?:gradient|direction|basis|embedding|features?|^X(?:_|\.)|^x_)", p.name, re.I) or any(re.fullmatch(r"(?:gradient|derivative|gradients|derivatives|feature_cache)s?", q, re.I) for q in p.parts)):
                reason = "Unbound historical derivative/feature cache; final outputs and audit conclusions retained"
        elif p.parts[0] == "platform":
            if p.suffix in {".zip", ".data", ".indices", ".indptr"} or re.search(r"\.part\d+$", p.name):
                reason = "Superseded upload/export copy; current V169 preparation retained"
            elif p.suffix == ".parquet" and ("v32_bundle_20260912" in p.parts or "upload_bundle" in p.parts):
                reason = "Old upload preparation/official input copy; unique official data retained"
        if reason:
            current = checked(rel)
            s = current.stat()
            if s.st_size != int(row["bytes"]) or s.st_mtime_ns != int(row["mtime_ns"]):
                raise ValueError("Inventory changed: " + rel)
            files.append({"path": rel, "bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "reason": reason})
    counts, sizes = Counter(), Counter()
    for f in files:
        counts[f["reason"]] += 1
        sizes[f["reason"]] += f["bytes"]
    manifest = {"status": "reviewed_plan_not_yet_deleted", "root": str(ROOT), "seal_sha256": sha(SEAL), "protected_paths": protected, "selected_or_terminal_model_paths": sorted(keep_numeric), "preserve_full_latest_prefixes": ["v159", "v164", "v168", "v169"], "files": files, "logical_bytes": sum(f["bytes"] for f in files), "categories": {k: {"files": counts[k], "bytes": sizes[k]} for k in counts}}
    save("deletion_plan.json", manifest)
    print(json.dumps({"files": len(files), "logical_GiB": round(manifest["logical_bytes"] / 2**30, 3), "categories": manifest["categories"], "protected_paths": len(protected), "selected_terminal_window_paths": len(keep_numeric)}, ensure_ascii=False), flush=True)


def verify_protected(label, protected):
    bad = []
    for i, (p, expected) in enumerate(protected.items()):
        try:
            path = checked(p)
            actual = sha(path)
            if expected and actual != expected:
                bad.append({"path": p, "expected": expected, "actual": actual})
        except Exception as e:
            bad.append({"path": p, "error": str(e)})
        if i % 3000 == 0:
            print(label, i, "of", len(protected), flush=True)
    save(label + ".json", {"checked": len(protected), "passed": not bad, "failures": bad})
    if bad:
        raise ValueError(label + " failed; see report")


def apply():
    m = load(OUT / "deletion_plan.json")
    if (OUT / "deletion_journal.jsonl").exists():
        raise ValueError("Existing deletion journal; do not blindly repeat")
    if m["seal_sha256"] != sha(SEAL):
        raise ValueError("Active seal changed")
    # This is the final review before the first deletion, not a best-effort guard.
    for f in m["files"]:
        p = checked(f["path"])
        s = p.stat()
        if f["path"] in m["protected_paths"] or s.st_size != f["bytes"] or s.st_mtime_ns != f["mtime_ns"]:
            raise ValueError("Deletion identity/precondition mismatch: " + f["path"])
    verify_protected("protected_before", m["protected_paths"])
    initial_disk = shutil.disk_usage(ROOT)
    deleted_bytes = 0
    with (OUT / "deletion_journal.jsonl").open("x", encoding="utf-8") as journal:
        for i, f in enumerate(m["files"]):
            p = checked(f["path"])
            s = p.stat()
            if s.st_size != f["bytes"] or s.st_mtime_ns != f["mtime_ns"]:
                raise ValueError("File changed during cleanup: " + f["path"])
            receipt = {**f, "sha256": sha(p), "deleted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            # Persist review evidence before irreversible unlink.
            journal.write(json.dumps({**receipt, "event": "reviewed_before_delete"}, ensure_ascii=False) + "\n")
            journal.flush()
            p.unlink()
            if p.exists():
                raise ValueError("Deletion failed: " + f["path"])
            journal.write(json.dumps({"event": "deleted", "path": f["path"], "bytes": f["bytes"]}, ensure_ascii=False) + "\n")
            journal.flush()
            deleted_bytes += f["bytes"]
            if i % 200 == 0:
                print("deleted", i, "of", len(m["files"]), "GiB", round(deleted_bytes / 2**30, 3), flush=True)
    removed_dirs = []
    # Remove empty shells only in disposable artifact/export branches; no recursive delete.
    for top in (ROOT / "artifacts", ROOT / "platform"):
        dirs = sorted((p for p in top.rglob("*") if p.is_dir() and not p.is_symlink() and not p.is_junction()), key=lambda p: len(p.parts), reverse=True)
        for p in dirs:
            rel = p.relative_to(ROOT).as_posix()
            if any(k.startswith(rel + "/") for k in m["protected_paths"]) or "/v169" in rel:
                continue
            try:
                checked(rel)
                p.rmdir()
                removed_dirs.append(rel)
            except OSError:
                pass
    verify_protected("protected_after", m["protected_paths"])
    missing = [f["path"] for f in m["files"] if (ROOT / f["path"]).exists()]
    final_disk = shutil.disk_usage(ROOT)
    result = {"status": "cleanup_completed" if not missing else "incomplete", "deleted_files": len(m["files"]), "logical_bytes_deleted": deleted_bytes, "disk_free_bytes_before": initial_disk.free, "disk_free_bytes_after": final_disk.free, "observed_disk_free_gain_bytes": final_disk.free - initial_disk.free, "empty_directories_removed": removed_dirs, "deletion_postcondition_failures": missing, "protected_before_and_after_passed": True, "model_execution_or_training": False, "old_archival_seals_may_reference_intentionally_removed_intermediates": True, "current_v169_seal_remains_intact": True}
    save("result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    {"plan": plan, "apply": apply}[sys.argv[1]]()
