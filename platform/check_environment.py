"""Portable, read-only environment/data probe; optional tiny synthetic CPU fit.

No package installation, network requests, full log loading, or model-file writes.
Synthetic fitting verifies execution only and reports no detection score.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def read_limit(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Actual mounted official parquet directory")
    parser.add_argument("--smoke-fit", action="store_true", help="Fit 12 trees on 96 synthetic rows using CPU")
    args = parser.parse_args()
    report = {
        "scope": "Environment and parquet metadata only; optional synthetic fit is not SOC quality validation",
        "python": sys.version.split()[0],
        "os": platform.system(),
        "architecture": platform.machine(),
        "working_directory": str(Path.cwd()),
        "logical_cpus_reported_by_os": os.cpu_count(),
        "container_limits_raw": {
            "cpu_max_v2": read_limit("/sys/fs/cgroup/cpu.max"),
            "cpuset_v2": read_limit("/sys/fs/cgroup/cpuset.cpus.effective"),
            "memory_max_v2": read_limit("/sys/fs/cgroup/memory.max"),
            "cpu_quota_v1": read_limit("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            "cpu_period_v1": read_limit("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
            "memory_limit_v1": read_limit("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        },
        "packages": {},
        "files": {},
        "synthetic_cpu_fit": {"requested": args.smoke_fit, "executed": False},
    }
    problems = []
    for name in ["numpy", "pandas", "pyarrow", "scikit-learn", "catboost", "PyYAML", "matplotlib"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
            problems.append("Missing package: " + name)
    if args.data_dir is not None:
        expected = {"train.parquet": 2056871, "valid_input.parquet": 2014052}
        for name, expected_rows in expected.items():
            path = args.data_dir / name
            info = {"path": str(path.resolve()), "exists": path.is_file()}
            report["files"][name] = info
            if not path.is_file():
                problems.append("Missing data: " + name)
                continue
            try:
                import pyarrow.parquet as pq
                file = pq.ParquetFile(path)
                info.update(rows=file.metadata.num_rows, columns=file.schema_arrow.names)
                info["expected_row_count_matches"] = info["rows"] == expected_rows
                if not info["expected_row_count_matches"]:
                    problems.append("Unexpected row count: " + name)
                required = {"event_id", "message_sanitized"}
                if name == "train.parquet":
                    required.add("label_binary")
                if not required.issubset(info["columns"]):
                    problems.append("Required columns missing: " + name)
                info["identity_note"] = "Matching schema/row count is not a file hash check"
            except Exception as exc:
                info["error"] = str(exc)
                problems.append("Cannot read parquet metadata: " + name)
    if args.smoke_fit:
        try:
            from catboost import CatBoostClassifier
            x = [[i % 13, str(i % 4)] for i in range(96)]
            y = [i % 3 for i in range(96)]
            model = CatBoostClassifier(iterations=12, depth=3, loss_function="MultiClass",
                                       task_type="CPU", thread_count=2, verbose=False,
                                       allow_writing_files=False, random_seed=20260911)
            model.fit(x, y, cat_features=[1])
            probabilities = model.predict_proba(x[:3])
            if probabilities.shape != (3, 3):
                raise RuntimeError("Unexpected prediction shape")
            report["synthetic_cpu_fit"].update(executed=True, succeeded=True,
                                               rows=96, trees=model.tree_count_, threads=2)
        except Exception as exc:
            report["synthetic_cpu_fit"].update(succeeded=False, error=str(exc))
            problems.append("Synthetic CPU fit failed")
    report["problems"] = problems
    report["requested_checks_passed"] = not problems
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
