"""Observe real context/retention reads without model construction or forwards."""
import builtins
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v169_platform_dependency_inventory_v1_20261002'
assert not OUT.exists()
OUT.mkdir()
observed = {}


def record(value):
    if not isinstance(value, (str, os.PathLike)):
        return
    path = Path(value).resolve()
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return
    if relative.parts[0].startswith('.venv') or not path.is_file() or path.is_relative_to(OUT):
        return
    key = relative.as_posix()
    observed[key] = observed.get(key, 0) + 1


original_open = builtins.open
original_io_open = io.open


def file_open(file, *args, **kwargs):
    record(file)
    return original_open(file, *args, **kwargs)


def io_open(file, *args, **kwargs):
    record(file)
    return original_io_open(file, *args, **kwargs)


builtins.open = file_open
io.open = io_open
started = time.monotonic()
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import v169_prior_pair_training_entry_v14 as entry


def wrap(function):
    def tracked(path, *args, **kwargs):
        record(path)
        return function(path, *args, **kwargs)
    return tracked


pd.read_parquet = wrap(pd.read_parquet)
pq.read_table = wrap(pq.read_table)
reports = []
for role in [0, 1, 2]:
    ctx = entry.load_context(role)
    protection = entry.apply_all_current_correct(ctx, role)
    arrays = {scope + '_' + name: np.load(entry.PRIOR / f'role{role}/endpoint/{scope}_{name}.npy') for scope in ['OOF', 'deployment'] for name in ['q', 'logq']}
    ctx['baseline_stats'] = entry.stats(ctx, arrays['OOF_q'], 'OOF')
    guard = entry.ActualBackend._guard(SimpleNamespace(ctx=ctx), arrays)
    assert guard['passed'], (role, guard)
    reports.append(dict(role=role, protection=protection, saved_data_guard=guard))
    for scope in ['OOF', 'deployment']:
        record(entry.PRIOR / f'role{role}/endpoint/{scope}_original_rows.parquet')
    record(entry.PRIOR / f'role{role}/endpoint.pt')
    fit_path = entry.PRIOR / f'role{role}/fit.json'
    fit = entry.read(fit_path)
    old_risk = entry.PRIOR / f"role{role}/parameter_point{fit['permanent_updates']}/class1_repeat0"
    for name in ['fixed_pure_error_contribution', 'full_original_class_CE']:
        record(old_risk / (name + '.npy'))
    safe = entry.SAFE[role]
    for name in ['direction.npy', 'v168_complete_probe_review.json' if role == 1 else 'probe.json']:
        record(safe / name)
    for scope in ['OOF', 'deployment']:
        for name in ['q', 'logq']:
            record(safe / f'{scope}_{name}.npy')
    record(ROOT / f'artifacts/v169_learnable_prior_pair_plan_20261002/role{role}_all_S_initial_prior_cohort.parquet')
    del ctx, arrays

# Manifest history is carried as history; actual platform closure is separately sealed.
historical = [entry.PLAN, entry.OUT / 'registration.json', entry.OUT / 'run_seal.json', ROOT / 'artifacts/v169_full_preseal_bundle_v6_20261002/bundle.json', ROOT / 'artifacts/v169_root_final_preseal_review_v2_20261002/review.json', ROOT / 'docs/EXPERIMENT_REVIEW_RULES.md', ROOT / 'AGENTS.md']
plan = entry.read(entry.PLAN)
for key in plan['source_sha256']:
    record(ROOT / key)
for path in historical:
    record(path)
bundle = entry.read(historical[3])
for key in bundle['AST_recursive_training_module_paths']:
    record(ROOT / key)
for module in tuple(sys.modules.values()):
    record(getattr(module, '__file__', None))

builtins.open = original_open
io.open = original_io_open
files = {key: dict(bytes=(ROOT / key).stat().st_size, observed_reads=count) for key, count in sorted(observed.items())}
report = dict(status='actual_context_and_all_retention_reads_observed_without_model', files=files, physical_files=len(files), logical_bytes=sum(row['bytes'] for row in files.values()), role_reports=reports, original_full_bundle_files=bundle['physical_dependency_files'], original_full_bundle_bytes=sum((ROOT / key).stat().st_size for key in bundle['source_sha256']), scientific_source_sha256=plan['source_sha256'], elapsed_seconds=time.monotonic()-started, official_heads=0, official_features=0, official_derivatives=0, fits=0, permanent_updates=0, observed_context_reads_do_not_prove_complete_platform_execution_closure=True, platform_numerical_qualification=False)
(OUT / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: report[key] for key in ['status', 'physical_files', 'logical_bytes', 'elapsed_seconds', 'official_heads', 'fits']}, ensure_ascii=False), flush=True)
