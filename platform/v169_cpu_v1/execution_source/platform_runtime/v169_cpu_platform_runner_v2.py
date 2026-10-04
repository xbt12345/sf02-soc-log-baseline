"""Actual platform preflight and prospective seal before the original counted fits."""
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'platform_runtime'), str(PROJECT/'training')]
STATE = PROJECT/'platform_run_v1'


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def child(filename, arguments=()):
    path = PROJECT/'platform_runtime'/filename
    wrapper = PROJECT/'platform_runtime/v169_cpu_qualification_child_v1.py'
    with (STATE/(path.stem+'_'+('_'.join(arguments) or 'run')+'.log')).open('x', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-I', '-B', '-X', 'utf8', str(wrapper), str(path), *arguments],
                                cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Platform qualification failed: '+filename+' '+str(result.returncode))


def runtime_bindings(sha):
    from root_offline_bootstrap_v1 import ENVIRONMENT
    paths = {p.resolve() for p in ENVIRONMENT.rglob('*') if p.is_file()}
    loaded = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[5].startswith('/'):
            continue
        if parts[5].endswith(' (deleted)'):
            raise RuntimeError('A loaded physical library/file has been deleted')
        path = Path(parts[5]).resolve()
        if path.is_file():
            if not path.is_relative_to(PROJECT) or path.is_relative_to(ENVIRONMENT):paths.add(path)
            loaded.add(str(path))
    paths.add(Path(sys.executable).resolve())
    result = {str(p):sha(p) for p in sorted(paths)}
    for record in STATE.glob('*_runtime.json'):
        child_loaded = json.loads(record.read_text(encoding='utf-8'))
        for path, expected in child_loaded.items():
            if sha(Path(path)) != expected:
                raise RuntimeError('Actual qualification loaded library/file changed: '+path)
            physical=Path(path)
            if not physical.is_relative_to(PROJECT) or physical.is_relative_to(ENVIRONMENT):result[path] = expected
            loaded.add(path)
    return result, sorted(loaded)


def main():
    from experiment_review import read, sha, check_bindings
    from root_offline_bootstrap_v1 import PACKAGES
    import numpy as np
    import torch
    import v169_cpu_training_entry_v1 as entry
    from v169_cpu_execution_review_v1 import CPU_RESOURCES, platform_identity, require_resources, review_plan
    entry.configure()
    identity = platform_identity()
    resources = require_resources(CPU_RESOURCES, 'initial')
    manifest_path = PROJECT/'platform_runtime/package_source_manifest.json'
    manifest = read(manifest_path)
    sources = manifest['project_source_sha256']
    check_bindings(sources)
    original = read(PROJECT/'training/review_policy/v169_prior_pair_execution_contract.json')
    root_path = PROJECT/'platform_runtime/independent_root_execution_review.json'
    root = read(root_path)
    if (root.get('supports_platform_execution_after_live_preflight') is not True
            or root.get('reviewed_source_manifest_sha256') != sha(manifest_path)
            or root.get('reviewed_entry_sha256') != sha(Path(entry.__file__))
            or root.get('source_sha256') != sources):
        raise RuntimeError('Independent root review for this physical CPU package is required')
    versions = {name:importlib.metadata.version(name) for name in PACKAGES}
    if versions != PACKAGES:
        raise RuntimeError('Installed environment differs from locked vendor')
    # Ensure all qualified CPU numerical and file-reading libraries are loaded before sealing.
    for name in ['scipy.linalg', 'scipy.optimize', 'scipy.sparse', 'pandas', 'pyarrow.parquet',
                 'sklearn', 'joblib', 'tabm', 'rtdl_num_embeddings', 'threadpoolctl']:
        importlib.import_module(name)
    child('v169_cpu_synthetic_qualification_v1.py')
    synthetic = read(PROJECT/'artifacts/v169_platform_CPU_synthetic_qualification_v1/qualification.json')
    if (synthetic['status'] != 'V169_explicit_CPU_actual_backend_synthetic_wiring_qualified'
            or len(synthetic['cases']) != 2 or synthetic['fits'] != 0
            or any(synthetic[key] != 0 for key in ['official_heads','official_features','official_derivatives','permanent_updates'])):
        raise RuntimeError('Incomplete actual synthetic CPU qualification')
    qualification_paths = ['artifacts/v169_platform_CPU_synthetic_qualification_v1/qualification.json']
    for arm in ['A','B']:
        child('v169_cpu_full_width_qualification_v1.py', ('--arm',arm))
        relative = f'artifacts/v169_platform_CPU_full_width_{arm}_qualification_v1/qualification.json'
        result = read(PROJECT/relative)
        if (result['passed'] is not True or result['official_calls'] != 0
                or result['complete_parameters'] != 1060832+(arm=='B') or result['normal_rank'] != 66):
            raise RuntimeError('Full-width 64-function CPU qualification failed')
        qualification_paths.append(relative)
    saved = []
    for role in [0,1,2]:
        ctx = entry.load_context(role)
        protection = entry.apply_all_current_correct(ctx, role)
        arrays = {scope+'_'+name:np.load(entry.PRIOR/f'role{role}/endpoint/{scope}_{name}.npy')
                  for scope in ['OOF','deployment'] for name in ['q','logq']}
        ctx['baseline_stats'] = entry.stats(ctx, arrays['OOF_q'], 'OOF')
        guard = entry.ActualBackend._guard(SimpleNamespace(ctx=ctx), arrays)
        if not guard['passed']:
            raise RuntimeError('Full original saved guard failed for role '+str(role))
        state = torch.load(entry.PRIOR/f'role{role}/endpoint.pt', map_location='cpu', weights_only=True)['state']
        if sum(p.numel() for p in state.values()) != 1060832:
            raise RuntimeError('Full V164 source parameter width mismatch')
        saved.append(dict(role=role, protection=protection, saved_complete_guard=guard,
                          source_checkpoint_sha256=sha(entry.PRIOR/f'role{role}/endpoint.pt')))
        del state, ctx, arrays;gc.collect()
    live_path = STATE/'live_preflight.json'
    write(live_path, dict(passed=True, platform=identity, resources=resources,
                         actual_qualification_sha256={p:sha(PROJECT/p) for p in qualification_paths},
                         qualifier_loaded_library_record_sha256={p.relative_to(PROJECT).as_posix():sha(p) for p in STATE.glob('*_runtime.json')},
                         saved_full_guards=saved, official_calls=0, fits=0, permanent_updates=0,
                         actual_registered_point0_replay_still_required=True, quality_acceptance=False))
    # Binding is prospective. The first actual original-row forwards remain in registered point0.
    if entry.OUT.exists():
        raise FileExistsError(entry.OUT)
    entry.OUT.mkdir()
    profile = dict(device='cpu',system='Linux',machine='aarch64',python='3.12.14',torch='2.7.1+cpu',
                   minimum_glibc='2.28',torch_threads=4,native_BLAS_threads=24,deterministic_algorithms=True,
                   no_official_preflight_outside_registered_point0=True,
                   no_global_torch_or_binding_monkeypatch=True,original_CUDA_seal_is_history_only=True)
    plan = dict(original, status='V169_CPU_platform_registered_before_original_counted_calls',
                execution_authority=True,new_fit_permission=True,
                entry='platform_runtime/v169_cpu_training_entry_v1.py', resources=CPU_RESOURCES,
                resource_review_path=live_path.relative_to(PROJECT).as_posix(), source_sha256=sources,
                independent_preseal_review=root_path.relative_to(PROJECT).as_posix(),
                independent_preseal_review_sha256=sha(root_path), dependency_bundle=manifest_path.relative_to(PROJECT).as_posix(),
                dependency_bundle_sha256=sha(manifest_path),preseal_resource_snapshot=resources,platform_profile=profile)
    review_plan(plan)
    write(entry.PLAN, plan)
    runtime, loaded = runtime_bindings(sha)
    write(entry.OUT/'run_seal.json', dict(status='V169_CPU_platform_physically_sealed_before_official_calls',
          trainer_path=plan['entry'], plan_path=entry.PLAN.relative_to(PROJECT).as_posix(),plan_sha256=sha(entry.PLAN),
          source_sha256=sources,runtime_sha256=runtime,actual_loaded_files=loaded,python_version=sys.version,
          package_versions=versions,root_review_path=plan['independent_preseal_review'],root_review_sha256=sha(root_path),
          live_preflight_path=live_path.relative_to(PROJECT).as_posix(),live_preflight_sha256=sha(live_path),
          official_calls_before_seal=0,quality_acceptance=False))
    write(entry.OUT/'registration.json', dict(protocol=plan['protocol'],new_caps=plan['new_caps'],
          role_caps=plan['role_caps'],original_historical_costs=plan['prior_actual_costs'],official_calls=0,
          fits=0,permanent_updates=0,quality_acceptance=False))
    entry.configure()
    entry.main()
    incomplete = entry.OUT/'partial_inconclusive.json'
    write(STATE/'runner_outcome.json', dict(status='inconclusive' if incomplete.exists() else 'bounded_entry_returned',
          full_task_quality_acceptance=False,model_promoted=False,
          independent_full_original_row_quality_audit_required=True))
    if incomplete.exists():
        raise RuntimeError('Registered training stopped inconclusively; preserved raw results required for review')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        failure = STATE/'runner_failure.json'
        if not failure.exists():
            write(failure,dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                              quality_acceptance=False,no_automatic_retry=True))
        raise
