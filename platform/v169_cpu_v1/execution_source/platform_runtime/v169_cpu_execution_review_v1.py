"""Prospective CPU environment binding, with the original scientific contract."""
import ctypes
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path, PurePosixPath
import torch
from experiment_review import ROOT, read, sha, check_bindings
from v169_pair_execution_review import review_plan as original_scope_review
from threadpoolctl import threadpool_info

ORIGINAL = ROOT / 'training/review_policy/v169_prior_pair_execution_contract.json'
ENVIRONMENT_KEYS = {
    'status', 'execution_authority', 'new_fit_permission', 'entry',
    'resource_review_path', 'resources', 'independent_preseal_review',
    'independent_preseal_review_sha256', 'dependency_bundle',
    'dependency_bundle_sha256', 'preseal_resource_snapshot', 'source_sha256'}
OUTPUT_BOUND = 12136488300
RESERVE = 2147483648
ZIP_HEADER_BOUND = 256 * 1024**2
RESULT_INPUT_BOUND = 2 * 1024**3
CPU_RESOURCES = dict(
    full_run_worst_storage_bytes=OUTPUT_BOUND,
    fixed_free_disk_reserve_bytes=RESERVE,
    result_input_and_qualification_bound_bytes=RESULT_INPUT_BOUND,
    result_zip_maximum_bytes=OUTPUT_BOUND + RESULT_INPUT_BOUND + ZIP_HEADER_BOUND,
    minimum_free_disk_start_bytes=2*OUTPUT_BOUND+RESULT_INPUT_BOUND+ZIP_HEADER_BOUND+RESERVE,
    minimum_effective_available_RAM_bytes=8*1024**3,
    qualification_peak_RSS_bound_bytes=8*1024**3,
    minimum_cpu_cores=4)


def platform_identity():
    if platform.system() != 'Linux' or platform.machine() != 'aarch64' or sys.byteorder != 'little':
        raise RuntimeError('Requires little-endian Linux aarch64; no alternate backend')
    libc = ctypes.CDLL(None)
    version_function = libc.gnu_get_libc_version
    version_function.restype = ctypes.c_char_p
    version = version_function().decode('ascii')
    if tuple(map(int, version.split('.')[:2])) < (2, 28):
        raise RuntimeError('glibc >=2.28 required by the pinned CPU wheel')
    if sys.version_info[:3] != (3, 12, 14) or sys.flags.optimize:
        raise RuntimeError('Isolated Python3.12.14 without optimization required')
    if torch.__version__ != '2.7.1+cpu' or torch.version.cuda is not None:
        raise RuntimeError('Exact CPU PyTorch required; CUDA/Ascend are not this protocol')
    return dict(system=platform.system(), machine=platform.machine(), byteorder=sys.byteorder,
                glibc=version, python=sys.version, executable=str(Path(sys.executable).resolve()),
                torch=torch.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)))


def resource_snapshot():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, rest = line.split(':', 1)
        values[key] = int(rest.strip().split()[0]) * 1024
    available = values['MemAvailable']
    groups = []
    candidates = [Path('/sys/fs/cgroup')]
    memory_candidates = [Path('/sys/fs/cgroup/memory')]
    cpu_candidates = [Path('/sys/fs/cgroup/cpu'), Path('/sys/fs/cgroup/cpu,cpuacct')]
    cpu_v2_candidates = [Path('/sys/fs/cgroup')]
    def ancestors(root, relative):
        if '..' in PurePosixPath(relative).parts:
            raise RuntimeError('Cannot establish cgroup namespace ancestry safely: '+relative)
        current = root / relative.lstrip('/')
        result = []
        while current.is_relative_to(root):
            result.append(current)
            if current == root:
                break
            current = current.parent
        return result
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        separator = fields.index('-')
        kind = fields[separator+1]
        if kind in ['cgroup','cgroup2']:
            def unescape(value):
                for old,new in [('\\040',' '),('\\011','\t'),('\\012','\n'),('\\134','\\')]:
                    value=value.replace(old,new)
                return value
            mounts.append((kind,PurePosixPath(unescape(fields[3])),Path(unescape(fields[4])),
                           set(fields[separator+3].split(','))))
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        hierarchy, controllers, relative = line.split(':', 2)
        for kind, mount_root, mountpoint, options in mounts:
            if kind=='cgroup2' and not controllers or kind=='cgroup' and set(controllers.split(',')) & options:
                if '..' in PurePosixPath(relative).parts:
                    raise RuntimeError('Unresolved cgroup namespace membership: '+relative)
                group = PurePosixPath(relative)
                # A namespace may expose its own group as / while mountinfo names the host subtree.
                inside = group.relative_to(mount_root) if group.is_relative_to(mount_root) else PurePosixPath('.')
                visible = ancestors(mountpoint, str(inside))
                if kind=='cgroup2':candidates.extend(visible);cpu_v2_candidates.extend(visible)
                if 'memory' in controllers.split(','):memory_candidates.extend(visible)
                if 'cpu' in controllers.split(','):cpu_candidates.extend(visible)
        if not controllers:
            candidates.extend(ancestors(Path('/sys/fs/cgroup'), relative))
            cpu_v2_candidates.extend(ancestors(Path('/sys/fs/cgroup'), relative))
        if 'memory' in controllers.split(','):
            memory_candidates.extend(ancestors(Path('/sys/fs/cgroup/memory'), relative))
        if 'cpu' in controllers.split(','):
            for root in [Path('/sys/fs/cgroup/cpu'), Path('/sys/fs/cgroup/cpu,cpuacct')]:
                cpu_candidates.extend(ancestors(root, relative))
    for group in sorted(set(candidates)):
        limit, used = group/'memory.max', group/'memory.current'
        if limit.is_file() and used.is_file():
            raw = limit.read_text().strip()
            if raw != 'max':
                maximum, current = int(raw), int(used.read_text())
                headroom = max(0, maximum-current)
                groups.append(dict(path=str(group), limit=maximum, current=current, available=headroom))
                available = min(available, headroom)
    for group in sorted(set(memory_candidates)):
        limit, used = group/'memory.limit_in_bytes', group/'memory.usage_in_bytes'
        if limit.is_file() and used.is_file():
            maximum, current = int(limit.read_text()), int(used.read_text())
            headroom = max(0, maximum-current)
            groups.append(dict(path=str(group), limit=maximum, current=current, available=headroom))
            available = min(available, headroom)
    cpu_bounds = []
    effective_cpus = float(len(os.sched_getaffinity(0)))
    for group in sorted(set(cpu_v2_candidates)):
        path = group/'cpu.max'
        if path.is_file():
            quota, period = path.read_text().split()
            if quota != 'max':
                value = int(quota)/int(period)
                cpu_bounds.append(dict(path=str(path), quota=int(quota), period=int(period), cores=value))
                effective_cpus = min(effective_cpus, value)
    for group in sorted(set(cpu_candidates)):
        quota, period = group/'cpu.cfs_quota_us', group/'cpu.cfs_period_us'
        if quota.is_file() and period.is_file():
            maximum, interval = int(quota.read_text()), int(period.read_text())
            if maximum > 0:
                value = maximum/interval
                cpu_bounds.append(dict(path=str(group), quota=maximum, period=interval, cores=value))
                effective_cpus = min(effective_cpus, value)
    return dict(OS_mem_available_bytes=values['MemAvailable'], effective_available_RAM_bytes=available,
                cgroup_memory_bounds=groups, disk_free_bytes=shutil.disk_usage(ROOT).free,
                cpu_affinity_cores=len(os.sched_getaffinity(0)),
                effective_cpu_cores=effective_cpus, cgroup_CPU_bounds=cpu_bounds,
                Linux_commit_limit_bytes=values.get('CommitLimit'),
                Linux_committed_AS_bytes=values.get('Committed_AS'),
                Windows_commit_gate_not_reused_on_Linux=True)


def require_resources(resources, phase):
    if resources != CPU_RESOURCES or phase not in ['initial', 'running']:
        raise RuntimeError('Unregistered CPU resource profile/phase')
    state = resource_snapshot()
    disk = resources['minimum_free_disk_start_bytes'] if phase == 'initial' else resources['fixed_free_disk_reserve_bytes']
    if state['disk_free_bytes'] < disk or state['effective_available_RAM_bytes'] < resources['minimum_effective_available_RAM_bytes'] or state['effective_cpu_cores'] < resources['minimum_cpu_cores']:
        raise RuntimeError('Registered actual CPU memory/CPU/disk prerequisite not met: '+json.dumps(state))
    return state


def review_plan(plan):
    original = read(ORIGINAL)
    source_manifest = read(ROOT/'platform_runtime/package_source_manifest.json')
    expected_sources = source_manifest['project_source_sha256']
    if len(expected_sources) < 248 or plan['source_sha256'] != expected_sources:
        raise RuntimeError('Complete independently reviewed CPU source/data closure required')
    check_bindings(plan['source_sha256'])
    original_scope_review(plan)
    for key, value in original.items():
        if key not in ENVIRONMENT_KEYS and plan.get(key) != value:
            raise RuntimeError('Original scientific contract altered: '+key)
    if plan['resources'] != CPU_RESOURCES or plan['entry'] != 'platform_runtime/v169_cpu_training_entry_v1.py':
        raise RuntimeError('Exact separate CPU entry/resources required')
    profile = plan.get('platform_profile', {})
    expected = dict(device='cpu', system='Linux', machine='aarch64', python='3.12.14',
                    torch='2.7.1+cpu', minimum_glibc='2.28', torch_threads=4,
                    native_BLAS_threads=24, deterministic_algorithms=True,
                    no_official_preflight_outside_registered_point0=True,
                    no_global_torch_or_binding_monkeypatch=True,
                    original_CUDA_seal_is_history_only=True)
    if profile != expected:
        raise RuntimeError('Exact explicit platform profile required')
    return dict(status='original_science_and_explicit_CPU_migration_reviewed', quality_acceptance=False)


def require_run_seal(path, current_trainer, phase='running'):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError('CPU platform not sealed: zero official calls permitted')
    seal = read(path)
    if seal.get('status') != 'V169_CPU_platform_physically_sealed_before_official_calls':
        raise RuntimeError('A new platform seal is required; Windows seal is history')
    if (ROOT/seal['trainer_path']).resolve() != Path(current_trainer).resolve():
        raise RuntimeError('Wrong CPU trainer')
    platform_identity()
    check_bindings(seal['source_sha256'])
    for key, expected in seal['runtime_sha256'].items():
        if sha(Path(key)) != expected:
            raise RuntimeError('Actual installed runtime/system library changed: '+key)
    if sys.version != seal['python_version'] or {name:importlib.metadata.version(name) for name in seal['package_versions']} != seal['package_versions']:
        raise RuntimeError('Actual sealed runtime environment changed')
    plan_path = ROOT/seal['plan_path']
    if sha(plan_path) != seal['plan_sha256']:
        raise RuntimeError('Platform execution contract changed')
    plan = read(plan_path)
    review_plan(plan)
    if plan.get('execution_authority') is not True or plan.get('new_fit_permission') is not True:
        raise RuntimeError('Candidate plan has no execution authority')
    root_review = read(ROOT/seal['root_review_path'])
    if sha(ROOT/seal['root_review_path']) != seal['root_review_sha256'] or root_review.get('supports_platform_execution_after_live_preflight') is not True:
        raise RuntimeError('Independent static root review missing or changed')
    if (root_review.get('reviewed_source_manifest_sha256') != sha(ROOT/'platform_runtime/package_source_manifest.json')
            or root_review.get('reviewed_entry_sha256') != sha(current_trainer)
            or root_review.get('source_sha256') != plan['source_sha256']
            or seal['source_sha256'] != plan['source_sha256']):
        raise RuntimeError('Independent review/seal does not bind this exact complete CPU closure')
    if not seal['runtime_sha256'] or str(Path(sys.executable).resolve()) not in seal['runtime_sha256']:
        raise RuntimeError('Actual private runtime must be physically bound')
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts)==6 and parts[5].startswith('/'):
            if parts[5].endswith(' (deleted)'):
                raise RuntimeError('Actual mapped library/file was deleted')
            mapped = Path(parts[5]).resolve()
            if mapped.is_file() and str(mapped) not in seal['runtime_sha256']:
                try:relative=mapped.relative_to(ROOT).as_posix()
                except ValueError:relative=None
                expected=plan['source_sha256'].get(relative)
                if expected is None or sha(mapped)!=expected:
                    raise RuntimeError('New unsealed mapped library/file: '+str(mapped))
    live = read(ROOT/seal['live_preflight_path'])
    if sha(ROOT/seal['live_preflight_path']) != seal['live_preflight_sha256'] or live.get('passed') is not True or live.get('official_calls') != 0:
        raise RuntimeError('Actual platform qualification missing or changed')
    if torch.get_num_threads() != 4 or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('Registered CPU execution configuration changed')
    if any(pool['num_threads'] != 24 for pool in threadpool_info() if pool['user_api']=='blas'):
        raise RuntimeError('Registered native BLAS threads changed')
    require_resources(plan['resources'], phase)
    return plan
