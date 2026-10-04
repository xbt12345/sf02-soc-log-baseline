"""Python3.8 stdlib launcher: offline install, bound run, complete result collection."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tarfile
import traceback
import zipfile

PROJECT = Path(__file__).resolve().parent
STATE = PROJECT/'platform_run_v1'
RESULT = PROJECT.parent/'SF02_V169_CPU_result_v1.zip'
REVIEW = PROJECT.parent/'SF02_V169_CPU_review_v1.zip'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def safe_file(relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or '\\' in relative or not name.parts:
        raise RuntimeError('Unsafe manifest path: '+relative)
    path = PROJECT.joinpath(*name.parts)
    if path.is_symlink() or PROJECT.resolve() not in path.resolve().parents:
        raise RuntimeError('Manifest file escapes package: '+relative)
    return path


def verify_package():
    manifest = json.loads((PROJECT/'platform_runtime/package_manifest.json').read_text(encoding='utf-8'))
    if not manifest['files']:
        raise RuntimeError('Empty package manifest')
    for relative, item in manifest['files'].items():
        path = safe_file(relative)
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError('Package integrity failure: '+relative)
    return manifest


def environment():
    value = dict(os.environ)
    for key in list(value):
        if key.startswith('PIP_') or key in ['PYTHONHOME','PYTHONPATH','PYTHONSTARTUP','PYTHONOPTIMIZE',
                                           'LD_LIBRARY_PATH','LD_PRELOAD']:
            value.pop(key,None)
    value.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                 OPENBLAS_NUM_THREADS='24',MKL_NUM_THREADS='24',OMP_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    return value


def run(command, name):
    with (STATE/name).open('x',encoding='utf-8') as log:
        proc = subprocess.Popen(command,cwd=PROJECT,env=environment(),stdout=log,stderr=subprocess.STDOUT)
        try:
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    if code:
        raise RuntimeError('Child failed ('+str(code)+'): '+name)


def collect():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    files = {}
    original_manifest = PROJECT/'platform_runtime/package_manifest.json'
    baseline = json.loads(original_manifest.read_text(encoding='utf-8'))['files'] if original_manifest.is_file() else {}
    references = {}
    excluded = {'.v169_cpu_env_v1','__pycache__'}
    for path in sorted(PROJECT.rglob('*')):
        relative = path.relative_to(PROJECT)
        key = relative.as_posix()
        vendor = key.startswith('platform/v169_cpu_v1/vendor/')
        old_binary = key in baseline and (path.suffix in ['.npy','.npz','.pt','.gz','.whl'] or key=='data/official/train.parquet')
        if any(part in excluded for part in relative.parts) or vendor or old_binary:
            if key in baseline:references[key] = baseline[key]
            continue
        if path.is_symlink():
            raise RuntimeError('Result collector refuses unexpected project symlink: '+str(relative))
        if path.is_file():
            files[key] = dict(bytes=path.stat().st_size,sha256=sha(path))
    # Installation receipts accompany the hashes of actual installed libraries in the seal.
    env = PROJECT/'.v169_cpu_env_v1'
    for name in ['installation_started.json','environment_ready.json','requirements.lock']:
        path = env/name
        if path.is_file():
            files[path.relative_to(PROJECT).as_posix()] = dict(bytes=path.stat().st_size,sha256=sha(path))
    temporary = RESULT.with_suffix('.zip.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
        for relative in files:
            path = PROJECT/relative
            archive.write(path,'SF02/'+relative)
            if sha(path) != files[relative]['sha256']:
                raise RuntimeError('Result file changed while collecting: '+relative)
        archive.writestr('result_manifest.json',json.dumps(dict(files=files,full_task_quality_acceptance=False,
          model_promoted=False,includes_every_new_run_file=True,source_environment_and_raw_logs_retained=True,
          original_input_references=references,input_reconstruction='Use the original uploaded execution ZIP and exact source/package manifests; no platform input is deleted',
          independent_of_original_execution_package=False),indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Result ZIP CRC failure')
    temporary.rename(RESULT)
    print('RESULT_ZIP='+str(RESULT),flush=True)
    # A compact audit view accompanies the full lossless new-run evidence.
    latest = {}
    for relative in files:
        parts = PurePosixPath(relative).parts
        if len(parts)>=5 and parts[:2]==('artifacts','v169_platform_CPU_training_v1') and parts[3].startswith('accepted') and parts[-1]=='commit.json':
            state=int(parts[3][len('accepted'):]);latest[parts[2]]=max(state,latest.get(parts[2],0))
    selected = {}
    for relative,item in files.items():
        parts = PurePosixPath(relative).parts
        suffix = PurePosixPath(relative).suffix
        keep = suffix in ['.py','.json','.jsonl','.md','.txt','.log','.cpp'] or relative in baseline
        if len(parts)>=4 and parts[:2]==('artifacts','v169_platform_CPU_training_v1'):
            if parts[3]=='endpoint':keep=True
            if parts[3].startswith('accepted'):
                state=int(parts[3][len('accepted'):])
                keep=keep or (state>latest.get(parts[2],0)-5 and parts[2] in latest and suffix!='.pt')
                if suffix=='.pt':keep=parts[2] in latest and state==latest[parts[2]]
        if keep:selected[relative]=item
    if REVIEW.exists():raise FileExistsError(REVIEW)
    with zipfile.ZipFile(REVIEW,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
        for relative in selected:
            archive.write(PROJECT/relative,'SF02/'+relative)
            if sha(PROJECT/relative)!=selected[relative]['sha256']:
                raise RuntimeError('Review input changed during collection: '+relative)
        archive.writestr('review_manifest.json',json.dumps(dict(files=selected,full_result_zip=RESULT.name,
            full_result_zip_sha256=sha(RESULT),original_input_references=references,
            omitted_new_evidence={k:v for k,v in files.items() if k not in selected},
            full_task_quality_acceptance=False,complete_new_run_evidence_is_in_full_result=True),indent=2))
    with zipfile.ZipFile(REVIEW) as archive:
        if archive.testzip() is not None:raise RuntimeError('Review ZIP CRC failed')
        for relative,item in selected.items():
            h=hashlib.sha256()
            with archive.open('SF02/'+relative) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=item['sha256']:raise RuntimeError('Review ZIP member identity failure: '+relative)
    print('REVIEW_ZIP='+str(REVIEW)+' bytes='+str(REVIEW.stat().st_size),flush=True)
    if REVIEW.stat().st_size>150*1024**2:
        print('Review archive exceeds 150 MiB; use a file transfer for the full result',flush=True)


def main():
    if STATE.exists() or RESULT.exists() or REVIEW.exists() or RESULT.with_suffix('.zip.partial').exists():
        raise RuntimeError('Previous execution/result exists; automatic retry or overwrite is forbidden')
    STATE.mkdir()
    exit_code = 1
    try:
        manifest = verify_package()
        # The installer checks architecture/glibc before creating its private environment.
        sys.path.insert(0,str(PROJECT/'platform_runtime'))
        from root_offline_bootstrap_v1 import host_identity, VENDOR, RUNTIME
        host_identity()
        installation = 1024**3
        with tarfile.open(VENDOR/RUNTIME,'r:gz') as archive:
            installation += 2*sum(item.size for item in archive.getmembers() if item.isfile())
        for wheel in (VENDOR/'wheelhouse').glob('*.whl'):
            with zipfile.ZipFile(wheel) as archive:
                installation += sum(item.file_size for item in archive.infolist())
        minimum = manifest['minimum_free_disk_before_install_bytes'] + installation
        free = shutil.disk_usage(PROJECT).free
        if free < minimum:
            raise RuntimeError('Insufficient full-run/result-ZIP/installation disk: '+str((free,minimum)))
        print('Offline installation starting; raw logs in '+str(STATE),flush=True)
        run([sys.executable,'-B',str(PROJECT/'platform_runtime/root_offline_bootstrap_v1.py')],'offline_install.log')
        python = PROJECT/'.v169_cpu_env_v1/venv/bin/python'
        print('Platform qualification and registered CPU training starting',flush=True)
        run([str(python),'-I','-B','-X','utf8',str(PROJECT/'platform_runtime/v169_cpu_platform_runner_v2.py')],'platform_runner.log')
        exit_code = 0
    except BaseException as error:
        (STATE/'launcher_failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),
           traceback=traceback.format_exc(),full_task_quality_acceptance=False,no_automatic_retry=True),indent=2)+'\n',encoding='utf-8')
        print('Execution stopped: '+str(error),flush=True)
    finally:
        try:collect()
        except BaseException:
            exit_code = 1
            (STATE/'collector_failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print('Result ZIP collection failed; preserve full directory '+str(PROJECT),flush=True)
        print('Quality acceptance remains pending independent full-row review',flush=True)
    return exit_code


if __name__ == '__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    sys.exit(main())
