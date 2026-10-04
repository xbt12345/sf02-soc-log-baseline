"""Python 3.8 compatible offline installer; no model or training calls."""
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT / 'platform/v169_cpu_v1/vendor'
ENVIRONMENT = PROJECT / '.v169_cpu_env_v1'
PRIVATE = ENVIRONMENT / 'venv/bin/python'
TRUSTED_FILES = {'wheelhouse/cloudpickle-3.1.2-py3-none-any.whl': '9acb47f6afd73f60dc1df93bb801b472f05ff42fa6c84167d25cb206be1fbf4a', 'wheelhouse/filelock-3.32.6-py3-none-any.whl': '3f16ecd0117feae0dfc147e8c62eb5daeccd8bd800378c3ddf416de9b4feb6b1', 'wheelhouse/fsspec-2026.7.0-py3-none-any.whl': 'b57ddbafedfaef7018c1ecab32aa200a9d7ca26b77965f64e48b70061249d279', 'wheelhouse/jinja2-3.1.6-py3-none-any.whl': '85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67', 'wheelhouse/joblib-1.6.0-py3-none-any.whl': '3dbbf9f6e4b592a2357b854608e980fe6390d131d7a82f011a377ef2ebef7aba', 'wheelhouse/markupsafe-3.0.3-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl': '3a7e8ae81ae39e62a41ec302f972ba6ae23a5c5396c8e60113e9066ef893da0d', 'wheelhouse/mpmath-1.3.0-py3-none-any.whl': 'a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c', 'wheelhouse/networkx-3.6.1-py3-none-any.whl': 'd47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762', 'wheelhouse/numpy-2.2.6-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl': 'f2618db89be1b4e05f7a1a847a9c1c0abd63e63a1607d892dd54668dd92faf87', 'wheelhouse/packaging-26.3-py3-none-any.whl': 'd7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c', 'wheelhouse/pandas-2.2.3-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl': '5de54125a92bb4d1c051c0659e6fcb75256bf799a732a87184e5ea503965bce3', 'wheelhouse/pip-25.0.1-py3-none-any.whl': 'c46efd13b6aa8279f33f2864459c8ce587ea6a1a59ee20de055868d8f7688f7f', 'wheelhouse/pyarrow-19.0.1-cp312-cp312-manylinux_2_28_aarch64.whl': 'd383591f3dcbe545f6cc62daaef9c7cdfe0dff0fb9e1c8121101cabe9098cfa6', 'wheelhouse/python_dateutil-2.9.0.post0-py2.py3-none-any.whl': 'a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427', 'wheelhouse/pytz-2026.3.post1-py2.py3-none-any.whl': 'dd95840dd199baea12d9cc096a1d452caa6596a1c1e4b5f3dbd1541855d5e815', 'wheelhouse/rtdl_num_embeddings-0.0.12-py3-none-any.whl': '87fd61270118915cf40888f2164f3a1f0353edf702f15fd91e71a147d23c930d', 'wheelhouse/scikit_learn-1.6.1-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl': '1061b7c028a8663fb9a1a1baf9317b64a257fcb036dae5c8752b2abef31d136f', 'wheelhouse/scipy-1.15.3-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl': 'c05045d8b9bfd807ee1b9f38761993297b10b245f012b11b13b91ba8945f7e45', 'wheelhouse/setuptools-84.0.0-py3-none-any.whl': '51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670', 'wheelhouse/six-1.17.0-py2.py3-none-any.whl': '4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274', 'wheelhouse/sympy-1.14.0-py3-none-any.whl': 'e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5', 'wheelhouse/tabm-0.0.3-py3-none-any.whl': '6cabed3cb436fef5e8a45f46097fd745f51b3e16be16d555b9af16cffda7f944', 'wheelhouse/threadpoolctl-3.6.0-py3-none-any.whl': '43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb', 'wheelhouse/torch-2.7.1+cpu-cp312-cp312-manylinux_2_28_aarch64.whl': '3bf2db5adf77b433844f080887ade049c4705ddf9fe1a32023ff84ff735aa5ad', 'wheelhouse/typing_extensions-4.16.0-py3-none-any.whl': '481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8', 'wheelhouse/tzdata-2026.4-py2.py3-none-any.whl': 'c2169a8b0a7a5e9674da5a135ccdfb2b3e671b333ed9fed17b41f73c34476e81', 'cpython-3.12.14+20260924-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz': 'c8499b61252c433280f134df954464d19811527b31cb920c35fc6967c1222e35'}
LOCK_LINES = ['cloudpickle==3.1.2 --hash=sha256:9acb47f6afd73f60dc1df93bb801b472f05ff42fa6c84167d25cb206be1fbf4a', 'filelock==3.32.6 --hash=sha256:3f16ecd0117feae0dfc147e8c62eb5daeccd8bd800378c3ddf416de9b4feb6b1', 'fsspec==2026.7.0 --hash=sha256:b57ddbafedfaef7018c1ecab32aa200a9d7ca26b77965f64e48b70061249d279', 'jinja2==3.1.6 --hash=sha256:85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67', 'joblib==1.6.0 --hash=sha256:3dbbf9f6e4b592a2357b854608e980fe6390d131d7a82f011a377ef2ebef7aba', 'markupsafe==3.0.3 --hash=sha256:3a7e8ae81ae39e62a41ec302f972ba6ae23a5c5396c8e60113e9066ef893da0d', 'mpmath==1.3.0 --hash=sha256:a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c', 'networkx==3.6.1 --hash=sha256:d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762', 'numpy==2.2.6 --hash=sha256:f2618db89be1b4e05f7a1a847a9c1c0abd63e63a1607d892dd54668dd92faf87', 'packaging==26.3 --hash=sha256:d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c', 'pandas==2.2.3 --hash=sha256:5de54125a92bb4d1c051c0659e6fcb75256bf799a732a87184e5ea503965bce3', 'pip==25.0.1 --hash=sha256:c46efd13b6aa8279f33f2864459c8ce587ea6a1a59ee20de055868d8f7688f7f', 'pyarrow==19.0.1 --hash=sha256:d383591f3dcbe545f6cc62daaef9c7cdfe0dff0fb9e1c8121101cabe9098cfa6', 'python-dateutil==2.9.0.post0 --hash=sha256:a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427', 'pytz==2026.3.post1 --hash=sha256:dd95840dd199baea12d9cc096a1d452caa6596a1c1e4b5f3dbd1541855d5e815', 'rtdl-num-embeddings==0.0.12 --hash=sha256:87fd61270118915cf40888f2164f3a1f0353edf702f15fd91e71a147d23c930d', 'scikit-learn==1.6.1 --hash=sha256:1061b7c028a8663fb9a1a1baf9317b64a257fcb036dae5c8752b2abef31d136f', 'scipy==1.15.3 --hash=sha256:c05045d8b9bfd807ee1b9f38761993297b10b245f012b11b13b91ba8945f7e45', 'setuptools==84.0.0 --hash=sha256:51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670', 'six==1.17.0 --hash=sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274', 'sympy==1.14.0 --hash=sha256:e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5', 'tabm==0.0.3 --hash=sha256:6cabed3cb436fef5e8a45f46097fd745f51b3e16be16d555b9af16cffda7f944', 'threadpoolctl==3.6.0 --hash=sha256:43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb', 'torch==2.7.1+cpu --hash=sha256:3bf2db5adf77b433844f080887ade049c4705ddf9fe1a32023ff84ff735aa5ad', 'typing-extensions==4.16.0 --hash=sha256:481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8', 'tzdata==2026.4 --hash=sha256:c2169a8b0a7a5e9674da5a135ccdfb2b3e671b333ed9fed17b41f73c34476e81']
PACKAGES = {'cloudpickle': '3.1.2', 'filelock': '3.32.6', 'fsspec': '2026.7.0', 'jinja2': '3.1.6', 'joblib': '1.6.0', 'markupsafe': '3.0.3', 'mpmath': '1.3.0', 'networkx': '3.6.1', 'numpy': '2.2.6', 'packaging': '26.3', 'pandas': '2.2.3', 'pip': '25.0.1', 'pyarrow': '19.0.1', 'python-dateutil': '2.9.0.post0', 'pytz': '2026.3.post1', 'rtdl-num-embeddings': '0.0.12', 'scikit-learn': '1.6.1', 'scipy': '1.15.3', 'setuptools': '84.0.0', 'six': '1.17.0', 'sympy': '1.14.0', 'tabm': '0.0.3', 'threadpoolctl': '3.6.0', 'torch': '2.7.1+cpu', 'typing-extensions': '4.16.0', 'tzdata': '2026.4'}
RUNTIME = 'cpython-3.12.14+20260924-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def clean_environment():
    result = dict(os.environ)
    for key in list(result):
        if key.startswith('PIP_') or key in ['PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONOPTIMIZE']:
            result.pop(key, None)
    result['PYTHONNOUSERSITE'] = '1'
    return result


def host_identity():
    if platform.system() != 'Linux' or platform.machine() != 'aarch64' or sys.byteorder != 'little':
        raise RuntimeError('This package requires little-endian Linux aarch64')
    library = ctypes.CDLL(None)
    function = library.gnu_get_libc_version
    function.restype = ctypes.c_char_p
    version = function().decode('ascii')
    if tuple(map(int, version.split('.')[:2])) < (2, 28):
        raise RuntimeError('glibc >= 2.28 is required')
    if sys.flags.optimize:
        raise RuntimeError('Python optimization would disable source assertions')
    return {'system': 'Linux', 'machine': 'aarch64', 'glibc': version}


def validate_member(root, member):
    name = PurePosixPath(member.name)
    if name.is_absolute() or not name.parts or name.parts[0] != 'python' or '..' in name.parts:
        raise RuntimeError('Unsafe runtime archive member: ' + member.name)
    destination = (root / Path(*name.parts)).resolve()
    if destination != root and root not in destination.parents:
        raise RuntimeError('Runtime archive member escapes its directory')
    if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
        raise RuntimeError('Unsupported runtime archive member type')
    if member.issym() or member.islnk():
        link = PurePosixPath(member.linkname)
        if link.is_absolute():
            raise RuntimeError('Absolute runtime archive link')
        target = destination.parent / Path(*link.parts) if member.issym() else root / Path(*link.parts)
        resolved = target.resolve()
        if resolved != root and root not in resolved.parents:
            raise RuntimeError('Runtime archive link escapes its directory')


def package_versions():
    statement = ('import importlib.metadata,json; print(json.dumps({name: '
                 'importlib.metadata.version(name) for name in ' + repr(sorted(PACKAGES)) + '}))')
    output = subprocess.check_output([str(PRIVATE), '-I', '-c', statement], env=clean_environment(), text=True)
    observed = json.loads(output)
    if observed != PACKAGES:
        raise RuntimeError('Installed private package versions differ from the offline lock')
    subprocess.run([str(PRIVATE), '-I', '-m', 'pip', 'check'], env=clean_environment(), check=True)
    return observed


def main():
    identity = host_identity()
    for relative, expected in TRUSTED_FILES.items():
        path = VENDOR / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError('Missing or damaged offline vendor: ' + relative)
    ready = ENVIRONMENT / 'environment_ready.json'
    if ready.exists():
        previous = json.loads(ready.read_text(encoding='utf-8'))
        if previous.get('vendor_sha256') != TRUSTED_FILES or previous.get('package_versions') != PACKAGES:
            raise RuntimeError('An existing environment belongs to another offline package')
        package_versions()
        print(json.dumps({'status': 'private_offline_environment_verified', 'python': str(PRIVATE), 'official_model_calls': 0}), flush=True)
        return
    if ENVIRONMENT.exists():
        raise RuntimeError('Incomplete private installation is preserved; do not silently overwrite it')
    ENVIRONMENT.mkdir()
    (ENVIRONMENT / 'installation_started.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    standalone = ENVIRONMENT / 'standalone'
    standalone.mkdir()
    archive = VENDOR / RUNTIME
    with tarfile.open(archive, 'r:gz') as handle:
        members = handle.getmembers()
        root = standalone.resolve()
        names = set()
        for member in members:
            if member.name in names:
                raise RuntimeError('Duplicate runtime archive member')
            names.add(member.name)
            validate_member(root, member)
        handle.extractall(str(root), members=members)
    python = standalone / 'python/bin/python3.12'
    version = subprocess.check_output([str(python), '-I', '-c', 'import sys; print(".".join(map(str, sys.version_info[:3])))'], env=clean_environment(), text=True).strip()
    if version != '3.12.14':
        raise RuntimeError('Bundled runtime Python version differs')
    subprocess.run([str(python), '-I', '-m', 'venv', str(ENVIRONMENT / 'venv')], env=clean_environment(), check=True)
    lock = ENVIRONMENT / 'requirements.lock'
    lock.write_text('\n'.join(LOCK_LINES) + '\n', encoding='utf-8')
    subprocess.run([str(PRIVATE), '-I', '-m', 'pip', 'install', '--no-index', '--find-links', str(VENDOR / 'wheelhouse'), '--require-hashes', '--only-binary=:all:', '--disable-pip-version-check', '-r', str(lock)], env=clean_environment(), check=True)
    observed = package_versions()
    record = {'status': 'private_offline_environment_installed', 'python': str(PRIVATE), 'platform': identity, 'vendor_sha256': TRUSTED_FILES, 'package_versions': observed, 'official_model_calls': 0, 'scope': 'Offline installation only; no model quality or numerical qualification'}
    ready.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': record['status'], 'python': str(PRIVATE), 'official_model_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
