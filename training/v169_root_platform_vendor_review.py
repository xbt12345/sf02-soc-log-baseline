"""Independent artifact/provenance review; no imports of training or Torch."""
from concurrent.futures import ThreadPoolExecutor
from email import message_from_bytes
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import struct
import tarfile
import urllib.request
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'platform/v169_cpu_v1/vendor'
OUT = ROOT / 'artifacts/v169_root_platform_vendor_review_20261002'


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'SF02-independent-artifact-audit'})
    with urllib.request.urlopen(request, timeout=30) as stream:
        return json.load(stream)


def elf_arm64(header):
    return len(header) >= 20 and header[:4] == b'\x7fELF' and header[4:6] == b'\x02\x01' and struct.unpack_from('<H', header, 18)[0] == 183


def inspect_wheel(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, ('bad_crc', path.name)
        metadata_paths = [p for p in archive.namelist() if p.count('/') == 1 and p.endswith('.dist-info/METADATA')]
        assert len(metadata_paths) == 1, ('metadata_identity', path.name)
        metadata = message_from_bytes(archive.read(metadata_paths[0]))
        prefix = metadata_paths[0].rsplit('/', 1)[0]
        tags = message_from_bytes(archive.read(prefix + '/WHEEL')).get_all('Tag', [])
        assert tags and all(t.endswith('-any') or ('cp312' in t and 'aarch64' in t) for t in tags), (path.name, tags)
        requirement = metadata.get('Requires-Python', '')
        assert '3.12.14' in SpecifierSet(requirement), (path.name, requirement)
        binaries = 0
        launcher_resources = []
        for name in archive.namelist():
            pure = PurePosixPath(name)
            assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in name, ('unsafe_wheel_path', name)
            if name.lower().endswith(('.dll', '.pyd', '.exe')):
                # Universal packaging tools ship dormant Windows entry-point
                # launcher resources. They are not Linux extension modules.
                assert name.lower().endswith('.exe') and all(t.endswith('-any') for t in tags) and (
                    name.startswith('pip/_vendor/distlib/') or name.startswith('setuptools/')
                ), ('unexpected_windows_binary', path.name, name)
                launcher_resources.append(name)
            if name.endswith('.so') or '.so.' in name:
                with archive.open(name) as stream:
                    assert elf_arm64(stream.read(64)), ('wrong_elf_machine', path.name, name)
                binaries += 1
        return dict(file=path.name, name=canonicalize_name(metadata['Name']), version=metadata['Version'],
                    bytes=path.stat().st_size, sha256=digest(path), tags=tags, requires_python=requirement,
                    dependencies=metadata.get_all('Requires-Dist', []), aarch64_ELF_objects=binaries,
                    dormant_packaging_launcher_resources=launcher_resources)


def official_identity(item):
    if item['name'] == 'torch':
        url = 'https://download.pytorch.org/whl/cpu/torch/'
        request = urllib.request.Request(url, headers={'User-Agent': 'SF02-independent-artifact-audit'})
        with urllib.request.urlopen(request, timeout=30) as stream:
            html = stream.read().decode('utf-8')
        import re
        from urllib.parse import unquote
        links = re.findall(r'href="([^"]+)"', html)
        matched = [s for s in links if unquote(s.split('#')[0]).rsplit('/', 1)[-1] == item['file']]
        assert len(matched) == 1, ('official_torch_asset_missing', item['file'])
        expected = matched[0].rsplit('#sha256=', 1)[-1]
    else:
        url = 'https://pypi.org/pypi/{}/{}/json'.format(item['name'], item['version'])
        data = fetch_json(url)
        matched = [s for s in data['urls'] if s['filename'] == item['file']]
        assert len(matched) == 1, ('official_pypi_asset_missing', item['file'])
        expected = matched[0]['digests']['sha256']
    assert expected == item['sha256'], ('official_digest_mismatch', item['file'])
    return dict(file=item['file'], official_metadata_url=url, expected_sha256=expected, passed=True)


def main():
    OUT.mkdir(exist_ok=True)
    wheels = [inspect_wheel(p) for p in sorted((VENDOR / 'wheelhouse').glob('*.whl'))]
    assert len(wheels) == 26 and len({w['name'] for w in wheels}) == 26, 'Expected complete fixed wheel set'
    versions = {w['name']: w['version'] for w in wheels}
    env = dict(default_environment(), os_name='posix', sys_platform='linux', platform_system='Linux',
               platform_machine='aarch64', python_version='3.12', python_full_version='3.12.14',
               implementation_name='cpython', platform_python_implementation='CPython', extra='')
    dependencies = []
    for wheel in wheels:
        for raw in wheel['dependencies']:
            r = Requirement(raw)
            if r.marker is not None and not r.marker.evaluate(env):
                continue
            name = canonicalize_name(r.name)
            assert name in versions and versions[name] in r.specifier, ('unresolved_offline_dependency', wheel['name'], raw)
            dependencies.append(dict(parent=wheel['name'], requirement=raw, bundled_version=versions[name]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(official_identity, wheels))
    downloaded = json.loads((VENDOR / 'runtime_download.json').read_text())
    runtime = ROOT / downloaded['path']
    actual = digest(runtime)
    release = fetch_json('https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/20260924')
    assets = [a for a in release['assets'] if a['name'] == runtime.name]
    assert len(assets) == 1 and assets[0]['digest'] == 'sha256:' + actual, 'Independent runtime vendor digest'
    assert actual == downloaded['sha256'] and runtime.stat().st_size == downloaded['bytes']
    python_ELF = []
    with tarfile.open(runtime, 'r:gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and '..' not in path.parts, ('unsafe_runtime_path', member.name)
            if member.isfile() and (member.name.endswith('/bin/python3.12') or '/lib/libpython3.12.so' in member.name):
                with archive.extractfile(member) as stream:
                    assert elf_arm64(stream.read(64)), ('runtime_elf', member.name)
                python_ELF.append(member.name)
    assert python_ELF, 'No ARM64 Python ELF executable/library'
    report = dict(status='offline_vendor_artifacts_and_dependencies_independently_verified', passed=True,
                  wheels=wheels, active_dependency_checks=dependencies, official_wheel_receipts=receipts,
                  runtime=dict(file=runtime.name, bytes=runtime.stat().st_size, sha256=actual,
                               release_metadata_url=release['url'], actual_AArch64_ELF_objects=python_ELF),
                  official_model_calls=0, fits=0, updates=0, runtime_executed=False,
                  scope='CRC, full file hashes, official metadata digests, ARM64 ELF headers and Python3.12 dependency graph only; not Linux runtime or SOC quality acceptance')
    (OUT / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, wheels=len(wheels), active_dependencies=len(dependencies),
                          vendor_bytes=sum(w['bytes'] for w in wheels) + runtime.stat().st_size,
                          report=str(OUT / 'review.json'))))


if __name__ == '__main__':
    main()
