"""Fetch exactly one official portable runtime; no install or execution."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'platform/v169_cpu_v1/vendor'
OUT.mkdir(parents=True, exist_ok=True)
NAME = 'cpython-3.12.14+20260924-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz'
URL = 'https://github.com/astral-sh/python-build-standalone/releases/download/20260924/cpython-3.12.14%2B20260924-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz'
EXPECTED = 'c8499b61252c433280f134df954464d19811527b31cb920c35fc6967c1222e35'
target = OUT / NAME
assert not target.exists() and not (OUT / 'runtime_download.json').exists()
request = urllib.request.Request(URL, headers={'User-Agent': 'SF02-V169-platform-package'})
digest = hashlib.sha256()
with urllib.request.urlopen(request, timeout=60) as response, target.open('xb') as handle:
    while block := response.read(1024 * 1024):
        digest.update(block)
        handle.write(block)
assert digest.hexdigest() == EXPECTED
receipt = dict(status='official_aarch64_CPython_runtime_downloaded_and_digest_verified', url=URL, path=target.relative_to(ROOT).as_posix(), bytes=target.stat().st_size, sha256=digest.hexdigest(), expected_official_GitHub_asset_digest=EXPECTED, not_executed_on_windows=True, official_calls=0)
(OUT / 'runtime_download.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt), flush=True)
