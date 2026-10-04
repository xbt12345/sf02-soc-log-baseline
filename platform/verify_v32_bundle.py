"""Read-only bundle/environment check. Standard library; no package installation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir',default=str(Path(__file__).resolve().parent))
    args=parser.parse_args();root=Path(args.bundle_dir).resolve()
    manifest=json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))
    failed=[]
    for item in manifest['files']:
        path=(root/item['path']).resolve()
        if root not in path.parents or not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:failed.append(item['path'])
    packages={};missing=[]
    for package in ['numpy','scipy','scikit-learn','pyarrow','joblib']:
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None;missing.append(package)
    result={'bundle_files_verified':len(manifest['files']),'failed_files':failed,'packages':packages,
      'python':sys.version,'architecture':platform.machine(),'all_checks_passed':not failed and not missing and sys.version_info>=(3,8),
      'scope':'File identity and required package discovery only; not model quality, import compatibility or free-resource guarantee'}
    print(json.dumps(result,indent=2),flush=True)
    if not result['all_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
