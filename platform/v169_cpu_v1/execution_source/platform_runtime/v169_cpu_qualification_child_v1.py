"""Isolated qualifier execution followed by actual loaded file identity capture."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

PROJECT=Path(__file__).resolve().parents[1]
RUNTIME=PROJECT/'platform_runtime'
ALLOWED={'v169_cpu_synthetic_qualification_v1.py','v169_cpu_full_width_qualification_v1.py'}


def main():
    path=Path(sys.argv[1]).resolve()
    if path.parent!=RUNTIME.resolve() or path.name not in ALLOWED:
        raise RuntimeError('Only the two registered qualification entries may run here')
    sys.path[:0]=[str(RUNTIME),str(PROJECT/'training')]
    sys.argv=sys.argv[1:]
    runpy.run_path(str(path),run_name='__main__')
    native={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)!=6 or not parts[5].startswith('/'):continue
        if parts[5].endswith(' (deleted)'):raise RuntimeError('Deleted mapped qualification file')
        physical=Path(parts[5]).resolve()
        if not physical.is_file():continue
        h=hashlib.sha256()
        with physical.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        native[str(physical)]=h.hexdigest()
    output=PROJECT/'platform_run_v1'/(path.stem+'_'+'_'.join(sys.argv[1:])+'_runtime.json')
    with output.open('x',encoding='utf-8') as stream:json.dump(native,stream,indent=2)


if __name__=='__main__':main()
