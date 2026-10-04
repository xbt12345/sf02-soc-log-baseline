"""Single-file v3.6 official-data development run. No install or external data."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import zipfile

ARCHIVE_SHA = '__ARCHIVE_SHA__'
PAYLOAD = '__PAYLOAD__'


def unpack(destination):
    content=base64.b64decode(PAYLOAD)
    if hashlib.sha256(content).hexdigest()!=ARCHIVE_SHA:raise ValueError('Embedded upload is incomplete or damaged')
    archive=zipfile.ZipFile(io.BytesIO(content))
    manifest=json.loads(archive.read('manifest.json').decode('utf-8'))
    expected=set(manifest)|{'manifest.json'}
    if set(archive.namelist())!=expected or len(archive.namelist())!=len(expected):raise ValueError('Unexpected archive entries')
    destination.mkdir(parents=True,exist_ok=True)
    for name in archive.namelist():
        p=PurePosixPath(name)
        if p.is_absolute() or len(p.parts)!=1 or name in ('.','..') or '\\' in name:raise ValueError('Unsafe archive path')
        data=archive.read(name)
        if name!='manifest.json' and hashlib.sha256(data).hexdigest()!=manifest[name]:raise ValueError('Source entry damaged')
        target=destination/name
        if target.exists():
            if target.read_bytes()!=data:raise ValueError('Existing runtime differs: '+str(target))
        else:
            with target.open('xb') as f:f.write(data)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=str(Path(__file__).resolve().parent/'sf02_data'/'train.parquet'))
    p.add_argument('--work-root',default=str(Path(__file__).resolve().parent))
    p.add_argument('--previous');p.add_argument('--threads',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    root=Path(args.work_root).resolve()/('sf02_v36_'+ARCHIVE_SHA[:12])
    manifest=unpack(root)
    print(json.dumps({'bundle_files_verified':len(manifest),'runtime':str(root),'model_trained':False}),flush=True)
    if not args.verify_only:
        command=[sys.executable,'-u',str(root/'v36_pipeline.py'),'--train',str(Path(args.data).resolve()),
                 '--threads',str(args.threads),'--workers',str(args.workers)]
        if args.previous:command+=['--previous',args.previous]
        env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
        raise SystemExit(subprocess.call(command,env=env))
