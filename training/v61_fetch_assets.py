"""Download public, pinned assets with checked HTTP ranges and full SHA256."""
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from v61_common import MODEL_ID, REVISION, save, sha

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/v61_asset_cache'


def fetch(url, out, size, digest):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and out.stat().st_size==size and sha(out)==digest:
        print('Verified existing '+out.name,flush=True);return
    chunks=out.parent/(out.name+'.chunks');chunks.mkdir(exist_ok=True)
    step=16*1024*1024
    def part(start):
        end=min(start+step,size)-1;p=chunks/str(start)
        if p.exists() and p.stat().st_size==end-start+1:return
        for attempt in range(5):
            try:
                sep='&' if '?' in url else '?'
                req=urllib.request.Request(url+sep+'v61range='+str(start),headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=90) as r:
                    assert r.status==206,(r.status,start)
                    assert r.headers['Content-Range']==f'bytes {start}-{end}/{size}',r.headers['Content-Range']
                    data=r.read(end-start+2);assert len(data)==end-start+1
                p.write_bytes(data);return
            except Exception as e:
                if attempt==4:raise
                print(f'Retry {out.name} offset {start}: {type(e).__name__}',flush=True)
                time.sleep(2*(attempt+1))
    starts=list(range(0,size,step))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for n,_ in enumerate(pool.map(part,starts),1):
            if n%8==0 or n==len(starts):print(f'{out.name}: {n}/{len(starts)} chunks',flush=True)
    temp=out.with_suffix(out.suffix+'.partial')
    with temp.open('wb') as f:
        for start in starts:f.write((chunks/str(start)).read_bytes())
    assert temp.stat().st_size==size and sha(temp)==digest,'Full hash mismatch; never install this file'
    os.replace(temp,out)
    # Only this downloader's verified chunk files, never a user directory.
    for start in starts:(chunks/str(start)).unlink()
    chunks.rmdir()
    print('Verified SHA256 '+out.name,flush=True)


def torch_asset():
    name='torch-2.7.1+cu128-cp312-cp312-win_amd64.whl'
    fetch('https://download.pytorch.org/whl/cu128/'+name.replace('+','%2B'),CACHE/name,
          3273024349,'2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30')


def model_asset():
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true',timeout=60) as r:
        meta=json.load(r)
    assert meta['sha']==REVISION
    dest=CACHE/'securebert2';dest.mkdir(parents=True,exist_ok=True)
    save(dest/'upstream_metadata.json',meta)
    names=['config.json','model.safetensors','special_tokens_map.json','tokenizer.json','tokenizer_config.json']
    receipt={}
    for name in names:
        entry=next(s for s in meta['siblings'] if s['rfilename']==name)
        url=f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
        if 'lfs' in entry:
            fetch(url,dest/name,entry['size'],entry['lfs']['sha256'])
        else:
            with urllib.request.urlopen(url,timeout=60) as r:data=r.read()
            git=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert git==entry['blobId'],(name,git,entry['blobId'])
            (dest/name).write_bytes(data)
        receipt[name]=sha(dest/name)
    save(dest/'download_receipt.json',{'model':MODEL_ID,'revision':REVISION,'sha256':receipt})


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(torch_asset),pool.submit(model_asset)]
        for f in futures:f.result()
