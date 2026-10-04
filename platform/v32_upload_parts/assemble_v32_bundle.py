"""Verify upload parts and extract the bound v3.2 bundle; no training here."""
import hashlib,json
from pathlib import Path
import zipfile
PARTS=[{'name': 'sf02_v32_round.zip.part01', 'bytes': 83886080, 'sha256': '1a055a72a445bccb796017974c49b23364c68894d00543333e8ba724c3dc6247'}, {'name': 'sf02_v32_round.zip.part02', 'bytes': 76588586, 'sha256': '86c87296bdb5628a4f8f03f0a98ece154eb4cdd135c2e850672c29b94ffcb5dc'}]
ARCHIVE_SHA='3d8accb9f50dd33c3a0be9374d680662327505c15561796e0882509b9c02f8e8'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
root=Path(__file__).resolve().parent
for part in PARTS:
 path=root/part['name']
 if not path.is_file() or path.stat().st_size!=part['bytes'] or sha(path)!=part['sha256']:raise SystemExit('Missing or damaged part: '+part['name'])
archive=root/'sf02_v32_round.zip'
if archive.exists():
 if sha(archive)!=ARCHIVE_SHA:raise SystemExit('Existing ZIP differs; refusing overwrite')
else:
 with archive.open('xb') as out:
  for part in PARTS:
   with (root/part['name']).open('rb') as src:
    for b in iter(lambda:src.read(8388608),b''):out.write(b)
 if sha(archive)!=ARCHIVE_SHA:raise SystemExit('ZIP hash mismatch')
dest=root/'sf02_v32_round'
if dest.exists():raise SystemExit('Verified ZIP available; extraction directory already exists. No files overwritten.')
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  p=(dest/name).resolve()
  if dest.resolve() not in p.parents:raise SystemExit('Unsafe archive path')
 dest.mkdir()
 z.extractall(dest)
print(json.dumps({'archive_sha256':ARCHIVE_SHA,'extracted_to':str(dest),'model_trained':False}))
