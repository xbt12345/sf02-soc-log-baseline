"""Reuse same-point outputs only after exact byte equality, retain differing values."""
import hashlib,json
from pathlib import Path
import numpy as np

def digest(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()

def save(folder,repetition,value,point_file,scope,ids):
    folder=Path(folder);point_file=Path(point_file);ids=np.asarray(ids,np.int64)
    with np.load(point_file,allow_pickle=False) as point:q,lp=point[f'{scope}_q'][ids],point[f'{scope}_logq'][ids]
    exact=q.tobytes()==value['q'].tobytes() and lp.tobytes()==value['logq'].tobytes()
    metadata=dict(exact_reference=exact,source_name=str(point_file.resolve()),source_sha256=hashlib.sha256(point_file.read_bytes()).hexdigest(),scope=scope,ids=ids.tolist(),q_sha256=digest(value['q']),logq_sha256=digest(value['logq']),margin_hex=float(value['margin']).hex())
    if not exact:np.savez_compressed(folder/f'repeat{repetition}_different_outputs.npz',q=value['q'],logq=value['logq'])
    (folder/f'repeat{repetition}_outputs.json').write_bytes((json.dumps(metadata,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    q2,lp2,margin=load(folder,repetition)
    if q2.tobytes()!=value['q'].tobytes() or lp2.tobytes()!=value['logq'].tobytes() or margin.hex()!=float(value['margin']).hex():raise RuntimeError('Exact measurement reconstruction failed')

def load(folder,repetition):
    folder=Path(folder);spec=json.loads((folder/f'repeat{repetition}_outputs.json').read_text(encoding='utf-8'));source=Path(spec['source_name'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['source_sha256']:raise ValueError('Actual point output source changed')
    if spec['exact_reference']:
        with np.load(source,allow_pickle=False) as point:q,lp=point[f"{spec['scope']}_q"][spec['ids']],point[f"{spec['scope']}_logq"][spec['ids']]
    else:
        with np.load(folder/f'repeat{repetition}_different_outputs.npz',allow_pickle=False) as data:q,lp=data['q'],data['logq']
    if digest(q)!=spec['q_sha256'] or digest(lp)!=spec['logq_sha256']:raise ValueError('Complete measurement output bit digest mismatch')
    return q.copy(),lp.copy(),float.fromhex(spec['margin_hex'])
