"""Bit-exact same-point references with one bound chunk index array per function."""
import hashlib,json
from pathlib import Path
import numpy as np

def digest(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()

def save(folder,repetition,value,point_file,scope,ids):
    folder=Path(folder);point_file=Path(point_file);ids=np.asarray(ids,np.int64);index_file=folder/'chunk_local_ids.npy'
    if index_file.exists():
        if np.load(index_file).tobytes()!=ids.tobytes():raise ValueError('Measurement chunk identities changed')
    else:np.save(index_file,ids)
    with np.load(point_file,allow_pickle=False) as point:q,lp=point[f'{scope}_q'][ids],point[f'{scope}_logq'][ids]
    exact=q.tobytes()==value['q'].tobytes() and lp.tobytes()==value['logq'].tobytes()
    metadata=dict(exact_reference=exact,source_name=str(point_file.resolve()),source_sha256=hashlib.sha256(point_file.read_bytes()).hexdigest(),scope=scope,chunk_sha256=hashlib.sha256(index_file.read_bytes()).hexdigest(),q_sha256=digest(value['q']),logq_sha256=digest(value['logq']),margin_hex=float(value['margin']).hex())
    if not exact:np.savez_compressed(folder/f'repeat{repetition}_different_outputs.npz',q=value['q'],logq=value['logq'])
    encoded=(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    if len(encoded)>8192:raise RuntimeError('Measurement metadata exceeds registered overhead')
    (folder/f'repeat{repetition}_outputs.json').write_bytes(encoded)
    q2,lp2,margin=load(folder,repetition)
    if q2.tobytes()!=value['q'].tobytes() or lp2.tobytes()!=value['logq'].tobytes() or margin.hex()!=float(value['margin']).hex():raise RuntimeError('Exact measurement reconstruction failed')

def load(folder,repetition):
    folder=Path(folder);spec=json.loads((folder/f'repeat{repetition}_outputs.json').read_text(encoding='utf-8'));source=Path(spec['source_name']);index_file=folder/'chunk_local_ids.npy'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['source_sha256'] or hashlib.sha256(index_file.read_bytes()).hexdigest()!=spec['chunk_sha256']:raise ValueError('Actual point/chunk source changed')
    if spec['exact_reference']:
        ids=np.load(index_file)
        with np.load(source,allow_pickle=False) as point:q,lp=point[f"{spec['scope']}_q"][ids],point[f"{spec['scope']}_logq"][ids]
    else:
        with np.load(folder/f'repeat{repetition}_different_outputs.npz',allow_pickle=False) as data:q,lp=data['q'],data['logq']
    if digest(q)!=spec['q_sha256'] or digest(lp)!=spec['logq_sha256']:raise ValueError('Complete measurement output bit digest mismatch')
    return q.copy(),lp.copy(),float.fromhex(spec['margin_hex'])
