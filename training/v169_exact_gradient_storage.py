"""Lossless float64 complete vector storage, preserving negative zero/subnormals."""
import hashlib
from pathlib import Path
import numpy as np

def digest(vector):return hashlib.sha256(np.asarray(vector).tobytes()).hexdigest()

def save_vector(path,vector):
    v=np.asarray(vector)
    if v.dtype!=np.float64 or v.ndim!=1 or not v.size or not np.isfinite(v).all():raise ValueError('Complete finite native float64 vector required')
    bits=np.ascontiguousarray(v).view(np.uint64);ids=np.flatnonzero(bits!=0).astype(np.int64)
    target=Path(path)
    if target.exists():raise FileExistsError(target)
    # Index selection tests raw bits; -0 and all subnormals are retained.
    with target.open('xb') as stream:np.savez_compressed(stream,width=np.array([len(v)],np.int64),indices=ids,values=bits[ids],sha256=np.array([digest(v)]))
    return dict(width=len(v),nonzero_bit_patterns=len(ids),bytes=target.stat().st_size,dense_sha256=digest(v))

def load_vector(path,expected_width):
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(['width','indices','values','sha256']):raise ValueError('Unexpected lossless vector fields')
        w,ids,values,h=[archive[k] for k in ['width','indices','values','sha256']]
        if w.dtype!=np.int64 or w.shape!=(1,) or int(w[0])!=expected_width or ids.dtype!=np.int64 or values.dtype!=np.uint64 or ids.ndim!=1 or values.shape!=ids.shape or (len(ids) and (ids[0]<0 or ids[-1]>=expected_width or np.any(ids[1:]<=ids[:-1]))) or np.any(values==0):raise ValueError('Malformed complete sparse vector')
        bits=np.zeros(expected_width,np.uint64);bits[ids]=values;v=bits.view(np.float64)
        if not np.isfinite(v).all() or h.shape!=(1,) or digest(v)!=str(h[0]):raise ValueError('Complete vector digest/finite check failed')
        return v
