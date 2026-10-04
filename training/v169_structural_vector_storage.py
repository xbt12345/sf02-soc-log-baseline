"""Bit-exact full vector encoding with observed uniform signed off-support zeros."""
import hashlib
from pathlib import Path
import numpy as np

def digest(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()

def save_vector(path,vector,columns):
    v=np.asarray(vector);cols=np.asarray(columns,np.int64)
    if v.dtype!=np.float64 or v.shape not in [(1060832,),(1060833,)] or not np.isfinite(v).all() or cols.ndim!=1 or not cols.size or np.any(cols<0) or np.any(cols>=66287) or len(np.unique(cols))!=len(cols):raise ValueError('Full finite vector and actual observation support required')
    bits=np.ascontiguousarray(v).view(np.uint64);observed=bits[:1060592].reshape(66287,16);outside=np.ones(66287,bool);outside[cols]=False;off=observed[outside].ravel()
    default=int(off[0]);uniform=default in [0,1<<63] and bool(np.all(off==default))
    target=Path(path)
    if target.exists():raise FileExistsError(target)
    if not uniform:
        # Keep the actual full derivative/direction before refusing the assumed
        # structural storage bound. No sign or tiny coordinate is zeroed.
        with target.open('xb') as stream:np.savez_compressed(stream,width=np.array([len(v)],np.int64),dense_bits=bits,sha256=np.array([digest(v)]))
        raise RuntimeError('Off-support bits not a uniform signed zero; actual full vector preserved, resource qualification fails')
    ids=np.flatnonzero(bits!=np.uint64(default)).astype(np.int64)
    if len(ids)>len(cols)*16+(len(v)-1060592):raise AssertionError('Exact support encoding inconsistent')
    with target.open('xb') as stream:np.savez_compressed(stream,width=np.array([len(v)],np.int64),default_zero=np.array([default],np.uint64),indices=ids,values=bits[ids],sha256=np.array([digest(v)]))
    bound=len(ids)*16+8192
    if target.stat().st_size>bound:raise RuntimeError('Actual encoded vector exceeds declared serializer overhead; vector retained')
    return dict(complete_width=len(v),default_zero_bits=str(default),encoded_coordinates=len(ids),dense_sha256=digest(v),bytes=target.stat().st_size,max_bytes=bound)

def load_vector(path,expected_width):
    with np.load(path,allow_pickle=False) as archive:
        fields=set(archive.files);w=archive['width'];h=archive['sha256']
        if w.dtype!=np.int64 or w.shape!=(1,) or int(w[0])!=expected_width:raise ValueError('Complete width mismatch')
        if fields=={'width','dense_bits','sha256'}:
            bits=archive['dense_bits']
            if bits.dtype!=np.uint64 or bits.shape!=(expected_width,):raise ValueError('Dense retained failure vector malformed')
        elif fields=={'width','default_zero','indices','values','sha256'}:
            default,ids,values=archive['default_zero'],archive['indices'],archive['values']
            if default.dtype!=np.uint64 or default.shape!=(1,) or int(default[0]) not in [0,1<<63] or ids.dtype!=np.int64 or values.dtype!=np.uint64 or ids.ndim!=1 or values.shape!=ids.shape or (len(ids) and (ids[0]<0 or ids[-1]>=expected_width or np.any(ids[1:]<=ids[:-1]))) or np.any(values==default[0]):raise ValueError('Signed-zero complete vector encoding malformed')
            bits=np.full(expected_width,default[0],np.uint64);bits[ids]=values
        else:raise ValueError('Unknown complete vector representation')
        v=bits.view(np.float64)
        if h.shape!=(1,) or not np.isfinite(v).all() or digest(v)!=str(h[0]):raise ValueError('Complete raw-bit digest mismatch')
        return v.copy()

def reconstruction(recipe,root):
    """Exact float64 source recipe; original arrays and result digest bound."""
    kind=recipe['kind'];width=recipe['width']
    def source(path,expected):
        file=(Path(root)/path).resolve();file.relative_to(Path(root).resolve())
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('Recipe source file changed')
        return load_vector(file,width)
    if kind=='scaled':
        v=source(recipe['source'],recipe['source_sha256'])*float.fromhex(recipe['step_hex'])
    elif kind=='added':
        v=reconstruction(recipe['parent'],root)+source(recipe['correction'],recipe['correction_sha256'])
    elif kind=='stored':v=source(recipe['source'],recipe['source_sha256'])
    else:raise ValueError('Unknown lossless vector recipe')
    if v.shape!=(width,) or digest(v)!=recipe['dense_sha256']:raise ValueError('Reconstructed full displacement not bit-exact')
    return v
