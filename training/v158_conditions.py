"""All existing facts plus fixed signed projection of all original text CSR columns."""
import numpy as np
from scipy import sparse

def conditions(x):
    if x.shape[1]!=66287:raise ValueError('Registered input coordinates changed')
    ids=np.arange(65792,dtype=np.uint64)+np.uint64(15801)
    # SplitMix64 hash, fixed before labels; no fitted teacher coordinates.
    with np.errstate(over='ignore'):
        z=ids+np.uint64(0x9e3779b97f4a7c15);z=(z^(z>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        z=(z^(z>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);z=z^(z>>np.uint64(31))
    bucket=(z%np.uint64(32)).astype(np.int64);sign=np.where(z>>np.uint64(63),-1.,1.)
    projection=sparse.csr_matrix((sign,(np.arange(65792),bucket)),shape=(65792,32))
    text=(x[:,:65792].astype(np.float64)@projection).toarray()
    norms=np.sqrt(x[:,:65792].astype(np.float64).multiply(x[:,:65792]).sum(1)).A1
    text/=np.where(norms>0,norms,1)[:,None]
    result=np.concatenate([text,x[:,65792:].toarray().astype(np.float64)],1)
    assert result.shape==(len(x.indptr)-1,527) and np.isfinite(result).all()
    return result
