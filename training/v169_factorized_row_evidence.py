"""Complete original rows reconstructed from immutable reference plus full q/logq/mask."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def save_scope(folder,root,reference,q,logq,mask):
    root=Path(root).resolve();reference=Path(reference).resolve();reference.relative_to(root);folder=Path(folder)
    if not folder.exists():folder.mkdir(parents=True)
    rr=pd.read_parquet(reference)
    if q.shape!=logq.shape or q.shape!=(22546,3) or len(mask)!=len(rr) or np.asarray(mask).dtype!=np.bool_:raise ValueError('Complete local output and full original mask required')
    np.savez_compressed(folder/'complete_local_outputs_and_mask.npz',q=q,logq=logq,protected_correct=mask)
    spec=dict(reference=reference.relative_to(root).as_posix(),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),original_rows=len(rr),local_shape=[22546,3],payload_sha256=hashlib.sha256((folder/'complete_local_outputs_and_mask.npz').read_bytes()).hexdigest())
    (folder/'reference.json').write_bytes((json.dumps(spec,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    reconstructed=load_scope(folder,root)
    if not np.array_equal(reconstructed[['row_position','local','truth']].to_numpy(),rr[['row_position','local','truth']].to_numpy()):raise ValueError('Original row mapping changed')
    return spec

def load_scope(folder,root):
    folder=Path(folder);root=Path(root).resolve();spec=json.loads((folder/'reference.json').read_text(encoding='utf-8'));reference=(root/spec['reference']).resolve();reference.relative_to(root)
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=spec['reference_sha256'] or hashlib.sha256((folder/'complete_local_outputs_and_mask.npz').read_bytes()).hexdigest()!=spec['payload_sha256']:raise ValueError('Original reference/payload changed')
    rr=pd.read_parquet(reference)
    with np.load(folder/'complete_local_outputs_and_mask.npz',allow_pickle=False) as payload:q,lp,mask=[payload[k] for k in ['q','logq','protected_correct']]
    if len(rr)!=spec['original_rows'] or rr.row_position.duplicated().any() or q.shape!=lp.shape or q.shape!=(22546,3) or mask.shape!=(len(rr),) or mask.dtype!=np.bool_:raise ValueError('Full original population/shape mismatch')
    local=rr.local.to_numpy();truth=rr.truth.to_numpy()
    if not np.isfinite(q[local]).all() or not np.isfinite(lp[local]).all():raise ValueError('Missing actual original row outputs')
    rr=rr.copy();rr['protected_correct']=mask;rr['pred']=q[local].argmax(1)
    for cls in [0,1,2]:rr[f'p{cls}']=q[local,cls];rr[f'logp{cls}']=lp[local,cls]
    rr['stable_CE']=-lp[local,truth];rr['probability_clip_CE']=-np.log(np.maximum(q[local,truth],1e-300))
    return rr
