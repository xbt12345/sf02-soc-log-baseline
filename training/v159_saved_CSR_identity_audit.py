"""Bound saved CSR storage semantics; no classifier or model feature calls."""
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from experiment_review import ROOT,read,sha,check_bindings
import json

OUT=ROOT/'artifacts/v159_saved_CSR_identity_audit_20261002'
def main():
    assert not OUT.exists();OUT.mkdir();path=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    bindings={path.relative_to(ROOT).as_posix():sha(path),Path(__file__).resolve().relative_to(ROOT).as_posix():sha(__file__)}
    assert bindings[path.relative_to(ROOT).as_posix()]==read(ROOT/'artifacts/v158_fusion_trial_20261001/run_seal.json')['source_sha256'][path.relative_to(ROOT).as_posix()]
    (OUT/'pre_array_bindings.json').write_text(json.dumps(dict(source_sha256=bindings),indent=2)+'\n',encoding='utf-8')
    x=load_npz(path).tocsr();duplicate_entries=0;unsorted_rows=0
    for i in range(x.shape[0]):
        cols=x.indices[x.indptr[i]:x.indptr[i+1]];duplicate_entries+=len(cols)-len(np.unique(cols));unsorted_rows+=bool((np.diff(cols)<0).any())
    normalized=x.copy();normalized.sum_duplicates();normalized.eliminate_zeros();normalized.sort_indices()
    same_storage=np.array_equal(x.indptr,normalized.indptr) and np.array_equal(x.indices,normalized.indices) and np.array_equal(x.data,normalized.data)
    result=dict(status='current_CSR_exact_storage_and_semantic_identity_verified',shape=list(x.shape),dtype=str(x.dtype),nnz=x.nnz,has_canonical_format=bool(x.has_canonical_format),has_sorted_indices=bool(x.has_sorted_indices),duplicate_entries=int(duplicate_entries),explicit_zero_entries=int(np.count_nonzero(x.data==0)),unsorted_rows=int(unsorted_rows),sum_duplicates_eliminate_zeros_sort_preserves_exact_CSR_arrays=bool(same_storage),
        identity_convention='indices little-endian int64 plus unchanged values little-endian float32; exact bytes, not dense equivalence by rounding',official_classifier_calls=0,official_features_calls=0,official_gradients=0,official_fits=0,official_updates=0,source_sha256=bindings)
    assert x.shape==(22546,66287) and x.dtype==np.float32 and np.isfinite(x.data).all() and same_storage and not duplicate_entries
    check_bindings(bindings);(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
