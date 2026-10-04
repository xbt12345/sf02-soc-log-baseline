"""Recompute numerical identity independently; local indices are not unique inputs."""
import hashlib
import numpy as np
import pandas as pd
from v135_runtime import load_data,fit_context
from v142_runtime import ROOT,OUT,save,sha


def audit():
    x,d=load_data();x.sum_duplicates();x.sort_indices()
    actual=[hashlib.sha256(x.indices[x.indptr[i]:x.indptr[i+1]].astype('<i8').tobytes()+
                          x.data[x.indptr[i]:x.indptr[i+1]].astype('<f4').tobytes()).hexdigest() for i in range(x.shape[0])]
    assert np.array_equal(d.canonical_key,d.local.map(dict(enumerate(actual))))
    details=[];roles=[]
    for f in range(3):
        frame,_,_,_,_=fit_context(d,f)
        def floor(col):
            counts=frame.groupby([col,'truth']).size().unstack(fill_value=0)
            return int((counts.sum(1)-counts.max(1)).sum())
        roles.append(dict(fold=f,local_index_floor=floor('local'),actual_numeric_identity_floor=floor('canonical_key')))
        for key,g in frame.groupby('canonical_key'):
            if g.local.nunique()>1 and g.truth.nunique()>1:
                ids=g.local.unique();a=x[int(ids[0])]
                assert all((a!=x[int(i)]).nnz==0 for i in ids)
                for row in g.itertuples(index=False):
                    details.append(dict(training_role=f,canonical_key=key,row_position=row.row_position,local=row.local,truth=row.truth))
    assert [r['actual_numeric_identity_floor'] for r in roles]==[22,6,28]
    return dict(input_rows=x.shape[0],all_actual_input_identities_recomputed=True,roles=roles,
                reason='Distinct local indices can encode byte-identical sparse numerical inputs; grouping by local understates cross-index label conflicts.',
                new_fits=0,new_updates=0,source_sha256=sha(__file__)),pd.DataFrame(details)


if __name__=='__main__':
    report,rows=audit();rows.to_parquet(OUT/'cross_local_actual_input_conflicts.parquet',index=False)
    save(OUT/'input_identity_audit.json',report);print(report)
