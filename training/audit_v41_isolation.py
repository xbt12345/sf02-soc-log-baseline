"""All active matrices, all folds, before fitting any v41 classifier."""
import argparse
import gc
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse


def main(a):
    root=Path(a.root).resolve();run=Path(a.run).resolve();sys.path.insert(0,str(run/'frozen_training_runtime'))
    import v41_core as c
    from run_v39_prepare import save,sha
    stage=run/'prepared';receipt=json.loads((stage/'complete.json').read_text(encoding='utf-8'))
    for n,h in receipt['runtime_sources'].items():assert sha(run/'frozen_training_runtime'/n)==h
    assert not (run/'primary').exists(), 'This stage must precede all v41 fits'
    prep=Path(receipt['v39_prepared']);rows=pq.read_table(prep/'rows.parquet').to_pandas();rows=rows[rows.fold>=0].reset_index(drop=True)
    active,ids=np.unique(rows.projection_id.to_numpy(),return_inverse=True);pr=pq.read_table(prep/'projections.parquet').to_pandas().iloc[active]
    facts=[json.loads(v) for v in pr.facts];texts=pr.text.tolist();results=[]
    for fold in range(3):
        fit=rows.fold.to_numpy()!=fold;use=np.unique(ids[fit]);b=joblib.load(prep.parent/('primary/fold_%s/SEMANTIC/model.joblib'%fold))
        tx=b['text_encoder'].transform(texts);fx=b['fact_encoder'].transform(facts);xb=sparse.hstack([tx,fx],format='csr')
        pe=c.ParameterEffects().fit(facts,ids[fit],rows.body_group.to_numpy()[fit],3)
        i=joblib.load(root/('artifacts/v40_local_r1_20260913/primary/fold_%s/I/model.joblib'%fold))
        assert pe.term_names==i['parameter_encoder'].term_names and pe.support_lower_bounds==i['parameter_encoder'].support_lower_bounds
        del i
        comparisons={}
        for view in ['P','F','D']:
            if view=='P':x=sparse.hstack([xb,pe.transform(facts)],format='csr')
            else:
                fe=(c.FiniteOnlyFacts() if view=='F' else c.DeduplicatedFacts()).fit([facts[j] for j in use])
                x=sparse.hstack([tx,fe.transform(facts)],format='csr')
            comparisons[view]=c.isolation_check(view,xb,x,b['fact_encoder'].names(),tx.shape[1]);del x
        n=joblib.load(root/('artifacts/v40_local_r1_20260913/primary/fold_%s/N/model.joblib'%fold))
        dedup=[c.deduplicate(f) for f in facts];combined=c.FiniteOnlyFacts().fit([dedup[j] for j in use]).transform(dedup)
        difference=combined-n['fact_encoder'].transform(facts)
        assert difference.nnz==0
        results.append({'fold':fold,'comparisons':comparisons,'combined_F_D_matches_frozen_N_matrix':True,'parameter_support_matches_frozen_I':True})
        print(json.dumps(results[-1]),flush=True);del b,n,tx,fx,xb,combined,difference;gc.collect()
    updates=pq.read_table(stage/'native_updates.parquet').to_pandas();projections=pq.read_table(prep/'projections.parquet',columns=['text','facts']).to_pandas()
    before=[json.loads(v) for v in projections.facts.iloc[updates.old_projection_id.to_numpy()]]
    preexisting=sum('status' in f and 'substatus' in f for f in before)
    assert preexisting==357 and len(updates)-preexisting==21
    assert set(updates.row_position)<=set(rows.loc[rows.route=='windows_message','row_position'])
    save(run/'isolation_checks.json',{'all_checks_passed':True,'folds':results,'all_active_projections':len(active),'no_v41_fit_preceded_this_check':True,
        'native_preexisting_status_rows':preexisting,'native_new_status_rows':len(updates)-preexisting,
        'prepared_sha256':sha(stage/'complete.json'),'script_sha256':sha(__file__)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True);main(p.parse_args())
