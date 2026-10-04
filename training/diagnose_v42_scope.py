"""Post-result diagnostic only: coverage, constrained oracle and label conflict."""
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main(a):
    root=Path(a.root).resolve();run=Path(a.run).resolve();out=run/'scope_diagnosis';assert not out.exists()
    sys.path.insert(0,str(run/'frozen_training_runtime'))
    import v42_core as core
    from run_v42 import data,PROBS
    from run_v39_prepare import sha,save
    prep=root/'artifacts/v39_local_r2_20260913/prepared';rows,ids,texts,facts=data(prep)
    results=[];conflicts=[];reason_rows=[];bindings={}
    for fold in range(3):
        folder=run/('primary/fold_%s/R'%fold);bpath=prep.parent/('primary/fold_%s/SEMANTIC/evaluation.parquet'%fold)
        for p in [folder/'model.joblib',folder/'evaluation.parquet',bpath]:bindings[p.relative_to(root).as_posix()]=sha(p)
        d=pq.read_table(folder/'evaluation.parquet').to_pandas();b=pq.read_table(bpath).to_pandas();assert np.array_equal(d.row_position,b.row_position)
        bp=b[PROBS].to_numpy();p=d[PROBS].to_numpy();y=d.label_index.to_numpy();bc=bp.argmax(1)==y;eligible=d.context_eligible.to_numpy();asa=d.route.to_numpy()=='asa'
        delta=np.where(y==1,core.MAX_OFFSET,np.where(y==2,-core.MAX_OFFSET,0.))
        oracle=core.apply_offsets(bp,delta,eligible)
        # Labels are deliberately used ONLY to compute a hindsight diagnostic
        # upper bound. This oracle is never a predictor or a fitted candidate.
        possible=(~bc)&(oracle.argmax(1)==y)
        item={'fold':fold,'eligible_rows':int(eligible.sum()),'ASA_baseline_errors':int((asa&~bc).sum()),
            'eligible_baseline_errors':int((eligible&~bc).sum()),'eligible_errors_fixable_even_with_label_oracle_at_frozen_bound':int(possible.sum()),
            'changed_probability_rows':int(np.any(p!=bp,axis=1).sum()),'max_probability_change':float(np.abs(p-bp).max()),
            'baseline_wrong_margin_min_abs':float(np.abs(core.conditional_margin(bp)[eligible&~bc]).min()) if (eligible&~bc).any() else None}
        results.append(item)
        temp=d.loc[asa,['context_reason']].copy();temp['wrong']=~bc[asa]
        for key,sub in temp.groupby('context_reason'):reason_rows.append({'fold':fold,'reason':key,'rows':len(sub),'errors':int(sub.wrong.sum())})
        bundle=joblib.load(folder/'model.joblib');s=bundle['offset'].support;ctx,_=s.encode(texts,facts)
        fit=rows.fold.to_numpy()!=fold;fitctx=ctx[ids[fit]];fitrows=rows.loc[fit].reset_index(drop=True);fitrows['context_id']=fitctx
        for key,sub in d.loc[eligible&~bc].groupby('context_id'):
            tr=fitrows.loc[fitctx==key];ix=sub.index.to_numpy();first_pid=int(ids[~fit][ix[0]])
            conflicts.append({'fold':fold,'context_id':int(key),'text':texts[first_pid],'facts':facts[first_pid],
                'fit_class_counts':[int((tr.label_index==k).sum()) for k in range(3)],
                'fit_body_keys':int(tr.body_group.nunique()),'evaluation_wrong_class_counts':[int((sub.label_index==k).sum()) for k in range(3)],
                'evaluation_wrong_rows':len(sub),'evaluation_wrong_body_keys':int(sub.body_group.nunique()),
                'B_example_probabilities':bp[ix[0]].tolist(),'R_offset':float(bundle['offset'].delta[int(key)]),
                'has_opposite_pure_fit_label':bool(tr.label_index.nunique()==1 and not sub.label_index.isin(tr.label_index.unique()).all()),
                'scope':'Observed normalized context; not proof of erroneous original labels.'})
    out.mkdir();save(out/'diagnosis.json',{'training_executed':False,'oracle_is_posthoc_diagnostic_only':True,
        'folds':results,'ASA_reason_slices':reason_rows,'eligible_error_contexts':conflicts,'source_bindings':bindings,'script_sha256':sha(__file__),
        'scope':'Describes this reviewed evaluation; no tuning, new model, external data or blind generalization evidence.'})
    print(json.dumps({'folds':results,'eligible_error_contexts':len(conflicts),'pure_opposite_contexts':sum(v['has_opposite_pure_fit_label'] for v in conflicts)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True);main(p.parse_args())
