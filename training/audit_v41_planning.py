"""No fitting: inspect alias geometry, minority support and v40 regressions."""
import argparse
import gc
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pyarrow.parquet as pq


def main(a):
    root=Path(a.root).resolve(); out=Path(a.out).resolve(); assert not out.exists()
    run=root/'artifacts/v40_local_r1_20260913'
    sys.path.insert(0,str(run/'frozen_training_runtime'))
    import v40_core as core
    from run_v39_prepare import sha,save
    receipt=json.loads((root/'evidence/2026-09-13/v40_execution/delivery.json').read_text(encoding='utf-8'))
    # Check factual source artifacts; current entry docs will receive a plan pointer later.
    bindings={}
    for name,h in receipt['files'].items():
        if name.startswith('artifacts/') or name.endswith('V40_EXECUTION_REVIEW.md'):
            assert sha(root/name)==h,name; bindings[name]=h
    prep=root/'artifacts/v39_local_r2_20260913/prepared'
    inherited=json.loads((prep/'complete.json').read_text(encoding='utf-8'))
    for name in ['rows.parquet','projections.parquet']:
        assert sha(prep/name)==inherited['files'][name]
        bindings[(prep/name).relative_to(root).as_posix()]=sha(prep/name)
    rows=pq.read_table(prep/'rows.parquet').to_pandas()
    rows=rows[rows.fold>=0].reset_index(drop=True)
    active,ids=np.unique(rows.projection_id.to_numpy(),return_inverse=True)
    pr=pq.read_table(prep/'projections.parquet').to_pandas().iloc[active].reset_index(drop=True)
    facts=[json.loads(v) for v in pr.facts]; texts=pr.text.tolist()
    aliases=[('outcome=failure','auth_result=failure'),('outcome=success','auth_result=success'),
             ('response=missing','authentication_interaction=no_response')]
    folds=[]; tails=[]
    for fold in range(3):
        fit=rows.fold.to_numpy()!=fold; ev=~fit
        folder=prep.parent/('primary/fold_%s/SEMANTIC'%fold)
        old_receipt=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        assert sha(folder/'model.joblib')==old_receipt['model_sha256']
        bindings[(folder/'model.joblib').relative_to(root).as_posix()]=old_receipt['model_sha256']
        b=joblib.load(folder/'model.joblib')
        fx=b['fact_encoder'].transform(facts).toarray(); names=list(b['fact_encoder'].names())
        tcols=len(b['text_encoder'].names()); w=b['model'].coef_[:,tcols:]
        reparam_delta=np.zeros((len(pr),3)); old_penalty=0.; new_penalty=0.; alias_checks=[]
        for n1,n2 in aliases:
            j,k=names.index(n1),names.index(n2); mismatch=fx[:,j]!=fx[:,k]
            info={'pair':[n1,n2],'fit_mismatched_rows':int(mismatch[ids[fit]].sum()),
                  'evaluation_mismatched_rows':int(mismatch[ids[ev]].sum()),
                  'fit_active_rows':int((fx[ids[fit],j]!=0).sum()),
                  'fit_class_support':np.bincount(rows.label_index.to_numpy()[fit&(fx[ids,j]!=0)],minlength=3).tolist(),
                  'coefficient_max_difference':float(np.abs(w[:,j]-w[:,k]).max())}
            if not mismatch.any():
                original=fx[:,j,None]*w[None,:,j]+fx[:,k,None]*w[None,:,k]
                merged=(np.sqrt(2)*fx[:,j,None])*((w[:,j]+w[:,k])/np.sqrt(2))[None,:]
                reparam_delta+=merged-original
                old_penalty+=float(np.square(w[:,[j,k]]).sum())
                new_penalty+=float(np.square((w[:,j]+w[:,k])/np.sqrt(2)).sum())
                info['scaled_compaction_exact_on_all_development_projections']=True
            else:
                info['scaled_compaction_exact_on_all_development_projections']=False
                info['mismatch_routes']=pr.loc[mismatch,'route_audit_only'].value_counts().to_dict()
            alias_checks.append(info)
        fold_result={'fold':fold,'aliases':alias_checks,'eligible_compaction_logit_max_difference':float(np.abs(reparam_delta).max()),
            'eligible_old_coefficient_L2_square':old_penalty,'eligible_compacted_coefficient_L2_square':new_penalty}
        folds.append(fold_result)
        ie=pq.read_table(run/('primary/fold_%s/I/evaluation.parquet'%fold)).to_pandas()
        be=pq.read_table(folder/'evaluation.parquet').to_pandas()
        assert np.array_equal(ie.row_position,be.row_position)
        bp=be[['p_benign','p_malicious','p_suspicious']].to_numpy(); ip=ie[['p_benign','p_malicious','p_suspicious']].to_numpy()
        y=ie.label_index.to_numpy(); reg=(bp.argmax(1)==y)&(ip.argmax(1)!=y)
        target=reg&((ie.route=='authentication')|(ie.route=='windows_message')|((ie.route=='asa')&ie.unseen_parameter))
        for item in ie.loc[target].itertuples():
            local=int(np.searchsorted(active,item.projection_id)); same=fit&(ids==local)
            evidence={'row_position':int(item.row_position),'fold':fold,'route':item.route,'projection_id':int(item.projection_id),
                'label_index':int(item.label_index),'facts':facts[local],'text':texts[local],
                'fit_same_projection_class_rows':np.bincount(rows.label_index.to_numpy()[same],minlength=3).tolist(),
                'fit_same_projection_class_body_keys':[int(rows.loc[same&(rows.label_index.to_numpy()==c),'body_group'].nunique()) for c in range(3)],
                'B_probability':bp[item.Index].tolist(),'I_probability':ip[item.Index].tolist(),
                'baseline_partition':int(item.baseline_partition),'unseen_parameter':bool(item.unseen_parameter),
                'parameter_terms':core.terms(facts[local])}
            tails.append(evidence)
        del b,fx,ie,be,bp,ip;gc.collect()
        print(json.dumps(fold_result),flush=True)
    ports=np.arange(65536,dtype=np.int64)
    flips=np.array([int(int(v^(v+1)).bit_count()) for v in ports])
    grouped={}
    for item in tails:
        key=item['route']+'|'+str(item['projection_id'])
        g=grouped.setdefault(key,{'route':item['route'],'projection_id':item['projection_id'],'rows':0,'facts':item['facts'],'text':item['text']})
        g['rows']+=1
    out.mkdir(parents=True)
    save(out/'diagnosis.json',{'new_soc_fits':0,'allowed_rows':len(rows),'active_projections':len(active),
        'alias_geometry_by_fold':folds,'targeted_regressions':tails,'targeted_projection_summary':list(grouped.values()),
        'port_recoding':{'valid_ports':65536,'observed_ports_changed':int((flips>0).sum()),'changed_multiple_bits':int((flips>1).sum()),'maximum_bits_changed':int(flips.max())},
        'source_bindings':bindings,'script_sha256':sha(__file__),
        'scope':'Frozen model and fit-side support inspection. Projection identity is stricter than some actual input equality; no claim of incident independence or isolated training causality.'})
    print(json.dumps({'new_soc_fits':0,'targeted_regression_rows':len(tails),'distinct_target_projections':len(grouped)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);main(p.parse_args())
