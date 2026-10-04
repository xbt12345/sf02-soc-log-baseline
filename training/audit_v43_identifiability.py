"""Read-only observed-information and cross-fold support audit; no new fit."""
import argparse
import collections
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main(a):
    root=Path(a.root).resolve();out=root/'artifacts/v43_information_audit_20260914';assert not out.exists()
    frozen=root/'artifacts/v42_local_r1_20260914/frozen_training_runtime';sys.path.insert(0,str(frozen))
    import v39_core as core
    from run_v39_prepare import sha,save
    prep=root/'artifacts/v39_local_r2_20260913/prepared';receipt=json.loads((prep/'complete.json').read_text(encoding='utf-8'));bindings={}
    for name in ['rows.parquet','projections.parquet']:
        p=prep/name;assert sha(p)==receipt['files'][name];bindings[p.relative_to(root).as_posix()]=sha(p)
    rows=pq.read_table(prep/'rows.parquet').to_pandas();rows=rows[(rows.fold>=0)&(rows.route=='asa')].copy().reset_index(drop=True)
    assert len(rows)==106953 and set(rows.label_index)=={1,2}
    projected=pq.read_table(prep/'projections.parquet').to_pandas()
    active,ids=np.unique(rows.projection_id,return_inverse=True);projected=projected.iloc[active]
    assert (projected.text=='').all()
    facts=[]
    for value in projected.facts:
        f=json.loads(value)
        f={k:v for k,v in f.items() if k not in core.learning.BIT_FIELDS or v!=core.learning.BIT_FIELDS[k][1]}
        facts.append(f)
    views={
      'all_observed':facts,
      'without_source_port':[{k:v for k,v in f.items() if not k.startswith('src_port')} for f in facts],
      'without_both_ports':[{k:v for k,v in f.items() if not k.startswith(('src_port','dst_port'))} for f in facts],
      'availability_only':[{k:True for k in f} for f in facts]
    }
    b=pd.concat([pq.read_table(prep.parent/('primary/fold_%s/SEMANTIC/evaluation.parquet'%f)).to_pandas() for f in range(3)])
    b=b[b.route=='asa'].sort_values('row_position').reset_index(drop=True);assert np.array_equal(b.row_position,rows.row_position)
    for f in range(3):
        p=prep.parent/('primary/fold_%s/SEMANTIC/evaluation.parquet'%f);r=json.loads((p.parent/'complete.json').read_text(encoding='utf-8'))
        assert sha(p)==r['predictions_sha256'];bindings[p.relative_to(root).as_posix()]=sha(p)
    y=rows.label_index.to_numpy();bp=b[['p_benign','p_malicious','p_suspicious']].to_numpy();correct=bp.argmax(1)==y
    results={};support_records=[];conflict_records=[]
    for name,ff in views.items():
        keys=np.array([core.canonical(f) for f in ff],dtype=object)[ids]
        d=rows[['row_position','fold','body_group','label_index']].copy();d['key']=keys;d['B_correct']=correct
        pooled=pd.crosstab(d.key,d.label_index).reindex(columns=[1,2],fill_value=0)
        pooled_conflict=(pooled>0).all(axis=1)
        fold_floors=[];supports=[]
        for fold in range(3):
            fit=d[d.fold!=fold];ev=d[d.fold==fold];ct=pd.crosstab(ev.key,ev.label_index).reindex(columns=[1,2],fill_value=0)
            floor=int((ct.sum(axis=1)-ct.max(axis=1)).sum());fold_floors.append(floor)
            counts=pd.crosstab(fit.key,fit.label_index).reindex(columns=[1,2],fill_value=0)
            bodies=fit.groupby(['key','label_index']).body_group.nunique().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
            body_support=fit.groupby('key').body_group.nunique()
            matched=counts.reindex(ev.key,fill_value=0).to_numpy();mb=bodies.reindex(ev.key,fill_value=0).to_numpy()
            seen=matched.sum(1)>0;both=(matched>0).all(1);paired=(mb>=3).all(1)
            pure=seen&~both;majority=matched.argmax(1)+1;opp=pure&(majority!=ev.label_index.to_numpy())
            scope={}
            for label,mask in [('seen',seen),('both_labels_seen',both),('at_least_3_body_keys_per_label',paired),('pure_fit_opposite_evaluation',opp),('unseen',~seen)]:
                scope[label]={'rows':int(mask.sum()),'B_errors':int((mask&~ev.B_correct.to_numpy()).sum()),
                    'class_counts':[int((mask&(ev.label_index.to_numpy()==k)).sum()) for k in [1,2]]}
            supports.append({'fold':fold,'scopes':scope})
            if name=='all_observed':
                # A fixed fold classifier must assign a single class per key.
                pred=pd.DataFrame({'key':ev.key,'pred':bp[rows.fold.to_numpy()==fold].argmax(1)})
                assert pred.groupby('key').pred.nunique().max()==1
                for key,r in ct[(ct>0).all(axis=1)].iterrows():
                    sub=ev[ev.key==key]
                    conflict_records.append({'fold':fold,'key':key,'evaluation_class_counts':[int(r[1]),int(r[2])],
                        'evaluation_body_keys':int(sub.body_group.nunique()),'empirical_minimum_errors':int(r.min()),
                        'B_errors':int((~sub.B_correct).sum()),'fit_class_counts':counts.reindex([key],fill_value=0).to_numpy()[0].astype(int).tolist()})
                for i,r in enumerate(ev.itertuples()):
                    support_records.append({'row_position':int(r.row_position),'fold':fold,'both_labels_seen':bool(both[i]),
                        'three_bodies_per_label':bool(paired[i]),'pure_fit_opposite':bool(opp[i]),'unseen':bool(not seen[i])})
        results[name]={'unique_keys':len(pooled),'pooled_mixed_keys':int(pooled_conflict.sum()),
            'pooled_rows_in_mixed_keys':int(pooled.loc[pooled_conflict].sum().sum()),
            'single_common_classifier_empirical_minimum_errors':int((pooled.sum(axis=1)-pooled.max(axis=1)).sum()),
            'fold_specific_empirical_minimum_errors':fold_floors,'sum_fold_specific_empirical_minimum_errors':sum(fold_floors),
            'fit_only_support_by_fold':supports,
            'scope':'Label-aware descriptive lower bound for deterministic single-record classifiers on these exact evaluated feature keys. Not population Bayes error, not a prediction, not a Macro-F1 ceiling.'}
    assert results['without_source_port']['sum_fold_specific_empirical_minimum_errors']>=results['all_observed']['sum_fold_specific_empirical_minimum_errors']
    assert results['without_both_ports']['sum_fold_specific_empirical_minimum_errors']>=results['without_source_port']['sum_fold_specific_empirical_minimum_errors']
    # Exact grouped soft-target cross entropy equals original-row CE; it does
    # not introduce new label information or inverse-repeat weights.
    n=np.array([20.,4.]);p=np.array([.7,.3]);q=n/n.sum()
    assert abs(-(n*np.log(p)).sum()-(-n.sum()*(q*np.log(p)).sum()))<1e-12
    out.mkdir();shutil.copyfile(__file__,out/Path(__file__).name)
    conflict_records.sort(key=lambda v:v['empirical_minimum_errors'],reverse=True)
    pd.DataFrame(support_records).to_parquet(out/'fit_support_by_row.parquet',index=False)
    save(out/'identifiability.json',{'new_models_fit':0,'labels_modified':False,'ASA_rows':len(rows),'class_counts':[int((y==k).sum()) for k in [0,1,2]],
        'unique_original_projection_ids':len(active),'nonempty_semantic_texts':0,'B_errors':int((~correct).sum()),
        'views':results,'conflicting_all_observed_keys_by_fold':conflict_records,'source_bindings':bindings,'source_sha256':sha(__file__),
        'weighted_soft_targets_equivalent_to_original_rows':True,
        'evaluation_status':'Previously reviewed official development data; diagnostic labels used, not blind or prospective scoring.'})
    print(json.dumps({'B_errors':int((~correct).sum()),'views':{k:{x:v[x] for x in ['unique_keys','pooled_mixed_keys','pooled_rows_in_mixed_keys','fold_specific_empirical_minimum_errors','sum_fold_specific_empirical_minimum_errors']} for k,v in results.items()}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args())
