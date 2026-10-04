"""Independent support identities and selected cosine witnesses; no features."""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v135_runtime import load_data,fit_context
from v156_conditional_neighborhood import OUT,require,reference,save
from v156_neighborhood_geometry import TOLERANCE
from v148_observed_relation import FIELDS

def main():
    require();assert not (OUT/'verification.json').exists()
    audit=read(OUT/'audit.json');check_bindings(audit['source_sha256'])
    x,d,nodes=reference();actual=x.astype(np.float64);input_norm=np.sqrt(actual.multiply(actual).sum(1)).A1
    allrows=pd.read_parquet(OUT/'all_original_role_neighbors.parquet')
    prior_path=ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet'
    prior=pd.read_parquet(prior_path,columns=['row_position','known_578_cohort','same_family_and_outer_fold_control_S'])
    assert np.array_equal(prior.row_position,d.row_position)
    assert allrows.groupby(['training_role','space']).size().eq(112807).all()
    verdicts=[];cosine_witnesses=0;cosine_max_gap=0.;summaries=[];fields=[]
    trace=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    facts=pd.read_parquet(trace,columns=['facts_json']).facts_json.map(json.loads)
    fact_by_local={int(i):facts.iloc[int(g.index[0])] for i,g in d.groupby('local')}
    for f in range(3):
        _,c,_,_,_=fit_context(d,f)
        h=np.load(ROOT/f'artifacts/v141_representation_evidence_20261001/fold{f}_h1.npy').astype(np.float64).reshape(len(nodes),-1)
        hn=np.linalg.norm(h,axis=1)
        for space in ['actual_CSR_input','all16_H1','all16_V146_A_H2']:
            path=OUT/f'fold{f}_{space}_local_neighbors.parquet';g=pd.read_parquet(path)
            assert np.array_equal(g.local,nodes.local) and np.array_equal(g.root,nodes.root) and np.array_equal(g.condition,nodes.condition)
            for cl in [1,2]:
                totals=nodes.assign(mass=c[:,cl]).groupby('condition').mass.sum()
                own=nodes.assign(mass=c[:,cl]).groupby(['condition','root']).mass.sum()
                expected=g.condition.map(totals).to_numpy()-np.array([own.get((k,r),0) for k,r in zip(g.condition,g.root)])
                assert np.array_equal(g[f'c{cl}_available_original_rows'],expected)
                selected=g[f'c{cl}_local'].to_numpy(np.int64);present=expected>0;ids=np.flatnonzero(present);chosen=selected[present]
                assert np.all(selected[~present]==-1) and g.loc[~present,f'c{cl}_cosine'].isna().all()
                assert np.all(c[chosen,cl]>0) and np.all(nodes.root.to_numpy()[chosen]!=nodes.root.to_numpy()[ids])
                assert np.all(nodes.condition.to_numpy()[chosen]==nodes.condition.to_numpy()[ids])
                assert np.all(g.loc[present,f'c{cl}_tied_original_rows'].to_numpy()>=c[chosen,cl])
                assert np.all(g.loc[present,f'c{cl}_tied_original_rows'].to_numpy()<=expected[present])
                assert np.all(g.loc[present,f'c{cl}_tied_roots'].to_numpy()>0)
                if space!='all16_V146_A_H2':
                    for start in range(0,len(ids),512):
                        ii=ids[start:start+512];jj=chosen[start:start+512]
                        if space=='actual_CSR_input':
                            dot=np.asarray(actual[ii].multiply(actual[jj]).sum(1)).ravel();den=input_norm[ii]*input_norm[jj]
                        else:dot=np.einsum('ij,ij->i',h[ii],h[jj]);den=hn[ii]*hn[jj]
                        want=np.clip(dot/np.where(den==0,1,den),-1,1)
                        gap=float(np.abs(want-g.loc[ii,f'c{cl}_cosine'].to_numpy()).max(initial=0))
                        assert gap<=1e-10;cosine_max_gap=max(cosine_max_gap,gap);cosine_witnesses+=len(ii)
                verdicts.append(dict(fold=f,space=space,neighbor_class=cl,query_locals=len(g),present_locals=int(present.sum()),
                    complete_available_original_mass_verified=True,selected_legal_TRAIN_root_and_condition_verified=True,
                    selected_cosine_recomputed=space!='all16_V146_A_H2'))
            rows=allrows[allrows.training_role.eq(f)&allrows.space.eq(space)].reset_index(drop=True)
            assert np.array_equal(rows.row_position,d.row_position) and np.array_equal(rows.truth,d.truth)
            masks={'all_ASA':np.ones(len(d),bool),'hard578':prior.known_578_cohort.to_numpy(),
                'strict51':prior.same_family_and_outer_fold_control_S.to_numpy(),
                'V155_new_M16':rows.truth.eq(1)&rows.pred_baseline.eq(rows.truth)&rows.pred_V155_B.ne(rows.truth)}
            for cohort,mask in masks.items():
                for role in ['legal_TRAIN','outer_HELD']:
                    chosen_rows=rows[np.asarray(mask)&rows.query_role.eq(role)]
                    for (cl,relation),part in chosen_rows.groupby(['truth','relation'],sort=True):
                        same_roots=np.where(part.truth.eq(1),part.c1_tied_roots,part.c2_tied_roots)
                        summaries.append(dict(fold=f,space=space,cohort=cohort,role=role,truth=int(cl),relation=relation,
                            original_rows=len(part),baseline_errors=int(part.pred_baseline.ne(part.truth).sum()),
                            V155_B_errors=int(part.pred_V155_B.ne(part.truth).sum()),same_nearest_multiroot_rows=int((same_roots>=2).sum())))
                if cohort!='all_ASA':
                    part=rows[np.asarray(mask)&rows.query_role.eq('outer_HELD')].drop_duplicates('local')
                    for row in part.itertuples():
                        same=int(getattr(row,f'c{row.truth}_local'));other=int(getattr(row,f'c{3-row.truth}_local'))
                        for kind,j in [('same_class',same),('other_class',other)]:
                            different=None if j<0 else [k for k in FIELDS if fact_by_local[row.local].get(k)!=fact_by_local[j].get(k)]
                            fields.append(dict(fold=f,space=space,cohort=cohort,query_local=int(row.local),query_root=int(row.root),truth=int(row.truth),
                                neighbor_kind=kind,neighbor_local=j,neighbor_root=None if j<0 else int(nodes.root.iloc[j]),
                                differing_observed_fields=None if different is None else json.dumps(different),
                                whole_fine_behavior_identical=j>=0 and not different,relation=row.relation))
        del h;gc.collect()
    pd.DataFrame(summaries).to_parquet(OUT/'fixed_cohort_relation_summary.parquet',index=False)
    pd.DataFrame(fields).to_parquet(OUT/'selected_factual_witnesses.parquet',index=False)
    result=dict(status='all_root_condition_available_class_mass_and_selected_input_H1_cosines_verified',
        exact_original_role_identity=True,verified_query_local_class_cells=len(verdicts)*22546,
        selected_input_H1_cosine_witnesses=cosine_witnesses,selected_cosine_max_gap=cosine_max_gap,
        selected_cosine_tolerance=1e-10,H2_cosine_not_independently_recomputed=True,
        H2_scope='Producer real feature computation under prior seal; selected root/state/class support verified here, not new feature replay.',
        feature_function_calls=0,model_forward_calls=0,gradients=0,classifier_fits=0,updates=0,verdicts=verdicts,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),prior_path,OUT/'audit.json',OUT/'all_original_role_neighbors.parquet',OUT/'fixed_cohort_relation_summary.parquet',OUT/'selected_factual_witnesses.parquet']} )
    save(OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','verdicts']},ensure_ascii=False))

if __name__=='__main__':main()
