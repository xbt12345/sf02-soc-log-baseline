"""Actual source-role/cost preparation and old saved-bank audit; zero model calls."""
import argparse,json,sys,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from experiment_review import ROOT,read,sha,check_bindings
from v131_common import load_data,INPUT,TRACE,OFFICIAL
from v158_fusion_contract import STAGES,inner_fold,role,counts,validate_stage_roots,pipeline_cost

OUT=ROOT/'artifacts/v158_nested_fusion_qualification_20261001'
PLAN=ROOT/'training/review_policy/v158_nested_fusion_qualification_plan.json'
OLD=ROOT/'artifacts/v128_nested_score_trial_20260929_r3'
OLD_ROLES=ROOT/'artifacts/v128_mechanism_review_20260929/nested_score_roles.parquet'
ORIGINAL=ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def validate(p):
    expected=dict(version='V158-nested-fusion-qualification-only',official_classifier_forward_calls=0,
        official_features_calls=0,new_fits=0,new_gradients=0,new_updates=0,
        outer_folds=[0,1,2],inner_folds=[0,1,2],split='unchanged_V128_root_hash_no_labels_no_reroll',
        all_supervised_base_stages=list(STAGES),planned_new_base_pipelines=9,
        planned_new_base_stage_fits=45,planned_new_fusion_fits=6,planned_new_formal_fits_total=51,
        current_V146_seen_TRAIN_arrays_are_not_OOF=True,old_V128_bank_is_not_same_current_pipeline=True,
        official_training_activation=False,automatic_reroll_or_fit=False)
    for k,v in expected.items():
        if p.get(k)!=v:raise ValueError('Unregistered qualification: '+k)
    check_bindings(p['source_sha256']);return p

def negative_cases(p):
    cases=[]
    for k,v in [('official_classifier_forward_calls',9),('planned_new_base_stage_fits',9),
        ('planned_new_formal_fits_total',15),('current_V146_seen_TRAIN_arrays_are_not_OOF',False),
        ('old_V128_bank_is_not_same_current_pipeline',False),('official_training_activation',True),('automatic_reroll_or_fit',True)]:
        trial=dict(p);trial[k]=v
        try:validate(trial)
        except ValueError:cases.append(dict(field=k,rejected=True))
        else:raise AssertionError('Unregistered qualification accepted')
    return cases

def main():
    p=validate(read(PLAN));assert not OUT.exists();OUT.mkdir()
    # Binding is prospective for metadata/saved-array work; no official model import or call.
    paths={PLAN,Path(__file__).resolve()}|{ROOT/k for k in p['source_sha256']}
    for f in range(3):
        for j in range(3):paths.update(OLD/f'teacher{f}_{j}'/n for n in ['fit.json','epoch25_model.pt','canonical_member_logits.npy'])
    paths.update([OLD/'run_seal.json',OLD_ROLES,INPUT,TRACE,OFFICIAL,ORIGINAL])
    save(OUT/'pre_execution_bindings.json',dict(status='bound_before_saved_output_and_role_audit',
        source_sha256={q.relative_to(ROOT).as_posix():sha(q) for q in sorted(paths)},official_model_calls=0,new_fits=0))
    x,d=load_data();d=d.reset_index(drop=True)
    facts=pd.read_parquet(TRACE,columns=['row_position','facts_json']);assert np.array_equal(facts.row_position,d.row_position)
    typed=facts.facts_json.map(json.loads);fine=facts.facts_json.map(lambda v:json.dumps(json.loads(v),sort_keys=True,separators=(',',':')))
    proto=typed.map(lambda z:z.get('transport_protocol','absent'))
    original=sparse.load_npz(ORIGINAL).tocsr();assert original.shape==x.shape
    different=(original.astype(np.float64)-x.astype(np.float64)).tocsr();different.eliminate_zeros()
    changed=np.flatnonzero(np.diff(different.indptr)>0)
    old_roles=pd.read_parquet(OLD_ROLES)
    role_rows=[];stages=[];support=[];reuse=[];saved_profiles=[];ooof=[]
    for f in range(3):
        outer_legal=d[d.fold.ne(f)].copy();outer_legal['outer_fold']=f;outer_legal['inner_fold']=outer_legal.root.map(lambda z:inner_fold(f,z))
        outer_legal['crossfit_teacher']=outer_legal.inner_fold;outer_legal['matched_infit_teacher']=(outer_legal.inner_fold+1)%3
        old=old_roles[old_roles.outer_fold.eq(f)].reset_index(drop=True)
        for col in ['row_position','local','root','truth','fold','inner_fold']:
            assert np.array_equal(outer_legal[col].to_numpy(),old[col].to_numpy()),col
        role_rows.append(outer_legal)
        for j in range(3):
            fit,query,held=role(d,f,j);c=counts(fit);qc=counts(query);used=int((c.sum(1)>0).sum());cost=pipeline_cost(used)
            root_map={name:sorted(set(fit.root)) for name in STAGES}
            validate_stage_roots(set(fit.root),set(query.root),set(held.root),root_map)
            save(OUT/f'outer{f}_inner{j}_all_supervised_stage_roots.json',dict(outer_fold=f,excluded_inner=j,fit_roots=sorted(set(fit.root)),query_roots=sorted(set(query.root)),outer_roots=sorted(set(held.root)),supervised_stage_roots=root_map))
            empirical_local_floor=int((c.sum(1)-c.max(1)).sum())
            stages.append(dict(outer_fold=f,excluded_inner=j,fit_rows=len(fit),query_rows=len(query),fit_roots=fit.root.nunique(),query_roots=query.root.nunique(),fit_locals=used,
                fit_M=int(c[:,1].sum()),fit_S=int(c[:,2].sum()),OOF_M=int(qc[:,1].sum()),OOF_S=int(qc[:,2].sum()),local_label_collision_floor=empirical_local_floor,**cost))
            for name,frame in [('fit',fit),('OOF',query)]:
                by_key=pd.DataFrame(dict(key=fine.iloc[frame.index].to_numpy(),root=frame.root.to_numpy(),truth=frame.truth.to_numpy()))
                for (key,cl),g in by_key.groupby(['key','truth']):
                    support.append(dict(outer_fold=f,excluded_inner=j,role=name,fine_key=key,truth=int(cl),original_rows=len(g),root_count=g.root.nunique()))
            receipt_path=OLD/f'teacher{f}_{j}/fit.json';receipt=read(receipt_path)
            assert receipt['status']=='fit_executed' and receipt['fit_rows']==len(fit) and receipt['fit_M']==int(c[:,1].sum()) and receipt['fit_S']==int(c[:,2].sum())
            folder=receipt_path.parent
            assert receipt['model_sha256']==sha(folder/'epoch25_model.pt') and receipt['logits_sha256']==sha(folder/'canonical_member_logits.npy')
            z=np.load(folder/'canonical_member_logits.npy');assert z.shape==(22546,16,3) and np.isfinite(z).all()
            probabilities=softmax(z,axis=-1);pred=probabilities.mean(1).argmax(1);member=z.argmax(-1)
            for label,frame in [('old_pipeline_fit',fit),('legal_OOF',query)]:
                for cl in [1,2]:
                    sub=frame[frame.truth.eq(cl)];loc=sub.local.to_numpy();correct=member[loc]==cl
                    saved_profiles.append(dict(outer_fold=f,excluded_inner=j,role=label,truth=cl,original_rows=len(sub),errors=int((pred[loc]!=cl).sum()),
                        all_members_correct=int(correct.all(1).sum()),no_correct_member=int((~correct).all(1).sum()),member_disagreement=int((member[loc].min(1)!=member[loc].max(1)).sum()),
                        correct_member_available_but_mean_wrong=int(((correct.any(1))&(pred[loc]!=cl)).sum())))
            rr=query[['row_position','local','root','fold','truth']].copy();rr['outer_fold']=f;rr['excluded_inner']=j;rr['old_uniform_pred']=pred[rr.local]
            rr['correct_members']=(member[rr.local]==rr.truth.to_numpy()[:,None]).sum(1);rr['member_disagreement']=member[rr.local].min(1)!=member[rr.local].max(1)
            rr['protocol']=proto.iloc[query.index].to_numpy();ooof.append(rr)
            reuse.append(dict(outer_fold=f,excluded_inner=j,root_roles_legal=True,saved_model_and_logits_hashes_match_receipt=True,
                training_input='N1_ASA.npz',saved_prediction_input='B_header_ASA.npz',training_epochs=25,
                compatible_as_same_V146_A_pipeline=False,reason='Different original fit input, 25-epoch base only, absent V135 100-epoch and four adaptation fits; legal predictions cannot be relabeled as current-pipeline OOF.',
                fit_receipt_sha256=sha(receipt_path),saved_original_frequency_profile_only=True))
    manifest=pd.concat(role_rows,ignore_index=True);assert len(manifest)==225614 and not manifest.duplicated(['outer_fold','row_position']).any()
    manifest.to_parquet(OUT/'nested_source_roles.parquet',index=False)
    pd.DataFrame(support).to_parquet(OUT/'complete_fine_class_root_support.parquet',index=False)
    pd.DataFrame(saved_profiles).to_parquet(OUT/'old_saved_member_role_profiles.parquet',index=False)
    scored=pd.concat(ooof,ignore_index=True);assert len(scored)==225614;scored.to_parquet(OUT/'old_saved_OOF_original_role_rows.parquet',index=False)
    costs=dict(planned_new_base_pipelines=9,full_network_fits=9,readout_fits=18,second_layer_fits=18,base_total_fits=45,fusion_fits_max=6,formal_total_fits_max=51,
        base_full_network_gradient_updates=sum(v['full_network_gradient_updates'] for v in stages),base_dense_full_gradient_cap=7200,base_dense_accepted_update_cap=7200,
        note='Prospective feasibility cost, not an activated fit budget. Fusion gradient/proposal/evaluation counts and all runtime entry/source bindings remain to implement.')
    direct=[OUT/n for n in ['nested_source_roles.parquet','complete_fine_class_root_support.parquet','old_saved_member_role_profiles.parquet','old_saved_OOF_original_role_rows.parquet','pre_execution_bindings.json']]
    save(OUT/'qualification.json',dict(status='actual_nested_source_roles_pipeline_lineage_cost_and_old_saved_member_support_audited',original_ASA_rows=112807,legal_fusion_original_role_rows=225614,
        official_classifier_forward_calls=0,official_features_calls=0,new_fits=0,new_gradients=0,new_updates=0,saved_softmax_array_recomputations=9,
        current_V146_A_fresh_inner_pipeline_needed=True,old_bank_legal_but_not_current_pipeline=True,old_fit_input_changed_actual_locals=len(changed),old_fit_input_changed_original_rows=int(d.local.isin(changed).sum()),
        old_input_difference_nnz=int(different.nnz),old_fact_coordinate_count=495,current_fact_coordinate_count=495,
        old_vs_current_fact_difference_nnz=int(different[:,65792:].nnz),old_vs_current_record_port18_difference_nnz=int(different[:,-18:].nnz),
        planned_costs=costs,roles=stages,reuse_reviews=reuse,negative_cases=negative_cases(p),
        formal_training_registered=False,quality_acceptance=False,full_goal_complete=False,latest_actual_training='V155',
        scope='Source-role and saved-output qualification, not new current-pipeline OOF supervision or model replay.',
        source_sha256={q.relative_to(ROOT).as_posix():sha(q) for q in [Path(__file__),PLAN]+direct}))
    print(json.dumps(dict(status='nested_qualification_finished',planned_total_fits=51,actual_new_fits=0,legal_original_role_rows=len(manifest),planned_base_full_network_updates=costs['base_full_network_gradient_updates']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
