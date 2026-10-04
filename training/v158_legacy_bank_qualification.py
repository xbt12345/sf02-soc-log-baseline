"""Legacy expert full-population nested roles and legal initialization algebra."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from experiment_review import ROOT,read,sha,check_bindings
from v107_matched_training import ROWS,FOLDS,FID,ASA_IDS,DEST,PREP,X
from v131_common import load_data,OFFICIAL
from v158_fusion_contract import inner_fold
from v158_bank_initialization import safe_legacy_prior

OUT=ROOT/'artifacts/v158_legacy_expert_bank_qualification_20261001'
PLAN=ROOT/'training/review_policy/v158_legacy_expert_bank_qualification_plan.json'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    p=read(PLAN);check_bindings(p['source_sha256']);assert p['classifier_calls']==p['new_fits']==p['new_updates']==0
    assert p['expert_count']==17 and p['planned_total_fits']==60 and not p['formal_training_activation']
    assert not OUT.exists();OUT.mkdir()
    actual=[ROWS,FOLDS,FID,ASA_IDS,OFFICIAL]
    for f in range(3):actual += [DEST/f'fold{f}_N1_teacher/{n}' for n in ['fit.json','teacher.joblib','scores_all_input_ids.npy']]
    save(OUT/'pre_saved_array_bindings.json',dict(source_sha256={q.relative_to(ROOT).as_posix():sha(q) for q in actual+[Path(__file__),PLAN]},model_calls=0))
    rr=pd.read_parquet(ROWS,columns=['row_position','route','label_index'])
    folds=pd.read_parquet(FOLDS,columns=['row_position','root','proposed_fold'])
    gold=pd.read_parquet(OFFICIAL,columns=['label_binary']).label_binary.map(dict(benign=0,malicious=1,suspicious=2)).to_numpy()
    assert len(rr)==len(folds)==len(gold)==2056871 and np.array_equal(rr.label_index,gold)
    assert np.array_equal(rr.row_position,np.arange(2056871)) and np.array_equal(folds.row_position,rr.row_position)
    assert folds.groupby('root').proposed_fold.nunique().max()==1
    fid=np.load(FID,mmap_mode='r');asaids=np.load(ASA_IDS);_,d=load_data();d=d.reset_index(drop=True)
    assert np.array_equal(fid[d.row_position],asaids[d.local]) and np.array_equal(rr.iloc[d.row_position].label_index,d.truth)
    review=[];initial=[];canonical=[]
    for f in range(3):
        legal=folds.proposed_fold.ne(f).to_numpy();inner=folds.root.map(lambda z:inner_fold(f,z)).to_numpy()
        for j in range(3):
            fit=legal&(inner!=j);query=legal&(inner==j)
            assert not set(folds.loc[fit,'root'])&set(folds.loc[query,'root'])
            assert not set(folds.loc[fit,'root'])&set(folds.loc[~legal,'root'])
            mass=np.bincount(gold[fit],minlength=3);assert (mass>0).all()
            subset=d[d.fold.ne(f)&d.root.map(lambda z:inner_fold(f,z)).ne(j)]
            assert np.array_equal(np.bincount(gold[fit&rr.route.eq('asa').to_numpy()],minlength=3),np.bincount(subset.truth,minlength=3))
            unique=int(np.unique(fid[fit]).size)
            review.append(dict(outer_fold=f,excluded_inner=j,fit_rows=int(fit.sum()),fit_class_mass=mass.tolist(),fit_roots=int(folds.loc[fit,'root'].nunique()),fit_actual_feature_ids=unique,
                OOF_full_rows=int(query.sum()),OOF_class_mass=np.bincount(gold[query],minlength=3).tolist(),OOF_ASA_rows=int((query&rr.route.eq('asa').to_numpy()).sum()),
                actual_existing_N1_fit_input=True,new_legacy_fits=1,gradient_evaluations_max=1000,accepted_iterations_max=1000,
                source_closure_all_stages=True,split_rerolled=False))
            cc=subset.groupby(['canonical_key','truth']).size().unstack(fill_value=0)
            local=subset.groupby(['local','truth']).size().unstack(fill_value=0)
            canonical.append(dict(outer_fold=f,excluded_inner=j,local_floor=int((local.sum(1)-local.max(1)).sum()),canonical_floor=int((cc.sum(1)-cc.max(1)).sum())))
        folder=DEST/f'fold{f}_N1_teacher';receipt=read(folder/'fit.json')
        assert sha(folder/'scores_all_input_ids.npy')==receipt['scores_sha256'] and sha(folder/'teacher.joblib')==receipt['model_sha256']
        raw=np.load(folder/'scores_all_input_ids.npy',mmap_mode='r')[fid[d.row_position]]
        old=softmax(raw,axis=1);current=np.load(ROOT/f'artifacts/v155_guarded_full_gradient_sam_20261001/fold{f}_zero_probability.npy')[d.local]
        prior=safe_legacy_prior(current,old,d.truth.to_numpy(),d.fold.ne(f).to_numpy())
        initial.append(dict(outer_fold=f,**prior,score_semantics='normalized_exp_of_saved_three_OVA_logits_preserves_old_argmax',
            OOF_or_outer_initial_fusion_quality_not_evaluated=True,probability_classifier_function_calls=0))
    save(OUT/'legacy_full_nested_roles_and_cost.json',dict(roles=review,new_legacy_fits=9,full_gradient_cap=9000,fit_input_parts='v79 X CSR physical arrays plus v101 N1_delta; both to bind in actual trainer before use',
        deployment_existing_A0_exact_role_reuse=True,ASA_IDS_mapping_verified=True))
    save(OUT/'initial_legacy_priors.json',dict(status='legal_FIT_only_closed_form_quarter_of_preserving_bound_no_outer_tuning',folds=initial,model_calls=0,new_fits=0,
        formula='alpha0=.25*min(1,min(current_true_margin/(current_true_margin-legacy_true_margin) over harmful slopes and current-correct legal FIT rows)); same alpha for A and B.',
        all_three_classes_preserved=True,learner_prior_moves_with_expert_probability_under_permutation=True))
    save(OUT/'actual_inner_canonical_floors.json',dict(roles=canonical,use_for_mastery='canonical_key_not_local_floor',old_qualification_local_floor_kept_as_descriptive_only=True))
    targets=[OUT/n for n in ['pre_saved_array_bindings.json','legacy_full_nested_roles_and_cost.json','initial_legacy_priors.json','actual_inner_canonical_floors.json']]
    save(OUT/'qualification.json',dict(status='legacy_bank_source_roles_full_population_cost_and_SAFE_FIT_prior_qualified',official_classifier_calls=0,official_features_calls=0,new_fits=0,new_gradients=0,new_updates=0,
        existing_score_array_softmax_recalculations=3,current_pipeline_fits_planned=45,new_legacy_N1_fits_planned=9,fusion_fits_max=6,total_planned_fits=60,
        expert_bank='16_current_V146_A_members_plus_one_same_role_legacy_A0_N1_vector',legacy_objective='original_V107_full_population_three_OVA_sigmoid_BCE_with_1e-6_L2_no_label_rewrite',
        decision='Include legacy expert in both A/B before any new fitting; no truth-based expert selection, no outer-score grid.',formal_training_registered=False,quality_acceptance=False,
        source_sha256={q.relative_to(ROOT).as_posix():sha(q) for q in [Path(__file__),PLAN]+targets}))
    print(json.dumps(dict(legacy_bank_qualification_complete=True,planned_total_fits=60,new_fits=0,priors=[z['legacy_prior'] for z in initial],legacy_fit_rows=[z['fit_rows'] for z in review]),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
