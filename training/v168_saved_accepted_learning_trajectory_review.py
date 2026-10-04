"""Actual committed V164 trajectory versus a temporary V168 candidate, saved data only."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v161_independent_all_finite_results_review import parameter_hash
from v164_saved_training_quality_review import compare
from v168_root_saved_prior_burden_review import weighted_quantiles

OUT=ROOT/'artifacts/v168_saved_accepted_learning_trajectory_review_20261002'
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
INITIAL=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'

def burden(frame,fixed,means,output):
    truth=frame.truth.to_numpy(np.int64);local=frame.local.to_numpy(np.int64);lp=frame[['logp0','logp1','logp2']].to_numpy();other=lp.copy();other[np.arange(len(frame)),truth]=-np.inf;rival=other.argmax(1);idx=np.arange(len(frame));prior=np.log(np.maximum(means[local,truth],1e-12))-np.log(np.maximum(means[local,rival],1e-12));margin=lp[idx,truth]-lp[idx,rival];learned=margin-prior
    fixed_deep=fixed.pred.ne(fixed.truth).to_numpy()&(means[local,truth]<=1e-12);wrong=frame.pred.ne(frame.truth).to_numpy();rows=[]
    for cls,name in [(1,'M'),(2,'S')]:
        selected=(truth==cls)&fixed_deep;bounds=math.fsum(abs(float(x)) for x in output[:,cls]-output[:,1 if cls==2 else 2]);rows.append(dict(class_name=name,fixed_deep_initial_error_rows=int(selected.sum()),fixed_deep_remaining_errors=int((selected&wrong).sum()),fixed_deep_actual_margin_quantiles=weighted_quantiles(margin[selected]),fixed_deep_learned_residual_margin_quantiles=weighted_quantiles(learned[selected]),fixed_deep_frozen_prior_margin_quantiles=weighted_quantiles(prior[selected]),current_readout_M_S_margin_bound=bounds,all_current_errors=int(((truth==cls)&wrong).sum()),all_current_pure_errors=int(((truth==cls)&wrong&frame.pure_current_input.to_numpy()).sum()),full_original_class_mass=int((truth==cls).sum())))
    return rows

def main():
    assert not OUT.exists();goldpath=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldpath,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();old=ROOT/'artifacts/v164_saved_training_quality_review_20261002/review.json';check_bindings(read(old)['source_sha256']);actual=ROOT/'artifacts/v168_independent_actual_decision_floor_review_20261002/review.json';check_bindings(read(actual.parent/'pre_review_bindings.json')['source_sha256']);assert read(actual)['all_three_actual_finite_candidates_safe']
    paths={Path(__file__).resolve(),goldpath,old,actual,actual.parent/'pre_review_bindings.json',ROOT/'training/v164_saved_training_quality_review.py',ROOT/'training/v168_root_saved_prior_burden_review.py',ROOT/'training/v161_independent_all_finite_results_review.py'};inputs=[]
    for role in range(3):
        folder=PRIOR/f'role{role}';fit=read(folder/'fit.json');initial=INITIAL/f'fold{role}_B/endpoint.pt';paths.update([initial,folder/'fit.json',folder/'started.json',folder/'baseline/OOF_original_rows.parquet',ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy']);states=[]
        for i in range(1,fit['permanent_updates']+1):
            current=folder/f'accepted{i}';paths.update([current/'checkpoint.pt',current/'commit.json',current/'OOF_original_rows.parquet']);states.append(current)
        inputs.append((role,folder,fit,initial,states))
    candidate=ROOT/'artifacts/v168_decision_floor_diagnostic_20261002/role1/probe0';paths.update([candidate/'direction.npy',candidate/'v168_complete_probe_review.json',candidate/'OOF_original_rows.parquet',PRIOR/'role1/endpoint.pt'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};result=[]
    for role,folder,fit,initial,states in inputs:
        fixed=pd.read_parquet(folder/'baseline/OOF_original_rows.parquet');assert fixed.row_position.is_unique and np.array_equal(fixed.truth,gold[fixed.row_position]);means=np.asarray(np.load(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy',mmap_mode='r')[:,:16],np.float64).mean(1);state=torch.load(initial,map_location='cpu',weights_only=True)['state'];assert parameter_hash(state)==read(folder/'started.json')['initial_parameter_sha256'];previous=fixed;trajectory=[dict(kind='actual_training_origin',state_index=0,parameter_sha256=parameter_hash(state),classes=compare(fixed,fixed,fixed),deep_error_burden=burden(fixed,fixed,means,state['output_weight'].numpy()))]
        for i,path in enumerate(states,1):
            frame=pd.read_parquet(path/'OOF_original_rows.parquet');receipt=read(path/'commit.json');state=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)['state'];identity=parameter_hash(state);assert identity==receipt['parameter_sha256'] and np.array_equal(frame.truth,gold[frame.row_position]);classes=compare(frame,previous,fixed);assert all(v['protected_regressions']==0 for v in classes.values());trajectory.append(dict(kind='actual_committed_training_state',state_index=i,parameter_sha256=identity,newly_repaired_original_rows=receipt['newly_repaired_original_rows'],classes=classes,deep_error_burden=burden(frame,fixed,means,state['output_weight'].numpy())));previous=frame
        temporary=None
        if role==1:
            endpoint=torch.load(PRIOR/'role1/endpoint.pt',map_location='cpu',weights_only=True)['state'];delta=np.load(candidate/'direction.npy');identity=parameter_hash(endpoint,delta,1.);assert identity==read(candidate/'v168_complete_probe_review.json')['probe_parameter_sha256'];frame=pd.read_parquet(candidate/'OOF_original_rows.parquet');assert np.array_equal(frame.truth,gold[frame.row_position]);temporary=dict(kind='temporary_finite_diagnostic_not_committed',parameter_sha256=identity,classes_vs_last_accepted=compare(frame,previous,fixed),deep_error_burden=burden(frame,fixed,means,endpoint['output_weight'].numpy()+delta[-48:].reshape(16,3)),counts_as_additional_accepted_training_state=False)
        result.append(dict(role=role,actual_accepted_updates=fit['permanent_updates'],actual_stop=fit['status'],actual_committed_trajectory=trajectory,temporary_V168_candidate=temporary,mastery=False))
    assert [r['actual_accepted_updates'] for r in result]==[1,1,3] and all(r['actual_stop']=='margin_normal_cap_stop' for r in result);s=[row['deep_error_burden'][1]['fixed_deep_remaining_errors'] for row in result[1]['actual_committed_trajectory']];assert s[0]==s[-1]==813 and result[1]['temporary_V168_candidate']['deep_error_burden'][1]['fixed_deep_remaining_errors']==813
    check_bindings(bindings);OUT.mkdir();(OUT/'review.json').write_bytes((json.dumps(dict(status='V164_actual_committed_trajectories_and_V168_uncommitted_deep_error_burden_reviewed',roles=result,actual_committed_updates_total=5,V168_candidate_not_added_to_training_trajectory=True,role1_frozen_deep_S_error_rows_still_wrong=813,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,no_new_training_authority=True,classification_mastery=False,full_quality_acceptance=False,scope='Saved supervised development trajectory; fixed NumPy prior means are diagnostic, not a new forward, acceptance tolerance or global model bound',source_sha256=bindings),ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status='V168_saved_actual_learning_trajectory_reviewed',actual_prior_commits=5,new_official_calls=0)))

if __name__=='__main__':main()
