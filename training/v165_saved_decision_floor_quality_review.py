"""Independent original-row candidate quality, restored tensors and readout."""
import json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v161_independent_all_finite_results_review import parameter_hash
from v164_saved_training_quality_review import compare
from v163_independent_readout_override_bound_review import review as readout_review

TRIAL=ROOT/'artifacts/v165_fixed_endpoint_decision_floor_diagnostic_20261002'
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
OUT=ROOT/'artifacts/v165_saved_decision_floor_quality_review_20261002'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/diagnostic.json').exists() for r in range(3));OUT.mkdir()
    goldpath=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldpath,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    sources={Path(__file__).resolve(),goldpath,ROOT/'training/v164_saved_training_quality_review.py',ROOT/'training/v161_independent_all_finite_results_review.py',ROOT/'training/v163_independent_readout_override_bound_review.py'}|{p for p in TRIAL.rglob('*') if p.is_file()}
    for role in range(3):
        sources.update([PRIOR/f'role{role}/endpoint.pt',PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet',PRIOR/f'role{role}/baseline/OOF_original_rows.parquet',ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));roles=[]
    for role in range(3):
        folder=TRIAL/f'role{role}';diagnostic=read(folder/'diagnostic.json');assert diagnostic['exception'] is None
        base=pd.read_parquet(folder/'baseline/OOF_original_rows.parquet');candidate=pd.read_parquet(folder/'treatment/OOF_original_rows.parquet');fixed=pd.read_parquet(PRIOR/f'role{role}/baseline/OOF_original_rows.parquet');previous=pd.read_parquet(PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet');endpoint=pd.read_parquet(folder/'endpoint/OOF_original_rows.parquet')
        for frame in [base,candidate,endpoint]:assert np.array_equal(frame.truth,gold[frame.row_position]) and np.array_equal(frame[['row_position','local','truth','root','pure_current_input']].to_numpy(),previous[['row_position','local','truth','root','pure_current_input']].to_numpy())
        assert np.array_equal(base.pred,previous.pred) and np.array_equal(endpoint.pred,base.pred) and np.array_equal(base.protected_correct,previous.protected_correct) and np.array_equal(candidate.protected_correct,base.protected_correct)
        for scope in ['OOF','deployment']:
            before=pd.read_parquet(folder/f'baseline/{scope}_original_rows.parquet');after=pd.read_parquet(folder/f'endpoint/{scope}_original_rows.parquet')
            assert np.array_equal(before.truth,gold[before.row_position]) and np.array_equal(before.pred,after.pred)
            for col in ['p0','p1','p2','logp0','logp1','logp2']:assert np.array_equal(before[col],after[col])
        state=torch.load(PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state'];restored=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)['state'];assert list(state)==list(restored) and all(torch.equal(value,restored[name]) for name,value in state.items())
        origin=parameter_hash(state);assert origin==diagnostic['restoration']['initial_parameter_sha256']==diagnostic['restoration']['restored_parameter_sha256'];delta=np.load(folder/'treatment/direction.npy');assert parameter_hash(state,delta,1.)==diagnostic['candidate']['probe_parameter_sha256']
        stats=compare(candidate,base,fixed);paired=compare(candidate,base,base);controlpath=ROOT/read(folder/'matched_control.json')['control_original_restoration'];control=pd.read_parquet(controlpath/'OOF_original_rows.parquet');controlstats=compare(control,base,base)
        opinions=np.load(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy',mmap_mode='r')[:,:16];prior=np.log(np.maximum(np.asarray(opinions,np.float64).mean(1),1e-12));output=state['output_weight'].numpy()+delta[-48:].reshape(16,3);readout=readout_review(candidate,prior,output)
        for item in readout['classes'].values():item['current_errors_not_blocked_by_fixed_readout_bound']=item['current_errors']-item['fixed_readout_cannot_override_prior'];item['pure_errors_not_blocked_by_fixed_readout_bound']=item['pure_current_errors']-item['pure_errors_fixed_readout_cannot_override']
        accepted=diagnostic['candidate']['accepted'];registered_errors=sum(v['protected_regressions'] for v in stats.values());all_new=sum(v['new_errors_vs_previous_accepted'] for v in stats.values())
        if accepted:assert registered_errors==0 and all(v['errors']<=int((base.truth.eq(cls)&base.pred.ne(base.truth)).sum()) for cls,v in zip([1,2],stats.values()))
        item=dict(role=role,actual_candidate_accepted=accepted,pure_errors_before=int((base.pred.ne(base.truth)&base.pure_current_input).sum()),pure_errors_after=int((candidate.pred.ne(candidate.truth)&candidate.pure_current_input).sum()),candidate_quality_relative_V159_and_V164=stats,paired_treatment_vs_V164=paired,paired_control_vs_V164=controlstats,all_new_errors_vs_V164=all_new,registered_cumulative_protection_regressions=registered_errors,candidate_fixed_readout_diagnostic=readout,candidate_parameter_sha256=diagnostic['candidate']['probe_parameter_sha256'],restored_parameter_sha256=origin,restored_full_tensor_and_predictions_exact=True,new_fits=0,permanent_updates=0,candidate_is_not_submitted_as_training_state=True)
        save(OUT/f'role{role}.json',item);roles.append(item)
    check_bindings(bindings);allpass=all(r['actual_candidate_accepted'] for r in roles);gain=any(r['pure_errors_after']<r['pure_errors_before'] for r in roles)
    save(OUT/'review.json',dict(status='all_actual_decision_floor_candidates_original_gold_full_quality_and_restored_tensors_reviewed',roles=roles,all_actual_finite_guards_passed=allpass,any_candidate_pure_classification_gain=gain,supports_separate_short_training_registration=allpass and gain,official_calls_by_review=0,new_fits=0,permanent_updates=0,classification_mastery=False,quality_acceptance=False,root_goal_complete=False,source_sha256=bindings));print(json.dumps(dict(status='V165_saved_decision_floor_quality_review_passed',all_actual_finite_guards_passed=allpass,official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
