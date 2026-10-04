"""Original-gold complete row quality and full restored state, saved data only."""
import json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v161_independent_all_finite_results_review import parameter_hash
from v159_float64_repeat_policy_v2 import repeat_values
from v164_saved_training_quality_review import compare
from v163_independent_readout_override_bound_review import review as readout_review

TRIAL=ROOT/'artifacts/v166_coverage_first_diagnostic_20261002'
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
CONTROL=ROOT/'artifacts/v165_fixed_endpoint_decision_floor_diagnostic_20261002'
OUT=ROOT/'artifacts/v166_saved_coverage_quality_review_20261002'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/diagnostic.json').exists() for r in range(3));OUT.mkdir()
    seal=read(TRIAL/'run_seal.json');check_bindings(seal['source_sha256']);plan=read(ROOT/seal['plan_path'])
    goldpath=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldpath,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();assert np.bincount(gold,minlength=3).tolist()==[1899723,111728,45420]
    sources={Path(__file__).resolve(),goldpath,ROOT/'training/v164_saved_training_quality_review.py',ROOT/'training/v161_independent_all_finite_results_review.py',ROOT/'training/v163_independent_readout_override_bound_review.py',ROOT/'training/v159_float64_repeat_policy_v2.py'}|{p for p in TRIAL.rglob('*') if p.is_file()}
    for role in range(3):sources.update([PRIOR/f'role{role}/endpoint.pt',PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet',PRIOR/f'role{role}/baseline/OOF_original_rows.parquet',CONTROL/f'role{role}/treatment/OOF_original_rows.parquet',ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));roles=[]
    for spec in plan['roles']:
        role=spec['role'];folder=TRIAL/f'role{role}';diagnostic=read(folder/'diagnostic.json');assert diagnostic['exception'] is None
        base=pd.read_parquet(folder/'baseline/OOF_original_rows.parquet');fixed=pd.read_parquet(PRIOR/f'role{role}/baseline/OOF_original_rows.parquet');previous=pd.read_parquet(PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet');endpoint=pd.read_parquet(folder/'endpoint/OOF_original_rows.parquet');has_candidate=diagnostic['candidate'] is not None;candidate=pd.read_parquet(folder/'treatment/OOF_original_rows.parquet') if has_candidate else base.copy()
        for frame in [base,candidate,endpoint]:assert np.array_equal(frame.truth,gold[frame.row_position]) and np.array_equal(frame[['row_position','local','truth','root','pure_current_input']].to_numpy(),previous[['row_position','local','truth','root','pure_current_input']].to_numpy())
        assert np.array_equal(base.pred,previous.pred) and np.array_equal(endpoint.pred,base.pred) and np.array_equal(base.protected_correct,previous.protected_correct) and np.array_equal(candidate.protected_correct,base.protected_correct)
        for scope in ['OOF','deployment']:
            before=pd.read_parquet(folder/f'baseline/{scope}_original_rows.parquet');after=pd.read_parquet(folder/f'endpoint/{scope}_original_rows.parquet');assert np.array_equal(before.truth,gold[before.row_position]) and np.array_equal(before.pred,after.pred)
            for keys,kind in [(['p0','p1','p2'],'probability'),(['logp0','logp1','logp2'],'log_probability')]:assert repeat_values(before[keys].to_numpy(),after[keys].to_numpy(),kind)['passed']
        state=torch.load(PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state'];restored=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)['state'];assert list(state)==list(restored) and all(torch.equal(value,restored[name]) for name,value in state.items());origin=parameter_hash(state);assert origin==diagnostic['initial_parameter_sha256']==diagnostic['restored_parameter_sha256'];delta=np.load(folder/'treatment/direction.npy') if has_candidate else np.zeros(1060832)
        if has_candidate:assert parameter_hash(state,delta,1.)==diagnostic['candidate']['probe_parameter_sha256']
        stats=compare(candidate,base,fixed);paired=compare(candidate,base,base);control=pd.read_parquet(CONTROL/f'role{role}/treatment/OOF_original_rows.parquet');controlstats=compare(control,base,base)
        opinions=np.load(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy',mmap_mode='r')[:,:16];prior=np.log(np.maximum(np.asarray(opinions,np.float64).mean(1),1e-12));output=state['output_weight'].numpy()+delta[-48:].reshape(16,3);readout=readout_review(candidate,prior,output)
        for item in readout['classes'].values():item['current_errors_not_blocked_by_fixed_readout_bound']=item['current_errors']-item['fixed_readout_cannot_override_prior'];item['pure_errors_not_blocked_by_fixed_readout_bound']=item['pure_current_errors']-item['pure_errors_fixed_readout_cannot_override']
        accepted=bool(has_candidate and diagnostic['candidate']['accepted']);registered_errors=sum(v['protected_regressions'] for v in stats.values());all_new=sum(v['new_errors_vs_previous_accepted'] for v in stats.values())
        if accepted:assert registered_errors==0 and all(v['errors']<=int((base.truth.eq(cls)&base.pred.ne(base.truth)).sum()) for cls,v in zip([1,2],stats.values()))
        events=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()];counts=diagnostic['counts'];expected=(2+int(has_candidate))*(spec['OOF_chunks']+12)+spec['fresh_margin_gradient_cap'];assert counts['head_completed']==counts['head_attempts']==counts['feature_completed']==counts['feature_attempts']==expected<=spec['head_cap'];assert counts['margin_attempts']==counts['margin_completed']==spec['fresh_margin_gradient_cap'];assert counts['gradient_attempts']==counts['gradient_completed']==0
        for kind,count in [('head',expected),('feature',expected),('full_parameter_margin_gradient',spec['fresh_margin_gradient_cap'])]:
            for event in ['attempt','completed']:assert [r['ordinal'] for r in events if r['kind']==kind and r['event']==event]==list(range(1,count+1))
        assert diagnostic['new_fits']==diagnostic['permanent_updates']==0
        item=dict(role=role,diagnostic_status=diagnostic['status'],actual_finite_candidate_present=has_candidate,actual_candidate_accepted=accepted,pure_errors_before=int((base.pred.ne(base.truth)&base.pure_current_input).sum()),pure_errors_after=int((candidate.pred.ne(candidate.truth)&candidate.pure_current_input).sum()),candidate_quality_relative_V159_and_V164=stats,paired_treatment_vs_V164=paired,paired_V165_control_vs_V164=controlstats,all_new_errors_vs_V164=all_new,registered_cumulative_protection_regressions=registered_errors,candidate_or_restored_origin_fixed_readout_diagnostic=readout,candidate_parameter_sha256=diagnostic['candidate']['probe_parameter_sha256'] if has_candidate else None,restored_parameter_sha256=origin,restored_full_tensors_exact_and_predictions_8eps_with_exact_argmax=True,new_fits=0,permanent_updates=0,candidate_is_not_submitted_as_training_state=True,actual_heads=expected,actual_margin_derivatives=spec['fresh_margin_gradient_cap'],actual_QP_solves=diagnostic['actual_QP_solves'],optimizer_iterations=diagnostic['optimizer_iterations'],local_unqualified_rows_are_origin_quality_not_unmeasured_candidate=True)
        save(OUT/f'role{role}.json',item);roles.append(item)
    check_bindings(bindings);allpass=all(r['actual_candidate_accepted'] for r in roles);gain=any(r['actual_finite_candidate_present'] and r['pure_errors_after']<r['pure_errors_before'] for r in roles)
    summary=dict(status='all_actual_coverage_or_local_stop_original_gold_full_quality_and_restored_tensors_reviewed',roles=roles,actual_new_heads=sum(r['actual_heads'] for r in roles),actual_new_margin_derivatives=sum(r['actual_margin_derivatives'] for r in roles),actual_new_QP_solves=sum(r['actual_QP_solves'] for r in roles),all_actual_finite_guards_passed=allpass,any_actual_pure_classification_gain=gain,supports_separate_short_training_registration=allpass and gain,official_calls_by_review=0,new_fits=0,permanent_updates=0,classification_mastery=False,quality_acceptance=False,root_goal_complete=False,source_sha256=bindings);save(OUT/'review.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['roles','source_sha256']}))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
