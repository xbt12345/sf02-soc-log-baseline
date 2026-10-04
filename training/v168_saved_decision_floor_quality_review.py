"""Complete original-gold quality of the single actual floor candidate and restoration."""
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

TRIAL=ROOT/'artifacts/v168_decision_floor_diagnostic_20261002'
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
OLD=ROOT/'artifacts/v167_trial_point_restoration_diagnostic_20261002'
OUT=ROOT/'artifacts/v168_saved_decision_floor_quality_review_20261002'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    folder=TRIAL/'role1';assert not OUT.exists() and (folder/'diagnostic.json').exists();seal=read(TRIAL/'run_seal.json');check_bindings(seal['source_sha256']);plan=read(ROOT/seal['plan_path']);diag=read(folder/'diagnostic.json');assert diag['exception'] is None and diag['new_fits']==diag['permanent_updates']==0
    goldfile=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldfile,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();assert np.bincount(gold,minlength=3).tolist()==[1899723,111728,45420]
    paths={Path(__file__).resolve(),goldfile,ROOT/'training/v164_saved_training_quality_review.py',ROOT/'training/v163_independent_readout_override_bound_review.py',ROOT/'training/v161_independent_all_finite_results_review.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/plan['additional_candidate_repair_goal']}|{p for p in TRIAL.rglob('*') if p.is_file()}
    for role in range(3):paths.update(p for p in (PRIOR/f'role{role}/endpoint').iterdir() if p.is_file());paths.add(PRIOR/f'role{role}/endpoint.pt')
    paths.update([PRIOR/'role1/baseline/OOF_original_rows.parquet',OLD/'role1/probe1/OOF_original_rows.parquet',ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001/fold1/OOF_probabilities.npy'])
    for role in [0,2]:paths.update(p for p in (OLD/f'role{role}/probe0').iterdir() if p.is_file())
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};OUT.mkdir();save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));base=pd.read_parquet(folder/'baseline/OOF_original_rows.parquet');previous=pd.read_parquet(PRIOR/'role1/endpoint/OOF_original_rows.parquet');fixed=pd.read_parquet(PRIOR/'role1/baseline/OOF_original_rows.parquet');control=pd.read_parquet(OLD/'role1/probe1/OOF_original_rows.parquet');endpoint=pd.read_parquet(folder/'endpoint/OOF_original_rows.parquet');state=torch.load(PRIOR/'role1/endpoint.pt',map_location='cpu',weights_only=True)['state'];restored=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)['state'];assert state.keys()==restored.keys() and all(torch.equal(state[k],restored[k]) for k in state);origin=parameter_hash(state);assert origin==diag['origin_parameter_sha256']==diag['restored_parameter_sha256'] and np.array_equal(base.pred,previous.pred) and np.array_equal(endpoint.pred,base.pred) and np.array_equal(base.protected_correct,previous.protected_correct)
    for scope in ['OOF','deployment']:
        before=pd.read_parquet(folder/f'baseline/{scope}_original_rows.parquet');after=pd.read_parquet(folder/f'endpoint/{scope}_original_rows.parquet');assert np.array_equal(before.truth,gold[before.row_position]) and np.array_equal(after.truth,gold[after.row_position]) and np.array_equal(before.pred,after.pred)
        for fields,kind in [(['p0','p1','p2'],'probability'),(['logp0','logp1','logp2'],'log_probability')]:assert repeat_values(before[fields].to_numpy(),after[fields].to_numpy(),kind)['passed']
    candidates=[];proposals=sorted(folder.glob('probe*'));assert len(proposals)<=1
    opinions=np.load(ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001/fold1/OOF_probabilities.npy',mmap_mode='r')[:,:16];prior=np.log(np.maximum(np.asarray(opinions,np.float64).mean(1),1e-12))
    for proposal in proposals:
        proof=read(proposal/'v168_complete_probe_review.json');frame=pd.read_parquet(proposal/'OOF_original_rows.parquet');assert np.array_equal(frame.truth,gold[frame.row_position]) and np.array_equal(frame[['row_position','local','truth','root','pure_current_input']].to_numpy(),previous[['row_position','local','truth','root','pure_current_input']].to_numpy()) and np.array_equal(frame.protected_correct,base.protected_correct);delta=np.load(proposal/'direction.npy');assert parameter_hash(state,delta,1.)==proof['probe_parameter_sha256'];stats=compare(frame,base,fixed);vs_origin=compare(frame,base,base);vs_tie=compare(frame,control,control);readout=readout_review(frame,prior,state['output_weight'].numpy()+delta[-48:].reshape(16,3));regress=sum(v['protected_regressions'] for v in stats.values())
        if proof['accepted']:assert regress==0 and proof['additional_prospective_guard_review']['passed'] and stats['M']['errors']<=864 and stats['S']['errors']<=1156
        for scope in ['OOF','deployment']:
            observed=pd.read_parquet(proposal/f'{scope}_original_rows.parquet');reference=pd.read_parquet(folder/f'baseline/{scope}_original_rows.parquet');assert np.array_equal(observed.truth,gold[observed.row_position]) and np.array_equal(observed.row_position,reference.row_position)
        candidates.append(dict(actual_candidate_accepted=proof['accepted'],pure_errors_before=int((base.pred.ne(base.truth)&base.pure_current_input).sum()),pure_errors_after=int((frame.pred.ne(frame.truth)&frame.pure_current_input).sum()),full_quality_relative_V159_and_V164=stats,paired_vs_V164=vs_origin,paired_vs_V167_final_tie=vs_tie,additional_prospective_guard=proof['additional_prospective_guard_review'],registered_protection_regressions=regress,candidate_fixed_readout_diagnostic=readout,candidate_parameter_sha256=proof['probe_parameter_sha256'],fixed_target_risk=np.load(proposal/'fixed_error_risk.npy').tolist(),full_original_class_risk=np.load(proposal/'full_original_class_risk.npy').tolist(),candidate_is_temporary_not_committed=True))
    heads=(2+len(candidates))*16+50;assert diag['counts']==dict(head_attempts=heads,head_completed=heads,feature_attempts=heads,feature_completed=heads,gradient_attempts=0,gradient_completed=0,margin_attempts=50,margin_completed=50);events=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
    for kind,n in [('head',heads),('feature',heads),('full_parameter_margin_gradient',50)]:
        for event in ['attempt','completed']:assert [r['ordinal'] for r in events if r['kind']==kind and r['event']==event]==list(range(1,n+1))
    safe=[]
    for role in [0,2]:
        accepted=read(OLD/f'role{role}/probe0/probe.json');assert accepted['accepted'];before=pd.read_parquet(PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet');after=pd.read_parquet(OLD/f'role{role}/probe0/OOF_original_rows.parquet');assert np.array_equal(after.truth,gold[after.row_position]);safe.append(dict(role=role,prior_safe_candidate=True,paired_vs_V164=compare(after,before,before),official_calls=0,not_new_gain=True))
    check_bindings(bindings);passed=bool(candidates and candidates[0]['actual_candidate_accepted']);save(OUT/'review.json',dict(status='V168_single_actual_candidate_complete_original_gold_quality_and_full_restoration_reviewed',role=1,status_actual=diag['status'],all_finite_candidates=candidates,unchanged_safe_references=safe,actual_new_heads=heads,actual_new_complete_margin_derivatives=50,actual_QP_solves=diag['actual_QP_solves'],actual_finite_proposals=len(candidates),optimizer_iterations=diag['optimizer_iterations'],all_actual_finite_guards_passed=passed,supports_separate_short_training_registration=passed,all_restored_full_tensors_exact=True,all_restored_original_argmax_exact=True,original_q_and_logq_8eps_passed=True,official_calls_by_review=0,new_fits=0,permanent_updates=0,classification_mastery=False,quality_acceptance=False,root_goal_complete=False,source_sha256=bindings));print(json.dumps(dict(status='V168_saved_quality_review_passed',actual_candidate_safe=passed,official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
