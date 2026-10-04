"""All finite candidate original-row quality, costs and exact restored states."""
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

TRIAL=ROOT/'artifacts/v167_trial_point_restoration_diagnostic_20261002'
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
CONTROL=ROOT/'artifacts/v166_coverage_first_diagnostic_20261002'
OUT=ROOT/'artifacts/v167_saved_trial_point_quality_review_20261002'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/diagnostic.json').exists() for r in range(3));OUT.mkdir();seal=read(TRIAL/'run_seal.json');check_bindings(seal['source_sha256']);plan=read(ROOT/seal['plan_path']);goldfile=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldfile,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();assert np.bincount(gold,minlength=3).tolist()==[1899723,111728,45420]
    paths={Path(__file__).resolve(),goldfile,ROOT/'training/v164_saved_training_quality_review.py',ROOT/'training/v163_independent_readout_override_bound_review.py',ROOT/'training/v161_independent_all_finite_results_review.py',ROOT/'training/v159_float64_repeat_policy_v2.py'}|{p for p in TRIAL.rglob('*') if p.is_file()}
    for role in range(3):paths.update([PRIOR/f'role{role}/endpoint.pt',PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet',PRIOR/f'role{role}/baseline/OOF_original_rows.parquet',CONTROL/f'role{role}/treatment/OOF_original_rows.parquet',ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));roles=[]
    for spec in plan['roles']:
        role=spec['role'];folder=TRIAL/f'role{role}';diag=read(folder/'diagnostic.json');assert diag['exception'] is None and diag['new_fits']==diag['permanent_updates']==0;base=pd.read_parquet(folder/'baseline/OOF_original_rows.parquet');previous=pd.read_parquet(PRIOR/f'role{role}/endpoint/OOF_original_rows.parquet');fixed=pd.read_parquet(PRIOR/f'role{role}/baseline/OOF_original_rows.parquet');control=pd.read_parquet(CONTROL/f'role{role}/treatment/OOF_original_rows.parquet');endpoint=pd.read_parquet(folder/'endpoint/OOF_original_rows.parquet');state=torch.load(PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state'];restored=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)['state'];assert state.keys()==restored.keys() and all(torch.equal(state[name],restored[name]) for name in state);origin=parameter_hash(state);assert origin==diag['origin_parameter_sha256']==diag['restored_parameter_sha256'];assert np.array_equal(base.pred,previous.pred) and np.array_equal(endpoint.pred,base.pred) and np.array_equal(base.protected_correct,previous.protected_correct)
        for scope in ['OOF','deployment']:
            before=pd.read_parquet(folder/f'baseline/{scope}_original_rows.parquet');after=pd.read_parquet(folder/f'endpoint/{scope}_original_rows.parquet');assert np.array_equal(before.truth,gold[before.row_position]) and np.array_equal(before.pred,after.pred)
            for fields,kind in [(['p0','p1','p2'],'probability'),(['logp0','logp1','logp2'],'log_probability')]:assert repeat_values(before[fields].to_numpy(),after[fields].to_numpy(),kind)['passed']
        opinions=np.load(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy',mmap_mode='r')[:,:16];prior=np.log(np.maximum(np.asarray(opinions,np.float64).mean(1),1e-12));candidates=[];proposals=sorted(folder.glob('probe*'),key=lambda p:int(p.name[5:]))
        for proposal in proposals:
            proof=read(proposal/'probe.json');frame=pd.read_parquet(proposal/'OOF_original_rows.parquet');assert np.array_equal(frame.truth,gold[frame.row_position]) and np.array_equal(frame[['row_position','local','truth','root','pure_current_input']].to_numpy(),previous[['row_position','local','truth','root','pure_current_input']].to_numpy());assert np.array_equal(frame.protected_correct,base.protected_correct);delta=np.load(proposal/'direction.npy');assert parameter_hash(state,delta,1.)==proof['probe_parameter_sha256'];stats=compare(frame,base,fixed);vs_origin=compare(frame,base,base);vs_control=compare(frame,control,control);readout=readout_review(frame,prior,state['output_weight'].numpy()+delta[-48:].reshape(16,3));accepted=proof['accepted'];regress=sum(v['protected_regressions'] for v in stats.values())
            if accepted:assert regress==0 and all(stats[k]['errors']<=int((base.truth.eq(cls)&base.pred.ne(base.truth)).sum()) for cls,k in [(1,'M'),(2,'S')])
            candidates.append(dict(proposal=proposal.name,actual_candidate_accepted=accepted,pure_errors_before=int((base.pred.ne(base.truth)&base.pure_current_input).sum()),pure_errors_after=int((frame.pred.ne(frame.truth)&frame.pure_current_input).sum()),full_quality_relative_V159_and_V164=stats,paired_vs_V164=vs_origin,paired_vs_V166=vs_control,registered_protection_regressions=regress,candidate_fixed_readout_diagnostic=readout,candidate_parameter_sha256=proof['probe_parameter_sha256'],fixed_target_risk=np.load(proposal/'fixed_error_risk.npy').tolist(),full_original_class_risk=np.load(proposal/'full_original_class_risk.npy').tolist(),candidate_is_temporary_not_committed=True))
        expected_margins=50*diag['joint_restoration_attempts'];expected_heads=(2+len(proposals))*(spec['OOF_chunks']+12)+expected_margins;counts=diag['counts'];assert counts==dict(head_attempts=expected_heads,head_completed=expected_heads,feature_attempts=expected_heads,feature_completed=expected_heads,gradient_attempts=0,gradient_completed=0,margin_attempts=expected_margins,margin_completed=expected_margins);events=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
        for kind,count in [('head',expected_heads),('feature',expected_heads),('full_parameter_margin_gradient',expected_margins)]:
            for event in ['attempt','completed']:assert [r['ordinal'] for r in events if r['kind']==kind and r['event']==event]==list(range(1,count+1))
        item=dict(role=role,status=diag['status'],actual_finite_accepted=diag['actual_finite_accepted'],all_finite_candidates=candidates,final_candidate=candidates[-1] if candidates else None,actual_heads=expected_heads,actual_margin_derivatives=expected_margins,actual_QP_solves=diag['actual_QP_solves'],optimizer_iterations=diag['optimizer_iterations'],all_restored_full_tensors_exact=True,all_restored_original_argmax_exact=True,probabilities_and_log_probabilities_use_original_8eps_policy=True,unchanged_V166_control_replay_not_new_gain=diag['joint_restoration_attempts']==0,new_fits=0,permanent_updates=0);save(OUT/f'role{role}.json',item);roles.append(item)
    check_bindings(bindings);safe=all(r['actual_finite_accepted'] for r in roles);gain=any(r['final_candidate'] and r['final_candidate']['pure_errors_after']<r['final_candidate']['pure_errors_before'] for r in roles);summary=dict(status='V167_all_four_actual_candidates_original_gold_full_row_quality_costs_and_restored_parameters_reviewed',roles=roles,actual_new_heads=sum(r['actual_heads'] for r in roles),actual_new_complete_margin_derivatives=sum(r['actual_margin_derivatives'] for r in roles),actual_QP_solves=sum(r['actual_QP_solves'] for r in roles),actual_finite_proposals=sum(len(r['all_finite_candidates']) for r in roles),all_actual_finite_guards_passed=safe,any_actual_pure_classification_gain_vs_V164=gain,supports_separate_short_training_registration=safe and gain,official_calls_by_review=0,new_fits=0,permanent_updates=0,classification_mastery=False,quality_acceptance=False,root_goal_complete=False,source_sha256=bindings);save(OUT/'review.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['roles','source_sha256']}))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
