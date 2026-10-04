"""Refuse wrong final point, stale predictions, repeats, drift and exhausted caps."""
import copy,json,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import assign,restore,tensor_hash
from v168_decision_floor_execution_review import prospective_plan
import v168_decision_floor_diagnostic_v2 as entry
import v167_trial_point_measurement as measurement

OUT=ROOT/'artifacts/v168_decision_floor_adversarial_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();q=ROOT/'artifacts/v168_decision_floor_lifecycle_qualification_v2_20261002/qualification.json';value=read(q);check_bindings(value['source_sha256']);paths={ROOT/p for p in value['source_sha256']}|{Path(__file__).resolve(),q};bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));plan=prospective_plan();spec=plan['role_spec'];ctx=entry.load_context(1);state=torch.load(entry.PRIOR/'role1/endpoint.pt',map_location='cpu',weights_only=True)['state'];records,gradients,u,proof,goal,tie_rows=entry.cached_inputs(plan,state,ctx);model=entry.CurrentInputBoundary().cpu();model.load_state_dict(state);base=tuple(p.detach().clone() for p in model.parameters());origin=tensor_hash(state);trial_path=ROOT/plan['actual_failed_trial_path'];actual_q={scope:np.load(trial_path/f'{scope}_q.npy') for scope in ['OOF','deployment']};actual_logs={scope:np.load(trial_path/f'{scope}_logq.npy') for scope in actual_q};origin_q={scope:np.load(entry.PRIOR/f'role1/endpoint/{scope}_q.npy') for scope in actual_q};origin_logs={scope:np.load(entry.PRIOR/f'role1/endpoint/{scope}_logq.npy') for scope in actual_q};synthetic=np.ones(1060832);cases=[]
    for mode in ['wrong_final_parameter_identity_before_measure','stale_origin_outputs_not_final_trial_outputs','gradient_repeat_mismatch','temporary_parameter_drift','zero_margin_budget_before_head']:
        assign(model,base,u,1.);trial=tensor_hash(model.state_dict());assert trial==plan['actual_failed_trial_parameter_sha256'];folder=OUT/mode;folder.mkdir();budget=copy.deepcopy(spec)
        if mode=='zero_margin_budget_before_head':budget['margin_gradient_cap']=0
        counter=entry.TrialCounter(model,budget,folder/'mock_calls.jsonl',origin);calls=0
        def mock_inputs(observed_ctx,scope,chunk):return scope,chunk
        def mocked_measure(observed_model,scope,chunk,query,truth,rival):
            nonlocal calls
            calls+=1
            for kind in ['head','feature']:counter.before(kind);counter.after(kind)
            gradient=synthetic.copy()
            if mode=='gradient_repeat_mismatch' and calls==2:gradient[-1]+=.1
            if mode=='temporary_parameter_drift':
                with torch.no_grad():next(observed_model.parameters()).flatten()[0].add_(1e-6)
            return dict(gradient=gradient,q=actual_q[scope][chunk].copy(),logq=actual_logs[scope][chunk].copy(),margin=float(actual_logs[scope][chunk[query],truth]-actual_logs[scope][chunk[query],rival]))
        advertised=entry.parameter_hash(state,u*.5,1.) if mode=='wrong_final_parameter_identity_before_measure' else trial;expected_q=origin_q if mode=='stale_origin_outputs_not_final_trial_outputs' else actual_q;expected_logs=origin_logs if mode=='stale_origin_outputs_not_final_trial_outputs' else actual_logs;failure=None
        try:
            with patch.object(measurement,'inputs',mock_inputs),patch.object(measurement,'measure',mocked_measure):measurement.measure_all(model,ctx,records,counter,folder/'normals',origin,advertised,0,origin_logs,expected_logs,expected_q)
        except (AssertionError,RuntimeError) as error:failure=dict(error_type=type(error).__name__,error=str(error))
        finally:restore(model,base);assert tensor_hash(model.state_dict())==origin
        assert failure is not None;expected_calls=0 if mode in ['wrong_final_parameter_identity_before_measure','zero_margin_budget_before_head'] else 1 if mode=='temporary_parameter_drift' else 2;assert calls==expected_calls and counter.margin_attempts==counter.margin_completed==expected_calls
        cases.append(dict(mode=mode,refused=True,mocked_head_calls=calls,complete_original_CPU_state_restored=True,failure=failure));counter.close()
    check_bindings(bindings);entry.save(OUT/'qualification.json',dict(status='V168_final_tie_measurement_point_stale_output_repeat_drift_and_budget_refusals_qualified',cases=cases,head_and_gradient_values_are_cached_or_synthetic=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,physical_run_seal_still_required=True,execution_authority=False,source_sha256=bindings));print(json.dumps(dict(status='V168_decision_floor_adversarial_qualified',cases=len(cases),official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
