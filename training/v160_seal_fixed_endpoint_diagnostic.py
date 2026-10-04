"""Seal complete fixed-endpoint diagnostic before official model calls."""
import ast,importlib.metadata,json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v160_fixed_endpoint_diagnostic_v3 import OUT,PLAN,PRIOR,PROTOCOL,save,require

def main():
    assert not OUT.exists() and not PLAN.exists()
    qualifications=[ROOT/'artifacts/v160_active_direction_numeric_qualification_v5_20261002/qualification.json',ROOT/'artifacts/v160_margin_normal_synthetic_qualification_20261002/qualification.json',ROOT/'artifacts/v160_diagnostic_entry_synthetic_qualification_v3_20261002/qualification.json']
    files={Path(__file__).resolve(),ROOT/'training/v160_fixed_endpoint_diagnostic_v3.py',ROOT/'training/v160_active_margin_direction_v5.py',ROOT/'training/v160_margin_normal.py',ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'docs/V160_FIXED_ENDPOINT_FINITE_DIAGNOSTIC_PLAN_20261002.md'}
    for path in qualifications:
        q=read(path);assert 'passed' in q['status'] and q.get('official_calls',q.get('official_heads',0))==0
        check_bindings(q['source_sha256']);files.add(path);files|={ROOT/k for k in q['source_sha256']}
    previous=ROOT/'artifacts/v159_evaluation_cached_continuation_20261002/run_seal.json';prior_seal=read(previous)
    check_bindings(prior_seal['source_sha256']);files.add(previous);files|={ROOT/k for k in prior_seal['source_sha256']}
    files|={p for p in PRIOR.rglob('*') if p.is_file()}
    # Preserve every executed numeric failure and qualification used in selection.
    files|={p for p in ROOT.joinpath('artifacts').glob('v160*/*') if p.is_file()}
    files|=set(ROOT.joinpath('training').glob('v160*.py'))
    cumulative=read(ROOT/'artifacts/v159_complete_result_records_20261002/actual_result_summary.json')
    files.add(ROOT/'artifacts/v159_complete_result_records_20261002/actual_result_summary.json')
    roles=[]
    for role,k in enumerate([10,4,10]):
        folder=PRIOR/f'fold{role}_B';fit=read(folder/'fit.json')
        direction=json.loads((folder/'directions.jsonl').read_text(encoding='utf-8').strip().splitlines()[-1]);last=json.loads((folder/'proposals.jsonl').read_text(encoding='utf-8').strip().splitlines()[-1])
        assert direction['parameter_sha256']==fit['endpoint_parameter_sha256'] and last['iteration']==direction['iteration'] and not last['accepted'] and last['OOF_stats']['protected_regressions']==2
        roles.append(dict(role=role,OOF_chunks=k,deployment_chunks=12,head_cap=64*k+804,feature_cap=64*k+804,class_gradient_cap=2,margin_gradient_cap=48,margin_normal_cap=24,QP_solve_cap=3,finite_proposal_cap=61,endpoint_path=(folder/'endpoint.pt').relative_to(ROOT).as_posix(),endpoint_file_sha256=sha(folder/'endpoint.pt'),endpoint_parameter_sha256=fit['endpoint_parameter_sha256'],last_failed_proposal=last,last_direction_record=direction))
    assert sum(r['head_cap'] for r in roles)==3948
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    free=shutil.disk_usage(ROOT).free;assert free>=12*1024**3
    plan=dict(protocol=PROTOCOL,activation_entries=['training/v160_fixed_endpoint_diagnostic_v3.py'],fixed_roles=[0,1,2],roles=roles,new_caps=dict(heads=3948,features=3948,class_gradients=6,margin_gradients=144,finite_proposals=183,QP_solves=9,fits=0,permanent_updates=0),
      prior_actual_V159_lineage=dict(heads=14586,features=14586,class_gradients=362,margin_gradients=0,completed_fits=6,accepted_updates=165),future_cumulative_actual_V159_V160_lineage_caps=dict(heads=18534,features=18534,class_gradients=368,margin_gradients=144,completed_fits=6,accepted_updates=165),prior_technical_head_cap=78226,new_technical_head_cap=82174,prior_technical_class_gradient_cap=2420,new_technical_class_gradient_cap=2426,new_margin_gradient_cap=144,
      preserved_model='training/v159_current_input_boundary_v3.py',preserved_numeric_policy='training/v159_float64_repeat_policy_v2.py',solver_module='training/v160_active_margin_direction_v5.py',solver=dict(common_class_scale_only=True,positive_normal_row_scale_cone_preserving=True,direct_original_basis=True,relative_rank_cutoff='eps * dual_variable_count',residual_refinements=3,max_active_set_iterations=256,no_Gram_free_solve=True,no_damping=True,no_QP_success_finite_safety_claim=True,original_unit_certificate=True,independent_fsum_sign_certificate=True,cancellation_residue_stop_without_global_infeasibility_claim=True),
      baseline_policy='One complete M and one complete S gradient at the same endpoint. Save both entire vectors. Compare risk/q/logq across those same-point class passes and endpoint saved q. Not claimed as two repeated measurements of each class gradient.',margin_policy='Each selected legal original blocker margin gradient measured twice on exactly its protection chunk; save both complete vectors before fixed 8eps segment/norm/sign repeat checks.',
      probes=dict(control='exact previously registered final failed step, original MGDA direction',active_rounds=3,backtracks_per_round=20,step_sequence='2**(-j), j=0..19',next_constraints='all newly violated protection margins from the last actually executed failed candidate',normal_cap_stop_without_row_selection=24,all_outputs_saved_before_guards=True,no_outer_truth_for_direction=True,no_warm_restart=True,no_permanent_update=True,all_exit_paths_restore=True),
      quality_acceptance=False,free_disk_bytes_before_registration=free,source_sha256=bindings)
    OUT.mkdir();save(OUT/'pre_registration_bindings.json',dict(status='before_any_official_diagnostic_model_feature_or_gradient_call',source_sha256=bindings));save(PLAN,plan)
    bindings[PLAN.relative_to(ROOT).as_posix()]=sha(PLAN)
    seal=dict(status='sealed_fixed_endpoint_diagnostic_before_official_calls',protocol=PROTOCOL,allowed_entries=plan['activation_entries'],plan_sha256=sha(PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings)
    save(OUT/'run_seal.json',seal);require();save(OUT/'registration.json',dict(status='fixed_endpoint_diagnostic_sealed_zero_official_calls',seal_sha256=sha(OUT/'run_seal.json'),physical_sources=len(bindings),new_caps=plan['new_caps'],official_heads=0,official_features=0,official_class_gradients=0,official_margin_gradients=0,official_fits=0,permanent_updates=0,quality_acceptance=False))
    print(json.dumps(dict(status='fixed_endpoint_diagnostic_sealed_before_calls',physical_sources=len(bindings),new_caps=plan['new_caps'])))

if __name__=='__main__':main()
