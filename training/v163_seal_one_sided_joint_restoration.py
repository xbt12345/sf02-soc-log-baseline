"""Seal uniform six-proposal joint QP restoration and cumulative costs."""
import ast,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
import v163_fixed_endpoint_one_sided_restoration_v2 as entry

def main():
    assert not entry.OUT.exists() and not entry.PLAN.exists()
    tail=ROOT/'artifacts/v162_cached_function_restoration_tail_20261002';old=read(tail/'run_seal.json');check_bindings(old['source_sha256'])
    summarypath=ROOT/'artifacts/v162_cached_tail_results_20261002/actual_result_summary.json';summary=read(summarypath);assert summary['cumulative_heads']==17334 and summary['cumulative_all_complete_derivatives']==422 and summary['new_fits']==summary['permanent_updates']==0;check_bindings(summary['source_sha256'])
    files={ROOT/k for k in old['source_sha256']}|{p for p in tail.rglob('*') if p.is_file()}|{ROOT/k for k in summary['source_sha256']}
    files|={summarypath,Path(__file__).resolve(),ROOT/'training/v163_fixed_endpoint_one_sided_restoration_v2.py',ROOT/'training/v163_one_sided_joint_restoration.py',ROOT/'docs/V163_ONE_SIDED_JOINT_FINITE_EXECUTION_PLAN_20261002.md',ROOT/'docs/V162_TAIL_ACTUAL_REVIEW_AND_V163_EXECUTION_ADDENDUM_20261002.md'}
    qualifiers=[('v163_one_sided_joint_restoration_qualification','qualification.json'),('v163_one_sided_nonlinear_fixture_qualification','review.json'),('v163_one_sided_entry_qualification_v2','qualification.json'),('v163_independent_inequality_regression_review','review.json')]
    for name,filename in qualifiers:
        folder=ROOT/f'artifacts/{name}_20261002';q=read(folder/filename);assert 'passed' in q['status'];assert q.get('official_heads',0)==q.get('official_gradients',0)==q.get('official_fits',0)==q.get('official_calls',0)==0;check_bindings(q['source_sha256'])
        files|={ROOT/k for k in q['source_sha256']}|{p for p in folder.rglob('*') if p.is_file()}
    for name in ['v162_saved_cached_restoration_tail_audit_v2','v162_independent_cached_restoration_tail_review','v162_independent_safe_progress_rate_review']:
        folder=ROOT/f'artifacts/{name}_20261002';assert folder.exists();files|={p for p in folder.rglob('*') if p.is_file()}|{ROOT/'training'/f'{name}.py'}
    previous=ROOT/'artifacts/v162_fixed_endpoint_finite_restoration_20261002';oldplan=read(ROOT/'training/review_policy/v162_fixed_endpoint_finite_restoration_contract.json');roles=[]
    for role,k,expected in [(0,10,15),(1,4,5),(2,10,1)]:
        last=sorted((previous/f'role{role}').glob('restoration*/all_measured_functions.json'))[-1];records=read(last);assert len(records)==expected;cached=[]
        for identity,metadata in records.items():
            fresh=previous/f'role{role}/fresh_normals/{identity}'
            if fresh.exists():spec=dict(metadata=(fresh/'input_binding.json').relative_to(ROOT).as_posix(),gradient=(fresh/'repeat0_gradient.npy').relative_to(ROOT).as_posix())
            else:
                matches=[v for v in oldplan['roles'][role]['cached_normals'] if read(ROOT/v['metadata'])['input_identity']==identity];assert len(matches)==1;spec=matches[0]
            origin=ROOT/spec['metadata'];assert read(origin)==metadata and np.load(ROOT/spec['gradient']).shape==(1060832,) and all(v['passed'] for v in read(origin.parent/'measurement_repeat_review.json').values());cached.append(spec)
        fit=read(entry.PRIOR/f'fold{role}_B/fit.json');margin_cap=2*(24-expected);roles.append(dict(role=role,OOF_chunks=k,deployment_chunks=12,head_cap=8*(k+12)+margin_cap,fresh_margin_gradient_cap=margin_cap,restoration_cap=6,finite_proposal_cap=6,QP_cap=6,endpoint_parameter_sha256=fit['endpoint_parameter_sha256'],cached_normals=cached,fixed_base_step=1/16,initial_saved_actual_failure=(entry.DIAG/f'role{role}/round0/probe4').relative_to(ROOT).as_posix(),target_gradients_already_repeated_at_same_endpoint=True))
    # Bind newly imported optimizer/SVD runtime files as well as old closure.
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).is_file():files.add(Path(name).resolve())
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    caps=dict(heads=582,features=582,fixed_error_target_gradients=0,full_original_class_gradients=0,margin_gradients=102,finite_proposals=18,restoration_solves=18,QP_solves=18,fits=0,permanent_updates=0)
    plan=dict(protocol=entry.PROTOCOL,allowed_entries=['training/v163_fixed_endpoint_one_sided_restoration_v2.py'],roles=roles,new_caps=caps,prior_actual_costs=dict(heads=17334,features=17334,full_original_class_gradients=368,fixed_error_target_gradients=12,margin_gradients=42,complete_parameter_derivatives_all_types=422),future_cumulative_actual_caps=dict(heads=17916,features=17916,full_original_class_gradients=368,fixed_error_target_gradients=12,margin_gradients=144,complete_parameter_derivatives_all_types=524),preserved_technical_head_cap=82174,preserved_technical_complete_derivative_cap=2426,single_factor='one-sided protected-margin restoration jointly constrained by both original target gradients',same_algorithm_and_step_for_all_roles=True,max_restorations_per_role=6,normal_cap_per_role=24,QP_and_restoration_are_same_outer_subproblem_not_double_cost=True,actual_QP_calls_count_entered_minimize=True,actual_internal_optimizer_iterations_saved=True,original_classification_and_numeric_gates_unchanged=True,full_displacement_Armijo_step=1.,all_original_mixed_rows_scored=True,paired_new_actual_blocker_normals_at_unchanged_origin=True,cached_normal_counts=[15,5,1],fixed_endpoint_pure_repairs_protected=True,full_deployment_and_joint_TRAIN_retention=True,all_candidate_outputs_saved_before_acceptance=True,restore_each_candidate_and_exit=True,no_new_target_gradient_fit_or_permanent_update=True,no_short_training_authority=True,source_sha256=bindings,quality_acceptance=False)
    assert sum(r['head_cap'] for r in roles)==582 and sum(r['fresh_margin_gradient_cap'] for r in roles)==102 and 17916<=82174 and 524<=2426
    entry.OUT.mkdir();entry.save(entry.OUT/'pre_registration_bindings.json',dict(source_sha256=bindings,official_calls=0));entry.save(entry.PLAN,plan);bindings[entry.PLAN.relative_to(ROOT).as_posix()]=sha(entry.PLAN)
    entry.save(entry.OUT/'run_seal.json',dict(status='three_role_one_sided_joint_restoration_sealed_before_calls',protocol=entry.PROTOCOL,allowed_entries=plan['allowed_entries'],plan_sha256=sha(entry.PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings))
    entry.require();entry.save(entry.OUT/'registration.json',dict(status='V163_one_sided_joint_restoration_registered',physical_sources=len(bindings),new_caps=caps,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,run_seal_sha256=sha(entry.OUT/'run_seal.json')));print(json.dumps(dict(status='V163_one_sided_joint_restoration_sealed',physical_sources=len(bindings),new_caps=caps)))

if __name__=='__main__':main()
