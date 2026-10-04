"""Versioned SF02 review protocol; historic experiment_review stays immutable.

Six fixed class-boundary fits with finite feasible stopping, never disguised
as the historic 25-epoch plus confirmation experiment.
"""
import importlib.metadata,json,sys
from pathlib import Path
from experiment_review import ROOT,ReviewError,read,sha,check_bindings,evaluate_primary,align_predictions,class_counts

PROTOCOL='V159-six-fixed-class-boundary-finite-descent-v2'
STOPS={'gradient_iteration_budget','accepted_update_budget','proposal_budget','zero_direction_no_automatic_fallback','no_strict_common_descent_no_automatic_fallback','no_feasible_step'}

def review_plan(plan):
    check_bindings(plan['source_sha256']);violations=[]
    registry=read(ROOT/plan['risk_registry'])
    expected=dict(protocol=PROTOCOL,candidate_count=1,candidate_module='training/v159_current_input_boundary_v3.py',arms=['A','B'],seed=15901,
        activation_entries=['training/v159_boundary_train_v2.py','training/v159_boundary_evaluate_v2.py'])
    for k,v in expected.items():
        if plan.get(k)!=v:violations.append(k)
    if plan.get('endpoint')!='last accepted state at registered budget/finite-guard stop; no label-selected checkpoint':violations.append('fixed_actual_endpoint')
    if plan['total_future_caps'].get('fits')!=6 or plan['total_future_caps'].get('full_class_gradients')!=2412 or plan['total_future_caps'].get('classifier_forward_chunks')!=77832:violations.append('bounded_cost')
    if plan.get('shape',{}).get('parameters')!=1060832 or plan.get('shape',{}).get('outputs')!=3:violations.append('complete_three_class_function')
    if plan['solver']['max_step']!=1 or plan['solver']['max_backtracks']!=40:violations.append('finite_solver_budget')
    if plan['probability_base']['floor']!=1e-12 or plan['probability_base']['origin_probability_tolerance']!=3e-12:violations.append('fixed_numerical_policy')
    for f,b in enumerate(plan['role_call_budgets']):
        k=(b['locals']+2047)//2048
        required=dict(role=f,chunks=k,preflight_classifier_cap_A=6*k+24,preflight_classifier_cap_B=2*k+24,preflight_full_class_gradients_A=4,preflight_full_class_gradients_B=0,fit_classifier_cap_per_arm=1603*k+12,fit_full_class_gradient_cap_per_arm=400,fit_gradient_iterations_per_arm=200,proposals_per_arm=600,accepted_updates_per_arm=200,evaluation_classifier_cap_per_arm=72+k)
        if any(b.get(key)!=value for key,value in required.items()) or sum(b['original_class_mass'])!=b['original_rows'] or b['original_class_mass'][0]!=0:violations.append('original_role_and_cost:'+str(f))
    if len(plan['role_call_budgets'])!=3 or [b['chunks'] for b in plan['role_call_budgets']]!=[10,4,10]:violations.append('complete_three_roles')
    if not plan['OOF_retention_refinement']['all_accepted_deployment_correct_and_joint_scopes_unchanged'] or not plan['OOF_retention_refinement']['all_mixed_rows_still_in_loss_and_score']:violations.append('population_and_old_abilities')
    for risk in registry['risks']:
        if plan['risk_actions'].get(risk['id'])!=risk['required_action']:violations.append('unhandled_risk:'+risk['id'])
    if plan['task_quality']['full_original_rows']!=2056871 or plan['task_quality']['classes']!=[0,1,2]:violations.append('complete_task_population')
    return dict(status='eligible_for_fixed_finite_trial' if not violations else 'blocked_before_official_functions',plan_review_passed=not violations,violations=violations,quality_acceptance=False,model_promoted=False)

def seal_run(plan_path,trainer_path,source_paths,destination):
    plan_path,trainer_path,destination=map(Path,(plan_path,trainer_path,destination));p=read(plan_path)
    if not review_plan(p)['plan_review_passed']:raise ReviewError('New finite protocol plan failed')
    actor=trainer_path.resolve().relative_to(ROOT).as_posix()
    if destination.exists() or actor not in p['activation_entries']:raise ReviewError('Existing seal or unregistered entry')
    files=set(Path(x).resolve() for x in source_paths)|{plan_path.resolve(),trainer_path.resolve(),Path(__file__).resolve(),ROOT/'training/experiment_review.py'}
    if any(not f.is_file() for f in files):raise ReviewError('Incomplete physical execution sources')
    bindings={f.relative_to(ROOT).as_posix() if f.is_relative_to(ROOT) else str(f):sha(f) for f in sorted(files)}
    result=dict(status='sealed_before_any_official_boundary_classifier_feature_gradient_or_update',protocol=PROTOCOL,allowed_entries=p['activation_entries'],plan_path=plan_path.resolve().relative_to(ROOT).as_posix(),plan_sha256=sha(plan_path),
        python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings,quality_acceptance=False,model_promoted=False,driver_scope='OS and GPU driver not physically snapshotted')
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return result

def require_run_seal(seal_path,current_trainer):
    if not Path(seal_path).is_file():raise ReviewError('Missing new-protocol seal')
    s=read(seal_path);actor=Path(current_trainer).resolve().relative_to(ROOT).as_posix()
    if s.get('status')!='sealed_before_any_official_boundary_classifier_feature_gradient_or_update' or s.get('protocol')!=PROTOCOL or actor not in s.get('allowed_entries',[]):raise ReviewError('Wrong seal protocol or entry')
    if s['source_sha256'].get(actor)!=sha(current_trainer):raise ReviewError('Changed actual entry')
    check_bindings(s['source_sha256']);path=ROOT/s['plan_path'];p=read(path)
    if s['plan_sha256']!=sha(path) or s['allowed_entries']!=p['activation_entries'] or not review_plan(p)['plan_review_passed']:raise ReviewError('Plan changed or no longer qualified')
    if s['python_version']!=sys.version or s['package_versions']!={n:importlib.metadata.version(n) for n in s['package_versions']}:raise ReviewError('Actual runtime changed')
    return p

def require_checkpoint(plan,r):
    if r.get('status')!='V159_boundary_fit_executed' or r.get('selected_by_score') or r.get('fold') not in [0,1,2] or r.get('arm') not in ['A','B']:raise ReviewError('Wrong actual fit or selected endpoint')
    iterations,gradients,proposals,updates=[r.get(k,-1) for k in ['gradient_iterations','full_class_gradients','proposal_evaluations','accepted_updates']]
    if not 0<=updates<=iterations<=200 or gradients!=2*iterations or not updates<=proposals<=600 or r.get('termination') not in STOPS:raise ReviewError('Incomplete or exceeded finite budget')
    stop=r['termination']
    if (stop=='gradient_iteration_budget' and iterations!=200) or (stop=='accepted_update_budget' and updates!=200) or (stop=='proposal_budget' and proposals!=600):raise ReviewError('Budget stopping reason inconsistent with actual cost')
    cap=plan['role_call_budgets'][r['fold']]['fit_classifier_cap_per_arm'];c=r['counts']
    if not c['head_attempts']==c['head_completed']==c['feature_attempts']==c['feature_completed']<=cap or not c['gradient_attempts']==c['gradient_completed']==gradients:raise ReviewError('Actual attempted/completed functions do not reconcile')
    if len(r['last5'])!=min(5,updates) or len({i['parameter_sha256'] for i in r['last5']})!=len(r['last5']):raise ReviewError('Fabricated or duplicate last window')
    if r['last5'] and r['last5'][-1]['parameter_sha256']!=r['endpoint_parameter_sha256']:raise ReviewError('Endpoint is not last actually accepted state')
    return dict(endpoint_review_passed=True,numerical_convergence_claimed=False,quality_acceptance=False)
