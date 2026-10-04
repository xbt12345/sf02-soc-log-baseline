"""Applicable resource-terminal review profile, integrated with experiment_review."""
import copy,sys,importlib.metadata
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha,check_bindings,ReviewError
from v138_runtime import save
OUT=ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001'
OLD=ROOT/'artifacts/v146_guarded_pair_training_20261001'
PLAN=ROOT/'training/review_policy/v155_guarded_full_gradient_sam_plan.json'
SOLVER=dict(max_updates=200,max_proposals=600,max_backtracks=20,initial_step=0.01,max_step=0.01,min_step=1e-8,
    shrink=0.5,grow=2.,armijo=1e-4,zero_gradient=1e-15)

def validate(p):
    expected=dict(version='V155-full-gradient-first-order-neighborhood',fits=6,folds=[0,1,2],arms=['A','B'],candidate='B',
        initialization='same_fold_V146_A_fixed_endpoint_for_both_arms',trainable='existing_second_layer_22528_only',new_parameters=0,
        loss_A='full_original_frequency_member_CE',loss_B='first_order_full_gradient_neighborhood_member_CE',
        radius='0.001_times_initial_parameter_L2_fixed_for_entire_fit_no_scan',solver=SOLVER,
        max_fit_full_gradients={'A':200,'B':400},max_fit_outer_gradient_attempts=200,
        preflight_full_gradients=9,preflight_dummy_gradients=1,
        direction='unit_L2_A_true_CE_gradient_B_first_order_shifted_CE_gradient_ignore_shift_derivative',
        proposal_proxy='current_iteration_epsilon_frozen_across_backtracks_L_theta_trial_plus_epsilon',
        accepted_trial='frozen_epsilon_proxy_Armijo_AND_unshifted_actual_all_correct_TRAIN_guard',
        endpoint='last_accepted_on_registered_budget_or_registered_no_feasible_step_no_checkpoint_selection',
        proxy_not_exact_ball_maximum=True,moving_epsilon_risk_monotonic_not_claimed=True,ordinary_CE_monotonic_required=False,
        preserve_input_class_mass=True,HELD_used_for_fitting_or_selection=False,automatic_repeat=False,automatic_issue_closure=False,
        old_budgets={'old_head_chain_fits':12,'old_head_additional':0,'V142_existing_fits':3,'V146_existing_fits':6,'new_head_fits':0},
        TRAIN_acceptance={'pure_errors':0,'M_errors':0,'S_by_fold':[22,6,28],'correct_row_new_errors':0,'last_distinct_states':5,'guard_entry':'training/v142_retention_check.py'},
        matched_effect_acceptance={'M_no_increase':True,'S_no_increase':True,'one_class_improves':True,'two_folds_improve':True,'outside_top3_S_no_increase':True},
        confirmation='none_automatic_new_confirmation_requires_separate_bound_plan_after_all_quality_gates',
        selection_labels='none_fixed_resource_terminal',evaluation_scope='previously_inspected_development',
        actual_compute_A_B_not_equal=True,new_independent_support_created=False)
    expected['initial_guard_containment']='actual_same_role_V146_A_correct_contains_V138_V140_V142_verified'
    for k,v in expected.items():
        if p.get(k)!=v:raise ReviewError('Unregistered first-order neighborhood factor: '+k)
    if p['task_quality']!=read(ROOT/'training/review_policy/v137_single_issue_plan.json')['task_adoption_quality']:raise ReviewError('Changed full task gates')
    registry=read(ROOT/'training/review_policy/history_cases.json')
    if p['risk_actions']!={z['id']:z['required_action'] for z in registry['risks']}:raise ReviewError('Unhandled historical risk')
    if p['hypothesis_scope']!='bounded_test_not_causal_proof_or_reproduction_of_stochastic_SAM':raise ReviewError('Unsupported qualification')
    check_bindings(p['evidence_sha256']);return p

def review():return validate(read(PLAN))

def adversaries(p):
    out=[]
    for name,k,value in [('radius scan','radius','fit_outer_selected_radius'),('false exact gradient','direction','true_ball_risk_gradient'),
        ('false moving proxy monotonic','moving_epsilon_risk_monotonic_not_claimed',False),('drop conflicts','preserve_input_class_mass',False),
        ('held answers','HELD_used_for_fitting_or_selection',True),('extra fits','fits',12),('early point selection','selection_labels','minimum_outer_error'),
        ('equal compute fiction','actual_compute_A_B_not_equal',False),('support fiction','new_independent_support_created',True)]:
        q=copy.deepcopy(p);q[k]=value
        try:validate(q)
        except ReviewError:out.append(dict(case=name,rejected=True))
        else:raise ReviewError('Accepted forbidden '+name)
    return out

def physical():
    files=set()
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name:
            q=Path(name).resolve()
            if q.suffix=='.pyc' and q.with_suffix('.py').is_file():q=q.with_suffix('.py')
            if q.is_file():files.add(q)
    return files

def seal(trainer,extras):
    review();assert not (OUT/'run_seal.json').exists()
    paths={PLAN,Path(trainer).resolve(),Path(__file__)}|set(extras)|physical()
    paths|={ROOT/z for z in read(OLD/'run_seal.json')['source_sha256']}
    paths.add(Path(sys.executable).resolve());paths.update((Path(torch.__file__).parent/'lib').glob('*.dll'))
    for n in ['python.exe','python3.dll','python311.dll','python312.dll','python313.dll']:
        q=Path(sys.base_prefix)/n
        if q.is_file():paths.add(q)
    bindings={q.relative_to(ROOT).as_posix() if q.is_relative_to(ROOT) else str(q):sha(q) for q in sorted(z.resolve() for z in paths)}
    save(OUT/'run_seal.json',dict(status='sealed_before_any_classifier_probe_or_update',trainer_path=Path(trainer).resolve().relative_to(ROOT).as_posix(),
        plan_sha256=sha(PLAN),source_sha256=bindings,python_version=sys.version,
        package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','pyarrow']},
        scope='Inherited V146 physical dependencies plus own sources/evaluator/runtime; OS/GPU driver not byte-snapshotted.'))

def require(trainer):
    p=review();s=read(OUT/'run_seal.json')
    if s['status']!='sealed_before_any_classifier_probe_or_update' or s['plan_sha256']!=sha(PLAN) or (ROOT/s['trainer_path']).resolve()!=Path(trainer).resolve():raise ReviewError('Seal/entry mismatch')
    # Existing experiment_review binding check integrated; legacy 25-epoch/6+12
    # profile is not mislabeled as applicable to this resource-terminal method.
    check_bindings(s['source_sha256'])
    if s['python_version']!=sys.version or s['package_versions']!={n:importlib.metadata.version(n) for n in s['package_versions']}:raise ReviewError('Changed runtime')
    return p

def endpoint(r):
    p=review();arm=r['arm'];factor=2 if arm=='B' else 1
    if r['status']!='fit_executed' or arm not in p['arms'] or r['fold'] not in p['folds'] or r['selected_by_score']:raise ReviewError('Wrong fixed resource endpoint')
    if not (0<=r['accepted_updates']<=r['outer_gradient_attempts']<=200 and r['full_gradient_evaluations']==factor*r['outer_gradient_attempts'] and r['accepted_updates']<=r['proposal_evaluations']<=600):raise ReviewError('Over budget or hidden gradients')
    if r['termination'] not in ['outer_gradient_budget','proposal_budget','accepted_update_budget','zero_gradient','no_feasible_step']:raise ReviewError('Unregistered stopping')
    return True
