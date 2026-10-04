"""New population source seal for nine complete OOF pipelines, not old fits resumed."""
import importlib.metadata,json,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v158_fusion_contract import STAGES

OUT=ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001'
PLAN=ROOT/'training/review_policy/v158_complete_nested_trial_v2_plan.json'
QUAL=ROOT/'artifacts/v158_nested_fusion_qualification_20261001'

def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def expected_contract():
    expected=dict(version='V158-complete-nested-member-fusion-logging-v2',outer_folds=[0,1,2],inner_folds=[0,1,2],
        split='unchanged_V128_root_hash_no_labels_no_reroll',stages=list(STAGES),base_pipelines=9,
        base_stage_fits=45,legacy_N1_fits=9,supervised_base_fits=54,fusion_fits_max=6,total_new_fits_max=60,
        base_full_network_epochs=100,base_full_network_gradient_updates=35600,
        dense_stage_gradients_per_fit=200,dense_stage_updates_per_fit=200,
        base_dense_gradient_cap=7200,base_dense_update_cap=7200,
        base_initialization='fresh_seed10201_no_supervised_prior_weights_or_caches',
        original_frequency=True,outer_labels_used_for_fit_guard_or_selection=False,
        query_labels_used_in_base_supervision_or_guard=False,base_phase_selection='last_registered_budget_state_no_score_selection',
        fusion_condition_width=527,fusion_text_sketch_width=32,fusion_hidden_width=16,
        fusion_member_index_input=False,fusion_A_condition_input=False,fusion_B_condition_input=True,
        fusion_original_mean_probability_CE=True,fusion_gradients_per_fit_max=200,
        fusion_proposals_per_fit_max=600,fusion_updates_per_fit_max=200,
        fusion_initialization_seed=15801,fusion_initial_shared_parameters='B_first11_hidden_columns_bias_and_zero_output_copy_A',
        fusion_A_parameters=208,fusion_B_parameters=8640,
        fusion_solver=dict(initial_step=.01,max_step=.01,min_step=1e-8,shrink=.5,grow=2.,armijo=1e-4,max_backtracks=20,zero_gradient=1e-15),
        fusion_guard='all_same_role_V146_A_correct_original_FIT_rows_plus_v142_joint_retention_every_accepted_state',
        fusion_endpoint='last_accepted_on_registered_budget_or_no_feasible_step_no_outer_checkpoint_selection',
        fusion_activation='separate_implemented_entry_and_all_actual_OOF_sources_sealed_before_forward_gradient_or_update',
        legacy_probability_representation='softmax_of_original_three_OVA_scores_preserves_argmax_not_claimed_calibrated',
        expert_count=17,legacy_full_gradient_cap=9000,legacy_iterations_per_fit_max=1000,
        initialization_prior='legal_FIT_quarter_closed_form_preserving_bound_fixed_before_OOF',
        execution_entries=['training/v158_nested_base_train_v2.py','training/v158_legacy_nested_train_v2.py'],
        full_network_row_checkpoints=[0,1,2,5,20,40,60,80,96,97,98,99,100],
        dense_row_checkpoints=[0,1,20,50,100,150,200],dense_last_distinct_row_states=5,
        every_step_class_and_source_statistics=True,canonical_floor_required=True,
        automatic_repeat=False,automatic_confirmation=False,quality_acceptance=False)
    return expected

def validate(p,verify_bindings=True):
    for k,v in expected_contract().items():
        if p.get(k)!=v:raise ValueError('Unregistered complete nested trial: '+k)
    q=read(QUAL/'qualification.json')
    floors=read(ROOT/'artifacts/v158_legacy_expert_bank_qualification_20261001/actual_inner_canonical_floors.json')['roles']
    import math
    expected_roles=[]
    for r,floor in zip(q['roles'],floors):
        n=r['fit_locals'];nb=math.ceil(n/256);k=math.ceil(n/2048)
        expected_roles.append(dict(outer_fold=r['outer_fold'],excluded_inner=r['excluded_inner'],fit_locals=n,
            fit_rows=r['fit_rows'],fit_class_mass=[0,r['fit_M'],r['fit_S']],canonical_floor=floor['canonical_floor'],
            full_network_gradient_updates=100*nb,full_network_forward_cap=201*nb+89,
            LBFGS_stage_forward_cap=402*k+12,Armijo_stage_forward_cap=1402*k+12))
    if p.get('role_budgets')!=expected_roles:raise ValueError('Incomplete original-mass role/call budget')
    legacy=read(ROOT/'artifacts/v158_legacy_expert_bank_qualification_20261001/legacy_full_nested_roles_and_cost.json')['roles']
    if p.get('legacy_role_budgets')!=legacy:raise ValueError('Wrong full-population N1 roles')
    registry=read(ROOT/'training/review_policy/history_cases.json')
    if p.get('risk_actions')!={r['id']:r['required_action'] for r in registry['risks']}:raise ValueError('Unhandled historical risk')
    if p.get('task_quality')!=read(ROOT/'training/review_policy/v137_single_issue_plan.json')['task_adoption_quality']:raise ValueError('Changed full task gates')
    if verify_bindings:
        check_bindings(p['source_sha256']);check_bindings(q['source_sha256'])
        check_bindings(read(QUAL/'pre_execution_bindings.json')['source_sha256'])
    if q['planned_costs']['formal_total_fits_max']!=51 or q['official_classifier_forward_calls']!=0:raise ValueError('Wrong source qualification')
    # Preserve the executed 51-fit preliminary audit; the new expert decision
    # adds nine new legal full-population fits, with its own prospective audit.
    bank=read(ROOT/'artifacts/v158_legacy_expert_bank_qualification_20261001/qualification.json')
    if verify_bindings:check_bindings(bank['source_sha256'])
    if bank['total_planned_fits']!=60 or bank['new_fits']!=0:raise ValueError('Legacy bank qualification changed')
    return p

def review():return validate(read(PLAN))

def physical():
    paths={Path(sys.executable).resolve()}
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if not value:continue
        q=Path(value).resolve()
        if q.suffix=='.pyc' and q.with_suffix('.py').is_file():q=q.with_suffix('.py')
        if q.is_file():paths.add(q)
    import torch
    paths.update(Path(torch.__file__).parent.joinpath('lib').glob('*.dll'))
    for name in ['numpy','scipy','sklearn']:
        package=__import__(name);root=Path(package.__file__).parent
        for base in [root,root.parent/(name+'.libs')]:
            if base.exists():paths.update(base.rglob('*.dll'));paths.update(base.rglob('*.pyd'))
    for name in ['python.exe','python3.dll','python311.dll','python312.dll','python313.dll']:
        q=Path(sys.base_prefix)/name
        if q.is_file():paths.add(q)
    return paths

def seal(entry,extra):
    p=review();target=OUT/'run_seal_v2.json';assert not target.exists()
    files=physical()|set(extra)|{PLAN,Path(entry).resolve(),Path(__file__).resolve()}|{ROOT/k for k in p['source_sha256']}
    save(target,dict(status='sealed_before_any_official_classifier_call_or_update',entry=Path(entry).resolve().relative_to(ROOT).as_posix(),
        allowed_entries=p['execution_entries'],
        plan_sha256=sha(PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','tabm','numpy','scipy','pandas','pyarrow']},
        source_sha256={q.relative_to(ROOT).as_posix() if q.is_relative_to(ROOT) else str(q):sha(q) for q in sorted(z.resolve() for z in files)},
        epoch_project_source_sha256={q.relative_to(ROOT).as_posix():sha(q) for q in sorted(z.resolve() for z in files)
            if q.is_relative_to(ROOT) and q.suffix=='.py' and q.relative_to(ROOT).parts[0]=='training'},
        driver_scope='OS and GPU driver are not physical byte snapshots'))

def require_epoch_sources():
    s=read(OUT/'run_seal_v2.json')
    if s['plan_sha256']!=sha(PLAN):raise ValueError('Plan changed during epoch')
    check_bindings(s['epoch_project_source_sha256'])

def require(entry):
    p=review();s=read(OUT/'run_seal_v2.json')
    actor=Path(entry).resolve().relative_to(ROOT).as_posix()
    if s['status']!='sealed_before_any_official_classifier_call_or_update' or s['plan_sha256']!=sha(PLAN) or s['allowed_entries']!=p['execution_entries'] or actor not in s['allowed_entries'] or s['source_sha256'].get(actor)!=sha(entry):raise ValueError('Nested trial seal/entry changed')
    if s['python_version']!=sys.version or s['package_versions']!={n:importlib.metadata.version(n) for n in s['package_versions']}:raise ValueError('Runtime changed')
    check_bindings(s['source_sha256']);return p
