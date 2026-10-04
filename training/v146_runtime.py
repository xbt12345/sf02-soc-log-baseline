"""One prospective guarded Armijo CE/auxiliary matched experiment."""
import copy,sys,importlib.metadata
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha,check_bindings,ReviewError
from v138_runtime import save
BASE=ROOT/'artifacts/v142_second_layer_training_20261001'
OUT=ROOT/'artifacts/v146_guarded_pair_training_20261001'
PLAN=ROOT/'training/review_policy/v146_guarded_pair_plan.json'
SOLVER=dict(max_gradients=200,max_proposals=600,max_updates=200,max_backtracks=20,initial_step=0.01,max_step=0.01,min_step=1e-8,shrink=0.5,grow=2.,armijo=1e-4,zero_gradient=1e-15)


def validate(p):
    def need(ok,msg):
        if not ok:raise ReviewError(msg)
    need(p['latest_actual']=='V142' and p['issue']=='FINE-SUPPORT-SUPERVISION','Wrong actual/issue')
    need(p['fits']==6 and p['arms']==['A','B'] and p['folds']==[0,1,2] and p['candidate']=='B','Wrong matched fit budget or candidate')
    need(p['initialization']=='same_fold_V142_fixed_endpoint' and p['trainable']=='existing_second_layer_only' and p['new_parameters']==0,'Wrong parameter/source scope')
    need(p['loss_A']=='original_frequency_member_CE' and p['loss_B']=='loss_A_plus_fixed_initial_lambda_restricted_SupCon','Unregistered loss')
    need(p['lambda_source']=='V144_legal_TRAIN_fixed_norm_ratio_0.1_no_scan' and p['temperature']==0.1,'Coefficient selection/adaptation')
    need(p['solver']==SOLVER and p['direction']=='true_objective_gradient_unit_L2_no_projection','Wrong optimizer or gradient semantics')
    need(p['accepted_trial']=='Armijo_total_objective_AND_actual_registered_TRAIN_classification_guard','Wrong step acceptance')
    need(not p['class_CE_monotonic_required'] and not p['HELD_used'] and not p['automatic_repeat'] and not p['automatic_issue_closure'],'Surrogate license, leakage or false closure')
    need(p['endpoint']=='last_accepted_on_budget_or_registered_no_feasible_step_no_checkpoint_selection','Checkpoint/endpoint choice')
    need(p['TRAIN_acceptance']==dict(pure_errors=0,M_errors=0,S_by_fold=[22,6,28],correct_row_new_errors=0,last_distinct_states=5,guard_entry='training/v142_retention_check.py'),'Lost mastered capability')
    need(p['matched_effect_acceptance']==dict(M_no_increase=True,S_no_increase=True,one_class_improves=True,two_folds_improve=True,outside_top3_S_no_increase=True),'Relaxed class/source gate')
    need(p['task_quality']==read(ROOT/'training/review_policy/v137_single_issue_plan.json')['task_adoption_quality'],'Changed original A0 task gate')
    need(p['old_budgets']==dict(LP_fits_used=12,LP_additional=0,V142_fits_used=3,V142_automatic_repeat=False,new_head_fits=0),'Reset prior budget')
    need(p['risk_actions']=={r['id']:r['required_action'] for r in read(ROOT/'training/review_policy/history_cases.json')['risks']},'Missing prior risks')
    need(p['new_risks']==dict(V145_failed_qualification_preserved=True,CE_rise_not_classification_failure=True,local_descent_not_finite_guarantee=True,known_578_S_direct_pair_coverage=0,aux_S_rows_by_fold=[34,18,26],all_original_classification_rows_retained=True,all_rejected_trials_logged=True),'Hidden evidence/compute')
    return True


def adversaries(p):
    items=[('extra fits',('fits',),12),('choose A',('candidate',),'A'),('held tuning',('HELD_used',),True),('reset LP',('old_budgets','LP_additional'),6),
        ('force class CE license',('class_CE_monotonic_required',),True),('weaken guard',('TRAIN_acceptance','correct_row_new_errors'),1),('projected gradient called true',('direction',),'projected'),
        ('best checkpoint',('endpoint',),'best_HELD'),('loss-only acceptance',('accepted_trial',),'loss_only'),('scan lambda',('lambda_source',),'best_lambda'),('auto repeat',('automatic_repeat',),True)]
    r=[]
    for label,path,value in items:
        q=copy.deepcopy(p);d=q
        for k in path[:-1]:d=d[k]
        d[path[-1]]=value
        try:validate(q)
        except (ValueError,KeyError):r.append(dict(case=label,rejected=True))
        else:raise ReviewError('Accepted forbidden '+label)
    validate(p);return r


def review():
    p=read(PLAN);validate(p);check_bindings(p['evidence_sha256']);return p


def seal(trainer,extras):
    review();old=read(BASE/'run_seal.json');check_bindings(old['source_sha256'])
    target=OUT/'run_seal.json'
    if target.exists():raise FileExistsError('Never overwrite seal')
    paths=set(extras)|{PLAN,Path(trainer).resolve(),Path(__file__).resolve(),ROOT/'training/experiment_review.py'}|{ROOT/p for p in old['source_sha256']}|{ROOT/p for p in read(PLAN)['evidence_sha256']};missing=[]
    for m in list(sys.modules.values()):
        name=getattr(m,'__file__',None)
        if name:
            q=Path(name).resolve()
            if q.suffix=='.pyc' and q.with_suffix('.py').is_file():q=q.with_suffix('.py')
            if q.is_file():paths.add(q)
            else:missing.append(str(q))
    paths.add(Path(sys.executable).resolve());paths.update((Path(torch.__file__).parent/'lib').glob('*.dll'))
    for n in ['python.exe','python3.dll','python311.dll','python312.dll','python313.dll']:
        q=Path(sys.base_prefix)/n
        if q.is_file():paths.add(q)
    bindings={q.relative_to(ROOT).as_posix() if q.is_relative_to(ROOT) else str(q):sha(q) for q in sorted(v.resolve() for v in paths)}
    save(target,dict(status='sealed_before_update',plan_sha256=sha(PLAN),trainer_path=Path(trainer).resolve().relative_to(ROOT).as_posix(),source_sha256=bindings,
        python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','tabm','numpy','pandas','scipy','pyarrow']},dynamic_nonphysical_metadata=sorted(set(missing)),
        scope='Loaded physical sources, binaries, input/fold/model/evaluator dependencies, Torch DLLs, Python runtime; OS/GPU driver not byte-snapshotted.'))


def require(trainer):
    p=review();s=read(OUT/'run_seal.json')
    if s['status']!='sealed_before_update' or s['plan_sha256']!=sha(PLAN) or (ROOT/s['trainer_path']).resolve()!=Path(trainer).resolve():raise ReviewError('Entry/plan mismatch')
    check_bindings(s['source_sha256'])
    if s['python_version']!=sys.version or s['package_versions']!={n:importlib.metadata.version(n) for n in s['package_versions']}:raise ReviewError('Runtime changed')
    return p


def endpoint(r):
    if r['status']!='fit_executed' or r['arm'] not in ['A','B'] or r['fold'] not in [0,1,2] or r['selected_by_score']:raise ReviewError('Invalid endpoint identity')
    if not 0<=r['accepted_updates']<=r['full_gradient_evaluations']<=200 or not r['accepted_updates']<=r['proposal_evaluations']<=600:raise ReviewError('Over budget')
    if r['termination'] not in ['gradient_budget','proposal_budget','accepted_update_budget','zero_gradient','no_feasible_step']:raise ReviewError('Wrong stopping')
    return True
