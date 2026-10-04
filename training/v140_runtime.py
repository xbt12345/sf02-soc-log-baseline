"""New objective review adapter; immutable historical execution remains intact."""
import copy
import sys
import importlib.metadata
from pathlib import Path
from experiment_review import ROOT, read, sha, check_bindings, ReviewError
from v138_runtime import OUT as OLD, save

PLAN = ROOT/'training/review_policy/v140_ensemble_training_plan.json'
OUT = ROOT/'artifacts/v140_ensemble_training_round2_20261001'


def validate(p):
    def need(ok, reason):
        if not ok: raise ReviewError(reason)
    need(p['issue']=='TRAIN-PURE-READOUT' and p['round']==2, 'Keep first issue and second round')
    need(p['budget']=={'old_fits':6,'new_fits_max':6,'chain_fits_max':12,'chain_rounds_max':2,
                      'gradient_evaluations_per_fit':200,'accepted_updates_per_fit':200}, 'No budget reset/extension')
    need(p['folds']==[0,1,2] and p['arms']==['C','E'] and p['candidate']=='E', 'No candidate/fold cherry-picking')
    need(p['changed_factor']=='member_mean_CE_vs_mean_probability_CE', 'Single registered factor')
    need(p['start']=='same_fold_V138_H_L_endpoint' and p['trainable_parameters']==7677, 'No initialization/capacity confound')
    need(p['solver']=={'name':'LBFGS','dtype':'float64','max_iter_per_call':1,'max_eval_per_call':200,'lr':1,'history_size':20,
                       'tolerance_grad':1e-7,'tolerance_change':1e-9,'line_search_fn':'strong_wolfe'}, 'Unreviewed numerical solver')
    need(p['original_TRAIN_frequency'] and not p['HELD_used_for_training_or_selection'], 'No label/frequency leakage')
    need(p['selection']=='last_accepted_state_budget_or_numerical_no_change', 'No intermediate selection')
    need(p['old_C_margin_eligible'] is False and p['old_C_margin_executed'] is False, 'No fabricated old certificate')
    need(p['registered_guard_new_errors_max']==0 and p['all_H_L_correct_pure_new_errors_max']==0, 'No relaxed retention')
    need(p['mastery']=={'pure_M_errors':0,'pure_S_errors':0,'full_M_errors':0,
                       'full_S_errors_by_fold':[22,6,28],'distinct_accepted_window':5}, 'No relaxed mastery')
    need(p['task_adoption_quality']==read(ROOT/'training/review_policy/v137_single_issue_plan.json')['task_adoption_quality'], 'No weakened task gate')
    history=read(ROOT/'training/review_policy/history_cases.json')
    need(p['risk_actions']=={r['id']:r['required_action'] for r in history['risks']}, 'Unhandled historical risk')
    need(p['inspected_development_not_blind'] and p['collusion_risk_reported'] and not p['automatic_more_fits'], 'No transfer fiction/automatic expansion')
    return True


def adversaries(p):
    cases=[('budget reset',('budget','old_fits'),0),('extra fits',('budget','new_fits_max'),7),
           ('held labels',('HELD_used_for_training_or_selection',),True),('late checkpoint',('selection',),'best_loss'),
           ('third issue',('issue',),'SOURCE-TRANSFER'),('forged LP eligibility',('old_C_margin_eligible',),True),
           ('guard relaxation',('registered_guard_new_errors_max',),1),('pure retention relaxation',('all_H_L_correct_pure_new_errors_max',),1),
           ('single member necessity',('mastery','pure_S_errors'),1),('hide collusion risk',('collusion_risk_reported',),False),
           ('pick arm after score',('candidate',),'C'),('no blind honesty',('inspected_development_not_blind',),False)]
    result=[]
    for name,path,value in cases:
        q=copy.deepcopy(p); dest=q
        for key in path[:-1]: dest=dest[key]
        dest[path[-1]]=value
        try: validate(q)
        except (ValueError,KeyError): result.append({'case':name,'rejected':True})
        else: raise ReviewError('Counterexample accepted: '+name)
    validate(p)
    return result


def review():
    p=read(PLAN); validate(p); check_bindings(p['evidence_sha256']); return p


def seal_run(trainer, extras):
    p=review(); target=OUT/'run_seal.json'
    if target.exists(): raise FileExistsError(target)
    # Transitive historical sources, input mappings, official data and all caches.
    old=read(OLD/'run_seal.json'); check_bindings(old['source_sha256'])
    paths={PLAN,Path(trainer).resolve(),Path(__file__).resolve(),ROOT/'training/experiment_review.py'}|set(extras)
    paths.update(ROOT/rel for rel in old['source_sha256'])
    paths.update(ROOT/rel for rel in p['evidence_sha256'])
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name:
            q=Path(name).resolve()
            if q.is_file() and q.suffix=='.py' and q.is_relative_to(ROOT) and not q.relative_to(ROOT).parts[0].startswith('.venv'):
                paths.add(q)
    bindings={q.resolve().relative_to(ROOT).as_posix():sha(q) for q in sorted(paths)}
    packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','pyarrow','tabm','torch']}
    save(target,{'status':'sealed_before_fit','version':'V140-round2-new-objective','trainer_path':Path(trainer).resolve().relative_to(ROOT).as_posix(),
                 'plan_sha256':sha(PLAN),'source_sha256':bindings,'packages':packages,'quality_acceptance':False,
                 'review_adapter':'experiment_review.check_bindings with reviewed non-epoch objective/budget dispatch'})


def require_run_seal(trainer):
    p=review(); s=read(OUT/'run_seal.json')
    if s['status']!='sealed_before_fit' or s['plan_sha256']!=sha(PLAN) or (ROOT/s['trainer_path']).resolve()!=Path(trainer).resolve():
        raise ReviewError('Invalid actual execution seal')
    check_bindings(s['source_sha256'])
    if s['packages']!={k:importlib.metadata.version(k) for k in s['packages']}: raise ReviewError('Dependencies changed')
    return p


def endpoint(r):
    if r['status']!='fit_executed' or r['arm'] not in ['C','E'] or r['fold'] not in [0,1,2] or r['candidate_selected_by_score']:
        raise ReviewError('Unregistered endpoint')
    if not 0<=r['accepted_updates']<=r['full_gradient_evaluations']<=200:
        raise ReviewError('Exceeded registered evaluations/updates')
    if r['termination'] not in ['gradient_budget','gradient_budget_trial_rolled_back','accepted_update_budget','numerical_no_change']:
        raise ReviewError('Invalid stopping')
    if r['termination']=='gradient_budget' and r['full_gradient_evaluations']!=200: raise ReviewError('Incomplete terminal')
    return True
