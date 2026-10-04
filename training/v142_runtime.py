"""Existing second-layer adaptation, separate bounded scope from exhausted LP."""
import copy
import sys
import importlib.metadata
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha,check_bindings,ReviewError
from v138_runtime import save
from v140_runtime import OUT as BASE
from v141_representation_probe import OUT as PROBE

PLAN=ROOT/'training/review_policy/v142_second_layer_plan.json'
OUT=ROOT/'artifacts/v142_second_layer_training_20261001'


def validate(p):
    def need(ok,why):
        if not ok:raise ReviewError(why)
    need(p['issue']=='TRAIN-PURE-READOUT' and p['latest_actual']=='V140','First issue must remain open until actual mastery')
    need(p['old_chain']=={'fits_used':12,'rounds_used':2,'remaining_fits':0,'readout_additional_fits':0},'Do not reopen old LP budget')
    need(p['new_scope']=={'trainable':'existing_second_layer_only','parameters':22528,'frozen':['first','head','facts_direct'],'new_parameters':0},'No unsupported scope or capacity expansion')
    need(p['fits']==3 and p['folds']==[0,1,2] and p['candidate']=='S2','No extra fits or candidate selection')
    need(p['baseline']=='all_V140_C_fixed_endpoints_no_new_fit_TRAIN_retention_initialization_only','No posthoc V140 candidate promotion or held-based init')
    need(p['loss']=='original_frequency_independent_member_CE' and p['HELD_labels_used']==0,'No objective/sampling leakage')
    need(p['full_gradient_max_per_fit']==p['accepted_updates_max_per_fit']==200,'Unregistered computation budget')
    need(p['selection']=='last_accepted_budget_or_numerical_no_change' and not p['automatic_repeat'],'No intermediate selection or unlimited new scopes')
    need(p['acceptance']=={'pure_M':0,'pure_S':0,'full_M':0,'full_S_by_fold':[22,6,28],'last_distinct_states':5,'guard_new_errors':0,'all_initial_correct_pure_new_errors':0},'No relaxed training/guard gate')
    need(p['task_adoption_quality']==read(ROOT/'training/review_policy/v137_single_issue_plan.json')['task_adoption_quality'],'No relaxed task quality')
    need(p['risk_actions']=={r['id']:r['required_action'] for r in read(ROOT/'training/review_policy/history_cases.json')['risks']},'Unhandled history')
    need(p['solver']=={'lr':1,'max_iter':1,'max_eval':200,'history_size':20,'tolerance_grad':1e-7,'tolerance_change':1e-9,'line_search_fn':'strong_wolfe'},'Unreviewed solver')
    need(p['inspected_development_not_blind'] and p['nonconvex_and_feature_distortion_risk'],'False guarantee or blind claim')
    return True


def adversaries(p):
    changes=[('old budget reset',('old_chain','remaining_fits'),3),('more head fits',('old_chain','readout_additional_fits'),3),
             ('held-label fitting',('HELD_labels_used',),1),('expand capacity',('new_scope','new_parameters'),128),
             ('unfreeze first',('new_scope','frozen'),['head']),('more fits',('fits',),6),('change candidate',('candidate',),'BEST'),
             ('loss reweight',('loss',),'balanced_CE'),('checkpoint picking',('selection',),'best_TRAIN'),
             ('guard relaxation',('acceptance','guard_new_errors'),1),('mastery relaxation',('acceptance','pure_S'),12),
             ('automatic search',('automatic_repeat',),True),('ignore feature distortion',('nonconvex_and_feature_distortion_risk',),False)]
    results=[]
    for name,path,v in changes:
        q=copy.deepcopy(p);d=q
        for k in path[:-1]:d=d[k]
        d[path[-1]]=v
        try:validate(q)
        except (ValueError,KeyError):results.append({'case':name,'rejected':True})
        else:raise ReviewError('Accepted forbidden design '+name)
    validate(p);return results


def review():
    p=read(PLAN);validate(p);check_bindings(p['evidence_sha256'])
    e=read(PROBE/'probe.json')
    if e['parameter_updates'] or e['positive_actual_probability_margin_descent_inputs']!=6:raise ReviewError('New representation evidence changed')
    return p


def seal_run(trainer,extras):
    review();target=OUT/'run_seal.json'
    if target.exists():raise FileExistsError(target)
    paths={PLAN,Path(trainer).resolve(),Path(__file__).resolve(),ROOT/'training/experiment_review.py'}|set(extras)
    old=read(BASE/'run_seal.json');check_bindings(old['source_sha256']);paths.update(ROOT/p for p in old['source_sha256'])
    paths.update(ROOT/p for p in read(PLAN)['evidence_sha256'])
    missing=[]
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.suffix=='.pyc' and p.with_suffix('.py').is_file():p=p.with_suffix('.py')
            if p.is_file():paths.add(p)
            else:missing.append(str(p))
    # Bind actual Torch CUDA/CPU binaries and the Python runtime before updates.
    paths.add(Path(sys.executable).resolve())
    torchlib=Path(torch.__file__).parent/'lib';paths.update(torchlib.glob('*.dll'))
    for name in ['python.exe','python311.dll','python312.dll','python313.dll','python3.dll']:
        p=Path(sys.base_prefix)/name
        if p.is_file():paths.add(p)
    bindings={p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p):sha(p) for p in sorted(q.resolve() for q in paths)}
    save(target,{'status':'sealed_before_fit','version':'V142-second-only','trainer_path':Path(trainer).resolve().relative_to(ROOT).as_posix(),
                 'plan_sha256':sha(PLAN),'source_sha256':bindings,'package_versions':{k:importlib.metadata.version(k) for k in ['torch','tabm','numpy','scipy','pandas','pyarrow']},
                 'python_version':sys.version,'loaded_dynamic_metadata_without_physical_file':sorted(set(missing)),
                 'source_and_runtime_binding':'All loaded physical module files, explicit evaluation dependencies, Torch lib DLLs and Python runtime; GPU driver/OS versions not byte-snapshotted.',
                 'quality_acceptance':False,'new_fits':0,'new_updates':0})


def require_run_seal(trainer):
    p=review();s=read(OUT/'run_seal.json')
    if s['status']!='sealed_before_fit' or s['plan_sha256']!=sha(PLAN) or (ROOT/s['trainer_path']).resolve()!=Path(trainer).resolve():raise ReviewError('Run binding changed')
    check_bindings(s['source_sha256'])
    if s['python_version']!=sys.version or s['package_versions']!={k:importlib.metadata.version(k) for k in s['package_versions']}:raise ReviewError('Runtime changed')
    return p


def endpoint(r):
    if r['status']!='fit_executed' or r['candidate_selected_by_score'] or r['arm']!='S2' or r['fold'] not in [0,1,2]:raise ReviewError('Wrong endpoint')
    if not 0<=r['accepted_updates']<=r['full_gradient_evaluations']<=200:raise ReviewError('Wrong evaluation budget')
    if r['termination'] not in ['gradient_budget','gradient_budget_trial_rolled_back','accepted_update_budget','numerical_no_change']:raise ReviewError('Wrong stopping')
    if r['termination']=='gradient_budget' and r['full_gradient_evaluations']!=200:raise ReviewError('Incomplete budget')
    return True
