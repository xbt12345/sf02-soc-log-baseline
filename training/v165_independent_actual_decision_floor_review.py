"""Independent original-gold V165 finite comparison and full CPU checkpoint audit."""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v160_independent_fixed_diagnostic_review import read, sha, rows_review
from v161_independent_all_finite_results_review import close, target_risk, parameter_hash, gradient_repeat
from v164_independent_actual_short_trajectory_review import proposal_review, tables

ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'artifacts/v165_fixed_endpoint_decision_floor_diagnostic_20261002'
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
COHORT=ROOT/'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OUT=ROOT/'artifacts/v165_independent_actual_decision_floor_review_20261002'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/diagnostic.json').exists() for r in range(3))
    OUT.mkdir()
    planpath=ROOT/'training/review_policy/v165_fixed_endpoint_decision_floor_diagnostic_contract.json'
    plan=read(planpath);seal=read(TRIAL/'run_seal.json')
    assert sha(planpath)==seal['plan_sha256']
    assert plan['new_caps']['fits']==plan['new_caps']['permanent_updates']==0
    goldpath=ROOT/'data/official/train.parquet'
    gold=pd.read_parquet(goldpath,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert np.bincount(gold,minlength=3).tolist()==[1899723,111728,45420]
    sources={p for p in TRIAL.rglob('*') if p.is_file()}|{Path(__file__).resolve(),planpath,goldpath}
    sources|={ROOT/'training'/name for name in ['v160_independent_fixed_diagnostic_review.py',
        'v161_independent_all_finite_results_review.py','v164_independent_actual_short_trajectory_review.py',
        'v165_fixed_endpoint_decision_floor_diagnostic.py','v165_execution_review.py','v165_decision_floor_restoration.py']}
    for spec in plan['roles']:
        role=spec['role'];previous=PRIOR/f'role{role}';point=previous/f"parameter_point{spec['parameter_point']}"
        sources|={p for p in (previous/'endpoint').rglob('*') if p.is_file()}
        sources.add(previous/'endpoint.pt');sources.add(point/'parameter_identity.json')
        sources.add(COHORT/f'role{role}/fixed_pure_error_targets.parquet')
        sources|={p for p in (point/'restoration0').rglob('*') if p.is_file()}
        sources|={p for p in (ROOT/spec['cached_treatment']).rglob('*') if p.is_file()}
        for c in [1,2]:
            for k in [0,1]:
                sources.add(point/f'class{c}_repeat{k}/complete_fixed_error_target_gradient.npy')
                sources.add(point/f'class{c}_repeat{k}/parameter_point.json')
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)}
    save(OUT/'pre_review_bindings.json',dict(source_sha256=bindings,official_calls=0))
    roles=[]
    for spec in plan['roles']:
        role=spec['role'];folder=TRIAL/f'role{role}';previous=PRIOR/f'role{role}'
        point=previous/f"parameter_point{spec['parameter_point']}";diag=read(folder/'diagnostic.json')
        assert diag['exception'] is None and diag['new_fits']==diag['permanent_updates']==diag['new_complete_derivatives']==0
        assert diag['new_CPU_QP_solves']==0 and diag['finite_proposals']==1
        state=torch.load(previous/'endpoint.pt',map_location='cpu',weights_only=True)['state']
        origin=parameter_hash(state)
        assert origin==spec['endpoint_parameter_sha256']==read(point/'parameter_identity.json')['parameter_sha256']
        reference=tables(previous/'endpoint');baseline=tables(folder/'baseline')
        for scope in baseline:
            observed,_,_,_=rows_review(folder/'baseline',scope,reference[scope],gold)
            close(observed[['p0','p1','p2']].to_numpy(),reference[scope][['p0','p1','p2']].to_numpy())
            close(observed[['logp0','logp1','logp2']].to_numpy(),reference[scope][['logp0','logp1','logp2']].to_numpy())
        gradients=[]
        for c in [1,2]:
            pair=[np.load(point/f'class{c}_repeat{k}/complete_fixed_error_target_gradient.npy') for k in [0,1]]
            gradient_repeat(*pair);gradients.append(pair[0])
            assert all(read(point/f'class{c}_repeat{k}/parameter_point.json')['parameter_sha256']==origin for k in [0,1])
        targets=pd.read_parquet(COHORT/f'role{role}/fixed_pure_error_targets.parquet').row_position
        treatment,frames=proposal_review(folder/'treatment',baseline,targets,gradients,state,gold)
        assert np.array_equal(np.load(folder/'treatment/direction.npy'),np.load(ROOT/spec['cached_treatment']/'displacement.npy'))
        assert read(folder/'treatment/probe.json')['step']==1.
        control,_=proposal_review(point/'restoration0/finite_probe',baseline,targets,gradients,state,gold)
        matched=read(folder/'matched_control.json')
        assert matched['control']==read(point/'restoration0/finite_probe/probe.json')
        assert matched['treatment']==read(folder/'treatment/probe.json')
        paired={};before=baseline['OOF'];after=frames['OOF']
        for c,name in [(1,'M'),(2,'S')]:
            population=before.truth.eq(c);oldwrong=before.pred.ne(before.truth);wrong=after.pred.ne(after.truth)
            repair=population&oldwrong&~wrong;regress=population&~oldwrong&wrong;pure=before.pure_current_input
            paired[name]=dict(original_rows=int(population.sum()),errors_before=int((population&oldwrong).sum()),
                errors_after=int((population&wrong).sum()),pure_errors_before=int((population&oldwrong&pure).sum()),
                pure_errors_after=int((population&wrong&pure).sum()),repairs_vs_V164_endpoint=int(repair.sum()),
                new_errors_vs_V164_endpoint_all_original_rows=int(regress.sum()),pure_repairs_vs_V164_endpoint=int((repair&pure).sum()),
                new_pure_errors_vs_V164_endpoint=int((regress&pure).sum()),new_mixed_errors_vs_V164_endpoint=int((regress&~pure).sum()),
                registered_cumulative_protection_regressions=int((population&wrong&before.protected_correct).sum()))
        assert paired==read(folder/'paired_treatment_vs_V164_endpoint.json')
        restored=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)['state']
        assert state.keys()==restored.keys() and all(torch.equal(state[k],restored[k]) for k in state)
        assert parameter_hash(restored)==origin
        restoration=read(folder/'restoration_review.json')
        assert restoration['initial_parameter_sha256']==restoration['restored_parameter_sha256']==origin
        for scope in baseline:
            observed,_,_,_=rows_review(folder/'endpoint',scope,baseline[scope],gold)
            close(observed[['p0','p1','p2']].to_numpy(),baseline[scope][['p0','p1','p2']].to_numpy())
            close(observed[['logp0','logp1','logp2']].to_numpy(),baseline[scope][['logp0','logp1','logp2']].to_numpy())
        events=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
        for kind in ['head','feature']:
            for event in ['attempt','completed']:
                selected=[e for e in events if e['kind']==kind and e['event']==event]
                assert [e['ordinal'] for e in selected]==list(range(1,spec['head_cap']+1))
        assert not any(e['kind'] in ['full_parameter_margin_gradient','fixed_error_target_gradient',
                                    'full_original_class_gradient','permanent_update','supervised_classifier_fit'] for e in events)
        finite=[e for e in events if e['kind']=='finite_decision_floor_proposal']
        assert [e['event'] for e in finite]==['attempt','completed'] and all(e['ordinal']==1 for e in finite)
        assert finite[0]['origin_parameter_sha256']==origin and finite[1]['probe_parameter_sha256']==diag['candidate']['probe_parameter_sha256']
        assert diag['counts']==dict(head_attempts=spec['head_cap'],head_completed=spec['head_cap'],
            feature_attempts=spec['head_cap'],feature_completed=spec['head_cap'],gradient_attempts=0,gradient_completed=0)
        assert treatment['accepted']==finite[1]['accepted']==diag['candidate']['accepted']
        roles.append(dict(role=role,origin_parameter_sha256=origin,control=control,treatment=treatment,
                          paired_classification=paired,restored_full_parameters_and_predictions=True,heads=spec['head_cap']))
    assert all(sha(ROOT/p)==value for p,value in bindings.items())
    common=all(r['treatment']['accepted'] for r in roles)
    any_gain=any(r['treatment']['metrics']['OOF']['pure_errors']<
                 sum(v['pure_errors_before'] for v in r['paired_classification'].values()) for r in roles)
    report=dict(status='all_three_actual_decision_floor_finite_probes_original_gold_guards_costs_and_full_restoration_verified',
                roles=roles,new_heads=sum(r['heads'] for r in roles),new_complete_derivatives=0,new_fits=0,permanent_updates=0,
                all_three_actual_finite_guards_passed=common,any_actual_pure_classification_gain=any_gain,
                supports_new_short_training_registration=common and any_gain,
                official_calls_by_this_review=0,training_issue_mastered=False,quality_acceptance=False,
                scope='Previously supervised development rows; no independent transfer or complete-task confirmation.')
    save(OUT/'review.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='roles'},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
