"""Audit saved real V162 probes; original gold and CPU vectors only.

No official forward, features, gradients, fit, or parameter update.
"""
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz

from v160_independent_fixed_diagnostic_review import read, sha, rows_review
from v161_independent_all_finite_results_review import (
    close, target_risk, parameter_hash, gradient_repeat, margin_progress, save,
)
from v160_margin_normal import input_identity
from v162_multi_function_finite_restoration_v2 import propose

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT/'artifacts/v162_fixed_endpoint_finite_restoration_20261002'
PRIOR = ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002'
COHORT = ROOT/'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OUT = ROOT/'artifacts/v162_independent_actual_restoration_review_20261002'
EPS = float(np.finfo(float).eps)


def actual_proposal(path, baseline, targets, gradients, state, gold):
    report = read(path/'probe.json')
    u = np.load(path/'direction.npy')
    assert report['step'] == 1. and report['actual_parameter_change']
    assert np.array_equal(u, np.load(path.parent/'displacement.npy'))
    assert report['probe_parameter_sha256'] == parameter_hash(state, u, 1.)
    slopes = np.array([math.fsum(g*u) for g in gradients])
    close(slopes, report['class_slopes'])
    frames, metrics = {}, {}
    for scope in baseline:
        f, fullrisk, _, met = rows_review(path, scope, baseline[scope], gold)
        frames[scope], metrics[scope] = f, met
        close(np.exp(f[['logp0','logp1','logp2']].to_numpy()), f[['p0','p1','p2']].to_numpy())
        for key in ['original_rows','M_errors','S_errors','pure_errors','total_errors',
                    'protected_regressions','new_errors_vs_initial','repairs_vs_initial']:
            assert met[key] == report[scope+'_stats'][key]
        if scope == 'OOF':
            after = target_risk(f, targets)
            close(after, np.load(path/'fixed_error_risk.npy'))
            close(fullrisk, np.load(path/'full_original_class_risk.npy'))
    before = target_risk(baseline['OOF'], targets)
    old = baseline['OOF']
    counts = all(metrics['OOF'][k] <= int((old.pred.ne(old.truth)&old.truth.eq(c)).sum())
                 for k,c in [('M_errors',1),('S_errors',2)])
    guard = (counts and metrics['OOF']['protected_regressions']==0 and
             metrics['deployment']['new_errors_vs_initial']==0 and
             report['deployment_stats']['mastered'] and report['joint_TRAIN_retention']['passed'])
    assert counts == report['full_original_M_S_error_count_guard']
    assert guard == report['classification_guard']
    resolution = 16*EPS*np.maximum(1., np.maximum(np.abs(before),np.abs(after)))
    drop = before-after
    slack = before+1e-4*slopes-after
    accepted = bool(guard and np.all(slopes<0) and np.all(drop>resolution) and np.all(slack>resolution))
    assert accepted == report['accepted'] == report['finite_error_target_review']['accepted']
    observed = pd.read_parquet(path/'actual_blocking_original_rows.parquet')
    expected = []
    for scope,f in frames.items():
        protection = f.protected_correct if scope=='OOF' else f.initial_correct
        expected += [(scope,int(r.row_position),int(r.local),int(r.truth),int(r.pred))
                     for r in f[protection&f.pred.ne(f.truth)].itertuples()]
    assert sorted(expected)==sorted((r.scope,int(r.row_position),int(r.local),int(r.truth),int(r.rival))
                                   for r in observed.itertuples())
    return dict(accepted=accepted, class_slopes=slopes.tolist(), fixed_target_drop=drop.tolist(),
                actual_blocking_original_rows=len(observed), metrics=metrics,
                progress={s:margin_progress(frames[s],baseline[s]) for s in frames},
                actual_classification_repair_safe=bool(accepted and metrics['OOF']['repairs_vs_fixed_endpoint']>0))


def main():
    assert not OUT.exists()
    assert all((TRIAL/f'role{r}/diagnostic.json').exists() for r in range(3))
    OUT.mkdir()
    plan = read(ROOT/'training/review_policy/v162_fixed_endpoint_finite_restoration_contract.json')
    goldpath = ROOT/'data/official/train.parquet'
    gold = pd.read_parquet(goldpath,columns=['label_binary']).label_binary.map(
        {'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert np.bincount(gold,minlength=3).tolist()==[1899723,111728,45420]
    x = load_npz(ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz').tocsr()
    x.sort_indices()
    files={p for p in TRIAL.rglob('*') if p.is_file()}|{Path(__file__).resolve(),goldpath}
    files|={ROOT/'training'/n for n in ['v160_independent_fixed_diagnostic_review.py',
            'v161_independent_all_finite_results_review.py','v160_margin_normal.py',
            'v162_multi_function_finite_restoration_v2.py']}
    files|={ROOT/'training/review_policy/v162_fixed_endpoint_finite_restoration_contract.json',
            ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz'}
    for role in range(3):
        files|={p for p in (COHORT/f'role{role}').glob('*') if p.is_file()}
        files.add(ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt')
        files|={ROOT/n[k] for n in plan['roles'][role]['cached_normals'] for k in ['gradient','metadata']}
        for c in [1,2]:
            files|={PRIOR/f'role{role}/baseline_error_class{c}_repeat{k}/complete_fixed_error_target_gradient.npy' for k in [0,1]}
        files|={p for p in (PRIOR/f'role{role}/round0/probe4').glob('*') if p.is_file()}
        files|={ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{s}_probabilities.npy'
                for s in ['OOF','deployment']}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    save(OUT/'pre_review_bindings.json',dict(source_sha256=bindings,official_calls=0))
    roles=[]
    for role,spec in enumerate(plan['roles']):
        folder=TRIAL/f'role{role}'
        diagnostic=read(folder/'diagnostic.json')
        assert diagnostic['exception'] is None and diagnostic['new_fits']==diagnostic['permanent_updates']==0
        state=torch.load(ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt',
                         map_location='cpu',weights_only=True)['state']
        parameter=parameter_hash(state)
        assert parameter==spec['endpoint_parameter_sha256']==diagnostic['initial_parameter_sha256']==diagnostic['restored_parameter_sha256']
        baseline={s:pd.read_parquet(folder/'baseline'/f'{s}_original_rows.parquet') for s in ['OOF','deployment']}
        for scope,f in baseline.items():
            rows_review(folder/'baseline',scope,f,gold)
            prior=PRIOR/f'role{role}'/('baseline_error_class1_repeat0' if scope=='OOF' else 'baseline_deployment')
            p=pd.read_parquet(prior/f'{scope}_original_rows.parquet')
            assert np.array_equal(f.row_position,p.row_position) and np.array_equal(f.pred,p.pred)
            close(f[['p0','p1','p2']].to_numpy(),p[['p0','p1','p2']].to_numpy())
            close(f[['logp0','logp1','logp2']].to_numpy(),p[['logp0','logp1','logp2']].to_numpy())
        targets=pd.read_parquet(COHORT/f'role{role}/fixed_pure_error_targets.parquet').row_position
        b=baseline['OOF']
        assert np.array_equal(targets,b.loc[b.pure_current_input&b.pred.ne(b.truth),'row_position'])
        gradients=[]
        for c in [1,2]:
            pairs=[np.load(PRIOR/f'role{role}/baseline_error_class{c}_repeat{k}/complete_fixed_error_target_gradient.npy') for k in [0,1]]
            gradient_repeat(*pairs);gradients.append(pairs[0])
        opinions={s:np.load(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{s}_probabilities.npy',mmap_mode='r')[:,:16]
                  for s in baseline}
        locals_by_scope={'OOF':np.sort(b.local.unique()),'deployment':np.arange(22546)}
        current_logs={s:np.load(PRIOR/f'role{role}/round0/probe4/{s}_logq.npy') for s in baseline}
        u=np.load(PRIOR/f'role{role}/round0/polished_direction.npy')/16
        assert np.array_equal(u,np.load(folder/'base_certified_direction_displacement.npy'))
        iterations=[]
        for iteration in range(diagnostic['restoration_solves']):
            target=folder/f'restoration{iteration}'
            records=read(target/'active_functions.json')
            a=np.load(target/'active_complete_margin_normals.npy')
            bm=np.load(target/'base_margins.npy');cm=np.load(target/'actual_current_margins.npy')
            assert len(records)==len(a) and np.array_equal(u,np.load(target/'current_displacement.npy'))
            for i,(identity,metadata) in enumerate(records.items()):
                assert metadata['base_parameter_sha256']==parameter
                scope=metadata['scope'];local=metadata['local'];t=metadata['truth'];r=metadata['rival']
                ids=locals_by_scope[scope];position=int(np.searchsorted(ids,local));start=position//2048*2048
                chunk=ids[start:start+2048];query=position-start
                assert input_identity(role,scope,chunk,x[chunk],np.asarray(opinions[scope][chunk],np.float64),query,t,r)==identity
                bp=np.load(folder/'baseline'/f'{scope}_logq.npy')
                assert bm[i]==bp[local,t]-bp[local,r] and cm[i]==current_logs[scope][local,t]-current_logs[scope][local,r]
                fresh=folder/'fresh_normals'/identity
                if fresh.exists():
                    assert read(fresh/'input_binding.json')==metadata
                    pair=[np.load(fresh/f'repeat{k}_gradient.npy') for k in [0,1]]
                    gradient_repeat(*pair);assert np.array_equal(a[i],pair[0])
                    assert all(v['passed'] for v in read(fresh/'measurement_repeat_review.json').values())
                else:
                    matching=[n for n in spec['cached_normals'] if read(ROOT/n['metadata'])['input_identity']==identity]
                    assert len(matching)==1 and np.array_equal(a[i],np.load(ROOT/matching[0]['gradient']))
            computed=propose(u,a,bm,cm,*gradients)
            recorded=read(target/'original_unit_restoration_review.json')
            assert {k:v for k,v in computed.items() if k not in ['displacement','correction']}==recorded
            for key in ['displacement','correction']:
                if key in computed:assert np.array_equal(computed[key],np.load(target/f'{key}.npy'))
            item=dict(iteration=iteration,status=computed['status'],active_function_count=len(records),
                      class_reviews=computed.get('class_reviews'),relative_correction_norm=computed.get('relative_correction_norm'),
                      residuals_passed=all(v['passed'] for v in computed.get('residual_reviews',[])))
            if (target/'finite_probe/probe.json').exists():
                assert computed['status']=='linear_restoration_candidate_requires_full_actual_finite_guard'
                item['actual']=actual_proposal(target/'finite_probe',baseline,targets,gradients,state,gold)
                u=computed['displacement']
                current_logs={s:np.load(target/'finite_probe'/f'{s}_logq.npy') for s in baseline}
            iterations.append(item)
        events=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
        counts={}
        for kind,prefix in [('head','head'),('feature','feature'),('full_class_gradient','gradient'),('full_parameter_margin_gradient','margin')]:
            for event,suffix in [('attempt','attempts'),('completed','completed')]:
                values=[v for v in events if v['kind']==kind and v['event']==event]
                assert [v['ordinal'] for v in values]==list(range(1,len(values)+1))
                counts[prefix+'_'+suffix]=len(values)
        assert counts==diagnostic['counts'] and counts['gradient_attempts']==0
        assert counts['head_attempts']==counts['head_completed']==counts['feature_completed']
        assert counts['head_completed']==(2+diagnostic['finite_proposals'])*(spec['OOF_chunks']+12)+counts['margin_completed']
        assert counts['head_completed']<=spec['head_cap'] and counts['margin_completed']<=spec['fresh_margin_gradient_cap']
        assert sum('actual' in it for it in iterations)==diagnostic['finite_proposals']
        assert any(it.get('actual',{}).get('accepted',False) for it in iterations)==diagnostic['actual_finite_restoration_pass']
        for scope in baseline:
            f,_,_,_=rows_review(folder/'restored',scope,baseline[scope],gold)
            assert np.array_equal(f.pred,baseline[scope].pred)
            close(f[['p0','p1','p2']].to_numpy(),baseline[scope][['p0','p1','p2']].to_numpy())
        assert diagnostic['restored_joint_TRAIN_retention']['passed']
        roles.append(dict(role=role,status=diagnostic['status'],actual_counts=counts,iterations=iterations,
                          restored_parameter_and_original_decisions_verified=True))
    assert all(sha(ROOT/k)==v for k,v in bindings.items())
    new_heads=sum(r['actual_counts']['head_completed'] for r in roles)
    new_margin=sum(r['actual_counts']['margin_completed'] for r in roles)
    report=dict(status='all_saved_actual_V162_restorations_original_gold_complete_vectors_and_costs_independently_verified',
                roles=roles,new_heads_and_features=new_heads,new_margin_gradients=new_margin,
                cumulative_heads_and_features=16908+new_heads,cumulative_all_complete_derivatives=390+new_margin,
                official_calls_by_this_review=0,new_fits=0,permanent_updates=0,quality_acceptance=False,
                all_three_finite_qualified=all(any(it.get('actual',{}).get('accepted',False) for it in r['iterations']) for r in roles),
                limitations='Independent saved-output review only. Finite mechanism qualification is not learned classification, promotion, or source-transfer validation.')
    save(OUT/'review.json',report)
    print(json.dumps({k:report[k] for k in ['status','new_heads_and_features','new_margin_gradients','all_three_finite_qualified','new_fits']},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
