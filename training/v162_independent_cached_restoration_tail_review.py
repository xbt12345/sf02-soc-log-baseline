"""Original-gold saved-output audit of the two-probe restoration tail.

No official forward, gradient, features, fit or parameter update.
"""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v160_independent_fixed_diagnostic_review import read, sha, rows_review
from v161_independent_all_finite_results_review import close, parameter_hash, save
from v162_independent_actual_restoration_review import actual_proposal
from v162_multi_function_finite_restoration_v2 import propose

ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'artifacts/v162_cached_function_restoration_tail_20261002'
PRIOR=ROOT/'artifacts/v162_fixed_endpoint_finite_restoration_20261002/role2'
DIAG=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role2'
OUT=ROOT/'artifacts/v162_independent_cached_restoration_tail_review_20261002'


def main():
    assert not OUT.exists() and (TRIAL/'role2/diagnostic.json').exists()
    OUT.mkdir()
    goldpath=ROOT/'data/official/train.parquet'
    gold=pd.read_parquet(goldpath,columns=['label_binary']).label_binary.map(
        {'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert np.bincount(gold,minlength=3).tolist()==[1899723,111728,45420]
    statepath=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002/fold2_B/endpoint.pt'
    targetpath=ROOT/'artifacts/v161_independent_frozen_error_cohort_review_20261002/role2/fixed_pure_error_targets.parquet'
    sources={p for p in TRIAL.rglob('*') if p.is_file()}|{Path(__file__).resolve(),goldpath,statepath,targetpath}
    sources|={p for p in (PRIOR/'restoration3').rglob('*') if p.is_file()}
    sources|={p for p in (PRIOR/'baseline').glob('*') if p.is_file()}
    sources|={ROOT/'training'/n for n in ['v162_independent_actual_restoration_review.py',
              'v161_independent_all_finite_results_review.py','v160_independent_fixed_diagnostic_review.py',
              'v162_multi_function_finite_restoration_v2.py']}
    sources|={DIAG/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1,2]}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)}
    save(OUT/'pre_review_bindings.json',dict(source_sha256=bindings,official_calls=0))
    folder=TRIAL/'role2';diagnostic=read(folder/'diagnostic.json')
    assert diagnostic['exception'] is None
    assert all(diagnostic[k]==0 for k in ['new_full_original_class_gradients','new_fixed_error_target_gradients',
                                         'new_margin_gradients','new_fits','permanent_updates'])
    state=torch.load(statepath,map_location='cpu',weights_only=True)['state']
    assert parameter_hash(state)==diagnostic['initial_parameter_sha256']==diagnostic['restored_parameter_sha256']
    baseline={s:pd.read_parquet(PRIOR/'baseline'/f'{s}_original_rows.parquet') for s in ['OOF','deployment']}
    for scope,b in baseline.items():
        f,_,_,_=rows_review(folder/'baseline',scope,b,gold)
        close(f[['p0','p1','p2']].to_numpy(),b[['p0','p1','p2']].to_numpy())
        close(f[['logp0','logp1','logp2']].to_numpy(),b[['logp0','logp1','logp2']].to_numpy())
    targets=pd.read_parquet(targetpath).row_position
    gradients=[np.load(DIAG/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]]
    u=np.load(PRIOR/'restoration3/displacement.npy')
    a=np.load(PRIOR/'restoration3/active_complete_margin_normals.npy')
    b=np.load(PRIOR/'restoration3/base_margins.npy')
    records=read(PRIOR/'restoration3/active_functions.json')
    assert len(records)==len(a)==1
    record=next(iter(records.values()))
    assert record['local']==21050 and record['truth']==2 and record['rival']==1
    logs=np.load(PRIOR/'restoration3/finite_probe/OOF_logq.npy')
    reports=[]
    for index in range(diagnostic['restoration_solves']):
        target=folder/f'restoration{index+4}'
        assert np.array_equal(u,np.load(target/'current_displacement.npy'))
        assert np.array_equal(a,np.load(target/'active_complete_margin_normals.npy'))
        assert np.array_equal(b,np.load(target/'base_margins.npy'))
        assert records==read(target/'active_functions.json')
        c=np.load(target/'actual_current_margins.npy')
        assert c[0]==logs[21050,2]-logs[21050,1]
        computed=propose(u,a,b,c,*gradients)
        assert {k:v for k,v in computed.items() if k not in ['displacement','correction']}==read(target/'original_unit_restoration_review.json')
        for key in ['displacement','correction']:
            if key in computed:assert np.array_equal(computed[key],np.load(target/f'{key}.npy'))
        item=dict(iteration=index+4,status=computed['status'],class_reviews=computed.get('class_reviews'))
        if (target/'finite_probe/probe.json').exists():
            item['actual']=actual_proposal(target/'finite_probe',baseline,targets,gradients,state,gold)
            logs=np.load(target/'finite_probe/OOF_logq.npy')
            item['actual_protected_margin']=float(logs[21050,2]-logs[21050,1])
            u=computed['displacement']
        reports.append(item)
    events=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
    counts={}
    assert not any(v['kind'] not in ['head','feature'] for v in events)
    for kind in ['head','feature']:
        for event,suffix in [('attempt','attempts'),('completed','completed')]:
            matching=[v for v in events if v['kind']==kind and v['event']==event]
            assert [v['ordinal'] for v in matching]==list(range(1,len(matching)+1))
            counts[kind+'_'+suffix]=len(matching)
    counts.update(gradient_attempts=0,gradient_completed=0)
    assert counts==diagnostic['counts']
    assert counts['head_completed']==counts['feature_completed']==44+22*diagnostic['finite_proposals']<=88
    assert sum('actual' in v for v in reports)==diagnostic['finite_proposals']<=2
    assert any(v.get('actual',{}).get('accepted',False) for v in reports)==diagnostic['actual_finite_restoration_pass']
    for scope,b in baseline.items():
        f,_,_,_=rows_review(folder/'restored',scope,b,gold)
        assert np.array_equal(f.pred,b.pred)
        close(f[['p0','p1','p2']].to_numpy(),b[['p0','p1','p2']].to_numpy())
    assert diagnostic['restored_joint_TRAIN_retention']['passed']
    assert all(sha(ROOT/k)==v for k,v in bindings.items())
    report=dict(status='saved_actual_tail_original_gold_complete_displacements_guards_and_costs_independently_verified',
                actual_finite_pass=diagnostic['actual_finite_restoration_pass'],stop=diagnostic['status'],
                reports=reports,new_heads_and_features=counts['head_completed'],new_derivatives=0,
                cumulative_heads_and_features=17246+counts['head_completed'],cumulative_all_complete_derivatives=422,
                official_calls_by_this_review=0,new_fits=0,permanent_updates=0,quality_acceptance=False,
                limitations='Actual finite mechanism only. No learned trajectory, full-task improvement, or transfer demonstrated.')
    save(OUT/'review.json',report)
    print(json.dumps({k:report[k] for k in ['status','actual_finite_pass','stop','new_heads_and_features','new_fits']},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
