"""Real V165 blocker coverage versus same-theta measured margin functions."""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from v160_independent_fixed_diagnostic_review import read, sha
from v160_margin_normal import input_identity

ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'artifacts/v165_fixed_endpoint_decision_floor_diagnostic_20261002'
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
OUT=ROOT/'artifacts/v165_independent_blocker_coverage_review_20261002'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    assert not OUT.exists()
    OUT.mkdir()
    planpath=ROOT/'training/review_policy/v165_fixed_endpoint_decision_floor_diagnostic_contract.json'
    plan=read(planpath)
    xfile=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    sources={Path(__file__).resolve(),ROOT/'training/v160_margin_normal.py',planpath,xfile,
             ROOT/'artifacts/v165_independent_actual_decision_floor_review_20261002/review.json'}
    prepared=[]
    for spec in plan['roles']:
        role=spec['role'];folder=TRIAL/f'role{role}';point=PRIOR/f"role{role}/parameter_point{spec['parameter_point']}"
        refs=read(point/'restoration0/active_normal_references.json')
        opaths={s:ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{s}_probabilities.npy'
                for s in ['OOF','deployment']}
        sources|={folder/'diagnostic.json',folder/'baseline/OOF_original_rows.parquet',
                  folder/'treatment/actual_blocking_original_rows.parquet',point/'restoration0/active_normal_references.json'}
        sources.update(opaths.values())
        for scope in opaths:
            sources.add(folder/f'treatment/{scope}_logq.npy')
            sources.add(folder/f'baseline/{scope}_logq.npy')
        prepared.append((spec,folder,refs,opaths))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)}
    save(OUT/'source_bindings.json',dict(source_sha256=bindings,official_calls=0))
    x=load_npz(xfile).tocsr();x.sort_indices();results=[]
    for spec,folder,refs,opaths in prepared:
        role=spec['role'];origin=spec['endpoint_parameter_sha256']
        assert all(ref['metadata']['base_parameter_sha256']==origin for ref in refs.values())
        base=pd.read_parquet(folder/'baseline/OOF_original_rows.parquet');ids0=np.sort(base.local.unique())
        opinions={s:np.asarray(np.load(p,mmap_mode='r')[:,:16],np.float64) for s,p in opaths.items()}
        logs={s:np.load(folder/f'treatment/{s}_logq.npy') for s in opaths}
        baselogs={s:np.load(folder/f'baseline/{s}_logq.npy') for s in opaths}
        known_actual=[]
        for identity,ref in refs.items():
            m=ref['metadata'];scope=m['scope'];local=m['local'];truth=m['truth'];rival=m['rival']
            origin_margin=baselogs[scope][local,truth]-baselogs[scope][local,rival]
            actual_margin=logs[scope][local,truth]-logs[scope][local,rival]
            known_actual.append(dict(identity=identity,scope=scope,local=local,truth=truth,rival=rival,
                                     origin_margin=float(origin_margin),actual_margin=float(actual_margin),
                                     actual_argmax_correct=bool(logs[scope][local].argmax()==truth)))
        blockers=pd.read_parquet(folder/'treatment/actual_blocking_original_rows.parquet')
        functions={}
        for (scope,local,truth,rival),group in blockers.groupby(['scope','local','truth','rival'],sort=True):
            ids=ids0 if scope=='OOF' else np.arange(22546);pos=int(np.searchsorted(ids,local));assert ids[pos]==local
            chunk=ids[pos//2048*2048:pos//2048*2048+2048]
            identity=input_identity(role,scope,chunk,x[chunk],opinions[scope][chunk],pos%2048,int(truth),int(rival))
            functions[identity]=dict(scope=scope,local=int(local),truth=int(truth),rival=int(rival),
                                    original_rows=len(group),covered_by_current_local_constraints=identity in refs)
        fresh=set(functions)-set(refs);covered=set(functions)&set(refs)
        # These are diagnostic facts, not gates selected to make a method pass.
        result=dict(role=role,actual_blocking_original_rows=len(blockers),
                    blocking_original_class_counts={str(int(c)):int(n) for c,n in blockers.truth.value_counts().items()},
                    actual_blocking_functions=len(functions),same_theta_measured_functions=len(refs),
                    blocked_measured_functions=len(covered),fresh_unmeasured_functions=len(fresh),
                    current_union_if_fresh_functions_added=len(refs)+len(fresh),previous_fixed_cap=24,
                    measured_functions_actual_argmax_failures=sum(not r['actual_argmax_correct'] for r in known_actual),
                    all_blocking_functions_not_in_local_restoration_constraints=not covered,
                    no_claim_unmeasured_functions_are_classification_cause_or_sufficient_repair=True)
        save(OUT/f'role{role}_function_coverage.json',dict(review=result,known_actual_margins=known_actual,
                                                        blocking_functions=functions,fresh_function_ids=sorted(fresh)))
        results.append(result)
    assert all(sha(ROOT/p)==value for p,value in bindings.items())
    report=dict(status='actual_V165_unprotected_local_function_coverage_and_measured_function_margins_reviewed',
                roles=results,official_calls=0,parameter_updates=0,
                scope='Same-theta local constraint coverage only; no new derivatives or repair proof.')
    save(OUT/'review.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
