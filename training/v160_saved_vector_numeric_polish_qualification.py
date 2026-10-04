"""Single bounded polish policy; synthetic and actual saved vectors only."""
import json,traceback,time
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v160_active_margin_direction_v5 import solve_direction
from v160_saved_vector_numeric_polish import polish,MAX_DECIMAL_DOT_TERMS,DECIMAL_PRECISION
OUT=ROOT/'artifacts/v160_saved_vector_numeric_polish_qualification_20261002'
TRIAL=ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002'

def short(r):return {k:v for k,v in r.items() if k not in ['raw','direction']}
def save(path,r):path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not OUT.exists();OUT.mkdir();start=time.monotonic()
    files={Path(__file__).resolve(),ROOT/'training/v160_saved_vector_numeric_polish.py',ROOT/'training/v160_active_margin_direction_v5.py',ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v159_float64_repeat_policy_v2.py',TRIAL/'run_seal.json',ROOT/'artifacts/v160_independent_fixed_diagnostic_review_20261002/audit.json',ROOT/'artifacts/v160_independent_saved_primal_gap_witness_20261002/decimal80_witness.json',ROOT/'artifacts/v160_independent_saved_cone_inward_witness_20261002/witness.json'}
    scopes=[(0,0),(1,0),(1,1),(2,0)]
    for role,round_number in scopes:
        folder=TRIAL/f'role{role}'
        files|={folder/f'baseline_class{c}/gradient.npy' for c in [1,2]}
        files|={folder/f'round{round_number}'/name for name in ['QP_certificate.json','direction.npy','raw_margin_normals.npy']}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    save(OUT/'pre_numeric_bindings.json',dict(source_sha256=bindings,decimal_precision=DECIMAL_PRECISION,maximum_decimal_dot_terms=MAX_DECIMAL_DOT_TERMS,inward_iterations=3,inward_arithmetic_error_multiplier=16,official_calls=0,no_new_policy_threshold_for_acceptance=True))
    actual=[]
    for role,round_number in scopes:
        folder=TRIAL/f'role{role}';source=folder/f'round{round_number}';gm,gs=[np.load(folder/f'baseline_class{c}/gradient.npy') for c in [1,2]];a=np.load(source/'raw_margin_normals.npy');seed=read(source/'QP_certificate.json')
        r=polish(gm,gs,a,seed);target=OUT/f'role{role}_round{round_number}';target.mkdir()
        for key in ['raw','direction']:
            if key in r:np.save(target/f'{key}.npy',r[key])
        save(target/'certificate.json',short(r));assert r['status']=='numeric_polish_certified_for_finite_probe_only',short(r)
        old=np.load(source/'direction.npy');actual.append(dict(role=role,round=round_number,original_status=seed['status'],polished_status=r['status'],maximum_direction_change=float(np.abs(r['direction']-old).max()),certificate=short(r)))
    toy=[]
    cases=[('safe',[-1,1],[-1,2],[[-1,0]]),('opposed',[-1,1],[-1,-1],[[-1,0]]),('impossible',[-1,0],[0,-1],[[-1,0],[0,-1]]),('redundant',[-1,1],[-1,2],[[-1,0],[-2,0]]),('useful_small',[-1,1e-11,0],[-1,0,2e-11],[[-1,0,0]])]
    for name,m,s,n in cases:
        for scale in [1.,1e-6,1e-30]:
            gm,gs,a=[np.array(x,np.float64)*scale for x in [m,s,n]];seed=solve_direction(gm,gs,a)
            r=polish(gm,gs,a,seed);toy.append(dict(case=name,scale=scale,status=r['status'],certificate=short(r)))
            if name in ['impossible','opposed']:assert r['status']=='direction_reconstruction_unresolved_stop'
            else:assert r['status']=='numeric_polish_certified_for_finite_probe_only',short(r)
    # Budget/invalid-seed stops must not substitute a truncated input problem.
    gm=np.full(100000,.01);gs=np.full(100000,.02);a=np.tile(gm,(24,1));seed=dict(alpha=.5,multipliers=[1.]*24)
    budget=polish(gm,gs,a,seed);assert budget['status']=='decimal_dot_term_budget_stop'
    invalid=polish(np.array([-1.,1.]),np.array([-1.,2.]),np.array([[-1.,0.]]),dict(alpha=.5,multipliers=[-1.]));assert invalid['status']=='invalid_seed_dual_bounds_stop'
    check_bindings(bindings)
    report=dict(status='bounded_numeric_polish_synthetic_and_all_four_actual_saved_direction_cases_passed',actual=actual,synthetic=toy,budget_negative_case=budget,invalid_seed_negative_case=invalid,elapsed_seconds=time.monotonic()-start,official_heads=0,official_features=0,official_class_gradients=0,official_margin_gradients=0,official_fits=0,permanent_updates=0,finite_step_safety_proven=False,quality_acceptance=False,source_sha256=bindings,limitations='Fixed existing active set only. Actual finite guard and classification quality require separately registered real probes. Decimal80 improves arithmetic on saved float64 measurements; no higher precision upstream gradient claim.')
    save(OUT/'qualification.json',report);print(json.dumps(dict(status=report['status'],actual_saved_cases=len(actual),synthetic_cases=len(toy),official_calls=0,elapsed_seconds=report['elapsed_seconds'])))

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),official_calls=0))
        raise
