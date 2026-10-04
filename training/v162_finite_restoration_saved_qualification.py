"""Complete saved SOC candidate and nonlinear toy qualification, zero calls."""
import json,traceback
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v159_float64_repeat_policy_v2 import repeat_values,finite_step_review
from v162_single_function_finite_restoration import propose
OUT=ROOT/'artifacts/v162_finite_restoration_saved_qualification_20261002'
DIAG=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role2'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def small_report(r):return {k:v for k,v in r.items() if k not in ['displacement','correction']}
def main():
    assert not OUT.exists();OUT.mkdir();files={Path(__file__).resolve(),ROOT/'training/v162_single_function_finite_restoration.py',ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v159_float64_repeat_policy_v2.py'}
    files|={DIAG/'round0/polished_direction.npy',DIAG/'round0/raw_margin_normals.npy',DIAG/'round0/normal_records.json',DIAG/'baseline_error_class1_repeat0/OOF_logq.npy',DIAG/'round0/probe4/OOF_logq.npy',DIAG/'round0/probe4/actual_blocking_original_rows.parquet'}
    files|={DIAG/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1,2]}
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};save(OUT/'pre_bindings.json',dict(source_sha256=binding,official_calls=0))
    a=np.load(DIAG/'round0/raw_margin_normals.npy');assert a.shape==(1,1060832)
    records=read(DIAG/'round0/normal_records.json');record=next(iter(records.values()));assert len(records)==1 and record['local']==21050 and record['truth']==2 and record['rival']==1
    import pandas as pd
    blockers=pd.read_parquet(DIAG/'round0/probe4/actual_blocking_original_rows.parquet');assert len(blockers)==2 and blockers.local.eq(21050).all() and blockers.scope.eq('OOF').all()
    before=np.load(DIAG/'baseline_error_class1_repeat0/OOF_logq.npy');after=np.load(DIAG/'round0/probe4/OOF_logq.npy');m0=float(before[21050,2]-before[21050,1]);failed=float(after[21050,2]-after[21050,1]);d=np.load(DIAG/'round0/polished_direction.npy');gs=[np.load(DIAG/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]]
    r=propose(d/16,a[0],m0,failed,*gs);assert r['status']=='linear_restoration_candidate_requires_full_actual_finite_guard'
    for key in ['displacement','correction']:np.save(OUT/f'actual_saved_{key}.npy',r[key])
    save(OUT/'actual_saved_candidate.json',small_report(r));reports=[]
    # Full width: changing normal and margin units must preserve displacement.
    for scale in [1.,1e-12,1e12,1e-24,1e24]:
        rr=propose(d/16,a[0]*scale,m0*scale,failed*scale,*gs)
        assert rr['status']==r['status'] and repeat_values(rr['displacement'],r['displacement'],'displacement')['passed']
        reports.append(dict(normal_and_margin_units=scale,passed=True))
    # Quadratic protected margin: tangent line search fails, nonlinear residual
    # correction restores the original margin while target losses decrease.
    toy=[]
    for width in [2,1060832]:
        def embed(v):z=np.zeros(width);z[-2:]=v;return z
        m0=3.85e-13;h=1/16;u=embed([h,0]);n=embed([0,1]);gm=embed([-1,.01]);gs=embed([-.5,.02]);failed=m0-.27*h*h
        corrected=propose(u,n,m0,failed,gm,gs);assert corrected['status']==r['status']
        x,y=corrected['displacement'][-2:];actual_margin=m0+y-.27*x*x;assert actual_margin>0 and failed<0
        def losses(xx,yy):return np.array([np.logaddexp(0.,1-xx+.01*yy),np.logaddexp(0.,2-.5*xx+.02*yy)])
        initial=losses(0.,0.);end=losses(x,y)
        # Exact base derivatives of the declared toy CE, not generic gm labels.
        gradients=[embed([-1,.01])/(1+np.exp(-1)),embed([-.5,.02])/(1+np.exp(-2))]
        slopes=[float(g@corrected['displacement']) for g in gradients];gate=finite_step_review(initial,end,slopes,1,1,'B',1.,True);assert gate['accepted']
        bad=propose(u,n,m0,failed,-gm,-gs);assert 'failed_stop' in bad['status']
        nofail=propose(u,n,m0,m0,gm,gs);assert nofail['status']=='no_actual_negative_protected_margin_to_restore_stop'
        zero=propose(u,n*0,m0,failed,gm,gs);assert zero['status']=='zero_actual_normal_stop'
        toy.append(dict(width=width,old_tangent_margin=failed,restored_actual_toy_margin=actual_margin,target_finite_review=gate,wrong_descent_rejected=True,no_failure_and_zero_normal_rejected=True))
    check_bindings(binding)
    result=dict(status='saved_full_width_single_function_finite_restoration_and_nonlinear_counterexample_qualified',actual_saved_candidate=small_report(r),unit_cases=reports,synthetic_nonlinear_cases=toy,actual_corrected_model_finite_probe_not_yet_executed=True,not_generic_QP_optimality_or_actual_safety=True,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,source_sha256=binding)
    save(OUT/'qualification.json',result);print(json.dumps(dict(status=result['status'],actual_linear_candidate=small_report(r),official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
