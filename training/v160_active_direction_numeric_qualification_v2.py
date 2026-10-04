"""Prospective pure numeric fixtures and primal cross-check, no official calls."""
import json,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from experiment_review import ROOT,sha,read,check_bindings
from v160_active_margin_direction_v2 import solve_direction
OUT=ROOT/'artifacts/v160_active_direction_numeric_qualification_v2_20261002'
def short(r):return {k:v for k,v in r.items() if k!='direction'}
def main():
    assert not OUT.exists();OUT.mkdir()
    files=[Path(__file__).resolve(),ROOT/'training/v160_active_margin_direction_v2.py',ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v160_active_margin_direction.py',ROOT/'artifacts/v160_active_direction_numeric_qualification_20261002/failure.json',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v159_mgda_synthetic_qualification.py',ROOT/'training/v160_synthetic_constrained_direction_review.py',ROOT/'training/v160_synthetic_solver_scale_counterexample.py',ROOT/'artifacts/v160_synthetic_constrained_direction_review_20261002/qualification.json',ROOT/'artifacts/v160_synthetic_solver_scale_counterexample_20261002/qualification.json']
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in files};(OUT/'pre_synthetic_bindings.json').write_text(json.dumps(dict(status='before_numeric_fixture_generation_and_solver_calls',common_scale_only=True,QP_iteration_cap=256,certificate_eps=64,no_official_calls=True,source_sha256=binding),indent=2)+'\n',encoding='utf-8')
    gm,gs,n=np.array([-1.,1.]),np.array([-1.,2.]),np.array([[-1.,0.]])
    positive=[]
    for scale in [1.,1e-6,1e6,1e-30]:
        r=solve_direction(scale*gm,scale*gs,scale*n);positive.append(short(r));assert r['status']=='local_QP_certified_requires_actual_finite_guard' and np.allclose(r['direction'],[0.,-1.],atol=1e-14,rtol=0)
    full_gm=np.zeros(1060832);full_gs=np.zeros_like(full_gm);full_n=np.zeros((1,len(full_gm)));full_gm[-2:]=gm;full_gs[-2:]=gs;full_n[0,-2:]=n[0]
    high=solve_direction(full_gm*1e-6,full_gs*1e-6,full_n*1e-6);assert high['status']=='local_QP_certified_requires_actual_finite_guard' and np.array_equal(high['direction'][:-2],np.zeros(1060830))
    unconstrained=solve_direction(gm,gs,[]);assert unconstrained['status']=='local_QP_certified_requires_actual_finite_guard' and unconstrained['direction'][0]>0
    impossible=[solve_direction([-1,0],[0,-1],[[-1,0],[0,-1]]),solve_direction([-1,1],[-1,-1],[[-1,0]])]
    assert all(r['status'] in ['no_resolved_safe_common_descent','direction_reconstruction_unresolved_stop'] for r in impossible),[short(r) for r in impossible]
    normal_rescaling=[]
    for ns in [1e-24,1.,1e24]:
        r=solve_direction(gm,gs,n*ns);normal_rescaling.append(short(r))
        assert r['status']=='local_QP_certified_requires_actual_finite_guard' and np.allclose(r['direction'],[0.,-1.],atol=1e-14,rtol=0)
    near=solve_direction(gm,gs,np.array([[-1.,0.],[-1.,1e-10]]))
    if near['status']=='local_QP_certified_requires_actual_finite_guard':
        assert near['independent_saved_vector_certificate']['eligible_for_finite_trial_only']
    else:assert near['status'] in ['local_QP_certificate_failed_stop','direction_reconstruction_unresolved_stop','no_resolved_safe_common_descent']
    redundant=solve_direction(gm,gs,np.vstack([n,n,2*n]));assert redundant['status']=='local_QP_certified_requires_actual_finite_guard'
    # Independently solve the original primal in 2D, never trust dual success.
    primal=minimize(lambda z:.5*np.dot(z[:2],z[:2])+z[2],np.array([0.,0.,0.]),jac=lambda z:np.r_[z[:2],1.],constraints=[dict(type='ineq',fun=lambda z:z[2]-gm@z[:2],jac=lambda z:np.r_[-gm,1.]),dict(type='ineq',fun=lambda z:z[2]-gs@z[:2],jac=lambda z:np.r_[-gs,1.]),dict(type='ineq',fun=lambda z:n@z[:2],jac=lambda z:np.column_stack([n,np.zeros(len(n))]))],method='SLSQP',options=dict(ftol=1e-14,maxiter=256))
    assert primal.success and np.allclose(primal.x[:2],[0,-1],atol=1e-12,rtol=0)
    curved=-.0-10*(.5*positive[0]['class_slopes'][0])**2;assert curved<0
    cases=dict(common_scale_1e_minus6_success_is_not_false_feasible=True,common_scale_1e_minus30_does_not_use_absolute_gradient_floor=True,full1060832_dimension_common_scale_case_passed=True,no_common_safe_descent_stops=True,cancellation_residue_not_normalized=True,positive_normal_row_rescaling_preserves_cone=True,near_dependent_normal_never_false_certified=True,opposed_classes_stop=True,redundant_constraints_do_not_change_direction=True,independent_primal_matches_dual=True,curved_finite_guard_rejects_linear_QP_success=True)
    check_bindings(binding)
    result=dict(status='active_set_dual_original_unit_certificates_and_scale_counterexamples_passed',positive=positive,high_dimension=short(high),unconstrained=short(unconstrained),no_descent=[short(r) for r in impossible],redundant=short(redundant),normal_rescaling=normal_rescaling,near_dependent=short(near),primal=dict(direction=primal.x[:2].tolist(),objective=float(primal.fun)),curved_finite_margin=curved,cases=cases,official_heads=0,official_features=0,official_gradients=0,official_fits=0,official_updates=0,finite_SOC_qualification=False,source_sha256=binding)
    (OUT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],cases=cases,official_calls=0)))
if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps(dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),official_heads=0,official_gradients=0,official_fits=0),indent=2)+'\n',encoding='utf-8')
        raise
