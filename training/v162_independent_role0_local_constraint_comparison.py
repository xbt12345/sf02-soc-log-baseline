"""Compare saved equality recovery with one-sided joint-descent recovery.

CPU saved-vector diagnostic only, not a finite model test or fit authority.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from v160_independent_fixed_diagnostic_review import read, sha
from v161_independent_all_finite_results_review import save

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v162_fixed_endpoint_finite_restoration_20261002/role0/restoration3'
PRIOR=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role0'
OUT=ROOT/'artifacts/v162_independent_role0_local_constraint_comparison_20261002'
EPS=float(np.finfo(float).eps)


def main():
    assert not OUT.exists();OUT.mkdir()
    files=[Path(__file__).resolve(),SOURCE/'current_displacement.npy',SOURCE/'active_complete_margin_normals.npy',
           SOURCE/'base_margins.npy',SOURCE/'actual_current_margins.npy',SOURCE/'correction.npy',
           SOURCE/'original_unit_restoration_review.json',PRIOR/'round0/polished_direction.npy']
    files += [PRIOR/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1,2]]
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    save(OUT/'source_bindings.json',dict(source_sha256=bindings,official_calls=0))
    u=np.load(SOURCE/'current_displacement.npy');a=np.load(SOURCE/'active_complete_margin_normals.npy')
    b=np.load(SOURCE/'base_margins.npy');c=np.load(SOURCE/'actual_current_margins.npy')
    gradients=np.stack([np.load(PRIOR/f'baseline_error_class{i}_repeat0/complete_fixed_error_target_gradient.npy') for i in [1,2]])
    seed=np.load(PRIOR/'round0/polished_direction.npy')/16
    seed_slopes=np.array([math.fsum(g*seed) for g in gradients]);assert np.all(seed_slopes<0)
    reports=[]
    for joint in [False,True]:
        matrix=a.copy();rhs=b-c
        if joint:
            matrix=np.vstack([matrix,-gradients])
            rhs=np.r_[rhs,[math.fsum(g*u)-desired for g,desired in zip(gradients,seed_slopes)]]
        scales=np.linalg.norm(matrix,axis=1);assert np.all(scales>0)
        scaled=matrix/scales[:,None];lower=rhs/scales
        left,singular,basis=np.linalg.svd(scaled,full_matrices=False)
        rank=int(np.count_nonzero(singular>EPS*len(matrix)*singular[0]));basis=basis[:rank]
        small=scaled@basis.T
        initial=np.linalg.lstsq(small,lower,rcond=EPS*len(matrix))[0]
        result=minimize(lambda z:.5*float(z@z),initial,jac=lambda z:z,method='SLSQP',
                        constraints=[LinearConstraint(small,lower,np.inf)],
                        options=dict(ftol=1e-14,maxiter=1000))
        z=result.x.copy()
        # Exact active-row arithmetic polish, not a change to constraints.
        for _ in range(3):
            slack=small@z-lower
            active=np.flatnonzero(slack<=64*EPS*np.maximum(1.,np.maximum(np.abs(lower),np.abs(small@z))))
            if not len(active):break
            z += np.linalg.lstsq(small[active],lower[active]-small[active]@z,rcond=EPS*len(matrix))[0]
        e=z@basis;candidate=u+e
        observed=np.array([math.fsum(row*e) for row in matrix])
        limits=16*EPS*np.maximum(np.abs(rhs),np.linalg.norm(matrix,axis=1)*np.linalg.norm(e))
        qualified=bool(np.all(observed-rhs>=-limits))
        slopes=np.array([math.fsum(g*candidate) for g in gradients])
        name='one_sided_with_joint_original_descent' if joint else 'one_sided_margin_only'
        np.save(OUT/(name+'_correction.npy'),e);np.save(OUT/(name+'_displacement.npy'),candidate)
        reports.append(dict(name=name,solver_success=bool(result.success),solver_message=str(result.message),iterations=int(result.nit),
                            rank=rank,one_sided_complete_vector_residual_qualified=qualified,
                            residual_gap=(observed-rhs).tolist(),residual_limits=limits.tolist(),
                            M_S_complete_displacement_slopes=slopes.tolist(),common_negative=bool(np.all(slopes<0)),
                            correction_norm=float(np.linalg.norm(e)),complete_displacement_norm=float(np.linalg.norm(candidate)),
                            fixed_original_seed_descent_retained=bool(np.all(slopes<=seed_slopes+16*EPS*np.maximum(1.,np.abs(seed_slopes)))),
                            no_official_finite_or_classification_evaluation=True))
    old=read(SOURCE/'original_unit_restoration_review.json')
    output=dict(status='saved_role0_local_equality_vs_one_sided_and_joint_descent_comparison_complete',
                original_equalities=dict(class_reviews=old['class_reviews'],correction_norm=old['correction_norm'],
                                         all_residuals_passed=all(r['passed'] for r in old['residual_reviews'])),
                already_safer_than_original_margin_forced_back_by_equalities=int(np.count_nonzero(c>b)),
                original_seed_class_slopes=seed_slopes.tolist(),alternatives=reports,official_heads=0,official_gradients=0,
                official_fits=0,permanent_updates=0,finite_trial_authority=False,quality_acceptance=False,
                limitations='Local old-point vector comparison only. Does not prove corrected nonlinear classifier safety, actual repairs, global feasibility, or transfer.')
    assert all(sha(ROOT/k)==v for k,v in bindings.items())
    save(OUT/'comparison.json',output);print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
