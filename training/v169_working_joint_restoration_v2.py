"""Prospective 25-function coverage and exact-zero decision-floor recovery.

Only original full-vector residuals and resolved class signs authorize a
finite probe. SLSQP success is not an optimality or nonlinear safety proof.
"""
import numpy as np
from scipy.optimize import LinearConstraint,minimize
from v159_float64_repeat_policy_v2 import EPS,STEP_EPS
from v160_independent_saved_direction_certificate import dot,norm
MAX_NORMALS=64
MAX_ITERATIONS=256
MAX_REFINEMENTS=3

def _joint_propose(displacement,normals,base_margins,current_margins,gm,gs):
    u,a,b,c,m,s=[np.asarray(v,np.float64) for v in [displacement,normals,base_margins,current_margins,gm,gs]]
    if u.ndim!=1 or not u.size or m.shape!=u.shape or s.shape!=u.shape or a.ndim!=2 or a.shape[1]!=u.size or not 0<len(a)<=MAX_NORMALS or b.shape!=c.shape or b.shape!=(len(a),) or np.any(b<0) or not all(np.isfinite(v).all() for v in [u,a,b,c,m,s]):raise ValueError('Bounded matched original-unit full arrays required')
    if not np.any(c<0):return dict(status='no_actual_negative_protected_margin_to_restore_stop',finite_step_authority=False)
    # -g e >= g u is exactly g (u+e) <= 0; no arbitrary target weighting.
    rhs=np.r_[b-c,dot(m,u)[0],dot(s,u)[0]]
    scales=np.r_[np.max(np.abs(a),axis=1),np.max(np.abs(m)),np.max(np.abs(s))]
    if np.any(scales==0):return dict(status='zero_actual_constraint_normal_stop',finite_step_authority=False)
    scaled=np.empty((len(a)+2,len(u)),np.float64)
    np.divide(a,scales[:-2,None],out=scaled[:-2])
    np.divide(-m,scales[-2],out=scaled[-2]);np.divide(-s,scales[-1],out=scaled[-1])
    lower=rhs/scales
    _,singular,basis=np.linalg.svd(scaled,full_matrices=False);cutoff=EPS*(len(a)+2);rank=int(np.count_nonzero(singular>cutoff*singular[0]));basis=basis[:rank]
    small=scaled@basis.T;coordinate_scale=float(np.max(np.abs(lower)))
    if not coordinate_scale>0:return dict(status='zero_local_recovery_scale_stop',finite_step_authority=False)
    target=lower/coordinate_scale;initial=np.linalg.lstsq(small,target,rcond=cutoff)[0]
    result=minimize(lambda z:.5*float(z@z),initial,jac=lambda z:z,method='SLSQP',constraints=[LinearConstraint(small,target,np.inf)],options=dict(ftol=64*EPS*(len(a)+2),maxiter=MAX_ITERATIONS))
    coordinate=result.x.copy();trace=[]
    # Fixed original-row arithmetic polishing, not new physical tolerances.
    for iteration in range(MAX_REFINEMENTS):
        values=small@coordinate;limits=64*EPS*(len(a)+2)*np.maximum(np.abs(target),np.linalg.norm(small,axis=1)*np.linalg.norm(coordinate));active=np.flatnonzero(values-target<=limits)
        trace.append(dict(iteration=iteration,active_constraints=active.tolist()))
        if len(active):coordinate+=np.linalg.lstsq(small[active],target[active]-values[active],rcond=cutoff)[0]
    correction=(coordinate@basis)*coordinate_scale;candidate=u+correction;nc=norm(correction);nd=norm(candidate);inequalities=[]
    for i in range(len(a)+2):
        row=a[i] if i<len(a) else (-m if i==len(a) else -s)
        value,error=dot(row,correction);limit=max(error,STEP_EPS*EPS*max(abs(float(rhs[i])),norm(row)*nc));slack=value-float(rhs[i])
        inequalities.append(dict(index=i,kind='protected_margin' if i<len(a) else 'joint_class_nonincrease',target=float(rhs[i]),linear_recovery=value,slack=slack,arithmetic_error=error,residual_limit=limit,passed=bool(slack>=-limit)))
    classes=[]
    for name,g in [('M',m),('S',s)]:
        value,error=dot(g,candidate);resolution=STEP_EPS*EPS*norm(g)*nd;classes.append(dict(class_name=name,linear_change=value,arithmetic_error=error,resolution=resolution,resolved_negative=bool(value<-max(error,resolution))))
    passed=bool(result.success) and all(v['passed'] for v in inequalities) and all(v['resolved_negative'] for v in classes)
    return dict(status='one_sided_joint_restoration_requires_full_actual_finite_guard' if passed else 'local_inequality_or_common_descent_unqualified_stop',displacement=candidate,correction=correction,optimizer_success_reported=bool(result.success),optimizer_message=str(result.message),optimizer_iterations=int(result.nit),normal_rank=rank,normalized_singular_values=singular.tolist(),relative_rank_cutoff=cutoff,coordinate_scale=coordinate_scale,normal_positive_scales=scales.tolist(),fixed_refinements=MAX_REFINEMENTS,refinement_trace=trace,inequality_reviews=inequalities,class_reviews=classes,correction_norm=nc,candidate_displacement_norm=nd,relative_correction_norm=nc/norm(u) if norm(u) else None,finite_step_authority=False,QP_optimality_claim=False,no_global_infeasibility_claim=True,no_parameter_update=True)

def propose(displacement,normals,origin_margins,current_margins,gm,gs):
    origin=np.asarray(origin_margins,np.float64)
    if origin.ndim!=1 or not np.isfinite(origin).all() or np.any(origin<0):
        raise ValueError('Matched nonnegative actual origin margins required')
    result=_joint_propose(displacement,normals,np.zeros_like(origin),current_margins,gm,gs)
    result['local_protection_target']='exact_zero_decision_floor'
    result['actual_origin_margins_for_identity_only']=origin.tolist()
    result['actual_argmax_all_original_rows_and_retention_guards_still_required']=True
    result['confidence_restoration_not_required_by_this_local_policy']=True
    return result
