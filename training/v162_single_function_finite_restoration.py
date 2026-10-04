"""One actual nonlinear residual, fixed qualified normal, complete displacement.

This is a new finite restoration mechanism, not the old homogeneous QP or
roundoff polish. No model, direction scan, gradient or update is performed.
"""
import math
import numpy as np
from v159_float64_repeat_policy_v2 import EPS,STEP_EPS
from v160_independent_saved_direction_certificate import dot as dot_review,norm

def propose(displacement,normal,base_margin,failed_margin,gm,gs):
    u,a,m,s=[np.asarray(v,np.float64) for v in [displacement,normal,gm,gs]]
    if u.ndim!=1 or not u.size or any(v.shape!=u.shape for v in [a,m,s]) or not all(np.isfinite(v).all() for v in [u,a,m,s]) or not np.isfinite(base_margin) or not np.isfinite(failed_margin):raise ValueError('Complete matched finite original-unit arrays and margins required')
    if base_margin<=0 or failed_margin>=0:return dict(status='no_actual_negative_protected_margin_to_restore_stop',finite_step_authority=False)
    norm_squared=math.fsum(float(v)*float(v) for v in a)
    if not norm_squared>0:return dict(status='zero_actual_normal_stop',finite_step_authority=False)
    residual=float(base_margin-failed_margin)
    correction=(residual/norm_squared)*a
    candidate=u+correction
    unorm=norm(candidate);class_reviews=[]
    for name,g in [('M',m),('S',s)]:
        value,error=dot_review(g,candidate);resolution=STEP_EPS*EPS*norm(g)*unorm
        class_reviews.append(dict(class_name=name,linear_change=value,arithmetic_error=error,resolution=resolution,resolved_negative=value<-max(error,resolution)))
    reproduced,error=dot_review(a,correction)
    residual_review=dict(requested_margin_recovery=residual,actual_linear_margin_recovery=reproduced,arithmetic_error=error,relative_reconstruction_error=abs(reproduced-residual)/residual)
    # This bound validates arithmetic reconstruction only; no margin tolerance
    # is supplied to the actual classification gate.
    reconstruction_pass=abs(reproduced-residual)<=max(error,STEP_EPS*EPS*residual)
    passed=reconstruction_pass and all(r['resolved_negative'] for r in class_reviews)
    return dict(status='linear_restoration_candidate_requires_full_actual_finite_guard' if passed else 'restoration_linear_descent_or_reconstruction_failed_stop',displacement=candidate,correction=correction,base_margin=float(base_margin),failed_margin=float(failed_margin),normal_norm_squared=norm_squared,correction_norm=norm(correction),candidate_displacement_norm=unorm,relative_correction_norm=norm(correction)/norm(u) if norm(u) else None,residual_review=residual_review,class_reviews=class_reviews,finite_step_authority=False,QP_optimality_claim=False,old_QP_certificate_applies_only_to_base_direction=True,no_parameter_update=True)
