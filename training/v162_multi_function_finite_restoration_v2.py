"""Full-space minimum-norm residual recovery, unchanged finite acceptance.

Original normals and margins share positive row units. Direct wide least
squares avoids squaring the condition number in a Gram system. Its local
residual certificate is not a proof of global feasibility or safe inference.
"""
import numpy as np
from v159_float64_repeat_policy_v2 import EPS,STEP_EPS
from v160_independent_saved_direction_certificate import dot,norm
MAX_NORMALS=24
MAX_REFINEMENTS=3

def propose(displacement,normals,base_margins,current_margins,gm,gs):
    u,a,b,c,m,s=[np.asarray(v,np.float64) for v in [displacement,normals,base_margins,current_margins,gm,gs]]
    if u.ndim!=1 or not u.size or m.shape!=u.shape or s.shape!=u.shape or a.ndim!=2 or a.shape[1]!=u.size or not 0<len(a)<=MAX_NORMALS or b.shape!=c.shape or b.shape!=(len(a),) or np.any(b<0) or not all(np.isfinite(v).all() for v in [u,a,b,c,m,s]):raise ValueError('Bounded complete original-unit arrays and correct base margins required')
    if not np.any(c<0):return dict(status='no_actual_negative_protected_margin_to_restore_stop',finite_step_authority=False)
    scales=np.max(np.abs(a),axis=1)
    if np.any(scales==0):return dict(status='zero_actual_normal_stop',finite_step_authority=False)
    target=b-c;normalized=a/scales[:,None];rhs=target/scales
    correction,_,rank,singular=np.linalg.lstsq(normalized,rhs,rcond=EPS*len(a))
    for _ in range(MAX_REFINEMENTS):
        residual=np.array([rhs[i]-dot(row,correction)[0] for i,row in enumerate(normalized)])
        extra,_,_,_=np.linalg.lstsq(normalized,residual,rcond=EPS*len(a));correction+=extra
    candidate=u+correction;reviews=[];unorm=norm(candidate)
    for name,g in [('M',m),('S',s)]:
        value,error=dot(g,candidate);resolution=STEP_EPS*EPS*norm(g)*unorm
        reviews.append(dict(class_name=name,linear_change=value,arithmetic_error=error,resolution=resolution,resolved_negative=value<-max(error,resolution)))
    residual_reviews=[];correction_norm=norm(correction)
    for i,row in enumerate(a):
        recovery,error=dot(row,correction)
        # Original unit certificate relative to the actual rhs and normal.
        limit=max(error,STEP_EPS*EPS*max(abs(float(target[i])),norm(row)*correction_norm))
        residual_reviews.append(dict(index=i,target=float(target[i]),linear_recovery=recovery,arithmetic_error=error,residual_limit=limit,passed=bool(abs(recovery-target[i])<=limit)))
    passed=all(r['passed'] for r in residual_reviews) and all(r['resolved_negative'] for r in reviews)
    return dict(status='linear_restoration_candidate_requires_full_actual_finite_guard' if passed else 'local_restoration_residual_or_common_descent_failed_stop',displacement=candidate,correction=correction,base_margins=b.tolist(),current_margins=c.tolist(),normal_rank=int(rank),normalized_singular_values=singular.tolist(),relative_rank_cutoff=EPS*len(a),fixed_refinements=MAX_REFINEMENTS,class_reviews=reviews,residual_reviews=residual_reviews,correction_norm=correction_norm,candidate_displacement_norm=unorm,relative_correction_norm=correction_norm/norm(u) if norm(u) else None,finite_step_authority=False,QP_optimality_claim=False,no_global_infeasibility_claim=True,no_parameter_update=True)
